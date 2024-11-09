#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
from facenet_pytorch import MTCNN
import os
import matplotlib.pyplot as plt
import json

BASE_DIR = os.getenv('BASE_DIR', '/app/Desktop/Dataset/pcit7')
VIDEO_NAME = os.path.basename(BASE_DIR)

# In[2]:


frame_id = 0
person_num = 0
video_clip=None
video_clip_set = []
vid_len = len(os.listdir(os.path.join(BASE_DIR, 'frames')))
while frame_id < vid_len:
    frame = cv2.imread(os.path.join(BASE_DIR, 'frames', f'{frame_id}.jpg'))
    w,h,c = frame.shape
    # txt_path = '/app/Desktop/MCGaze/MCGaze_demo/result/labels/%d.txt' % frame_id
    txt_path = os.path.join(BASE_DIR, 'result/labels', f'{frame_id}.txt')
    try:
        f = open(txt_path, 'r')
    except FileNotFoundError:
        print(f"File not found: {txt_path}, skipping...")
        frame_id += 1
        continue
    #遍历每一行
    face_bbox = []
    for line in f.readlines():
        line = line.strip()
        line = line.split(' ')
        for i in range(len(line)):
            line[i] = eval(line[i])
            #将每一行的数据存入字典
        if line[0]==1:
            face_bbox.append([(line[1]),(line[2]),(line[3]),(line[4])])
    f.close()
    #按第一维排序
    if face_bbox is not None:
        face_bbox = sorted(face_bbox, key= lambda x :x[0])
        cur_person_num = len(face_bbox)
    else:
        cur_person_num = 0
    if cur_person_num != person_num :
        if video_clip==None:
            video_clip={'frame_id': [], 'person_num': cur_person_num}
            video_clip['frame_id'].append(frame_id)
            for i in range(cur_person_num):
                video_clip['p'+str(i)]=[face_bbox[i]]
        else:
            video_clip_set.append(video_clip)
            video_clip={'frame_id': [], 'person_num': cur_person_num}
            video_clip['frame_id'].append(frame_id)
            for i in range(cur_person_num):
                video_clip['p'+str(i)]=[face_bbox[i]]
    else:
        video_clip['frame_id'].append(frame_id)
        for i in range(cur_person_num):
                video_clip['p'+str(i)].append(face_bbox[i])
    person_num = cur_person_num
    frame_id += 1

video_clip_set.append(video_clip)


# In[3]:


from mmdet.apis import init_detector
from mmdet.datasets.pipelines import Compose
import torch
from mmcv.parallel import collate, scatter
import numpy as np
model = init_detector(
        '/app/Desktop/MCGaze/configs/multiclue_gaze/multiclue_gaze_r50_l2cs.py',
        '/app/Desktop/MCGaze/ckpts/multiclue_gaze_r50_l2cs.pth',
        device="cuda:0",
        cfg_options=None,)
cfg = model.cfg


# In[4]:


print(cfg.data.test.pipeline[1:])
test_pipeline = Compose(cfg.data.test.pipeline[1:])

def load_datas(data, test_pipeline, datas):
    datas.append(test_pipeline(data))


# In[5]:


def infer(datas,model,clip,i):
    datas = sorted(datas, key=lambda x:x['img_metas'].data['filename']) # 按帧顺序 img名称从小到大
    datas = collate(datas, samples_per_gpu=len(frame_id)) # 用来形成batch用的
    datas['img_metas'] = datas['img_metas'].data
    datas['img'] = datas['img'].data
    datas = scatter(datas, ["cuda:0"])[0]
    with torch.no_grad():
        (det_bboxes, det_labels), det_gazes = model(
                return_loss=False,
                rescale=True,
                format=False,# 返回的bbox既包含face_bboxes也包含head_bboxes
                **datas)    # 返回的bbox格式是[x1,y1,x2,y2],根据return_loss函数来判断是forward_train还是forward_test.
    gaze_dim = det_gazes['gaze_score'].size(1)
    det_fusion_gaze = det_gazes['gaze_score'].view((det_gazes['gaze_score'].shape[0], 1, gaze_dim))
    clip['gaze_p'+str(i)].append(det_fusion_gaze.cpu().numpy())

# Process each video clip to extract gaze data
gaze_data = {}
max_len = 100

for clip in video_clip_set:
    frame_id = clip['frame_id']
    person_num = clip['person_num']
    for i in range(person_num):
        head_bboxes = clip['p'+str(i)]
        clip['gaze_p'+str(i)] = []
        datas = []
        for j,frame in enumerate(frame_id):
            cur_img = cv2.imread(os.path.join(BASE_DIR, 'frames', f'{frame}.jpg'))
            w,h,_ = cur_img.shape
            for xy in head_bboxes[j]:
                xy = int(xy)
            head_center = [int(head_bboxes[j][1]+head_bboxes[j][3])//2,int(head_bboxes[j][0]+head_bboxes[j][2])//2]
            l = int(max(head_bboxes[j][3]-head_bboxes[j][1],head_bboxes[j][2]-head_bboxes[j][0])*0.8)
            head_crop = cur_img[max(0,head_center[0]-l):min(head_center[0]+l,w),max(0,head_center[1]-l):min(head_center[1]+l,h),:]
            w_n,h_n,_ = head_crop.shape
            # if frame==0:
            #     plt.imshow(head_crop)
            # print(head_crop.shape)
            cur_data = dict(filename=j,ori_filename=111,img=head_crop,img_shape=(w_n,h_n,3),ori_shape=(2*l,2*l,3),img_fields=['img'])
            load_datas(cur_data,test_pipeline,datas)
            if len(datas)>max_len or j==(len(frame_id)-1):
                infer(datas,model,clip,i)
                datas = []
                if j==(len(frame_id)-1):
                    clip['gaze_p'+str(i)] = np.concatenate(clip['gaze_p'+str(i)],axis=0)

    # Save gaze data to dictionary
    for i, frame_id in enumerate(clip['frame_id']):
        if frame_id not in gaze_data:
            gaze_data[frame_id] = {}
        for j in range(person_num):
            gaze_data[frame_id][f'person_{j}'] = {
                'gaze': clip['gaze_p'+str(j)][i][0].tolist(),
                # 'gaze': clip['gaze_p'+str(j)][i].tolist(),
                'head_bbox': clip['p'+str(j)][i],
            }


with open(os.path.join(BASE_DIR, VIDEO_NAME+'-gaze.json'), 'w') as f:
    json.dump(gaze_data, f, indent=4)

print(f"Gaze data saved to {os.path.join(BASE_DIR, VIDEO_NAME+'-gaze.json')}")


# In[6]:


for vid_clip in video_clip_set:
    for i,frame_id in enumerate(vid_clip['frame_id']):  # 遍历每一帧
        cur_img = cv2.imread(os.path.join(BASE_DIR, 'frames', f"{vid_clip['frame_id'][i]}.jpg"))
        for j in range(vid_clip['person_num']):  # 遍历每一个人
            gaze = vid_clip['gaze_p'+str(j)][i][0]
            head_bboxes = vid_clip['p'+str(j)][i]
            for xy in head_bboxes:
                xy = int(xy)
            head_center = [int(head_bboxes[1]+head_bboxes[3])//2,int(head_bboxes[0]+head_bboxes[2])//2]
            l = int(max(head_bboxes[3]-head_bboxes[1],head_bboxes[2]-head_bboxes[0])*1)
            gaze_len = l*1.0
            thick = 3
            cv2.arrowedLine(cur_img,(head_center[1],head_center[0]),
                        (int(head_center[1]-gaze_len*gaze[0]),int(head_center[0]-gaze_len*gaze[1])),
                        (230,253,11),thickness=thick)
        cv2.imwrite(os.path.join(BASE_DIR, 'new_frames', f'{frame_id}.jpg'), cur_img)
