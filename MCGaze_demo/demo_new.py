#!/usr/bin/env python
# coding: utf-8

import cv2
from facenet_pytorch import MTCNN
import os
import json
from mmdet.apis import init_detector
from mmdet.datasets.pipelines import Compose
import torch
from mmcv.parallel import collate, scatter
import numpy as np

# Initialize variables
frame_id = 0
person_num = 0
video_clip = None
video_clip_set = []
vid_len = len(os.listdir('/app/Desktop/MCGaze/MCGaze_demo/frames'))

# Step 1: Process each frame and organize face bounding boxes
while frame_id < vid_len:
    frame = cv2.imread('/app/Desktop/MCGaze/MCGaze_demo/frames/%d.jpg' % frame_id)
    w, h, c = frame.shape
    txt_path = '/app/Desktop/MCGaze/MCGaze_demo/result/labels/%d.txt' % frame_id
    try:
        f = open(txt_path, 'r')
    except FileNotFoundError:
        print(f"File not found: {txt_path}, skipping...")
        frame_id += 1
        continue

    face_bbox = []
    for line in f.readlines():
        line = line.strip()
        line = line.split(' ')
        for i in range(len(line)):
            line[i] = eval(line[i])
        if line[0] == 1:
            face_bbox.append([(line[1]), (line[2]), (line[3]), (line[4])])
    f.close()

    if face_bbox is not None:
        face_bbox = sorted(face_bbox, key=lambda x: x[0])
        cur_person_num = len(face_bbox)
    else:
        cur_person_num = 0

    if cur_person_num != person_num:
        if video_clip is None:
            video_clip = {'frame_id': [], 'person_num': cur_person_num}
            video_clip['frame_id'].append(frame_id)
            for i in range(cur_person_num):
                video_clip['p' + str(i)] = [face_bbox[i]]
        else:
            video_clip_set.append(video_clip)
            video_clip = {'frame_id': [], 'person_num': cur_person_num}
            video_clip['frame_id'].append(frame_id)
            for i in range(cur_person_num):
                video_clip['p' + str(i)] = [face_bbox[i]]
    else:
        video_clip['frame_id'].append(frame_id)
        for i in range(cur_person_num):
            video_clip['p' + str(i)].append(face_bbox[i])
    person_num = cur_person_num
    frame_id += 1

video_clip_set.append(video_clip)

# Initialize gaze detection model
model = init_detector(
    '/app/Desktop/MCGaze/configs/multiclue_gaze/multiclue_gaze_r50_l2cs.py',
    '/app/Desktop/MCGaze/ckpts/multiclue_gaze_r50_l2cs.pth',
    device="cuda:0",
    cfg_options=None,
)
cfg = model.cfg

test_pipeline = Compose(cfg.data.test.pipeline[1:])

def load_datas(data, test_pipeline, datas):
    datas.append(test_pipeline(data))

def infer(datas, model, clip, i):
    datas = sorted(datas, key=lambda x: x['img_metas'].data['filename'])
    datas = collate(datas, samples_per_gpu=len(datas))
    datas['img_metas'] = datas['img_metas'].data
    datas['img'] = datas['img'].data
    datas = scatter(datas, ["cuda:0"])[0]
    with torch.no_grad():
        (det_bboxes, det_labels), det_gazes = model(
            return_loss=False,
            rescale=True,
            format=False,
            **datas
        )
    gaze_dim = det_gazes['gaze_score'].size(1)
    det_fusion_gaze = det_gazes['gaze_score'].view((det_gazes['gaze_score'].shape[0], 1, gaze_dim))
    clip['gaze_p' + str(i)].append(det_fusion_gaze.cpu().numpy())

# Process each video clip to extract gaze data
gaze_data = {}
max_len = 100

for clip in video_clip_set:
    frame_ids = clip['frame_id']
    person_num = clip['person_num']
    for i in range(person_num):
        head_bboxes = clip['p' + str(i)]
        clip['gaze_p' + str(i)] = []
        datas = []
        for j, frame in enumerate(frame_ids):
            cur_img = cv2.imread("/app/Desktop/MCGaze/MCGaze_demo/frames/" + str(frame) + ".jpg")
            w, h, _ = cur_img.shape
            for xy in head_bboxes[j]:
                xy = int(xy)
            head_center = [int(head_bboxes[j][1] + head_bboxes[j][3]) // 2, int(head_bboxes[j][0] + head_bboxes[j][2]) // 2]
            l = int(max(head_bboxes[j][3] - head_bboxes[j][1], head_bboxes[j][2] - head_bboxes[j][0]) * 0.8)
            head_crop = cur_img[max(0, head_center[0] - l):min(head_center[0] + l, w), max(0, head_center[1] - l):min(head_center[1] + l, h), :]
            w_n, h_n, _ = head_crop.shape
            cur_data = dict(filename=j, ori_filename=111, img=head_crop, img_shape=(w_n, h_n, 3), ori_shape=(2 * l, 2 * l, 3), img_fields=['img'])
            load_datas(cur_data, test_pipeline, datas)
            if len(datas) > max_len or j == (len(frame_ids) - 1):
                infer(datas, model, clip, i)
                datas = []
                if j == (len(frame_ids) - 1):
                    clip['gaze_p' + str(i)] = np.concatenate(clip['gaze_p' + str(i)], axis=0)

    # Save gaze data to dictionary
    for i, frame_id in enumerate(clip['frame_id']):
        if frame_id not in gaze_data:
            gaze_data[frame_id] = {}
        for j in range(person_num):
            gaze_data[frame_id][f'person_{j}'] = {
                'gaze': clip['gaze_p' + str(j)][i].tolist(),
                'head_bbox': [int(coord) for coord in clip['p' + str(j)][i]]
            }

# Step 3: Save the gaze data to a JSON file
output_dir = '/app/Desktop/MCGaze/MCGaze_demo/gaze_output/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(os.path.join(output_dir, 'gaze_data.json'), 'w') as f:
    json.dump(gaze_data, f, indent=4)

print(f"Gaze data saved to {os.path.join(output_dir, 'gaze_data.json')}")
