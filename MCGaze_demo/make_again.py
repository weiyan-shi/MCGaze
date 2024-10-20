import cv2
import os
import re

# 自然排序的辅助函数
def natural_sort_key(filename):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', filename)]

# 定义 base 目录和 pcit 名称
BASE_DIR = os.getenv('BASE_DIR', '/app/Desktop/Dataset/pcit7')
VIDEO_NAME = os.path.basename(BASE_DIR)

# 定义图片文件夹路径和其他路径
image_folder = os.path.join(BASE_DIR, 'onlyhead-segment')
output_video = os.path.join(BASE_DIR, f'{VIDEO_NAME}-bbox.mp4')
original_video = os.path.join(BASE_DIR, f'{VIDEO_NAME}.mp4')

# 获取原视频的帧率
cap = cv2.VideoCapture(original_video)
fps = round(cap.get(cv2.CAP_PROP_FPS))
cap.release()  # 记得释放视频文件

print(f"Original video FPS: {fps}")

# 获取所有图片文件名，按自然顺序排序
images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")], key=natural_sort_key)
# 确保图片列表非空
if not images:
    print("No images found in the folder.")
else:
    # 读取第一张图片来获取视频帧的宽高
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # 定义视频编码和帧率 (FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码器
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # 将每张图片写入视频
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    # 释放视频写入对象
    video.release()
    print(f'Video saved as {output_video}')
