import cv2
import os
import re

# 自然排序的辅助函数
def natural_sort_key(filename):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', filename)]

# 定义图片文件夹路径
image_folder = 'new_frames'  # 替换为实际的 newframes 文件夹路径
output_video = 'output_video.mp4'   # 输出视频的名称

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
    fps = 24  # 你可以调整帧率
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # 将每张图片写入视频
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    # 释放视频写入对象
    video.release()
    print(f'Video saved as {output_video}')
