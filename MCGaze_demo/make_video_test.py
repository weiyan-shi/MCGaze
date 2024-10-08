import os
import re

# 自然排序的辅助函数
def natural_sort_key(filename):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', filename)]

# 定义图片文件夹路径
image_folder = '/app/Desktop/Dataset/pcit1/new_frames'

# 获取所有图片文件名，按自然顺序排序
images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")], key=natural_sort_key)

# 提取文件名中的数字部分
def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

# 检查图片是否连续
def check_image_continuity(images):
    previous_number = None
    for image in images:
        image_number = extract_number(image)
        if image_number is not None:
            if previous_number is not None and image_number != previous_number + 1:
                print(f"Image sequence broken between {previous_number}.jpg and {image_number}.jpg")
            previous_number = image_number
        else:
            print(f"Could not extract number from image: {image}")

# 如果没有图片，给出提示
if not images:
    print(f"Error: No images found in folder: {image_folder}")
else:
    # 调用函数检查图片是否连续
    check_image_continuity(images)
