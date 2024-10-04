import os
import shutil

# 文件夹路径
frames_folder = 'frames'       # 这里替换为你的frames文件夹路径
newframes_folder = 'new_frames' # 这里替换为你的newframes文件夹路径

# 获取frames和newframes中的文件名
frames_files = set(os.listdir(frames_folder))
newframes_files = set(os.listdir(newframes_folder))

# 找出newframes中缺失的文件
missing_files = frames_files - newframes_files

# 将缺失的文件从frames复制到newframes
for file_name in missing_files:
    source = os.path.join(frames_folder, file_name)
    destination = os.path.join(newframes_folder, file_name)
    shutil.copy(source, destination)
    print(f'Copied {file_name} to newframes.')

print(f'Total missing files copied: {len(missing_files)}')
