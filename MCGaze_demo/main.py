import subprocess
import os

# 定义 ROOT_DIR
ROOT_DIR = '/app/Desktop/dataset-weiyan-latest-gaze-26'

# 定义脚本执行顺序
scripts = [
    '/app/Desktop/MCGaze/MCGaze_demo/head_det.py',
    '/app/Desktop/MCGaze/MCGaze_demo/refill.py',
    '/app/Desktop/MCGaze/MCGaze_demo/demo.py',
    '/app/Desktop/MCGaze/MCGaze_demo/make_again.py',
    # '/app/Desktop/MCGaze/MCGaze_demo/combine_sgmt.py',
]

# 遍历 ROOT_DIR 下的一级子目录
for subdir in os.listdir(ROOT_DIR):
    BASE_DIR = os.path.join(ROOT_DIR, subdir)
    
    # 只处理文件夹，跳过非目录项
    if not os.path.isdir(BASE_DIR):
        continue

    # 获取子文件夹中的所有文件
    files = os.listdir(BASE_DIR)

    # 检查子文件夹中是否存在 -gaze.mp4 文件
    new_mp4_exists = any(file.endswith('-gaze.mp4') for file in files)

    # 如果不存在 -gaze.mp4 文件，则运行脚本
    if not new_mp4_exists:
        print(f"No -gaze.mp4 file found in {BASE_DIR}. Executing scripts...")
        os.environ['BASE_DIR'] = BASE_DIR  # 设置 BASE_DIR 环境变量
        # 挨个执行脚本
        for script in scripts:
            print(f"Executing {script} with BASE_DIR = {BASE_DIR}")
            result = subprocess.run(['python3', script], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"{script} executed successfully.")
            else:
                print(f"Error executing {script}. Error: {result.stderr}")
    else:
        print(f"-gaze.mp4 file found in {BASE_DIR}, skipping execution.")
