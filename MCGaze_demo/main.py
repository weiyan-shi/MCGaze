import subprocess
import os

# 定义 BASE_DIR
BASE_DIR = '/app/Desktop/Dataset/pcit2'

# 定义脚本执行顺序
scripts = [
    '/app/Desktop/MCGaze/MCGaze_demo/head_det.py',
    '/app/Desktop/MCGaze/MCGaze_demo/refill.py',
    '/app/Desktop/MCGaze/MCGaze_demo/demo.py',
    '/app/Desktop/MCGaze/MCGaze_demo/make_again.py'
]

# 设置环境变量
os.environ['BASE_DIR'] = BASE_DIR

# 挨个执行脚本
for script in scripts:
    print(f"Executing {script} with BASE_DIR = {BASE_DIR}")
    result = subprocess.run(['python3', script], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"{script} executed successfully.")
    else:
        print(f"Error executing {script}. Error: {result.stderr}")
