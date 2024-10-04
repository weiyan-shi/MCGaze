import sys
import os
import cv2

# Configuration section
# You can modify these paths and settings as needed
BASE_DIR = '/app/Desktop/Dataset/pcit2'
YOLO_HEAD_PATH = '/app/Desktop/MCGaze/MCGaze_demo/yolo_head'
VIDEO_PATH = os.path.join(BASE_DIR, 'pcit2.mp4')
FRAMES_DIR = os.path.join(BASE_DIR, 'frames')
NEW_FRAMES_DIR = os.path.join(BASE_DIR, 'new_frames')
RESULT_LABELS_DIR = os.path.join(BASE_DIR, 'result', 'labels')

# Add yolo_head path to system path
sys.path.insert(0, YOLO_HEAD_PATH)
sys.path.append(YOLO_HEAD_PATH)

from yolo_head.detect import det_head

# Function to delete all files in a folder
def delete_files_in_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist")
        return

    files = os.listdir(folder_path)
    for file in files:
        file_path = os.path.join(folder_path, file)

        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
        elif os.path.isdir(file_path):
            delete_files_in_folder(file_path)

# Clean up directories
delete_files_in_folder(RESULT_LABELS_DIR)
delete_files_in_folder(FRAMES_DIR)
delete_files_in_folder(NEW_FRAMES_DIR)

# Capture frames from video
cap = cv2.VideoCapture(VIDEO_PATH)
frame_id = 0

while True:
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(os.path.join(FRAMES_DIR, f'{frame_id}.jpg'), frame)
        frame_id += 1
    else:
        break

# Run head detection on the saved frames
imgset = os.path.join(FRAMES_DIR, '*.jpg')
det_head(imgset, os.path.join(BASE_DIR, 'result'))
