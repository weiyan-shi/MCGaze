# MCGaze demo
This code is inspired by [gaze360-demo](https://colab.research.google.com/drive/1SJbzd-gFTbiYjfZynIfrG044fWi6svbV?usp=sharing) and [yolov5-crowdhuman](https://github.com/deepakcrk/yolov5-crowdhuman), thanks for their efforts for computer vision society.
## Quick start
We have included a test video as video_1.mp4 in this folder. You can use this or any other test video directly.
1. Create folders for the demo run.
   ```bash
   cd MCGaze_demo
   mkdir frames
   mkdir new_frames
   mkdir result
   mkdir result/labels
   ```
2. Download checkpoint for yolov5 head detector from [link](https://drive.google.com/file/d/1gglIwqxaH2iTvy6lZlXuAcMpd_U0GCUb/view?usp=sharing) 
3. Run head_det.py to crop the head image for each person in each frame. The result bboxes will be generated in result/lables.
4. Run refill.py to fill no-face frames
5. Running demo.py to generate the gaze vector.
6. Running demo_new.py to generate the gaze vector json (optional)
7. Running make_again.py to generate new_video.mp4
