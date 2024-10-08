import json
import numpy as np
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, AudioFileClip
import os

# 定义 base 目录和 pcit 名称
BASE_DIR = os.getenv('BASE_DIR')
VIDEO_NAME = os.path.basename(BASE_DIR)

# 定义图片文件夹路径和其他路径
video_path = os.path.join(BASE_DIR, f'{VIDEO_NAME}-new.mp4')
audio_path = os.path.join(BASE_DIR, f'{VIDEO_NAME}.wav')
segment_path = os.path.join(BASE_DIR, f'{VIDEO_NAME}-segment.json')
output_path = os.path.join(BASE_DIR, f'{VIDEO_NAME}-combine.mp4')



# 加载原始视频
video = VideoFileClip(video_path)

# 加载外部音频
audio = AudioFileClip(audio_path)

# 从 key-event.json 读取片段信息
with open(segment_path, 'r') as f:
    segments = json.load(f)['segments']  # 从 JSON 文件中的 'segments' 键读取数据

# 将时间戳转换为秒，处理格式: "00:03:33,032 - 00:03:47,577"
def time_to_seconds(time_str):
    time_str = time_str.replace(",", ".")  # 将逗号替换为点号处理毫秒
    h, m, s = time_str.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)

# 创建透明蒙版（50%透明度的黄色）
def add_highlight_mask(frame, t, segments):
    for segment in segments:
        start_time_str, end_time_str = segment['time'].split(" - ")  # 使用 " - " 分割开始和结束时间
        start_time = time_to_seconds(start_time_str)
        end_time = time_to_seconds(end_time_str)
        if start_time <= t <= end_time:
            # 添加透明黄色蒙版
            mask = np.full_like(frame, (255, 255, 0), dtype=np.uint8)  # 黄色蒙版
            return (frame * 0.7 + mask * 0.3).astype('uint8')  # 50%透明度
    return frame

# 创建标题叠加函数
def add_title_clip(clip, segments):
    # 创建标题视频层
    clips = [clip]  # 保留原始视频的所有部分
    for segment in segments:
        start_time_str, end_time_str = segment['time'].split(" - ")  # 使用 " - " 分割开始和结束时间
        start_time = time_to_seconds(start_time_str)
        end_time = time_to_seconds(end_time_str)
        
        # 创建黑色字体的标题文本（使用 'description' 而不是 'title'）
        title_text = TextClip(segment['description'], fontsize=20, color='black', bg_color='yellow', method='caption').set_position('top').set_duration(end_time - start_time)
        
        # 创建只有在这个时间段才出现的子视频
        title_subclip = title_text.set_start(start_time).set_end(end_time)
        
        # 将标题层加入列表
        clips.append(title_subclip)
    
    return CompositeVideoClip(clips, size=clip.size)

# 对整个视频应用高亮蒙版，保留音频
highlighted_video = video.fl(lambda gf, t: add_highlight_mask(gf(t), t, segments))

# 添加标题叠加效果
annotated_video = add_title_clip(highlighted_video, segments)

# 将外部音频与视频结合
final_video = annotated_video.set_audio(audio)

# 输出视频，确保音频保留
final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")

print(f"Annotated video saved to {output_path}")