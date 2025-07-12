import numpy as np
import os
from moviepy.editor import VideoFileClip, concatenate_videoclips
import subprocess

input_path = "./result_DualTalk/"
output_path = "./result_DualTalk_stitching"

os.makedirs(output_path, exist_ok=True)

for i in os.listdir(input_path):
    if i.endswith("speaker1_render.mp4"):
        video_name1 = os.path.join(input_path, i)
        npy_path1 = i.replace("speaker1_render.mp4", "speaker1.npy")
        i = i.replace("speaker1", "speaker2")
        video_name2 = os.path.join(input_path, i)
        npy_path2 = i.replace("speaker2_render.mp4", "speaker2.npy")
        npy1 = np.load(os.path.join(input_path, npy_path1))
        npy2 = np.load(os.path.join(input_path, npy_path2))
        neck_pose1 = npy1[:, 53:56]
        neck_pose2 = npy2[:, 53:56]
        neck_pose1_mean = np.mean(neck_pose1, axis=0)
        neck_pose2_mean = np.mean(neck_pose2, axis=0)
        video1_path = os.path.join(input_path, video_name1)
        video2_path = os.path.join(input_path, video_name2)
        if neck_pose1_mean[1] > neck_pose2_mean[1]:
            # Use ffmpeg to place video1 on the left, video2 on the right, and merge the two videos together, as well as their audio.
            output_video = os.path.join(output_path, f"combined_{os.path.basename(video1_path)}")
            ffmpeg_command = [
                "ffmpeg",
                "-i", video1_path,
                "-i", video2_path,
                "-filter_complex", "[0:v]pad=iw*2:ih[left];[left][1:v]overlay=w[video];[0:a][1:a]amix=inputs=2[audio]",
                "-map", "[video]",
                "-map", "[audio]",
                output_video
            ]
            subprocess.run(ffmpeg_command, check=True)
        else:
            # If neck_pose2_mean[1] is larger, swap the positions of video1 and video2
            output_video = os.path.join(output_path, f"combined_{os.path.basename(video2_path)}")
            ffmpeg_command = [
                "ffmpeg",
                "-i", video2_path,
                "-i", video1_path,
                "-filter_complex", "[0:v]pad=iw*2:ih[left];[left][1:v]overlay=w[video];[0:a][1:a]amix=inputs=2[audio]",
                "-map", "[video]",
                "-map", "[audio]",
                output_video
            ]
            subprocess.run(ffmpeg_command, check=True)

        print(f"Output video: {output_video}")
