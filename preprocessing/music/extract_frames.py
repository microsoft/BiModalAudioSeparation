# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import shutil
import subprocess
import os
import argparse
import glob


def extract_frames(video, dst):
    command1 = 'ffmpeg '
    command1 += '-i ' + video + " "
    command1 += "-vf "
    command1 += "fps=1/1 "
    command1 += '{0}/%06d.jpg'.format(dst)
    print(command1)
    os.system(command1)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', dest='out_dir', type=str, default='data/music/frames')
    parser.add_argument('--video_path', dest='video_path', type=str, default='data/music/videos')
    args = parser.parse_args()

    vid_list = os.listdir(args.video_path)

    for vid in vid_list:
        vid_pth = os.path.join(args.video_path, vid)
        lis = os.listdir(vid_pth)
        if not os.path.exists(os.path.join(args.out_dir, vid)):
            os.makedirs(os.path.join(args.out_dir, vid))

        for vid_id in lis:
            name = os.path.join(args.video_path, vid, vid_id)
            dst = os.path.join(args.out_dir, vid, vid_id)
            print(dst)
            if not os.path.exists(dst):
                os.makedirs(dst)
            extract_frames(name, dst)
            print("finish video id: " + vid_id)

