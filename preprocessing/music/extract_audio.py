# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import moviepy
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.editor import VideoFileClip

video_pth = 'data/music/video'

sound_list = os.listdir(video_pth)

save_pth = 'data/music/audio'

print("Total videos: ", len(sound_list))

for sound in sound_list:
    audio_pth = os.path.join(video_pth, sound)
    lis = os.listdir(audio_pth)
    if not os.path.exists(os.path.join(save_pth, sound)):
        os.makedirs(os.path.join(save_pth, sound))
    exist_lis = os.listdir(os.path.join(save_pth, sound))
    for audio_id in lis:
        name = os.path.join(video_pth, sound, audio_id)
        video = VideoFileClip(name)
        audio = video.audio
        audio_name = audio_id[:-4] + '.wav'
        if audio_name in exist_lis:
            print("already exist!")
            continue
        audio.write_audiofile(os.path.join(save_pth, sound, audio_name), fps=16000)
        print("finish video id: " + audio_name)


