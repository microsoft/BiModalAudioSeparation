# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pytube import YouTube
import os
import subprocess
import argparse
import json
import contextlib
import subprocess as sp
import warnings

def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download audios from YouTube."
    )
    parser.add_argument(
        "--source_file",
        type=str,
        default="data/audiocaps/annotations/parsed_all_caps.json",
        help="input filename",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        default="data/audiocaps/audio",
        help="output directory",
    )
    parser.add_argument(
        "--duration", type=int, default=10, help="audio duration"
    )

    parser.add_argument(
        "--rate", type=int, default=16000, help="sampling rate"
    )

    return parser.parse_args(args=args, namespace=namespace)


def ignore_exceptions(func):
    """Decorator that ignores all errors and warnings."""

    def inner(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return func(*args, **kwargs)
            except Exception:
                return None

    return inner


def suppress_outputs(func):
    """Decorator that suppresses writing to stdout and stderr."""

    def inner(*args, **kwargs):
        devnull = open(os.devnull, "w")
        with contextlib.redirect_stdout(devnull):
            with contextlib.redirect_stderr(devnull):
                return func(*args, **kwargs)

    return inner


@suppress_outputs
@ignore_exceptions
def download_audio(youtube_id, start_time, duration, sampling_rate, output_filename):
    # Download video from YouTube
    youtube_url = "https://www.youtube.com/watch?v={}".format(youtube_id)
    yt = YouTube(youtube_url)
    stream = yt.streams.filter(only_audio=True).first()
    download_path = stream.download()

    # Define output path for the trimmed and resampled audio
    output_path = f"{output_filename}.wav"

    # Command to extract, trim, and resample the audio using ffmpeg
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', download_path,  # Input video file
        '-ss', str(start_time),  # Start time
        '-t', str(duration),  # Duration
        '-ar', str(sampling_rate),  # Sampling rate
        '-ac', '1',  # Set audio channels to 1 (mono)
        '-vn',  # No video
        '-y',  # Overwrite output file if it exists
        output_path  # Output audio file
    ]
    
    # Execute the command
    subprocess.run(ffmpeg_cmd)

    # Remove the original download
    os.remove(download_path)

    print(f"Audio extracted and saved to {output_path}")

# Example usage
args = parse_args()

with open(args.source_file) as f:
    source_json = json.load(f)

for k, v in source_json.items():
    filename = v["file_id"]
    youtube_id = v["file_id"].split('_')[0]
    start_time = v["file_id"].split('_')[1]

    path = os.path.join(args.out_dir, filename)
    download_audio(youtube_id, start_time=start_time, duration=args.duration, sampling_rate=args.rate, output_filename=path)
