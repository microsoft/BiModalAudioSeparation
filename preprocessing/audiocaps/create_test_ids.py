# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import csv 
import os
import librosa
import random


csv_lis_path = "/home/tm36864/Music/CLIPSep/data/audiocaps/annotations/test.csv"
audio_path = "/home/tm36864/Music/CLIPSep/data/audiocaps/audio"

i = 0

# file_path = "/home/tm36864/Music/CLIPSep/data/audiocaps/annotations/train_ids.csv"

test_file_ids = []

for row in csv.reader(open(csv_lis_path, 'r'), delimiter=','):
    if i == 0:
        i += 1
        continue
    
    ac_id = row[0]
    name = row[1] + "_" + row[2]

    if os.path.isfile(os.path.join(audio_path, name + ".wav")):
        try:
            audio_raw, rate = librosa.load(os.path.join(audio_path, name + ".wav"), sr=None, mono=True)
            test_file_ids.append(ac_id)
        except:
            continue

csv_lis_path = "/home/tm36864/Music/CLIPSep/data/audiocaps/annotations/val.csv"
audio_path = "/home/tm36864/Music/CLIPSep/data/audiocaps/audio"

i = 0

# file_path = "/home/tm36864/Music/CLIPSep/data/audiocaps/annotations/train_ids.csv"

val_file_ids = []

for row in csv.reader(open(csv_lis_path, 'r'), delimiter=','):
    if i == 0:
        i += 1
        continue
    
    ac_id = row[0]
    name = row[1] + "_" + row[2]

    if os.path.isfile(os.path.join(audio_path, name + ".wav")):
        try:
            audio_raw, rate = librosa.load(os.path.join(audio_path, name + ".wav"), sr=None, mono=True)
            val_file_ids.append(ac_id)
        except:
            continue

file_path = "/home/tm36864/Music/CLIPSep/data/audiocaps/annotations/test_sep2_ids.csv"

with open(file_path, 'w') as f:
    for test_id in test_file_ids:
        indexN = random.randint(0, len(val_file_ids)-1)
        f.write(test_id + ',' + val_file_ids[indexN] + '\n' )