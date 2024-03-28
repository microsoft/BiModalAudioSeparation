# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import csv 
import os
import librosa

csv_lis_path = "./data/annotations/train.csv"
audio_path = "./data/audio"

i = 0

file_path = "./data/annotations/train_ids.csv"

with open(file_path, 'w') as f:
    for row in csv.reader(open(csv_lis_path, 'r'), delimiter=','):
        if i == 0:
            i += 1
            continue
        
        ac_id = row[0]
        name = row[1] + "_" + row[2]

        if os.path.isfile(os.path.join(audio_path, name + ".wav")):
            try:
                audio_raw, rate = librosa.load(os.path.join(audio_path, name + ".wav"), sr=None, mono=True)
                f.write(ac_id + '\n')
            except:
                continue