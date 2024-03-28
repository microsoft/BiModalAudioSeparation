# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import random

source_file = "./data/annotations/parsed_all_caps.json"
train_id_path = "./data/annotations/sample_train_ids.csv"
test_id_path =  "./data/annotations/sample_test_sep2_ids.csv"
test_fraction = 0.1
test_repeat = 3

# load source_file
with open(source_file) as f:
    source_json = json.load(f)

# extract all ids
all_ids = [k for k, v in source_json.items()]
N = len(all_ids)
train_N = int(N * (1.0 - test_fraction))
test_N = int(N * test_fraction)

# split train and test ids
test_ids = random.sample(all_ids, test_N)
train_ids = [id for id in all_ids if id not in test_ids]

# make sure test id file not present in train set
# extract all train file_ids
train_file_ids = [source_json[id]['file_id'] for id in train_ids]
test_ids = [id for id in test_ids if source_json[id]['file_id'] not in train_file_ids]

# write train ids
with open(train_id_path, 'w') as f:
    for train_id in train_ids:
        f.write(train_id + '\n')

# make a combination of test set
with open(test_id_path, 'w') as f:
    for _ in range(test_repeat):
        for test_id in test_ids:
            indexN = random.randint(0, len(test_ids)-1)
            f.write(test_id + ',' + test_ids[indexN] + '\n' )