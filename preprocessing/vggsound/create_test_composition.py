# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import csv
import random
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='data/vggsound/annotations',
                        help="path to output index files")
    parser.add_argument('--dum_time', default=2, type=int,
                        help="dummy times")
    parser.add_argument('--num_sources', default=2, type=int,
                        help="dummy times")

    args = parser.parse_args()
    filename = 'test.csv'

    info = []
    with open(os.path.join(args.path, filename), 'r') as f:
        for item in f:
            info.append(item)
    num = len(info)
    print(num)

    N = args.num_sources
    
    with open(os.path.join(args.path, f'test_sep_{N}.csv'), 'w') as sep:
        for k in range(args.dum_time):
            for i in range(num):
                class_list = []
                infos = [[] for n in range(N)]

                infos[0] = info[i]
                cls = ",".join(infos[0].split(',')[1:])
                class_list.append(cls)

                for n in range(1, N):
                    indexN = random.randint(0, (num) - 1)
                    sample = info[indexN]
                    while ",".join(sample.split(',')[1:]) in class_list:
                        indexN = random.randint(0, num - 1)
                        sample = info[indexN]
                    infos[n] = sample
                    class_list.append(",".join(sample.split(',')[1:]))

                s = []
                for n in range(N):
                    s.append(infos[n].split(',')[0])

                s = ",".join(s)
                
                sep.write(s+'\n')