# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import random
import numpy as np
import csv
import torch
import json
from transformers import RobertaTokenizer
import librosa
import torchaudio
import torch.utils.data as torchdata
import torchaudio.transforms as Transforms 

class VGGSoundDataset(torchdata.Dataset):
    def __init__(self, list_sample, num_mix, opt, split):
        super(VGGSoundDataset, self).__init__()
            
        self.num_mix = num_mix
        self.audLen = opt.audLen
        self.audRate = opt.audRate
        self.n_sources = opt.n_sources
        self.tokenize = RobertaTokenizer.from_pretrained('roberta-base')
        self.audio_dir = opt.audio_dir
        self.split = split
        self.seed = opt.seed
        random.seed(self.seed)

        if self.split == "test":
            solo_list = "data/vggsound/annotations/test.csv"
            self.test_files = []
            for row in csv.reader(open(solo_list, 'r'), delimiter=','):
                if len(row) < 2:
                    continue
                self.test_files.append([row[0], ",".join(row[1:])])

        ##############################
        # list_sample can be a python list or a csv file of list
        if isinstance(list_sample, str):
            self.list_sample = []
            for row in csv.reader(open(list_sample, 'r'), delimiter=','):
                self.list_sample.append([row[0], ",".join(row[1:])])
            
            if self.split == "train":
                self.list_sample = self.list_sample

        num_sample = len(self.list_sample)
        assert num_sample > 0
        print('# {} samples: {}'.format(self.split, num_sample))

    def __len__(self):
        return len(self.list_sample)

    def tokenizer(self, text):
        result = self.tokenize(
            text,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in result.items()}

    def dummy_data(self):
        ret_dict = {}
        
        ret_dict['mix'] = torch.zeros(self.audLen)
        ret_dict['audios'] = torch.zeros(self.num_mix, self.audLen)
        ret_dict['conds'] = {'input_ids': torch.zeros(self.num_mix, 77), 'attention_mask': torch.zeros(self.num_mix, 77)}

        if self.split == "train":
            ret_dict['source_conds'] = {'input_ids': torch.zeros(self.num_mix, self.n_sources, 77), 'attention_mask': torch.zeros(self.num_mix, self.n_sources, 77)}
            ret_dict["source_weights"] = torch.zeros(self.num_mix, self.n_sources)
        
        return ret_dict

    def _load_audio_file(self, path):
        if path.endswith('.mp3'):
            audio_raw, rate = torchaudio.load(path)
            audio_raw = audio_raw.numpy().astype(np.float32)

            # range to [-1, 1]
            audio_raw *= (2.0**-31)

            # convert to mono
            if audio_raw.shape[1] == 2:
                audio_raw = (audio_raw[:, 0] + audio_raw[:, 1]) / 2
            else:
                audio_raw = audio_raw[:, 0]
        else:
            audio_raw, rate = torchaudio.load(path)
            
            # convert to mono
            if audio_raw.shape[0] == 2:
                audio_raw = (audio_raw[0, :] + audio_raw[1, :]) / 2
            
            if len(audio_raw.shape) == 2:
                audio_raw = audio_raw[0]

            if rate != self.audRate:
                resampler = Transforms.Resample(rate, self.audRate, dtype=audio_raw.dtype).to(audio_raw.device, non_blocking=True)
                audio_raw = resampler(audio_raw)
                rate = self.audRate

        return audio_raw, rate

    def _load_audio(self, path, center_timestamp, nearest_resample=False):
        audio = np.zeros(self.audLen, dtype=np.float32)

        # silent
        if path.endswith('silent'):
            return audio

        # load audio
        audio_raw, rate = self._load_audio_file(path)

        if len(audio_raw.shape) == 2:
            audio_raw, rate = self._load_audio_file(path)

        # repeat if audio is too short
        if audio_raw.shape[0] < self.audLen:
            n = int(self.audLen / audio_raw.shape[0]) + 1
            audio_raw = np.tile(audio_raw, n)

        # crop N seconds
        len_raw = audio_raw.shape[0]
        center = int(center_timestamp * self.audRate)

        if center > len_raw:
            center = len_raw // 2

        start = max(0, center - self.audLen // 2)
        end = min(len_raw, center + self.audLen // 2)

        audio[self.audLen//2-(center-start): self.audLen//2+(end-center)] = \
            audio_raw[start:end]

        # randomize volume
        if self.split == 'train':
            scale = random.random() + 0.5     # 0.5-1.5
            audio *= scale
        audio[audio > 1.] = 1.
        audio[audio < -1.] = -1.

        return audio

    def _mix_n(self, audios):
        # mix N sounds
        N = len(audios)
    
        for n in range(N):
            audios[n] /= N

        audio_mix = sum(audios)

        return audio_mix

    def get_text_prompt(self, label):
        """Get the text prompt for a label."""
        if "_" in label:
            label = " ".join(label.split("_"))
        return f"The sound of {label}"
    
    def get_text_mix_prompt(self, labels):
        """Get the text prompt for mixture of labels."""
        for n in range(len(labels)):
            if "_" in labels[n]:
                labels[n] = " ".join(labels[n].split("_"))        
        labels = " and ".join(labels)

        return f"The sound mixture of {labels}"


    def __getitem__(self, index):
        N = self.num_mix
        P = self.n_sources

        if self.split == "test":
            P = 1

        T = N * P

        audios = [None for _ in range(T)]
        infos = [[] for _ in range(T)]
        classes = [[] for n in range(T)]
        source_prompts = [[] for n in range(T)]

        path_audios = ['' for n in range(T)]
        center_times = [0 for n in range(T)]
        class_list = []

        mix_conds = [[] for n in range(N)]
        mix_audios = [[] for n in range(N)]
        source_conds_mix = [[] for n in range(N)]
        source_conds_weights = [[1.0 for p in range(P)] for n in range(N)]
        

        if self.split == 'train':
            # the first mixture audio info
            infos[0] = self.list_sample[index]
            cls = infos[0][1]
            class_list.append(cls)

            for n in range(1, T):
                indexN = random.randint(0, len(self.list_sample)-1)
                sample = self.list_sample[indexN]
                while sample[1] in class_list:
                    indexN = random.randint(0, len(self.list_sample) - 1)
                    sample = self.list_sample[indexN]
                infos[n] = sample
                class_list.append(sample[1])
        else:
            samples = self.list_sample[index]

            for n in range(N):
                sample = samples[n].replace(" ", "")
                for i in range(len(self.test_files)):
                    data = self.test_files[i]
                    
                    if sample in data:
                        infos[n] = data
                        break

        for n, infoN in enumerate(infos):
            #print(infoN)
            path_audioN, _ = infoN
            path_audios[n] = os.path.join("data/vggsound/audio", path_audioN)
            center_times[n] = 5.0

        # # load frames and audios, STFT
        try:
            for n, infoN in enumerate(infos):
                prompt = self.get_text_prompt(
                    infoN[1]
                )
                classes[n] = infoN[1]
                source_prompts[n] = prompt
                center_timeN = center_times[n] - 0.5
                audios[n] = self._load_audio(path_audios[n], center_timeN)

            mix = self._mix_n(audios)

            if self.split == 'train':
                for n in range(N):
                    mix_conds[n] = self.get_text_mix_prompt([classes[n] for n in range(n * P, (n+1) * P)])
                    source_conds_mix[n] = source_prompts[n*P : (n+1)*P]
                    mix_audios[n] = self._mix_n([audios[n] for n in range(n * P, (n+1) * P)])

                tokenized_mix_conds = self.tokenizer(mix_conds)

                tokenized_source_conds = {}
                for k, v in tokenized_mix_conds.items():
                    tokenized_source_conds[k] = []

                for n in range(N):
                    tokens = self.tokenizer(source_conds_mix[n])
                    
                    for k, v in tokens.items():
                        tokenized_source_conds[k].append(v.unsqueeze(0))

                for k, v in tokenized_source_conds.items():
                    tokenized_source_conds[k] = torch.cat(v, dim=0)
            
                ret_dict = {'mix': torch.tensor(mix), 'conds': tokenized_mix_conds, 'source_conds': tokenized_source_conds, 'source_weights': torch.tensor(source_conds_weights)}        
                ret_dict['audios'] = torch.tensor(np.array(mix_audios))

            else:
                for n in range(N):
                    mix_conds[n] = self.get_text_prompt(classes[n])
                
                tokenized_mix_conds = self.tokenizer(mix_conds)
                
                ret_dict = {'mix': torch.tensor(mix), 'conds': tokenized_mix_conds}
            
                ret_dict['audios'] = torch.tensor(np.array(audios))

            return ret_dict

        except Exception as e:
            print('Failed loading audio: {}'.format(e))

            # create dummy data
            return self.dummy_data()