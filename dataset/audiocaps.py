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

class AudioCapsDataset(torchdata.Dataset):
    def __init__(self, list_sample, num_mix, opt, split):
        super(AudioCapsDataset, self).__init__()
            
        self.num_mix = num_mix
        self.audLen = opt.audLen
        self.audRate = opt.audRate
        self.n_sources = opt.n_sources
        self.tokenize = RobertaTokenizer.from_pretrained('roberta-base')
        self.audio_dir = opt.audio_dir
        self.split = split
        self.seed = opt.seed
        random.seed(self.seed)

        with open(opt.parsed_sources_path) as f:
            self.anns_json = json.load(f)

        ##############################
        # list_sample can be a python list or a csv file of list
        if isinstance(list_sample, str):
            self.list_sample = []
            for row in csv.reader(open(list_sample, 'r'), delimiter=','):
                self.list_sample.append(row)

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

    def _mix_n(self, audios):
        # mix N sounds
        N = len(audios)
    
        for n in range(N):
            audios[n] /= N

        audio_mix = sum(audios)

        return audio_mix

    def _load_audio(self, path):
        audio = np.zeros(self.audLen, dtype=np.float32)

        # load audio
        audio_raw, rate = self._load_audio_file(path)

        len_raw = audio_raw.shape[0]

        center = int(len_raw//2)
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

    def __getitem__(self, index):
        N = self.num_mix
        P = self.n_sources

        if self.split == "test":
            P = 1

        audios = [None for n in range(N)]
        infos = [[] for n in range(N)]
        mix_conds = [[] for n in range(N)]
        path_audios = ['' for n in range(N)]

        source_conds_mix = [[] for n in range(N)]
        weights_mix = [[1.0 for p in range(P)] for n in range(N)]

        if self.split == 'train':
            # the first mixture audio info
            infos[0] = self.anns_json[self.list_sample[index][0]]

            # other mixture audio info
            for n in range(1, N):
                indexN = random.randint(0, len(self.list_sample)-1)
                sample = self.list_sample[indexN][0]
                infos[n] =  self.anns_json[sample]
        else:
            ids = self.list_sample[index]
            for i, audio_id in enumerate(ids):
                infos[i] = self.anns_json[audio_id]

        # extracting file ids of each audio
        for n, infoN in enumerate(infos):
            path_audioN = infoN['file_id']
            path_audios[n] = os.path.join(self.audio_dir, path_audioN + ".wav")

        # load audios
        try:
            for n, infoN in enumerate(infos):
                mix_prompt = infoN['caption']
                mix_conds[n] = mix_prompt
                audios[n] = self._load_audio(path_audios[n])

                if self.split == "train":
                    # extract audio source n_sources and weights from each mixture
                    sources = infoN["sources"]
                    
                    # adjusting for the target number of sources for batch training
                    if len(sources) == P:
                        source_conds_mix[n] = sources
                    elif len(sources) > P:
                        inds = random.sample(range(len(sources)), P)
                        source_conds_mix[n] = [sources[ind] for ind in inds]
                    else:
                        extra = P - len(sources)
                        extra_inds = [random.sample(range(len(sources)), 1)[0] for _ in range(extra)]

                        all_inds = list(range(len(sources))) + extra_inds
                        source_conds_mix[n] = [sources[ind] for ind in all_inds]
                        # updates weights
                        weights_mix[n] = [1./all_inds.count(all_inds[i]) for i in range(P)]

            mix = self._mix_n(audios)

        except Exception as e:
            print('Failed loading audio: {}'.format(e))

            # create dummy data
            return self.dummy_data()
        
        tokenized_mix_conds = self.tokenizer(mix_conds)

        if self.split == "train":
            tokenized_source_conds = {}
            for k, v in tokenized_mix_conds.items():
                tokenized_source_conds[k] = []

            source_conds_weights = []

            for n in range(N):
                tokens = self.tokenizer(source_conds_mix[n])
                
                for k, v in tokens.items():
                    tokenized_source_conds[k].append(v.unsqueeze(0))

                source_conds_weights.append(torch.tensor(weights_mix[n]).unsqueeze(0))

            for k, v in tokenized_source_conds.items():
                tokenized_source_conds[k] = torch.cat(v, dim=0)

            source_conds_weights = torch.cat(source_conds_weights, dim=0)
        
            ret_dict = {'mix': torch.tensor(mix), 'conds': tokenized_mix_conds, 'source_conds': tokenized_source_conds, 'source_weights': source_conds_weights}        
        
        else:
            ret_dict = {'mix': torch.tensor(mix), 'conds': tokenized_mix_conds}
    
        ret_dict['audios'] = torch.tensor(np.array(audios))

        return ret_dict
