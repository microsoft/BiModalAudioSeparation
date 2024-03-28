# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


# System libs
import gradio as gr
import os
import copy
import csv
import random
import time
import builtins
import sys

# Numerical libs
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import soundfile as sf
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from imageio import imsave
from torch_mir_eval import bss_eval_sources as tbss_eval_sources
import torchaudio.transforms as Transforms 

#torch librosa tools
from torchlibrosa.stft import Spectrogram, LogmelFilterBank, STFT, ISTFT, magphase
from torchlibrosa.augmentation import SpecAugmentation
from transformers import RobertaTokenizer
import librosa
import torchaudio
from itertools import repeat

# muliprocessing tools
import torch.utils.data
import torch.utils.data.distributed
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
from enum import Enum
import shutil
import time
import torch.backends.cudnn as cudnn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Our libs
from arguments import ArgParser
from models import ModelBuilder, activate
from models.criterion import L1Loss
from utils import AverageMeter, \
    recover_rgb, magnitude2heatmap, \
    istft_reconstruction, warpgrid, \
    combine_video_audio, save_video, makedirs

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


best_sdr = -1000

def realimag(mag, cos, sin):

    real = mag * cos
    imag = mag * sin

    return real, imag


# Network wrapper, defines forward pass
class NetWrapper(torch.nn.Module):
    def __init__(self, nets, args):
        super(NetWrapper, self).__init__()
        self.net_sound, self.net_clap, self.net_synthesizer = nets

        self.N = args.num_mix

        self.stft = STFT(n_fft=args.stft_frame, hop_length=args.stft_hop, freeze_parameters=True)
        self.istft = ISTFT(n_fft=args.stft_frame, hop_length=args.stft_hop, freeze_parameters=True)

        self.src_rate = args.audRate
        self.trg_rate = 48000
        self.crit = L1Loss()
        self.crit2 = nn.CrossEntropyLoss()
        self.audLen = args.audLen
        self.device = self.net_clap.fc.weight.device

    def forward(self, audio, prompt, args):
        mix = audio.unsqueeze(0).to(self.device, non_blocking=True)

        cond = prompt

        real, imag = self.stft(mix)
        mag_mix, cos, sin = magphase(real, imag)

        # reshape mag_mix
        mag_mix = mag_mix.transpose(-1, -2)
        
        B = mag_mix.size(0)
        T = mag_mix.size(3)

        out_mag_mix = magnitude2heatmap(mag_mix.detach().squeeze().cpu().numpy())[::-1, :, :]
        
        # LOG magnitude
        log_mag_mix = torch.log1p(mag_mix).detach()
        weight = torch.clamp(log_mag_mix, 1e-3, 10)

        text_embed, hidden_state, feat_cond = self.net_clap.get_text_conditioning(cond, tokenize=False, return_hidden_states=True)
        feat_cond = activate(feat_cond, args.cond_activation)

        feat_sound = self.net_sound(log_mag_mix, hidden_state)
        feat_sound = activate(feat_sound, args.sound_activation)

        mask = self.net_synthesizer(feat_cond, feat_sound)
        mask = activate(mask, args.output_activation)

        pred_mask = mask

        #aggregate the mag_mix for reconstruction
        pred_mag = mag_mix * pred_mask

        out_pred_mag = magnitude2heatmap(pred_mag.detach().squeeze().cpu().numpy())[::-1, :, :]

        pred_mag = pred_mag.transpose(-1, -2)
        real, imag = realimag(pred_mag, cos, sin)
        pred_wav = self.istft(real, imag, self.audLen)
        out_wav = pred_wav

        return [out_wav[0].cpu().numpy(), out_mag_mix, out_pred_mag]


def main():
    parser = ArgParser()
    args = parser.parse_train_arguments()

    ## fixed the seed
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    np.random.seed(args.seed)

    args.distributed = args.multiprocessing_distributed

    ngpus_per_node = args.ngpu if args.ngpu else torch.cuda.device_count()

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)



def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for Demo".format(args.gpu))

    if args.multiprocessing_distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        dist.barrier()

    # Network Builders
    builder = ModelBuilder()
    device = torch.device('cuda:{}'.format(args.gpu))

    net_sound = builder.build_sound_net(
        fc_dim=args.num_channels,
        args=args)

    # net_clap = None
    net_clap = builder.build_custom_clap(
        enable_fusion=False,
        device=device,
        amodel='HTSAT-base',
        tmodel='roberta',
        channels=args.num_channels)

    net_synthesizer = builder.build_synthesizer(
        fc_dim=args.num_channels)

    dist.barrier()

    ## counting parameters
    sound_params = sum(p.numel() for p in net_sound.parameters())
    synth_params = sum(p.numel() for p in net_synthesizer.parameters())
    clap_params = sum(p.numel() for p in net_clap.parameters())

    print(f"Total parameters of Unet: {sound_params/1000000 : 6.3f}M")
    print(f"Total parameters of Synthesizer: {synth_params/1000000 : .4e}M")
    print(f"Total parameters of clap: {clap_params/1000000 : 6.3f}M")

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')
    elif args.multiprocessing_distributed:
        if torch.cuda.is_available():
            if args.gpu is not None:
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)

                torch.cuda.set_device(args.gpu)
                net_sound.cuda(args.gpu)
                net_clap.cuda(args.gpu)
                net_synthesizer.cuda(args.gpu)
                
                nets = (net_sound, net_clap, net_synthesizer)

                # Wrap networks
                model = NetWrapper(nets, args)
                model.cuda(args.gpu)

                model_params = sum(p.numel() for p in model.parameters())
                train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

                print(f"Total parameters of Model: {model_params/1000000 : 6.3f}M")
                print(f"Total trainable parameters: {train_params/1000000 : 6.3f}M")

                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

            else:
                model.cuda()
                model = torch.nn.parallel.DistributedDataParallel(model)

    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
    else:
        model = torch.nn.DataParallel(model).cuda()

    if args.load:
        if os.path.isfile(args.load):
            print("=> loading checkpoint '{}'".format(args.load))
            if args.gpu is None:
                checkpoint = torch.load(args.load)
            elif torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.load, map_location=loc)

            model.load_state_dict(checkpoint['state_dict'], strict=False)

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.load, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.load))


    def separate_audio(audio, prompt):
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

        tokenize_prompt = tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )

        # process the audio 
        (sr, src_audio) = audio
        dtype = src_audio.dtype

        if src_audio.dtype == 'int16':
            src_audio = src_audio / 2 ** 15

        if src_audio.ndim == 2:
            if dtype == 'int16':
                src_audio = librosa.to_mono(src_audio.transpose(-1, -2).astype(np.float32))
            else:
                src_audio = librosa.to_mono(src_audio.astype(np.float32))
        
        #resample audio        
        if sr != SR_TRG:
            src_audio = librosa.resample(src_audio.astype(np.float32), orig_sr=sr, target_sr=SR_TRG).astype(np.float32)
            sr = SR_TRG

        src_audio[src_audio > 1.] = 1.
        src_audio[src_audio < -1.] = -1.

        # padding src_audio
        len_src = len(src_audio)
        r = 131070 - len_src % 131070
        src_audio = np.concatenate([src_audio, np.zeros(r)])
        output_audio = np.zeros_like(src_audio)
        len_src = len(src_audio)
        duration = len_src // SR_TRG

        start = 0
        src_specs = []
        output_specs = []

        while start < len_src:
            sample = src_audio[start : start + 131070]
            
            with torch.no_grad():
                output, spec_in, spec_out = model(torch.tensor(sample.astype(np.float32)), tokenize_prompt, args)
                src_specs.append(spec_in)
                output_specs.append(spec_out)

            output_audio[start : start + 131070] = output

            start += 131070

        output_audio = output_audio[:-(r)]
        
        if len(output_specs) > 1:
            output_spec = np.concatenate(output_specs, axis=1)
            src_spec = np.concatenate(src_specs, axis=1)

        else:
            output_spec = output_specs[-1]
            src_spec = src_specs[-1]

        rI = int(r * 512 / 131070)

        output_spec = output_spec[:, :-rI, :]
        src_spec = src_spec[:, :-rI, :]

        return src_spec, output_spec, (SR_TRG, output_audio)


    def launch_gradio():
        with gr.Blocks() as demo:
            gr.Markdown("Separate audiofrom mixtures using text queries with this demo.")

            with gr.Row():
                with gr.Column():
                    input_audio = gr.Audio()
                    input_text =  gr.Textbox(placeholder="Provide the text prompt")
                    button = gr.Button("Submit")

                with gr.Column():
                    with gr.Row():
                        input_spec = gr.Image(label="Mixture Spectrogram") 
                        output_spec = gr.Image(label="Separated Spectrogram")

                    output_audio = gr.Audio()

            button.click(separate_audio, inputs=[input_audio, input_text], outputs=[input_spec, output_spec, output_audio])
            examples = gr.Examples(examples=[
                [f"{args.samples_dir}/sample1.wav", "a man is talking, clean sound with no background noises"],
                [f"{args.samples_dir}/sample2.wav", "A woman speaks, clean sound with no background noises"],
                [f"{args.samples_dir}/sample2.wav", "A cat meows in a silent room, clean sound with no background noises"],
                [f"{args.samples_dir}/sample2.wav", "A girl is coughing in a silent room, clean sound with no other noises"]
                ],
                inputs=[input_audio, input_text])

        demo.launch(share=True)


    SR_TRG = args.audRate

    launch_gradio()


if __name__ == '__main__':
    main()