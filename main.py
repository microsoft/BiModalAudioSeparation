# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# System libs
import os
import copy
import csv
import random
import time
import builtins
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import soundfile as sf
from imageio import imsave
from torch_mir_eval import bss_eval_sources as tbss_eval_sources
import torchaudio.transforms as Transforms 

#torch librosa tools
from torchlibrosa.stft import Spectrogram, LogmelFilterBank, STFT, ISTFT, magphase
from torchlibrosa.augmentation import SpecAugmentation

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
from dataset import AudioCapsDataset, MusicDataset, VGGSoundDataset
from models import ModelBuilder, activate
from models.criterion import L1Loss
from utils import AverageMeter, \
    recover_rgb, magnitude2heatmap, \
    istft_reconstruction, warpgrid, \
    combine_video_audio, save_video, makedirs, realimag
from viz_tools import HTMLVisualizer, smooth
import math
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


best_sdr = - math.inf

# Network wrapper, defines forward pass
class NetWrapper(torch.nn.Module):
    def __init__(self, nets, args):
        super(NetWrapper, self).__init__()
        self.net_sound, self.net_clap, self.net_synthesizer = nets

        self.num_mix = args.num_mix
        self.test_num_mix = args.test_num_mix

        self.stft = STFT(n_fft=args.stft_frame, hop_length=args.stft_hop, freeze_parameters=True)
        self.istft = ISTFT(n_fft=args.stft_frame, hop_length=args.stft_hop, freeze_parameters=True)

        self.src_rate = args.audRate
        self.trg_rate = 48000
        self.crit = L1Loss()
        self.crit2 = nn.CrossEntropyLoss()
        self.audLen = args.audLen


    def forward(self, batch_data, device, args):
        mix = batch_data['mix'].to(device, non_blocking=True)
        
        if self.training:
            N = self.num_mix
        else:
            N = self.test_num_mix

        conds = batch_data['conds']

        if self.training:
            source_conds = batch_data['source_conds']
            source_weights = batch_data['source_weights']

        # processing mixture of mixtures
        real, imag = self.stft(mix)
        mag_mix, cos, sin = magphase(real, imag)
        mag_mix = mag_mix.transpose(-1, -2)        
        B = mag_mix.size(0)
        T = mag_mix.size(3)

        if not self.training:
            out_mag_mix = mag_mix

        # extracting raw source mixtures
        raw_mags = [None for _ in range(N)]
        raw_wavs = [None for _ in range(N)]
        audios = batch_data['audios'].to(device, non_blocking=True)

        # processing raw source mixtures       
        for i in range(N):
            real, imag = self.stft(audios[:, i, :])
            raw_mags[i], _, _ = magphase(real, imag)
            raw_mags[i] = raw_mags[i].transpose(-1, -2)
            raw_wavs[i] = audios[:, i, :]
            
        # LOG magnitude of mixture of mixture magnitude spectrum
        log_mag_mix = torch.log1p(mag_mix).detach()

        # predicting mixture audios from the mixtrue of mixtures
        feat_conds = [None for n in range(N)]
        feat_sound = [None for n in range(N)]
        hidden_states = [None for n in range(N)]
        text_embed = [None for n in range(N)]

        for n in range(N):                
            cond = {}
            for key in conds.keys():
                cond[key] = conds[key][:, n, :]

            text_embed[n], hidden_states[n], feat_conds[n] = self.net_clap.get_text_conditioning(cond, tokenize=False, return_hidden_states=True)
            feat_conds[n] = activate(feat_conds[n], args.cond_activation)

            # 1. forward net_sound -> BxCxHxW
            feat_sound[n] = self.net_sound(log_mag_mix, hidden_states[n])
            feat_sound[n] = activate(feat_sound[n], args.sound_activation)

        # applying sound synthesizer to produce mask of source mixtures
        masks = [None for n in range(N)]
        text_embed = torch.cat(text_embed, dim=0).to(device, non_blocking=True)

        for n in range(N):
            masks[n] = self.net_synthesizer(feat_conds[n], feat_sound[n])
            masks[n] = activate(masks[n], args.output_activation)

        # pred masks aggregation
        pred_masks = masks
        pred_masks = torch.cat(pred_masks, dim=1)

        # extracting attention weights for loss 
        weight = torch.log1p(mag_mix)
        weight = torch.clamp(weight, 1e-3, 10)

        # pred mags of each source mixture
        mag_mix = torch.cat([mag_mix for _ in range(N)], dim=1)
        pred_mags = mag_mix * pred_masks

        if self.training:
            # calculating unsupervised reconstruction loss from mixture prediction
            unsup_recons_loss = 0
            for n in range(N):
                pred_mix = pred_mags[:, n, :, :].unsqueeze(1)
                unsup_recons_loss += self.crit(pred_mix, raw_mags[n], weight)

        if not self.training:
            # extracting sound wavs from the prediction
            out_pred_mags = list(torch.split(pred_mags.detach(), 1, dim=1))
            pred_mags = pred_mags.transpose(-1, -2)
            real, imag = realimag(pred_mags, cos, sin)
            pred_wavs = [None for n in range(N)]

            for n in range(N):
                real_n, imag_n = real[:, n, :, :].unsqueeze(1), imag[:, n, :, :].unsqueeze(1)
                pred_wavs[n] = self.istft(real_n, imag_n, self.audLen)

            out_wavs = pred_wavs

        if self.training:
            # Applying bimodal semantic loss in each sound source component present is source mixtures

            # preparing contrastive loss and consistency reconstruction loss
            cnt_loss = 0
            cns_recons_loss = 0

            # extracting the mag_mix (mixture of mixtures)
            mag_mix = mag_mix[:, 0, :, :].unsqueeze(1)
            mag_mix = torch.cat([mag_mix for _ in range(args.n_sources)], dim=1)

            for n in range(N):
                src_mix = raw_mags[n]

                # extracting loss attention weights for each source mixture
                loss_weight = torch.log1p(src_mix)
                loss_weight = torch.clamp(loss_weight, 1e-3, 10)
                
                feat_conds = [None for _ in range(args.n_sources)]
                feat_sound = [None for _ in range(args.n_sources)]
                hidden_states = [None for _ in range(args.n_sources)]
                text_embed = [None for _ in range(args.n_sources)]
                
                # extracting audio and text features of each source present in source mixtures
                for k in range(args.n_sources):                
                    cond = {}

                    for key in source_conds.keys():
                        cond[key] = source_conds[key][:, n, k, :]

                    text_embed[k], hidden_states[k], feat_conds[k] = self.net_clap.get_text_conditioning(cond, tokenize=False, return_hidden_states=True)
                    feat_conds[k] = activate(feat_conds[k], args.cond_activation)

                    feat_sound[k] = self.net_sound(log_mag_mix, hidden_states[k])
                    feat_sound[k] = activate(feat_sound[k], args.sound_activation)

                # applyinng sound synthesizer
                masks = [None for _ in range(args.n_sources)]
                text_embed = torch.cat(text_embed, dim=0).to(device, non_blocking=True)

                for k in range(args.n_sources):
                    masks[k] = self.net_synthesizer(feat_conds[k], feat_sound[k])
                    masks[k] = activate(masks[k], args.output_activation)

                # pred masks aggregation of each source component in source mixtures
           
                pred_masks = torch.cat(masks, dim=1)

                # pred mags aggregation of each source component in source mixtures
                pred_mags = mag_mix * pred_masks

                recons_mix = 0
                source_weight = source_weights[:, n]

                for k in range(args.n_sources):
                    recons_mix += source_weight[:, k].unsqueeze(-1).unsqueeze(-1).expand_as(pred_mags[:, k, :, :]) * pred_mags[:, k, :, :]

                # calculating consistency recnstruction loss                    
                recons_mix = recons_mix.unsqueeze(1)
                cns_recons_loss += self.crit(recons_mix, src_mix, loss_weight)

                # predicting each source sound wavs in source mixtures
                pred_mags = pred_mags.transpose(-1, -2)
                real, imag = realimag(pred_mags, cos, sin)
                pred_wavs = [None for _ in range(args.n_sources)]

                for k in range(args.n_sources):
                    real_k, imag_k = real[:, k, :, :].unsqueeze(1), imag[:, k, :, :].unsqueeze(1)
                    pred_wavs[k] = self.istft(real_k, imag_k, self.audLen)
                
                # preparing pred wavs for CLAP contrastive loss
                pred_wavs = torch.cat(pred_wavs, dim=0)

                # resampling the pred wavs to match CLAP encoder
                resampler = Transforms.Resample(self.src_rate, self.trg_rate, dtype=pred_wavs.dtype).to(device, non_blocking=True)
                pred_wavs = resampler(pred_wavs)

                # get the audio embedding of pred source wavs of mixture n
                audio_embed = self.net_clap.get_audio_embedding(pred_wavs)

                cnt_loss += self.net_clap.contrastive_loss(audio_embed, text_embed)

            loss = 0.5 * cnt_loss + args.recons_weight * (cns_recons_loss + unsup_recons_loss)

            return loss, cnt_loss, cns_recons_loss, unsup_recons_loss

        else:
                return \
                    {
                        "pred_wavs": out_wavs,
                        "raw_wavs": raw_wavs,
                        "mix_wav": batch_data['mix'],
                        "mag_mix": out_mag_mix,
                        "pred_mags": out_pred_mags,
                        "raw_mags": raw_mags
                    }

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

    # experiment name
    args.id += '-{}mix'.format(args.num_mix)
    args.id += '-{}src'.format(args.n_sources)
    args.id += '-rw{}'.format(args.recons_weight)
    args.id += '-({})attn'.format(args.cond_layer)
    args.id += '-nres{}'.format(args.num_res_layers)
    args.id += '-nblk{}'.format(args.num_cond_blocks)
    args.id += '-ndowns{}'.format(args.num_downs)
    args.id += '-batch{}'.format(args.batch_size)
    args.id += '-epoch{}'.format(args.num_epoch)
    args.id += '-lr{}'.format(args.lr)
    args.id += '-ngpu{}'.format(args.ngpu)
    args.id += '-step{}'.format(args.lr_step)
    print('Model ID: {}'.format(args.id))
    
    # paths to save/load output
    args.ckpt = os.path.join(args.ckpt, args.id)

    # logger
    args.log_fn = f"{args.ckpt}/{args.mode}_mix{args.num_mix}.log"

    if os.path.isfile(args.log_fn):
        os.remove(args.log_fn)

    if args.mode == 'train':
        makedirs(args.ckpt, remove=False)

    args.testing = False

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
    global best_sdr

    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    def print_and_log(*content, **kwargs):
		# suppress printing if not first GPU on each node
        if args.multiprocessing_distributed and (args.gpu != 0 or args.rank != 0):
            return
        msg = ' '.join([str(ct) for ct in content])
        sys.stdout.write(msg+'\n')
        sys.stdout.flush()
        with open(args.log_fn, 'a') as f:
            f.write(msg+'\n')
    builtins.print = print_and_log

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

    # define optimizer, and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                    weight_decay=args.weight_decay)

    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=0.1)

    # History of peroformance
    history = {
        'train': {'epoch': [], 'loss': [], 'cnt_loss': [], 'cns_recons_loss': [], 'unsup_recons_loss': []},
        'test': {'epoch': [], 'sdr': [], 'sir': [], 'sar': []}}


    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            elif torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            
            args.start_epoch = checkpoint['epoch'] + 1
            
            best_sdr = checkpoint['best_sdr']

            model.load_state_dict(checkpoint['state_dict'], strict=False)

            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            history = checkpoint['history']

            for mode in history:
                for key in history[mode]:
                    values = history[mode][key]

                    for i, value in enumerate(values):                  
                        if torch.is_tensor(value):
                            history[mode][key][i] = value.item()

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    elif args.load:
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

    # Dataset and Loader

    if args.dataset == "audiocaps":
        dataset_train = AudioCapsDataset(
            args.list_train, args.num_mix, args, split='train')

        dataset_test = AudioCapsDataset(
            args.list_test, args.test_num_mix, args, split='test')
    
    elif args.dataset == "music":
        dataset_train = MusicDataset(
            args.list_train, args.num_mix, args, split='train')

        dataset_test = MusicDataset(
            args.list_test, args.test_num_mix, args, split='test')
    else:
        dataset_train = VGGSoundDataset(
            args.list_train, args.num_mix, args, split='train')

        dataset_test = VGGSoundDataset(
            args.list_test, args.test_num_mix, args, split='test')

    if args.multiprocessing_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        test_sampler = None

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=int(args.workers),
        drop_last=True,
        pin_memory=True,
        sampler=train_sampler)

    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        sampler=test_sampler)

    args.epoch_iters = len(loader_train)

    print('1 Epoch = {} iters'.format(args.epoch_iters))

    # Eval mode
    if args.mode == 'test':
        args.testing = True

        print("Running Test Evaluations on 2-Mix")
        evaluate(model, loader_test, history, 0, device, args, prefix="Test-2Mix")

        print('Test Evaluation Done!')

        return

    args.checkpoint_interval = args.num_epoch // args.num_check if args.num_epoch > 7 else 2
    
    # Training loop
    for epoch in range(args.start_epoch, args.num_epoch + 1):
        if args.multiprocessing_distributed:
            train_sampler.set_epoch(epoch)

        train(model, loader_train, optimizer, history, epoch, device, args)

        # Running Evaluation
        if epoch % args.eval_epoch == 0:
            args.testing = True
            sdr = evaluate(model, loader_test, history, epoch, device, args, prefix="Test")
            args.testing = False

            # remember best sdr and save checkpoint
            is_best = sdr > best_sdr
            best_sdr = max(sdr, best_sdr)

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank % ngpus_per_node == 0):
                checkpoint = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_sdr': best_sdr,
                    'optimizer' : optimizer.state_dict(),
                    'scheduler' : scheduler.state_dict(),
                    'history' : history}

                save_checkpoint(checkpoint, history, epoch, is_best, args)
            
            dist.barrier()

        scheduler.step()

    print('Training Done!')

    print("#" * 60)

    print(f"Best Test SDR: {best_sdr}")

    print("Loading best model")
    loc = 'cuda:{}'.format(args.gpu)

    path = os.path.join(args.ckpt, 'model_best.pth.tar')

    checkpoint = torch.load(path, map_location=loc)

    model.load_state_dict(checkpoint['state_dict'], strict=False)

    args.testing = True

    print("Running Test Evaluations on 2-Mix")
    evaluate(model, loader_test, history, 0, device, args, prefix="Test-2Mix")

    print('Test Evaluation Done!')


# train one epoch
def train(model, loader, optimizer, history, epoch, device, args):
    # torch.set_grad_enabled(True)
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    lr = AverageMeter('lr', ':6.5f')
    losses = AverageMeter('Loss', ':6.3f')
    cnt_losses = AverageMeter('Cnt Loss', ':6.3f')
    cns_recons_losses = AverageMeter('Cns Recons Loss', ':6.3f')
    unsup_recons_losses = AverageMeter('Unsup Recons Loss', ':6.3f')

    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time, lr, losses, cnt_losses, cns_recons_losses, unsup_recons_losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, batch_data in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        optimizer.zero_grad()

        # forward pass
        loss, cnt_loss, cns_recons_loss, unsup_recons_loss = model.forward(batch_data, device, args)
 
        loss, cnt_loss, cns_recons_loss, unsup_recons_loss = loss.mean(), cnt_loss.mean(), cns_recons_loss.mean(), unsup_recons_loss.mean()
        losses.update(loss.item(), batch_data['mix'].size(0))
        cnt_losses.update(cnt_loss.item(), batch_data['mix'].size(0))
        cns_recons_losses.update(cns_recons_loss.item(), batch_data['mix'].size(0))
        unsup_recons_losses.update(unsup_recons_loss.item(), batch_data['mix'].size(0))

        # backward
        loss.backward()
        optimizer.step()

        lr.val = lr.avg = get_lr(optimizer)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # display
        if i % args.disp_iter == 0:
            progress.display(i + 1)

        fractional_epoch = epoch - 1 + 1. * i / args.epoch_iters
        history['train']['epoch'].append(fractional_epoch)
        history['train']['loss'].append(loss.mean().item())
        history['train']['cnt_loss'].append(cnt_loss.mean().item())
        history['train']['cns_recons_loss'].append(cns_recons_loss.mean().item())
        history['train']['unsup_recons_loss'].append(unsup_recons_loss.mean().item())


def evaluate(model, loader, history, epoch, device, args, prefix='test'):
    print('Evaluating at {} epochs...'.format(epoch))

    def run_evaluate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, batch_data in enumerate(loader):
                i = base_progress + i

                # forward pass
                outputs = model.forward(batch_data, device, args)

                # calculate metrics
                sdr_mix, sdr, sir, sar = calc_metrics(batch_data, outputs, device, args)

                sdr_mix_meter.update(sdr_mix)
                sdr_meter.update(sdr)
                sir_meter.update(sir)
                sar_meter.update(sar)

                if i % (args.disp_iter // 4) == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    sdr_mix_meter = AverageMeter('SDR Mix', ':6.2f', Summary.AVERAGE)
    sdr_meter = AverageMeter('SDR', ':6.2f', Summary.AVERAGE)
    sir_meter = AverageMeter('SIR', ':6.2f', Summary.AVERAGE)
    sar_meter = AverageMeter('SAR', ':6.2f', Summary.AVERAGE)

    progress = ProgressMeter(
        len(loader) + (args.multiprocessing_distributed and (len(loader.sampler) * args.world_size < len(loader.dataset))),
        [batch_time, sdr_mix_meter, sdr_meter, sir_meter, sar_meter],
        prefix=f'{prefix}: ')

    # switch to eval mode
    model.eval()

    run_evaluate(loader)

    if args.multiprocessing_distributed:
        sdr_mix_meter.all_reduce()
        sdr_meter.all_reduce()
        sir_meter.all_reduce()
        sar_meter.all_reduce()
        

    if args.multiprocessing_distributed and (len(loader.sampler) * args.world_size < len(loader.dataset)):
        aux_dataset = Subset(loader.dataset,
                                 range(len(loader.sampler) * args.world_size, len(loader.dataset)))
        aux_loader = torch.utils.data.DataLoader(
            aux_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        run_evaluate(aux_loader, len(loader))


    history['test']['epoch'].append(epoch)
    history['test']['sdr'].append(sdr_meter.avg)
    history['test']['sir'].append(sir_meter.avg)
    history['test']['sar'].append(sar_meter.avg)

    #aggregate all history
    for mode in history.keys():
        for key in history[mode]:
            if "loss" in key:
                val = torch.tensor(history[mode][key], dtype=torch.float32, device=device) * (args.batch_size)
                dist.all_reduce(val, dist.ReduceOp.SUM, async_op=False)
                val = (val / (args.batch_size * args.world_size)).tolist()
                history[mode][key] = val

    # plotting and saving
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
            and args.rank % args.world_size == 0):
        
        # Plot figure
        if epoch > 0:
            print('Plotting figures...')
            plot_loss_metrics(args.ckpt, history, n=1)

    progress.display_summary()

    return sdr_meter.avg


def plot_loss_metrics(path, history, n=1):
    fig = plt.figure()
    loss = history['train']['loss']
    loss = smooth(loss, n)
    plt.plot(history['train']['epoch'], loss,
             color='b', label='train')
                 
    plt.legend()
    fig.savefig(os.path.join(path, f'loss.png'), dpi=200)
    plt.close('all')

    fig = plt.figure()
    plt.plot(history['train']['epoch'], history['train']['loss'],
             color='r', label='Total Loss')
    plt.plot(history['train']['epoch'], history['train']['cnt_loss'],
             color='c', label='Cnt Loss')
    plt.plot(history['train']['epoch'], history['train']['cns_recons_loss'],
             color='b', label='Cns Recons Loss')
    plt.plot(history['train']['epoch'], history['train']['unsup_recons_loss'],
             color='g', label='Unsup` Recons Loss')

    plt.legend()
    fig.savefig(os.path.join(path, 'loss_details.png'), dpi=200)
    plt.close('all')

    fig = plt.figure()
    plt.plot(history['test']['epoch'], history['test']['sdr'],
             color='r', label='SDR')
    plt.plot(history['test']['epoch'], history['test']['sir'],
             color='g', label='SIR')
    plt.plot(history['test']['epoch'], history['test']['sar'],
             color='b', label='SAR')
    plt.legend()
    fig.savefig(os.path.join(path, 'metrics.png'), dpi=200)
    plt.close('all')


# Calculate metrics
def calc_metrics(batch_data, outputs, device, args):
    sdr_mix_meter = AverageMeter('SDR Mix', ':6.2f', Summary.AVERAGE)
    sdr_meter = AverageMeter('SDR', ':6.2f', Summary.AVERAGE)
    sir_meter = AverageMeter('SIR', ':6.2f', Summary.AVERAGE)
    sar_meter = AverageMeter('SAR', ':6.2f', Summary.AVERAGE)

    src_audios = outputs['raw_wavs']
    pred_audios = outputs['pred_wavs']
    mix_audio = outputs['mix_wav']

    # unwarp log scale
    N = args.num_mix

    B = outputs['mag_mix'].size(0)
    mix_audio = mix_audio.cpu().numpy()

    for j in range(B):
        valid = True

        gt_wavs = [None for n in range(N)]
        pred_wavs = [None for n in range(N)]

        for n in range(N):
            gt_wavs[n] = src_audios[n][j].cpu().numpy()
            pred_wavs[n] = pred_audios[n][j].cpu().numpy()

            valid *= np.sum(np.abs(gt_wavs[n])) > 1e-5
            valid *= np.sum(np.abs(pred_wavs[n])) > 1e-5

        if valid:
            sdr, sir, sar, _ = tbss_eval_sources(
                torch.tensor(np.asarray(gt_wavs)).to(src_audios[0].device),
                torch.tensor(np.asarray(pred_wavs)).to(src_audios[0].device),
                False)
            sdr_mix, _, _, _ = tbss_eval_sources(
                torch.tensor(np.asarray(gt_wavs)).to(src_audios[0].device),
                torch.tensor(np.asarray([mix_audio[j] for n in range(N)])).to(src_audios[0].device),
                False)

            sdr_mix_meter.update(sdr_mix.mean())
            sdr_meter.update(sdr.mean())
            sir_meter.update(sir.mean())
            sar_meter.update(sar.mean())


    return [sdr_mix_meter.avg,
            sdr_meter.avg,
            sir_meter.avg,
            sar_meter.avg]


def save_checkpoint(checkpoint, history, epoch, is_best, args):
    print('Saving checkpoints at {} epochs.'.format(epoch))

    suffix_latest = 'latest.pth.tar'
    suffix_best = 'best.pth.tar'

    # aggregate history
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    torch.save(history,
               '{}/history_{}'.format(args.ckpt, suffix_latest))
    torch.save(checkpoint,
               '{}/model_{}'.format(args.ckpt, suffix_latest))

    if is_best:
        torch.save(checkpoint,
                '{}/model_{}'.format(args.ckpt, suffix_best))

    if epoch % args.checkpoint_interval == 0:
        torch.save(checkpoint,
                '{}/model_epoch{}.pth.tar'.format(args.ckpt, epoch))

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

if __name__ == '__main__':
    main()