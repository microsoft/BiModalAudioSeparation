# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse

class ArgParser(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        
        # Model related arguments
        parser.add_argument('--id', default='',
                            help="a name for identifying the model")
        parser.add_argument('--num_mix', default=2, type=int,
                            help="number of sounds to mix")
        parser.add_argument('--test_num_mix', default=2, type=int,
                            help="number of sounds to mix for testing")
        parser.add_argument('--num_channels', default=32, type=int,
                            help='number of channels')
        parser.add_argument('--cond_activation', default='sigmoid',
                            help="activation on the condition features")
        parser.add_argument('--sound_activation', default='no',
                            help="activation on the sound features")
        parser.add_argument('--output_activation', default='sigmoid',
                            help="activation on the output")
        parser.add_argument('--binary_mask', default=0, type=int,
                            help="whether to use bianry masks")
        parser.add_argument('--mask_thres', default=0.5, type=float,
                            help="threshold in the case of binary masks")
        parser.add_argument('--weighted_loss', default=0, type=int,
                            help="weighted loss")
        parser.add_argument('--split', default='train',
                            help="train or test")

        # Data related arguments
        parser.add_argument('--batch_size', default=32, type=int,
                            help='input batch size')
        parser.add_argument('--workers', default=16, type=int,
                            help='number of data loading workers')
        parser.add_argument('--audLen', default=65535, type=int,
                            help='sound length')
        parser.add_argument('--audRate', default=11025, type=int,
                            help='sound sampling rate')
        parser.add_argument('--stft_frame', default=1022, type=int,
                            help="stft frame length")
        parser.add_argument('--stft_hop', default=256, type=int,
                            help="stft hop length")

        # Misc arguments
        parser.add_argument('--seed', default=1234, type=int,
                            help='manual seed')
        parser.add_argument('--ckpt', default='./outputs',
                            help='folder to output checkpoints')
        parser.add_argument('--disp_iter', type=int, default=20,
                            help='frequency to display')
        parser.add_argument('--eval_epoch', type=int, default=1,
                            help='frequency to evaluate')
        parser.add_argument('--loss', default="classification",
                            help='what loss to use: classification or contrastive')
        parser.add_argument('--n_sources', default=4, type=int,
                            help='num of n_sources')
        parser.add_argument('--parsed_sources_path', default='./data/annotations/parsed_phrases.json',
                            help='location of all parsed phrases of data captions')
        parser.add_argument('--audio_dir', default='./data/audiocaps/audio_16k',
                            help='location of all audio files')
        parser.add_argument('--samples_dir', default='./samples',
                            help='location of all audio samples for demo')

        self.parser = parser

    def add_train_arguments(self):
        parser = self.parser

        parser.add_argument('--mode', default='train',
                            help="train/eval")
        parser.add_argument('--dataset', type=str, default='audiocaps')
        parser.add_argument('--list_train',
                            default='data/train.csv')
        parser.add_argument('--sup_list_train',
                            default='data/train.csv')
        parser.add_argument('--list_val',
                            default='data/val.csv')
        parser.add_argument('--list_test',
                            default='data/test.csv')
        parser.add_argument('--list_test_3',
                            default='data/test.csv')
        parser.add_argument('--list_test_4',
                            default='data/test.csv')

        parser.add_argument('--dup_trainset', default=40, type=int,
                            help='duplicate so that one epoch has more iters')
        parser.add_argument('--recons_weight', default=5, type=int,
                            help='reconstruction loss weight')
        # optimization related arguments
        parser.add_argument('--num_epoch', default=100, type=int,
                            help='epochs to train for')
        parser.add_argument('--start_epoch', default=1, type=int,
                            help='epochs to start training for')

        parser.add_argument('--lr',
                            default=1e-3, type=float, help='LR')

        parser.add_argument('--lr_step',
                            default=30, type=int,
                            help='steps to drop LR in epochs')
        parser.add_argument('--beta1', default=0.9, type=float,
                            help='momentum for sgd, beta1 for adam')
        parser.add_argument('--weight_decay', default=1e-4, type=float,
                            help='weights regularizer')

        parser.add_argument('--aug_rate',
                            default=0.0, type=float, help='augmentation rate')
        parser.add_argument('--maxN', default=4, type=int,
                            help='maximum number of audios for mixture')                            

        parser.add_argument('--num_downs', default=4, type=int,
                            help='num of downs in unet') 
        parser.add_argument('--num_res_layers', default=2, type=int,
                            help='num of res layers in unet') 
        parser.add_argument('--num_head', default=8, type=int,
                            help='num of heads in attention units')                            
        parser.add_argument('--num_cond_blocks', default=1, type=int,
                            help='number of attention blocks')                            
        parser.add_argument('--cond_dim', default=512, type=int,
                            help='conditional feature dimension')                            
        parser.add_argument('--cond_layer', default='ca',
                            help="mode for conditioning")        
        parser.add_argument('--warmup_epochs', default=5, type=int,
                            help='num of warmup epochs before distillation')                            
        parser.add_argument('--ema_rate', default=0.9, type=float,
                            help='rate for ema on distillation')
        
        
        self.parser = parser


    def add_mgpu_arguments(self):
        parser = self.parser

        parser.add_argument('--world-size', default=1, type=int,
                            help='number of nodes for distributed training')
        parser.add_argument('--rank', default=0, type=int,
                            help='node rank for distributed training')
        parser.add_argument('--dist-url', default='tcp://127.0.0.1:1234', type=str,
                            help='url used to set up distributed training')
        parser.add_argument('--dist-backend', default='nccl', type=str,
                            help='distributed backend')
        parser.add_argument('--gpu', default=None, type=int,
                            help='GPU id to use.')
        parser.add_argument('--multiprocessing_distributed', action='store_true',
                            help='Use multi-processing distributed training to launch '
                                'N processes per node, which has N GPUs. This is the '
                                'fastest way to use PyTorch for either single node or '
                                'multi node data parallel training')
        parser.add_argument('--ngpu', default=None, type=int,
                            help='total number of gpus. ')
        parser.add_argument('--num_check', default=4, type=int,
                            help='num of intermediate checkpointing. ')
        parser.add_argument('--resume', default='', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        parser.add_argument('--load', default='', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                            help='use pre-trained model')
        parser = self.parser

    def print_arguments(self, args):
        print("Input arguments:")
        for key, val in vars(args).items():
            print("{:16} {}".format(key, val))

    def parse_train_arguments(self):
        self.add_train_arguments()
        self.add_mgpu_arguments()
        args = self.parser.parse_args()
        self.print_arguments(args)
        return args