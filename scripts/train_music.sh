# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

python main.py --id Proposed_music --mode train --dataset music --list_train data/music/annotations/train.csv \
                --list_test data/music/annotations/test_sep_2.csv --audio_dir data/music/audio \
                --cond_layer sca --num_cond_blocks 1 --num_res_layers 1 --num_head 8 \
                --cond_dim 768 --num_downs 7 --num_channels 32 --num_mix 2 --audLen 131070 \
                --audRate 16000 --workers 4 --batch_size 16 --lr 1e-4 --num_epoch 200 \
                --lr_step 15 --disp_iter 20 --ckpt outputs --multiprocessing_distributed \
                --ngpu 8 --recons_weight 5 --disp_iter 20 --dist-url tcp://localhost:12341 \
                --warmup_epochs 1 --eval_epoch 2 --n_sources 2 --test_num_mix 2