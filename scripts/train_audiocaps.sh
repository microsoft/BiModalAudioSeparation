# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

python main.py --id Proposed_AC --mode train --list_train data/annotations/train_ids.csv \
                --list_test data/annotations/test_sep2_ids.csv --audio_dir data/audio \
                --cond_layer sca --num_cond_blocks 1 --num_res_layers 1 --num_head 8 \
                --cond_dim 768 --num_downs 7 --num_channels 32 --num_mix 2 --audLen 131070 \
                --audRate 16000 --workers 4 --batch_size 16 --lr 1e-4 --num_epoch 200 \
                --lr_step 15 --disp_iter 20 --ckpt outputs --multiprocessing_distributed \
                --ngpu 8 --recons_weight 5 --disp_iter 20 --dist-url tcp://localhost:12341 \
                --parsed_sources_path data/annotations/parsed_all_caps.json \
                --warmup_epochs 1 --eval_epoch 2 --n_sources 3 