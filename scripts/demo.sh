# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

python demo.py  --cond_layer sca --num_cond_blocks 1 --num_res_layers 1 --num_head 8 \
                --cond_dim 768 --num_downs 7 --num_channels 32 --audLen 131070 \
                --audRate 16000 --workers 4 --multiprocessing_distributed \
                --ngpu 1 --dist-url tcp://localhost:12342 --samples_dir demo_samples \
                --load pretrained_weights/model_weights.pth.tar