#!/bin/bash -x

kill -9 `ps -x | grep main_amp | awk '{ print $1 }'`

# python -m torch.distributed.launch --nproc_per_node=8 main_amp.py -a resnet50 -b 128 -g 1 -j 16 --opt-level O0 ./ >& test.log
python -m torch.distributed.launch --nproc_per_node=8 main_amp.py --resume checkpoint.pth.tar -a resnet50 -b 128 -g 1 -j 16 --opt-level O0 ./ >& test.log

