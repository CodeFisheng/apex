#bin
python -m torch.distributed.launch --nproc_per_node=8 main_amp.py -a resnet50 -b 128 -g 1 -j 8 --workers 4 --opt-level O0 ./
# python -m torch.distributed.launch --nproc_per_node=8 main_amp.py --resume -a resnet50 -b 128 -g 1 --workers 4 --opt-level O0 ./

