CUDA_VISIBLE_DEVICES=0 python entry.py -m moco &
CUDA_VISIBLE_DEVICES=1 python entry.py -m byol &
CUDA_VISIBLE_DEVICES=2 python entry.py -m simclr &
CUDA_VISIBLE_DEVICES=3 python entry.py -m eqco &