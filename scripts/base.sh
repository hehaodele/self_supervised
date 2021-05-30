CUDA_VISIBILE_DEVICES=0 python entry.py -m moco &
CUDA_VISIBILE_DEVICES=1 python entry.py -m byol &
CUDA_VISIBILE_DEVICES=2 python entry.py -m simclr &
CUDA_VISIBILE_DEVICES=3 python entry.py -m eqco &