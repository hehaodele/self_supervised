CUDA_VISIBLE_DEVICES=0 python entry.py -m moco   -d stl10-id --id-type strip --id-weight 0.5 &
CUDA_VISIBLE_DEVICES=1 python entry.py -m byol   -d stl10-id --id-type strip --id-weight 0.5 &
CUDA_VISIBLE_DEVICES=2 python entry.py -m simclr -d stl10-id --id-type strip --id-weight 0.5 &
CUDA_VISIBLE_DEVICES=3 python entry.py -m eqco   -d stl10-id --id-type strip --id-weight 0.5 &