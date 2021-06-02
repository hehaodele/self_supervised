CUDA_VISIBLE_DEVICES=0 python entry.py -m moco   -d stl10-id --id-type strip-h --id-weight 0.5 --strip-len 96 &
CUDA_VISIBLE_DEVICES=1 python entry.py -m moco   -d stl10-id --id-type strip-h --id-weight 0.5 --strip-len 64 &
CUDA_VISIBLE_DEVICES=2 python entry.py -m moco   -d stl10-id --id-type strip-h --id-weight 0.5 --strip-len 32 &
CUDA_VISIBLE_DEVICES=3 python entry.py -m moco   -d stl10-id --id-type strip-h --id-weight 0.5 --strip-len 8 &