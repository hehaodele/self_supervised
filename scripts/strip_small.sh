CUDA_VISIBLE_DEVICES=0 python entry.py -m moco   -d stl10-id --id-type strip-h --id-weight 0.5 --strip-len 4 &
CUDA_VISIBLE_DEVICES=1 python entry.py -m moco   -d stl10-id --id-type strip-h --id-weight 0.5 --strip-len 2 &
CUDA_VISIBLE_DEVICES=2 python entry.py -m moco   -d stl10-id --id-type strip-hv --id-weight 0.5 --strip-len 4 &
CUDA_VISIBLE_DEVICES=3 python entry.py -m moco   -d stl10-id --id-type strip-hv --id-weight 0.5 --strip-len 2 &