#CUDA_VISIBLE_DEVICES=0 python entry.py -m moco   -d stl10-id --id-type strip-hv --id-weight 1.0 --strip-len 2 &
#CUDA_VISIBLE_DEVICES=1 python entry.py -m moco   -d stl10-id --id-type strip-hv --id-weight 0.75 --strip-len 2 &
#CUDA_VISIBLE_DEVICES=2 python entry.py -m moco   -d stl10-id --id-type strip-hv --id-weight 0.25 --strip-len 2 &
#CUDA_VISIBLE_DEVICES=3 python entry.py -m moco   -d stl10-id --id-type strip-hv --id-weight 1.0 --strip-len 32 &
CUDA_VISIBLE_DEVICES=1 python entry.py -m moco   -d stl10-id --id-type strip-hv --id-weight 0.75 --strip-len 32 &
CUDA_VISIBLE_DEVICES=2 python entry.py -m moco   -d stl10-id --id-type strip-hv --id-weight 0.25 --strip-len 32 &
CUDA_VISIBLE_DEVICES=0 python entry.py -m moco   -d stl10-id --id-type strip-hv --id-weight 1.0 --strip-len 8 &
CUDA_VISIBLE_DEVICES=3 python entry.py -m moco   -d stl10-id --id-type strip-hv --id-weight 1.0 --strip-len 64 &