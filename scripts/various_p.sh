CUDA_VISIBLE_DEVICES=4 python entry.py -m moco   -d stl10-id --id-type 2d --id-weight 0.25 &
CUDA_VISIBLE_DEVICES=5 python entry.py -m moco   -d stl10-id --id-type 2d --id-weight 0.75 &
CUDA_VISIBLE_DEVICES=6 python entry.py -m moco   -d stl10-id --id-type 2d --id-weight 1.0 &
CUDA_VISIBLE_DEVICES=7 python entry.py -m moco   -d stl10-id --id-type strip --id-weight 1.0 &