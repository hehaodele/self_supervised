CUDA_VISIBLE_DEVICES=0 python main.py --lmbda 0.0078125 --corr_zero --batch_size 128 --feature_dim 128 --dataset stl10 &
CUDA_VISIBLE_DEVICES=1 python main.py --lmbda 0.0078125 --corr_zero --batch_size 128 --feature_dim 128 --dataset stl10-id --id-weight 0.5 --id-type strip &
CUDA_VISIBLE_DEVICES=2 python main.py --lmbda 0.0078125 --corr_zero --batch_size 128 --feature_dim 128 --dataset stl10-id --id-weight 0.5 --id-type 2d &
CUDA_VISIBLE_DEVICES=3 python main.py --lmbda 0.0078125 --corr_zero --batch_size 128 --feature_dim 128 --dataset stl10-id --id-weight 1.0 --id-type 2d &
