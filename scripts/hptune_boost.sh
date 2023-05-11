echo "id: $1"
CUDA_VISIBLE_DEVICES=0 wandb agent carrtesy/OnlineTSAD/$1 &
CUDA_VISIBLE_DEVICES=1 wandb agent carrtesy/OnlineTSAD/$1 &
CUDA_VISIBLE_DEVICES=2 wandb agent carrtesy/OnlineTSAD/$1 &
CUDA_VISIBLE_DEVICES=3 wandb agent carrtesy/OnlineTSAD/$1 &
CUDA_VISIBLE_DEVICES=4 wandb agent carrtesy/OnlineTSAD/$1 &
CUDA_VISIBLE_DEVICES=5 wandb agent carrtesy/OnlineTSAD/$1 &
CUDA_VISIBLE_DEVICES=6 wandb agent carrtesy/OnlineTSAD/$1 &
CUDA_VISIBLE_DEVICES=7 wandb agent carrtesy/OnlineTSAD/$1 &
CUDA_VISIBLE_DEVICES=8 wandb agent carrtesy/OnlineTSAD/$1 &
