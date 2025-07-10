export NCCL_P2P_DISABLE=1
CUDA_VISIBLE_DEVICES=0,1 python -W ignore tools/train_val.py --config configs/monodinodetr.yaml --batch_size 16 --num_gpus 2