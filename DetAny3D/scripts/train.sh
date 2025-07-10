export NCCL_P2P_DISABLE=1
torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=6006 --nnodes=1 --node_rank=0 train.py --config_path detect_anything/configs/train.yaml
