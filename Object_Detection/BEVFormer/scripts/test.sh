export NCCL_P2P_DISABLE=1
bash tools/dist_test.sh projects/configs/bevformer/bevformer_tiny.py ckpts/bevformer_tiny_epoch_24.pth 1
bash tools/dist_test.sh projects/configs/bevformer/bevformer_small.py ckpts/bevformer_small_epoch_24.pth 1
bash tools/dist_test.sh projects/configs/bevformer/bevformer_base.py ckpts/bevformer_r101_dcn_24ep.pth 1