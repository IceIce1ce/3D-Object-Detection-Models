export NCCL_P2P_DISABLE=1
bash tools/dist_test.sh projects/configs/simpb_nus_r50_img_704x256.py ckpts/simpb_r50_img.pth 2 --eval bbox
