export NCCL_P2P_DISABLE=1
bash tools/dist_train.sh projects/configs/simpb_nus_r50_img_704x256.py 2 --no-validate
