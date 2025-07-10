export NCCL_P2P_DISABLE=1
bash tools/dist_test.sh configs/mv2d/exp/mv2d_r50_frcnn_two_frames_1408x512_ep24.py weights/epoch_72.pth 2 --eval bbox