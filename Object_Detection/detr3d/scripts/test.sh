export NCCL_P2P_DISABLE=1
# test Object DGCNN model
# bash tools/dist_test.sh projects/configs/obj_dgcnn/pillar.py checkpoints/pillar.pth 2 --eval=bbox
# bash tools/dist_test.sh projects/configs/obj_dgcnn/voxel.py checkpoints/voxel.pth 2 --eval=bbox

# test DETR3D model
# bash tools/dist_test.sh projects/configs/detr3d/detr3d_res101_gridmask.py checkpoints/detr3d_resnet101.pth 2 --eval=bbox
#b ash tools/dist_test.sh projects/configs/detr3d/detr3d_res101_gridmask_cbgs.py checkpoints/detr3d_resnet101_cbgs.pth 2 --eval=bbox
bash tools/dist_test.sh projects/configs/detr3d/detr3d_vovnet_gridmask_det_final_trainval_cbgs.py checkpoints/detr3d_vovnet_trainval.pth 2 --eval=bbox
