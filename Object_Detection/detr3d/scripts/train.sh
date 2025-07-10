export NCCL_P2P_DISABLE=1
# train Object DGCNN model
bash tools/dist_train.sh projects/configs/obj_dgcnn/pillar.py 2
bash tools/dist_train.sh projects/configs/obj_dgcnn/voxel.py 2

# train DETR3D model
bash tools/dist_train.sh projects/configs/detr3d/detr3d_res101_gridmask.py 2
bash tools/dist_train.sh projects/configs/detr3d/detr3d_res101_gridmask_cbgs.py 2
bash tools/dist_train.sh projects/configs/detr3d/detr3d_vovnet_gridmask_det_final_trainval_cbgs.py 2
