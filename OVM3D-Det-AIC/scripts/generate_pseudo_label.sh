# DATASET=$1
DATASET=Warehouse_008
# CUDA_VISIBLE_DEVICES=0 python -W ignore third_party/UniDepth/run_unidepth.py --dataset $DATASET
# CUDA_VISIBLE_DEVICES=0 python -W ignore third_party/Grounded-Segment-Anything/grounded_sam_detect.py --dataset $DATASET
CUDA_VISIBLE_DEVICES=0 python -W ignore third_party/Grounded-Segment-Anything/grounded_sam_detect_ground.py --dataset $DATASET
# python -W ignore tools/generate_pseudo_bbox.py --config-file configs/Base_Omni3D_${DATASET}.yaml OUTPUT_DIR output/generate_pseudo_label/$DATASET
# python -W ignore tools/transform_to_coco.py --dataset_name $DATASET