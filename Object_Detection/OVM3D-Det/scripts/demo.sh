DATASET=KITTI
python -W ignore demo/demo.py --config-file checkpoints/$DATASET/config.yaml --input-folder "datasets/KITTI_object/testing/image_2" --threshold 0.25 MODEL.WEIGHTS checkpoints/$DATASET/model_recent.pth OUTPUT_DIR output/demo
