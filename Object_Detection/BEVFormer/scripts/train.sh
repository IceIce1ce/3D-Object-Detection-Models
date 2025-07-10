# fix missing file: https://huggingface.co/spaces/facebook/sapiens-pose/blob/2d4ff18f0b4690b2c70b8bbe10889bc846ebcaba/external/engine/mmengine/hub/torchvision_0.12.json
export NCCL_P2P_DISABLE=1
bash tools/dist_train.sh projects/configs/bevformer/bevformer_tiny.py 2
# bash tools/dist_train.sh projects/configs/bevformer/bevformer_small.py 2
# bash tools/dist_train.sh projects/configs/bevformer/bevformer_base.py 2
# bash tools/fp16/dist_train.sh projects/configs/bevformer_fp16/bevformer_tiny_fp16.py 2