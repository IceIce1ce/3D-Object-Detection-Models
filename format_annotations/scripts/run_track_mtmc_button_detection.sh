#!/bin/bash

# Add python path for MLKit
export PYTHONPATH=$PYTHONPATH:$PWD
#export CUDA_LAUNCH_BLOCKING=1
export MKL_SERVICE_FORCE_INTEL=1

START_TIME="$(date -u +%s.%N)"

# Full path of the current script
THIS=$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null||echo $0)
# The directory where current script resides
DIR_CURRENT=$(dirname "${THIS}")

DATASET="synthehicle_town05"
DATASETSUBSET="N-rain"
#DATASET="aic22_mtmc"
#DATASETSUBSET="dataset_a"

# NOTE: IMAGE EXTRACTION
python projects/tss/runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --images_extraction \
--config "c01.yaml"
python projects/tss/runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --images_extraction \
--config "c02.yaml"
python projects/tss/runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --images_extraction \
--config "c03.yaml"
python projects/tss/runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --images_extraction \
--config "c04.yaml"
python projects/tss/runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --images_extraction \
--config "c05.yaml"
wait

# NOTE: DETECTION
seqs1=(C01 C02 C03 C04 C05)
img_size=1280
conf_thres=0.1
gpu_id=0
# shellcheck disable=SC2068
for seq in ${seqs1[@]}
do
    CUDA_VISIBLE_DEVICES=${gpu_id} python projects/tss/src/detectors/yolov5v4/detect.py   \
    --name   ${seq}  \
		--output projects/tss/data/${DATASET}/outputs/  \
		--source projects/tss/data/${DATASET}/outputs/images/${seq}  \
		--weights models_zoo/pretrained/yolov5/yolov5x.pt  \
		--roi_region projects/tss/data/${DATASET}/${DATASETSUBSET}/${seq}/roi.jpg \
		--conf ${conf_thres}  \
		--agnostic  \
		--save-txt  \
		--save-conf  \
		--classes 2 5 7 \
		--img-size ${img_size}
done
wait


END_TIME="$(date -u +%s.%N)"

ELAPSED="$(bc <<<"$END_TIME-$START_TIME")"
echo "Total of $ELAPSED seconds elapsed."
