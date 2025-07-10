#!/bin/bash

# Add python path for MLKit
export PYTHONPATH=$PYTHONPATH:$PWD
#export CUDA_LAUNCH_BLOCKING=1
export MKL_SERVICE_FORCE_INTEL=1

START_TIME="$(date -u +%s.%N)"

# Full path of the current script
THIS=$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null||echo $0)
# The directory where current script resides
# shellcheck disable=SC2034
DIR_CURRENT=$(dirname "${THIS}")

DATASET="synthehicle_town05"
DATASETSUBSET="N-rain"
#DATASET="aic22_mtmc"
#DATASETSUBSET="dataset_a"


# NOTE: FEATURE EXTRACTION - 1 - resnet101_ibn_a_2
python projects/tss/runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --feature_extraction \
--config "reid/c01_resnet101_ibn_a_2.yaml" 
python projects/tss/runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --feature_extraction \
--config "reid/c02_resnet101_ibn_a_2.yaml"
python projects/tss/runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --feature_extraction \
--config "reid/c03_resnet101_ibn_a_2.yaml"
python projects/tss/runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --feature_extraction \
--config "reid/c04_resnet101_ibn_a_2.yaml"
python projects/tss/runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --feature_extraction \
--config "reid/c05_resnet101_ibn_a_2.yaml"
wait

# NOTE: MERGE ALL FEATURES
python3 projects/tss/runs/main_aic21_track_mtmc.py  --dataset ${DATASET} \
    --config "default.yaml" --feature_merge --config_feat_merge "reid_merge.yaml"
wait


END_TIME="$(date -u +%s.%N)"

ELAPSED="$(bc <<<"$END_TIME-$START_TIME")"
echo "Total of $ELAPSED seconds elapsed."
