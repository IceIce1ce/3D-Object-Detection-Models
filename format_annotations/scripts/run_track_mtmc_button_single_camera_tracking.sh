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
#python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --images_extraction \
#--config "c01.yaml" &
#python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --images_extraction \
#--config "c02.yaml" &
#python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --images_extraction \
#--config "c03.yaml" &
#python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --images_extraction \
#--config "c04.yaml" &
#python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --images_extraction \
#--config "c05.yaml"
#wait

# NOTE: RUN TRACKING
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --tracking  \
   --config "c01.yaml"
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --tracking  \
   --config "c02.yaml"
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --tracking  \
   --config "c03.yaml"
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --tracking  \
   --config "c04.yaml"
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --tracking  \
   --config "c05.yaml"
wait

# NOTE: DRAW MOTS
#python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --save_tracks_img \
#--config "c01.yaml" &
#python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --save_tracks_img \
#--config "c02.yaml" &
#python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --save_tracks_img \
#--config "c03.yaml" &
#python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --save_tracks_img \
#--config "c04.yaml" &
#python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --save_tracks_img \
#--config "c05.yaml"
#wait

# NOTE: MATCHING ZONE
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --matching_zone --use_mots_feat_raw \
   --config "c01.yaml"
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --matching_zone --use_mots_feat_raw \
   --config "c02.yaml"
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --matching_zone --use_mots_feat_raw \
   --config "c03.yaml"
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --matching_zone --use_mots_feat_raw \
   --config "c04.yaml"
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --matching_zone --use_mots_feat_raw \
   --config "c05.yaml"
wait

# NOTE: DRAW MOTS WITH ZONE
#python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --save_tracks_zone \
# --config "c01.yaml" &
#python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --save_tracks_zone \
# --config "c02.yaml" &
#python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --save_tracks_zone \
# --config "c03.yaml" &
#python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --save_tracks_zone \
# --config "c04.yaml" &
#python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --save_tracks_zone \
# --config "c05.yaml"
#wait


END_TIME="$(date -u +%s.%N)"

ELAPSED="$(bc <<<"$END_TIME-$START_TIME")"
echo "Total of $ELAPSED seconds elapsed."
