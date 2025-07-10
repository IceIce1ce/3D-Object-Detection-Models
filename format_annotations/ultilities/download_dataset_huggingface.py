# from datasets import load_dataset
# imdb_dataset = load_dataset("KAIST-Multispectral-Pedestrian-Detection-Dataset")

# from huggingface_hub import snapshot_download
# snapshot_download(repo_id="richidubey/KAIST-Multispectral-Pedestrian-Detection-Dataset", repo_type="dataset")

from datasets import load_dataset

# init cai nay la download nguyen dataset luon :|, con khong chi download danh sach
# git lfs install
# git clone https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces


# find /home/vsw/Downloads/PhysicalAI-SmartSpaces/MTMC_Tracking_2025/val/ -type f -name "*" > val_2025.txt
# cp -L -R MTMC_Tracking_2024/ /media/vsw/hdd-02/AI_City_Challenge/2024/Track_1/
# cp -L -R README.md ~/Downloads/

# data_files = [
# 	"MTMC_Tracking_2025/train/*/videos/*.mp4",
#     # "MTMC_Tracking_2025/train/*/depth_maps/*.h5",
# 	"MTMC_Tracking_2025/train/*/*.json",
# 	"MTMC_Tracking_2025/train/*/*.png",
# 	"MTMC_Tracking_2025/val/*/videos/*.mp4",
#     # "MTMC_Tracking_2025/val/*/depth_maps/*.h5",
# 	"MTMC_Tracking_2025/val/*/*.json",
# 	"MTMC_Tracking_2025/val/*/*.png",
# 	]

# data_files = [
# 	"MTMC_Tracking_2024/train/*/*.json",
#     "MTMC_Tracking_2024/train/*/*.txt",
# 	"MTMC_Tracking_2024/train/*/*/*.mp4",
#     "MTMC_Tracking_2024/train/*/*/*.json",
# 	"MTMC_Tracking_2024/val/*/*.json",
#     "MTMC_Tracking_2024/val/*/*.txt",
# 	"MTMC_Tracking_2024/val/*/*/*.mp4",
#     "MTMC_Tracking_2024/val/*/*/*.json",
#     "MTMC_Tracking_2024/test/*/*.json",
#     "MTMC_Tracking_2024/test/*/*.txt",
# 	"MTMC_Tracking_2024/test/*/*/*.mp4",
#     "MTMC_Tracking_2024/test/*/*/*.json",
# 	]
data_files = [
	"MTMC_Tracking_2025/val/Lab_000/depth_maps/*.h5",
]
ds = load_dataset("nvidia/PhysicalAI-SmartSpaces", data_files=data_files)
# data_dir = "MTMC_Tracking_2024/val"
# ds = load_dataset("nvidia/PhysicalAI-SmartSpaces", data_dir=data_dir)

