import os
import sys
import json
import glob
from functools import cmp_to_key

from loguru import logger
from tqdm import tqdm

import numpy as np

scene_id_table ={
	"Train" : {
		0 : "Warehouse_000",
		1 : "Warehouse_001",
		2 : "Warehouse_002",
		3 : "Warehouse_003",
		4 : "Warehouse_004",
		5 : "Warehouse_005",
		6 : "Warehouse_006",
		7 : "Warehouse_007",
		8 : "Warehouse_008",
		9 : "Warehouse_009",
		10: "Warehouse_010",
		11: "Warehouse_011",
		12: "Warehouse_012",
		13: "Warehouse_013",
		14: "Warehouse_014",
	},
	"Val" : {
		15: "Warehouse_015",
		16: "Warehouse_016",
		22: "Lab_000",
		23: "Hospital_000",
	},
	"Test" : {
		17: "Warehouse_017",
		18: "Warehouse_018",
		19: "Warehouse_019",
		20: "Warehouse_020",
	}
}

object_type_name = {
	0 : "Person", # green
	1 : "Forklift", # green
	2 : "NovaCarter", # pink
	3 : "Transporter", # yellow
	4 : "FourierGR1T2", # purple
	5 : "AgilityDigit", # blue
}

object_type_id = {
	"Person"       : 0, # green
	"Forklift"     : 1, # green
	"NovaCarter"   : 2, # pink
	"Transporter"  : 3, # yellow
	"FourierGR1T2" : 4, # purple
	"AgilityDigit" : 5, # blue
}

color_chart = {
	"Person"      : (77, 109, 163), # brown
	"Forklift"    : (162, 245, 214), # light yellow
	"NovaCarter"  : (245, 245, 245), # light pink
	"Transporter" : (0  , 255, 255), # yellow
	"FourierGR1T2": (164, 17 , 157), # purple
	"AgilityDigit": (235, 229, 52) , # blue
}


def find_scene_name(scene_id):
	"""
	Finds the scene name based on the given scene ID.
	"""
	if scene_id in scene_id_table["Train"]:
		return scene_id_table["Train"][scene_id]
	elif scene_id in scene_id_table["Val"]:
		return scene_id_table["Val"][scene_id]
	elif scene_id in scene_id_table["Test"]:
		return scene_id_table["Test"][scene_id]
	else:
		logger.error(f"Scene ID {scene_id} not found in any dataset split.")
		return None


def find_scene_id(scene_name):
	"""
	Finds the scene ID based on the given scene name.
	"""
	for split in scene_id_table:
		for scene_id, name in scene_id_table[split].items():
			if name == scene_name:
				return scene_id
	logger.error(f"Scene name {scene_name} not found in any dataset split.")
	return None


def custom_result_sort(part_a, part_b):
	"""frame_id->object_id->class_id"""
	# <scene_id> <class_id> <object_id> <frame_id> <x> <y> <z> <width> <length> <height> <yaw>
	if part_a[3] != part_b[3]:
		return int(part_a[3]) - int(part_b[3])
	if part_a[2] != part_b[2]:
		return int(part_a[2]) - int(part_b[2])
	if part_a[1] != part_b[1]:
		return int(part_a[1]) - int(part_b[1])
	return 0


def main():
	# initialize paths
	folder_info_in    = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/lookup_table/"
	folder_in         = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/image_bev_track/bev_track_1_size_full/"
	final_result_path = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/lookup_table/final_result_avg.txt"

	folder_out  = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/lookup_table/"
	list_scene  = ["Warehouse_017", "Warehouse_018", "Warehouse_019", "Warehouse_020"]

	for scene_name in tqdm(list_scene):
		# DEBUG:
		# if scene_name not in ["Warehouse_017", "Warehouse_019"]:
		# 	continue

		folder_img_in = os.path.join(folder_in, scene_name)
		folder_img_ou = os.path.join(folder_out, scene_name)
		scene_id      = find_scene_id(scene_name)

		# load calibration info
		json_calibration_path = os.path.join(folder_info_in, f"{scene_name}_calibration.json")
		with open(json_calibration_path, 'r') as f:
			json_data_calibration = json.load(f)

		scale_factor                      = float(json_data_calibration["sensors"][0]["scaleFactor"])
		translation_to_global_coordinates = json_data_calibration["sensors"][0]["translationToGlobalCoordinates"]

		# load final result
		# <scene_id> <class_id> <object_id> <frame_id> <x> <y> <z> <width> <length> <height> <yaw>
		final_result = []
		with open(final_result_path, 'r') as f:
			lines = f.readlines()
			for line in lines:
				parts = np.array(line.split(), dtype=np.float32)
				if int(parts[0]) == scene_id:
					final_result.append(parts)

		final_result = sorted(final_result, key=cmp_to_key(custom_result_sort))

if __name__ == "__main__":
	main()