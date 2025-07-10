import math
import os
import queue
import random
import shutil
import sys
import glob
import json
import time
from copy import deepcopy

from loguru import logger
from sympy.codegen.fnodes import elemental
from tqdm import tqdm

import numpy as np

from ultilities.change_calibration_file_information import adjust_camera_id_calibration
from ultilities.process_final_result import load_final_result

################################################################################
# REGION: Hyperparameter
################################################################################


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



################################################################################
# REGION: Functions
################################################################################


# Create a JSON Encoder class
class json_serialize(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.integer):
			return int(obj)
		if isinstance(obj, np.floating):
			return float(obj)
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return json.JSONEncoder.default(self, obj)


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


def fill_middle_point_between_two_appear(data_object_preprocess):
	number_image_per_camera = 9000
	img_start_info = None
	img_end_info   = None
	is_need_fill   = False
	for img_index in range(number_image_per_camera):
		frame_name = str(img_index)
		if frame_name in data_object_preprocess["frames"]:
			if is_need_fill is True and img_start_info is not None:
				img_end_info = {
					"img_index" : str(img_index),
					"points"    : np.array([data_object_preprocess["frames"][frame_name]["x"],
					                        data_object_preprocess["frames"][frame_name]["y"]], dtype=np.float32)
				}
				# Add between points
				period_start = int(img_start_info["img_index"])
				period_end   = int(img_end_info["img_index"])
				if period_end - period_start < 100:
					for index in range(period_start + 1, period_end):
						# calculate the filling point
						# points_middle = (points_start + (float(frame_id - period_start) / float(period_end - period_start)) * (points_end - points_start))
						points_middle = (img_start_info["points"] + (float(index - period_start) / float(period_end - period_start)) * (img_end_info["points"] - img_start_info["points"]))
						data_object_preprocess["frames"][str(index)] = {
							"x": points_middle[0],
							"y": points_middle[1]
						}

				is_need_fill = False

			img_start_info = {
				"img_index" : str(img_index),
				"points"    : np.array([data_object_preprocess["frames"][frame_name]["x"],
				                        data_object_preprocess["frames"][frame_name]["y"]], dtype=np.float32)
			}
		elif img_start_info is not None:
			is_need_fill = True # start to find the filling point

	return data_object_preprocess



def optimize_add_object_to_trajectory_middle_result_postprocess(scene_name, json_data):
	number_image_per_camera = 9000
	warehouse_info          = None
	if scene_name in ["Warehouse_019"]:
		warehouse_info =  [
			{
				"object_type_id": 1,
				"x": 4.1413405809049735,
				"y": -1.678615078264393,
				"z": 1.0755656460918708,
				"width": 1.2138070722294594,
				"length": 2.3133653342342373,
				"height": 2.154944661372646,
				"pitch": 0.0,
				"roll": 0.0,
				"yaw": -1.0122909435268244
			},
			{
				"object_type_id": 1,
				"x": -7.253448251575037,
				"y": -2.840932607650757,
				"z": 1.0755656460918708,
				"width": 1.2138067355646172,
				"length": 2.313365426807877,
				"height": 2.154944661372646,
				"pitch": 0.0,
				"roll": 0.0,
				"yaw": -1.5707963267948963
			},
		]
	elif scene_name in ["Warehouse_020"]:
		warehouse_info = [
			{
				"object_type_id": 1,
				"x": 2.589342578738723,
				"y": 10.800000190734863,
				"z": 1.0755656460918708,
				"width": 1.2138061252130683,
				"length": 2.3133651216321027,
				"height": 2.154944661372646,
				"pitch": 0.0,
				"roll": 0.0,
				"yaw": 1.5707963267948963
			},
			{
				"object_type_id": 1,
				"x": -25.224881057729135,
				"y": -6.948549302286722,
				"z": 1.4133285626111793,
				"width": 1.129382298975088,
				"length": 1.8731548653582877,
				"height": 2.767817697648307,
				"pitch": 0.0,
				"roll": -0.0,
				"yaw": -3.141592653589793
			},
			{
				"object_type_id": 1,
				"x": 3.013970038228422,
				"y": 28.885050659169565,
				"z": 1.4133285626111793,
				"width": 1.129382298975088,
				"length": 1.873155170534062,
				"height": 2.767817697648307,
				"pitch": 0.0,
				"roll": 0.0,
				"yaw": 1.570796326794897
			},
		]
	if warehouse_info is not None:
		object_id = 0
		for object_info in warehouse_info:
			# check object_id is not in json_data
			while str(object_id) in json_data:
				object_id += 1

			# initialize object data
			json_data[str(object_id)] = {
				"object_type_id": object_info["object_type_id"],
				"frames": {}
			}

			# add object
			for img_index in range(number_image_per_camera):
				json_data[str(object_id)]["frames"][str(int(img_index))] = {
					"x": object_info["x"],
					"y": object_info["y"],
					"z": object_info["z"],
					"yaw": object_info["yaw"],
					"w": object_info["width"],
					"h": object_info["height"],
					"l": object_info["length"],
				}

	return json_data


def optimize_rotated_3d_bbox_object(scene_name, json_data):
	number_image_per_camera = 9000

	for object_id in json_data:
		object_type_index = json_data[object_id]["object_type_id"]
		object_type       = object_type_name[object_type_index]
		if object_type in ["AgilityDigit"]:
			for frame in json_data[object_id]["frames"]:
				# get x, y, z, yaw, w, h, l
				x = json_data[object_id]["frames"][frame]["x"]
				y = json_data[object_id]["frames"][frame]["y"]
				z = json_data[object_id]["frames"][frame]["z"]
				yaw = json_data[object_id]["frames"][frame]["yaw"]
				w = json_data[object_id]["frames"][frame]["w"]
				h = json_data[object_id]["frames"][frame]["h"]
				l = json_data[object_id]["frames"][frame]["l"]

				# update frame data
				json_data[object_id]["frames"][frame] = {
					"x": x,
					"y": y,
					"z": z,
					"yaw": yaw,
					"w": l,  # width is length for agility digit
					"h": h,
					"l": w   # length is width for agility digit
				}


	return json_data


def main_optimize_postprocess():
	"""
		Structure of the postrocessed JSON file:
			{
				"object_id" : {
					"object_type_id": .,
					"frames" : {
						"frame_id" : {
							"x":,
							"y":,
							"z":,
							"yaw":,
							"w":
							"h":
							"l":
						},
						...
					}
				}
				...
			}
	Returns:

	"""
	folder_data_version = "MTMC_Tracking_2025_20250614"
	folder_input        = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/{folder_data_version}/ExtractFrames/lookup_table/"
	list_scene          = ["Warehouse_017", "Warehouse_018", "Warehouse_019", "Warehouse_020"]
	postfix             = "avg"  # or "max", "min", etc. based on your requirement

	pbar = tqdm(list_scene)
	for scene_name in tqdm(list_scene):
		# DEBUG:
		# if scene_name not in ["Warehouse_017"]:
		# 	continue

		pbar.set_description(f"Optimizing postprocess for {scene_name}")

		# json_path_result = os.path.join(folder_input, f"{scene_name}_postprocess_{postfix}.json")
		json_path_result = os.path.join(folder_input, f"{scene_name}_postprocess.json")

		# load lookup table
		with open(json_path_result, 'r') as f:
			json_data = json.load(f)

		# copy json_data to back up
		shutil.copy(json_path_result, json_path_result.replace(".json", "_backup.json"))

		# add forklift
		json_data = optimize_add_object_to_trajectory_middle_result_postprocess(scene_name, json_data)

		# rotated 3d bounding box for agility digit
		json_data = optimize_rotated_3d_bbox_object(scene_name, json_data)

		# write output JSON file
		with open(json_path_result, 'w') as f:
			json.dump(json_data, f)

		pbar.update(1)
	pbar.close()


def main():
	main_optimize_postprocess()


if __name__ == "__main__":
	# Get the current script directory
	main()