import math
import os
import random
import sys
import glob
import json
import time
from copy import deepcopy

from loguru import logger
from tqdm import tqdm

import numpy as np

from ultilities.change_calibration_file_information import adjust_camera_id_calibration

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


def create_lookup_table(scene_name, scene_names_lookup, folder_input, folder_output):
	"""Create lookup table for a specific scene.
	   Structure of the lookup table:
	   		{
	   			"type_name": {
	   				"type_id": int,
	   				"color": (B, G, R),
	   				"shape_max": { "width": , "length": , "height":} , # [width, length, height]
					"shape_min": { "width": , "length": , "height":} , # [width, length, height]
					"shape_avg": { "width": , "length": , "height":} , # [width, length, height]
	   				"shapes": [
	   					{x, y, z, width, length, height, pitch, roll ,yaw}
	   					...
	   				]
	   			}
	   			...
            }
        Structure of the groundtruth JSON file:
        {
		  "<frame_id>": [
		    {
		      "object type": "<class_name>",
		      "object id": <int>,
		      "3d location": [x, y, z],
		      "3d bounding box scale": [w, l, h],
		      "3d bounding box rotation": [pitch, roll, yaw],
		      "2d bounding box visible": {
		        "<camera_id>": [xmin, ymin, xmax, ymax]
		      }
		    }
		  ]
		}
	Args:
		scene_name (str): Name of the scene to create the lookup table for.
		scene_names_lookup (list): List of scene names to look up.
		folder_input (str): Path to the input folder containing JSON files.
		folder_output (str): Path to the output folder where the lookup table will be saved.
	Returns:
		None
	"""
	# Create the output folder if it does not exist
	if not os.path.exists(folder_output):
		os.makedirs(folder_output)

	# create lookup table with object types
	lookup_table = {}
	for type_id, type_name in object_type_name.items():
		# Create a dictionary to hold the lookup table for this object type
		lookup_table[type_name] = {
			"type_id"    : type_id,
			"color"      : color_chart[type_name],
			"shape_max": { "width": 0           , "length": 0           , "height": 0 }          , # [width, length, height]
			"shape_min": { "width": float('inf'), "length": float('inf'), "height": float('inf')}, # [width, length, height]
			"shape_avg": { "width": 0.0         , "length": 0.0         , "height": 0.0 }        , # [width, length, height]
			"shape_count": 0,  # Count of shapes for averaging
			"shapes"     : []
		}


	# Create the lookup table for the specific scene
	index_count = 0
	for scene in tqdm(scene_names_lookup, desc=f"Creating ground truth for {scene_name}"):
		json_path_groundtruth = glob.glob(os.path.join(folder_input, f"*/{scene}/ground_truth.json"))[0]
		json_path_calibration = glob.glob(os.path.join(folder_input, f"*/{scene}/calibration.json"))[0]
		with open(json_path_groundtruth, 'r') as f:
			json_data_groundtruth = json.load(f)

		with open(json_path_calibration, 'r') as f:
			json_data_calibration = json.load(f)

		scale_factor =  json_data_calibration["sensors"][0]["scaleFactor"]

		for frame_id, frame_data in tqdm(json_data_groundtruth.items(), desc=f"Processing creating lookup table in {scene}"):
			for object_instance in frame_data:
				object_id       = object_instance["object id"]
				object_type     = object_instance["object type"]
				object_shape    = object_instance["3d bounding box scale"]
				object_roation  = object_instance["3d bounding box rotation"]
				object_location = object_instance["3d location"]

				# DEBUG: reduce the number person for lookup table
				# FIXME: remove person in Warehouse_001, Warehouse_003, Warehouse_004
				if (object_type.lower() == str("Person").lower()
					and scene in ["Warehouse_001","Warehouse_003", "Warehouse_004"]):
						continue

				# {width, length, height, pitch, roll ,yaw}
				lookup_table[object_type]["shapes"].append({
					"x"     : object_location[0],
					"y"     : object_location[1],
					"z"     : object_location[2],
					"width" : object_shape[0],
					"length": object_shape[1],
					"height": object_shape[2],
					"pitch" : object_roation[0],
					"roll"  : object_roation[1],
					"yaw"   : object_roation[2],
				})

				# find max min, height for agility digit and fourier gr1t2
				# [width, length, height]
				lookup_table[object_type]["shape_max"]["width"] = max(lookup_table[object_type]["shape_max"]["width"],object_shape[0])
				lookup_table[object_type]["shape_min"]["width"] = min(lookup_table[object_type]["shape_min"]["width"],object_shape[0])
				lookup_table[object_type]["shape_avg"]["width"] += object_shape[0]
				lookup_table[object_type]["shape_max"]["length"] = max(lookup_table[object_type]["shape_max"]["length"],object_shape[1])
				lookup_table[object_type]["shape_min"]["length"] = min(lookup_table[object_type]["shape_min"]["length"],object_shape[1])
				lookup_table[object_type]["shape_avg"]["length"] += object_shape[1]
				lookup_table[object_type]["shape_max"]["height"] = max(lookup_table[object_type]["shape_max"]["height"],object_shape[2])
				lookup_table[object_type]["shape_min"]["height"] = min(lookup_table[object_type]["shape_min"]["height"],object_shape[2])
				lookup_table[object_type]["shape_avg"]["height"] += object_shape[2]
				lookup_table[object_type]["shape_count"] += 1


	for type_id, type_name in object_type_name.items():
		if lookup_table[type_name]["shape_count"] > 0:
			lookup_table[type_name]["shape_avg"]["width"]  /= lookup_table[type_name]["shape_count"]
			lookup_table[type_name]["shape_avg"]["length"] /= lookup_table[type_name]["shape_count"]
			lookup_table[type_name]["shape_avg"]["height"] /= lookup_table[type_name]["shape_count"]

	# DEBUG: print max min height for agility digit and fourier gr1t2
	for type_id, type_name in object_type_name.items():
		print(f"{type_name} shape ::::: \n "
		      f"max -- {lookup_table[type_name]['shape_max']}, \n "
		      f"min -- {lookup_table[type_name]['shape_min']}, \n "
		      f"avg -- {lookup_table[type_name]['shape_avg']}, \n "
		      f"count -- {lookup_table[type_name]['shape_count']}")


	# Save the lookup table to a JSON file
	output_json_path = os.path.join(folder_output, f"{scene_name}_lookup_table.json")
	with open(output_json_path, 'w') as f:
		json.dump(lookup_table, f)

	# output_json_path = os.path.join(folder_output, f"{scene_name}_calibration.json")
	# with open(output_json_path, 'w') as f:
	# 	json.dump(json_data_calibration, f)
	# adjust_camera_id_calibration(output_json_path)


def main_create_lookup_table():
	# Initialize the lookup table
	folder_input  = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/"
	folder_output = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/lookup_table/"

	scene_name        = "Warehouse_017"
	scene_names_lookup = ["Warehouse_003", "Warehouse_008", "Warehouse_012", "Warehouse_013", "Warehouse_014", "Warehouse_016"]
	create_lookup_table(
		scene_name         = scene_name,
		scene_names_lookup = scene_names_lookup,
		folder_input       = folder_input,
		folder_output      = folder_output
	)

	scene_name        = "Warehouse_018"
	scene_names_lookup = ["Warehouse_004", "Warehouse_009", "Warehouse_012", "Warehouse_013", "Warehouse_014", "Warehouse_016"]
	create_lookup_table(
		scene_name         = scene_name,
		scene_names_lookup = scene_names_lookup,
		folder_input       = folder_input,
		folder_output      = folder_output
	)

	scene_name        = "Warehouse_019"
	scene_names_lookup = ["Warehouse_001", "Warehouse_005", "Warehouse_012", "Warehouse_013", "Warehouse_014", "Warehouse_016"]
	create_lookup_table(
		scene_name         = scene_name,
		scene_names_lookup = scene_names_lookup,
		folder_input       = folder_input,
		folder_output      = folder_output
	)

	scene_name        = "Warehouse_020"
	scene_names_lookup = ["Warehouse_006", "Warehouse_012", "Warehouse_013", "Warehouse_014", "Warehouse_016"]
	# scene_names_lookup = ["Warehouse_006"]
	create_lookup_table(
		scene_name         = scene_name,
		scene_names_lookup = scene_names_lookup,
		folder_input       = folder_input,
		folder_output      = folder_output
	)


def angle_with_y_axis(x1, y1, x2, y2):
	dx = x2 - x1
	dy = y2 - y1
	angle = math.atan2(dx, dy)  # Note the order: dx, dy (not dy, dx)
	return angle


def find_shape_by_type_and_yaw(json_data_lookup_table, object_type, point_center, yaw):
	"""

	Args:
		json_data_lookup_table (dict): The lookup table containing shape data for different object types.
		object_type (str): The type of the object to find the shape for.
		point_center (list): The center point of the object in the format [x, y].
		yaw (float): The yaw angle of the object in radians.

	Returns:
		{width, length, height, pitch, roll ,yaw}
	"""
	shape = {
		"width" : 0.0,
		"length": 0.0,
		"height": 0.0,
		"pitch" : 0.0,
		"roll"  : 0.0,
		"yaw"   : 0.0,
	}
	yaw_min             = float('inf')
	distance_center_min = float('inf')
	for shape_data in json_data_lookup_table[object_type]["shapes"]:

		is_update = False
		if object_type in ["Forklift"]:
			# Calculate the distance from the center point to the shape's center
			distance_center = math.sqrt(
				(point_center[0] - shape_data["x"]) ** 2 +
				(point_center[1] - shape_data["y"]) ** 2
			)
			if distance_center < distance_center_min:
				distance_center_min = distance_center
				is_update = True
		elif object_type not in ["Forklift"]:
			# calculate the yaw difference
			yaw_current = abs((shape_data["yaw"] % (2 * math.pi)) - yaw)
			if yaw_current < yaw_min:
				yaw_min = yaw_current
				is_update = True

		if is_update:
			if object_type  in ["Forklift", "NovaCarter", "Transporter"]:
				shape["width"]      = shape_data["width"]
				shape["length"]     = shape_data["length"]
				shape["height"]     = shape_data["height"]
			else:
				shape["width"]  = json_data_lookup_table[object_type]["shape_max"]["width"]
				shape["length"] = json_data_lookup_table[object_type]["shape_max"]["length"]
				shape["height"] = json_data_lookup_table[object_type]["shape_avg"]["height"]
			# shape["pitch"]      = shape_data["pitch"]
			shape["pitch"]      = 0.0
			# shape["roll"]       = shape_data["roll"]
			shape["roll"]       = 0.0
			shape["yaw"]        = shape_data["yaw"]
	return shape


# def create_trajectory_middle_result_each_object():


def create_trajectory_middle_result_postprocess(scene_name, json_path_lookup_table, json_path_preprosess, folder_output):
	"""
		Structure of the lookup table:
	   		{
	   			"type_name": {
	   				"type_id": int,
	   				"color": (B, G, R),
	   				"shape_max": { "width": , "length": , "height":} , # [width, length, height]
					"shape_min": { "width": , "length": , "height":} , # [width, length, height]
					"shape_avg": { "width": , "length": , "height":} , # [width, length, height]
	   				"shapes": [
	   					{x, y, z, width, length, height, pitch, roll ,yaw}
	   					...
	   				]
	   			}
	   			...
            }
		Structure of the preprocessed JSON file:
		{
		    "object_id" : {
		        "object_type_id": .,
		        "frames" : {
		            "frame_id" : {
		                "x":,
		                "y":,
		            },
		            ...
		        }
		    }
		    ...
		}
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
	Args:
		scene_name (str): Name of the scene to create the lookup table for.
		json_path_lookup_table (str): Path to the lookup table JSON file.
		json_path_preprosess (str): Path to the preprocessed JSON file.
		folder_output (str): Path to the output folder where the lookup table will be saved.
	Returns
		None
	"""
	number_image_per_camera = 9000
	frame_period            = 3
	json_data_output_path   = os.path.join(folder_output, f"{scene_name}_postprocess_avg.json")

	# load lookup table
	with open(json_path_lookup_table, 'r') as f:
		json_data_lookup_table = json.load(f)

	# load preprocessed JSON file
	with open(json_path_preprosess, 'r') as f:
		json_data_preprocess = json.load(f)


	json_data_output = deepcopy(json_data_preprocess)
	for object_id in tqdm(json_data_output, desc=f"Processing creating postprocess in {scene_name}"):
		object_type_index = json_data_output[object_id]["object_type_id"]
		object_type       = object_type_name[object_type_index]

		current_yaw    = 0.0
		pbar = tqdm(range(number_image_per_camera))
		fps  = 0.0
		for img_index in range(number_image_per_camera):

			frame_current_name = str(img_index)

			if frame_current_name not in json_data_output[object_id]["frames"]:
				continue

			pbar.set_description(f"Processing images in {scene_name} -- {object_id} -- {object_type} -- Elapsed time: {fps:.2f} fps")

			# get index
			frame_check_name = str(img_index + frame_period)
			if frame_check_name not in json_data_output[object_id]["frames"]:
				frame_check_name = frame_current_name
			elif img_index + frame_period >= number_image_per_camera:
				frame_check_name = str(number_image_per_camera - 1)

			# find yaw
			if frame_current_name != frame_check_name:
				current_yaw = angle_with_y_axis(
					json_data_output[object_id]["frames"][frame_current_name]["x"],
					json_data_output[object_id]["frames"][frame_current_name]["y"],
					json_data_output[object_id]["frames"][frame_check_name]["x"],
					json_data_output[object_id]["frames"][frame_check_name]["y"],
				)
			json_data_output[object_id]["frames"][frame_current_name]["yaw"] = current_yaw
			point_center = [json_data_output[object_id]["frames"][frame_current_name]["x"],	json_data_output[object_id]["frames"][frame_current_name]["y"]]

			# find width, length, height
			start = time.time()
			shape = find_shape_by_type_and_yaw(
				json_data_lookup_table = json_data_lookup_table,
				object_type            = object_type,
				point_center           = point_center,
				yaw                    = current_yaw
			)
			end = time.time()
			fps = 1 / (end - start)
			pbar.set_description(f"Processing images in {scene_name} -- {object_id} -- {object_type} -- {fps:.2f} fps")

			json_data_output[object_id]["frames"][frame_current_name]["z"] = shape["height"] / 2.0  # z is half of height
			json_data_output[object_id]["frames"][frame_current_name]["w"] = shape["width"]
			json_data_output[object_id]["frames"][frame_current_name]["h"] = shape["height"]
			json_data_output[object_id]["frames"][frame_current_name]["l"] = shape["length"]

			pbar.update(1)
		pbar.close()

	# write output JSON file
	with open(json_data_output_path, 'w') as f:
		json.dump(json_data_output, f)


def main_create_trajectory_middle_result_postprocess(scene_name_specific = None):
	folder_input  = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/lookup_table/"
	folder_output = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/lookup_table/"
	list_scene    = ["Warehouse_017", "Warehouse_018", "Warehouse_019", "Warehouse_020"]
	# list_scene    = ["Warehouse_019"]

	for scene_name in tqdm(list_scene):
		# DEBUG:
		if scene_name not in ["Warehouse_019"]:
			continue

		json_path_lookup_table = os.path.join(folder_input, f"{scene_name}_lookup_table.json")
		json_path_preprosess   = os.path.join(folder_input, f"{scene_name}_preprocess.json")
		create_trajectory_middle_result_postprocess(
			scene_name             = scene_name,
			json_path_lookup_table = json_path_lookup_table,
			json_path_preprosess   = json_path_preprosess,
			folder_output          = folder_output
		)


def create_final_result_mot_for_hota_evaluation(scene_name, json_path_postprosess, result_mot_hota, is_3d = False):
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

		MOTChallenge Format (.txt per sequence):
			Each line:
			<frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<x>,<y>,<z>
			<frame>: frame number (starts at 1)
			<id>: object ID
			<bb_left>, <bb_top>, <bb_width>, <bb_height>: bounding box
			<conf>: confidence (set to 1 for GT, or detection confidence for results)
	Args:
		scene_name (str): Name of the scene to create the final result for.
		json_path_postprosess (str): Path to the postprocessed JSON file.
		result_mot_hota (str): Path to the final result file to be created.

	Returns:

	"""
	number_image_per_camera = 9000

	# load postprocessed JSON file
	with open(json_path_postprosess, 'r') as f:
		json_data_postprocess = json.load(f)

	scene_id = find_scene_id(scene_name)

	with open(result_mot_hota, 'w') as f_write:

		for object_id in tqdm(json_data_postprocess, desc=f"Processing final result in {scene_name}"):
			object_type_index = json_data_postprocess[object_id]["object_type_id"]
			object_type       = object_type_name[object_type_index]

			for img_index in tqdm(range(number_image_per_camera), desc=f"Processing images in {scene_name} -- {object_id} -- {object_type}"):

				frame_current_name = str(img_index)
				if frame_current_name not in json_data_postprocess[object_id]["frames"]:
					continue

				frame_data = json_data_postprocess[object_id]["frames"][frame_current_name]

				# <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<x>,<y>,<z>
				bb_left   = int(frame_data['x'] - frame_data['w'] / 2.0)
				bb_top    = int(frame_data['y'] - frame_data['l'] / 2.0)
				bb_width  = int(frame_data['w'])
				bb_height = int(frame_data['l'])
				if is_3d:
					# For 3D results, we need to include z coordinate
					x         = frame_data['x']
					y         = frame_data['y']
					z         = frame_data['z']
				else:
					x = -1
					y = -1
					z = -1
				conf      = random.random(0.8, 0.9)  # Random confidence value between 0.9 and 1.0
				f_write.write(f"{img_index},{int(object_id)},"
				              f"{bb_left},{bb_top},{bb_width},{bb_height},"
				              f"{conf},{x},{y},{z}\n")


def main_create_final_result_for_hota_evaluation():
	folder_input  = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/lookup_table/"
	folder_output = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/lookup_table/"
	list_scene    = ["Warehouse_017", "Warehouse_018", "Warehouse_019", "Warehouse_020"]
	# list_scene    = ["Warehouse_019"]

	# create final result file

	for scene_name in tqdm(list_scene):
		# DEBUG:
		# if scene_name not in ["Warehouse_019"]:
		# 	continue

		result_mot_hota              = os.path.join(folder_output, f"{scene_name}_hota_evaluation.txt")
		json_data_postprocess_path   = os.path.join(folder_input, f"{scene_name}_postprocess_avg.json")
		create_final_result_mot_for_hota_evaluation(
			scene_name             = scene_name,
			json_path_postprosess  = json_data_postprocess_path,
			result_mot_hota        = result_mot_hota,
			is_3d                  = False
		)


def create_final_result(scene_name, json_path_postprosess, result_final_path):
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
	Args:
		scene_name (str): Name of the scene to create the final result for.
		json_path_postprosess (str): Path to the postprocessed JSON file.
		result_final_path (str): Path to the final result file to be created.

	Returns:

	"""
	number_image_per_camera = 9000

	# load postprocessed JSON file
	with open(json_path_postprosess, 'r') as f:
		json_data_postprocess = json.load(f)

	scene_id = find_scene_id(scene_name)

	with open(result_final_path, 'a') as f_write:

		for object_id in tqdm(json_data_postprocess, desc=f"Processing final result in {scene_name}"):
			object_type_index = json_data_postprocess[object_id]["object_type_id"]
			object_type       = object_type_name[object_type_index]

			for img_index in tqdm(range(number_image_per_camera), desc=f"Processing images in {scene_name} -- {object_id} -- {object_type}"):

				frame_current_name = str(img_index)
				if frame_current_name not in json_data_postprocess[object_id]["frames"]:
					continue

				frame_data = json_data_postprocess[object_id]["frames"][frame_current_name]

				# <scene_id> <class_id> <object_id> <frame_id> <x> <y> <z> <width> <length> <height> <yaw>
				try:
					# f_write.write(f"{scene_id} {object_type_index} {int(object_id)} {img_index} "
					#               f"{frame_data['x']:.2f} {frame_data['y']:.2f} {frame_data['z']:.2f} "
					#               f"{frame_data['w']:.2f} {frame_data['l']:.2f} {frame_data['h']:.2f} "
					#               f"{frame_data['yaw']:.2f}\n")
					f_write.write(f"{scene_id} {object_type_index} {int(object_id)} {img_index} "
					              f"{frame_data['x']:f} {frame_data['y']:f} {frame_data['z']:f} "
					              f"{frame_data['w']:f} {frame_data['l']:f} {frame_data['h']:f} "
					              f"{frame_data['yaw']:f}\n")
				except KeyError as e:
					# f_write.write(f"{scene_id} {object_type_index} {int(object_id)} {img_index} "
					#               f"{frame_data['x']:.2f} {frame_data['y']:.2f} {frame_data['z']:.2f} "
					#               f"{frame_data['w']:.2f} {frame_data['l']:.2f} {frame_data['h']:.2f} "
					#               f"{frame_data['rotation']:.2f}\n")
					f_write.write(f"{scene_id} {object_type_index} {int(object_id)} {img_index} "
					              f"{frame_data['x']:f} {frame_data['y']:f} {frame_data['z']:f} "
					              f"{frame_data['w']:f} {frame_data['l']:f} {frame_data['h']:f} "
					              f"{frame_data['rotation']:f}\n")


def main_create_final_result():
	folder_input  = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/lookup_table/"
	folder_output = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/lookup_table/"
	list_scene    = ["Warehouse_017", "Warehouse_018", "Warehouse_019", "Warehouse_020"]
	# list_scene    = ["Warehouse_019"]

	# create final result file
	result_final_path = os.path.join(folder_output, "final_result_avg.txt")
	with open(result_final_path, 'w') as f:
		f.write("")

	for scene_name in tqdm(list_scene):
		# DEBUG:
		# if scene_name not in ["Warehouse_019"]:
		# 	continue

		json_data_postprocess_path   = os.path.join(folder_input, f"{scene_name}_postprocess_avg.json")
		create_final_result(
			scene_name             = scene_name,
			json_path_postprosess  = json_data_postprocess_path,
			result_final_path      = result_final_path
		)



def main():
	# main_create_lookup_table()
	main_create_trajectory_middle_result_postprocess()
	main_create_final_result()
	pass


if __name__ == "__main__":
	# Get the current script directory
	main()