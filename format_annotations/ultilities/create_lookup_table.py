import math
import os
import queue
import random
import sys
import glob
import json
import time
from copy import deepcopy

from loguru import logger
from sympy.codegen.fnodes import elemental
from tqdm import tqdm

import numpy as np

# from ultilities.change_calibration_file_information import adjust_camera_id_calibration
from ultilities.create_lookup_table_optimize import main_optimize_postprocess, fill_middle_point_between_two_appear
from ultilities.create_w_h_l_object import find_shape_by_type_id
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
	for scene, object_type_specs in tqdm(scene_names_lookup, desc=f"Creating ground truth for {scene_name}"):
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

				# DEBUG:
				# print(object_type)

				if object_type not in object_type_specs:
					continue

				# DEBUG: reduce the number person for lookup table
				# FIXME: remove person in Warehouse_001, Warehouse_003, Warehouse_004
				# if (object_type.lower() == str("Person").lower()
				# 	and scene in ["Warehouse_001","Warehouse_003", "Warehouse_004"]):
				# 		continue

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
	# 	"Person"      : (77, 109, 163), # brown
	# 	"Forklift"    : (162, 245, 214), # light yellow
	# 	"NovaCarter"  : (245, 245, 245), # light pink
	# 	"Transporter" : (0  , 255, 255), # yellow
	# 	"FourierGR1T2": (164, 17 , 157), # purple
	# 	"AgilityDigit": (235, 229, 52) , # blue
	# Initialize the lookup table
	folder_data_version = "MTMC_Tracking_2025_20250614"
	folder_input  = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/{folder_data_version}/"
	folder_output = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/{folder_data_version}/ExtractFrames/lookup_table/"

	scene_name        = "Warehouse_017"
	# scene_names_lookup = ["Warehouse_003", "Warehouse_008", "Warehouse_012", "Warehouse_013", "Warehouse_014", "Warehouse_016"]
	scene_names_lookup = [
		["Warehouse_003", ["Person", "Forklift", "NovaCarter", "Transporter"]],
		["Warehouse_012", ["AgilityDigit", "FourierGR1T2"]],
		["Warehouse_013", ["AgilityDigit", "FourierGR1T2"]],
		["Warehouse_014", ["AgilityDigit", "FourierGR1T2"]],
		["Warehouse_016", ["AgilityDigit", "FourierGR1T2"]]
	]
	create_lookup_table(
		scene_name         = scene_name,
		scene_names_lookup = scene_names_lookup,
		folder_input       = folder_input,
		folder_output      = folder_output
	)

	scene_name        = "Warehouse_018"
	# scene_names_lookup = ["Warehouse_004", "Warehouse_009", "Warehouse_012", "Warehouse_013", "Warehouse_014", "Warehouse_016"]
	scene_names_lookup =[
		["Warehouse_004", ["Person", "Forklift", "NovaCarter", "Transporter"]],
		["Warehouse_012", ["AgilityDigit", "FourierGR1T2"]],
		["Warehouse_013", ["AgilityDigit", "FourierGR1T2"]],
		["Warehouse_014", ["AgilityDigit", "FourierGR1T2"]],
		["Warehouse_016", ["AgilityDigit", "FourierGR1T2"]]
	]
	create_lookup_table(
		scene_name         = scene_name,
		scene_names_lookup = scene_names_lookup,
		folder_input       = folder_input,
		folder_output      = folder_output
	)

	scene_name        = "Warehouse_019"
	# scene_names_lookup = ["Warehouse_001", "Warehouse_005", "Warehouse_012", "Warehouse_013", "Warehouse_014", "Warehouse_016"]
	# scene_names_lookup = ["Warehouse_001", "Warehouse_012", "Warehouse_013", "Warehouse_014", "Warehouse_016"]
	scene_names_lookup = [
		["Warehouse_001", ["Person", "Forklift", "NovaCarter", "Transporter"]],
		["Warehouse_012", ["AgilityDigit", "FourierGR1T2"]],
		["Warehouse_013", ["AgilityDigit", "FourierGR1T2"]],
		["Warehouse_014", ["AgilityDigit", "FourierGR1T2"]],
		["Warehouse_016", ["AgilityDigit", "FourierGR1T2"]]
	]
	create_lookup_table(
		scene_name         = scene_name,
		scene_names_lookup = scene_names_lookup,
		folder_input       = folder_input,
		folder_output      = folder_output
	)

	scene_name        = "Warehouse_020"
	# scene_names_lookup = ["Warehouse_000", "Warehouse_012", "Warehouse_013", "Warehouse_014", "Warehouse_016"]
	# scene_names_lookup = ["Warehouse_006"]
	scene_names_lookup = [
		["Warehouse_000", ["Person", "Forklift", "Transporter"]],
		["Warehouse_012", ["NovaCarter", "AgilityDigit", "FourierGR1T2"]],
		["Warehouse_013", ["NovaCarter", "AgilityDigit", "FourierGR1T2"]],
		["Warehouse_014", ["NovaCarter", "AgilityDigit", "FourierGR1T2"]],
		["Warehouse_016", ["NovaCarter", "AgilityDigit", "FourierGR1T2"]]
	]
	create_lookup_table(
		scene_name         = scene_name,
		scene_names_lookup = scene_names_lookup,
		folder_input       = folder_input,
		folder_output      = folder_output
	)


class KalmanFilter2D:
	def __init__(self, process_variance=1e-2, initial_estimate=np.zeros((4, 1)), measurement_variance=1.0):
		# State: [x, y, vx, vy]
		self.x = initial_estimate
		self.P = np.eye(4) * 500.0  # Initial uncertainty

		# State transition matrix (assuming dt=1)
		self.F = np.array([[1, 0, 1, 0],
		                   [0, 1, 0, 1],
		                   [0, 0, 1, 0],
		                   [0, 0, 0, 1]])

		# Measurement matrix
		self.H = np.array([[1, 0, 0, 0],
		                   [0, 1, 0, 0]])

		# Process noise covariance
		self.Q = np.eye(4) * process_variance

		# Measurement noise covariance
		self.R = np.eye(2) * measurement_variance

	def update(self, x, y):
		z = np.array([[x], [y]])

		# Predict
		self.x = self.F @ self.x
		self.P = self.F @ self.P @ self.F.T + self.Q

		# Update
		y_k = z - self.H @ self.x
		S = self.H @ self.P @ self.H.T + self.R
		K = self.P @ self.H.T @ np.linalg.inv(S)
		self.x = self.x + K @ y_k
		self.P = (np.eye(4) - K @ self.H) @ self.P

		return float(self.x[0]), float(self.x[1])


class YawKalmanFilter:
	def __init__(self, process_noise=0.0001, measurement_noise=0.01, initial_estimate=0.0, initial_uncertainty=1.0):
		self.x = initial_estimate       # Initial yaw estimate in radians
		self.P = initial_uncertainty    # Initial uncertainty
		self.Q = process_noise          # Process noise variance (rad^2)
		self.R = measurement_noise      # Measurement noise variance (rad^2)

	def update(self, measurement_rad):
		# Prediction
		self.P = self.P + self.Q

		# Kalman Gain
		K = self.P / (self.P + self.R)

		# Update estimate
		self.x = self.x + K * (measurement_rad - self.x)

		# Update uncertainty
		self.P = (1 - K) * self.P

		# Return filtered estimate in radians
		return self.x


def angle_with_y_axis(x1, y1, x2, y2):
	dx = x2 - x1
	dy = y2 - y1
	angle = math.atan2(dx, dy)  # Note the order: dx, dy (not dy, dx)
	return angle


def angle_with_linear_regression(x, y):
	# np.random.seed(0)
	# x = np.random.rand(30) * 10    # 30 random x values between 0 and 10
	# y = 2 * x + 1 + np.random.randn(30)  # y = 2x + 1 + noise

	# 2. Linear regression to fit y = m*x + b
	m, b = np.polyfit(x, y, 1)

	# 3. Find the angle with respect to x-axis
	theta_rad = math.atan(m)               # angle in radians
	# theta_deg = math.degrees(theta_rad)    # angle in degrees
	return theta_rad  # return angle in radians


def main_rotated_3d_bbox_object_txt():
	# initialize paths
	folder_data_version  = "MTMC_Tracking_2025_20250614"
	final_result_path_in = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/{folder_data_version}/ExtractFrames/lookup_table/final_result.txt"

	final_result_ou      = os.path.join(os.path.dirname(final_result_path_in), f"{os.path.splitext(os.path.basename(final_result_path_in))[0]}_rotated.txt")

	# <scene_id> <class_id> <object_id> <frame_id> <x> <y> <z> <width> <length> <height> <yaw>
	with open(final_result_ou, 'w') as f_out:
		with open(final_result_path_in, 'r') as f_in:
			lines = f_in.readlines()
			for line in tqdm(lines, desc=f"Processing rotation of 3D bounding boxes"):
				parts = line.split()
				object_type = object_type_name[int(parts[1])]
				if object_type in ["AgilityDigit"]:
					f_out.write(f"{parts[0]} {parts[1]} {parts[2]} {parts[3]} {parts[4]} {parts[5]} {parts[6]} {parts[8]} {parts[7]} {parts[9]} {parts[10]}\n")
				else:
					f_out.write(f"{parts[0]} {parts[1]} {parts[2]} {parts[3]} {parts[4]} {parts[5]} {parts[6]} {parts[7]} {parts[8]} {parts[9]} {parts[10]}\n")


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
			# shape["pitch"]      = shape_data["pitch"]
			shape["pitch"]      = 0.0
			# shape["roll"]       = shape_data["roll"]
			shape["roll"]       = 0.0
			shape["yaw"]        = shape_data["yaw"]
			if object_type  in ["Forklift", "NovaCarter", "Transporter"]:
				shape["width"]      = shape_data["width"]
				shape["length"]     = shape_data["length"]
				shape["height"]     = shape_data["height"]
			else:
				shape["width"]  = json_data_lookup_table[object_type]["shape_avg"]["width"]
				shape["length"] = json_data_lookup_table[object_type]["shape_avg"]["length"]
				shape["height"] = json_data_lookup_table[object_type]["shape_avg"]["height"]
				return shape

	return shape


def create_trajectory_middle_result_postprocess(scene_name, json_path_lookup_table, data_preprocess, folder_output, postfix ="avg"):
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
		data_preprocess (dict): the preprocessed JSON file.
		folder_output (str): Path to the output folder where the lookup table will be saved.
	Returns
		None
	"""
	number_image_per_camera = 9000
	frame_period            = 60
	# json_data_output_path   = os.path.join(folder_output, f"{scene_name}_postprocess_{postfix}.json")
	json_data_output_path   = os.path.join(folder_output, f"{scene_name}_postprocess.json")

	# load lookup table
	with open(json_path_lookup_table, 'r') as f:
		json_data_lookup_table = json.load(f)

	# load preprocessed JSON file
	# with open(json_path_preprosess, 'r') as f:
	# 	data_preprocess = json.load(f)

	json_data_output = deepcopy(data_preprocess)
	for object_id in tqdm(json_data_output, desc=f"Processing creating postprocess in {scene_name}"):
		object_type_index = json_data_output[object_id]["object_type_id"]
		object_type       = object_type_name[object_type_index]

		current_yaw    = 0.0
		pbar = tqdm(range(number_image_per_camera))
		fps  = 0.0
		yaw_kf      = YawKalmanFilter(initial_estimate=0.0)

		for img_index in range(frame_period):
			if str(img_index) in json_data_output[object_id]["frames"]:
				location_kf = KalmanFilter2D(initial_estimate = np.array([
					[float(json_data_output[object_id]["frames"][str(img_index)]["x"])], # x coordinate
					[float(json_data_output[object_id]["frames"][str(img_index)]["y"])], # y coordinateq
					[0.0],  # initial velocity x
					[0.0]   # initial velocity y
				]))
				break

		# Initialize a queue to store the last frame_period frames' x and y coordinates
		queue_xs = []
		queue_ys = []
		for img_index in range(frame_period):
			if str(img_index) in json_data_output[object_id]["frames"]:
				measured_x = json_data_output[object_id]["frames"][str(img_index)]["x"]
				measured_y = json_data_output[object_id]["frames"][str(img_index)]["y"]
				queue_xs.append(measured_x)
				queue_ys.append(measured_y)
				# filtered_x, filtered_y = location_kf.update(measured_x, measured_y)
				# queue_xs.append(filtered_x)
				# queue_ys.append(filtered_y)


		for img_index in range(number_image_per_camera):

			frame_current_name = str(img_index)

			if frame_current_name not in json_data_output[object_id]["frames"]:
				continue

			pbar.set_description(f"{scene_name} -- {object_id} -- {object_type} -- {fps:.2f} fps")

			# get index
			frame_check_name = str(img_index + frame_period)
			if frame_check_name not in json_data_output[object_id]["frames"]:
				frame_check_name = frame_current_name
			elif img_index + frame_period >= number_image_per_camera:
				frame_check_name = str(number_image_per_camera - 1)

			# find yaw
			# if frame_current_name != frame_check_name:
				# Find the angle between the current frame and the next-period frame (2 points)
				# current_yaw = angle_with_y_axis(
				# 	json_data_output[object_id]["frames"][frame_current_name]["x"],
				# 	json_data_output[object_id]["frames"][frame_current_name]["y"],
				# 	json_data_output[object_id]["frames"][frame_check_name]["x"],
				# 	json_data_output[object_id]["frames"][frame_check_name]["y"],
				# )

				# Find the angle with linear regression (multiple points)
				# measured_x = json_data_output[object_id]["frames"][frame_current_name]["x"]
				# measured_y = json_data_output[object_id]["frames"][frame_current_name]["y"]
			measured_x = json_data_output[object_id]["frames"][frame_check_name]["x"]
			measured_y = json_data_output[object_id]["frames"][frame_check_name]["y"]
			queue_xs.append(measured_x)
			queue_ys.append(measured_y)
			# filtered_x, filtered_y = location_kf.update(measured_x, measured_y)
			# queue_xs.append(filtered_x)
			# queue_ys.append(filtered_y)
			if len(queue_xs) > frame_period:
				queue_xs.pop(0)
				queue_ys.pop(0)
			current_yaw = angle_with_linear_regression(np.array(queue_xs), np.array(queue_ys))

			# Apply Kalman filter to smooth the yaw angle
			# current_yaw = yaw_kf.update(current_yaw)

			json_data_output[object_id]["frames"][frame_current_name]["yaw"] = current_yaw
			# point_center = [json_data_output[object_id]["frames"][frame_current_name]["x"],	json_data_output[object_id]["frames"][frame_current_name]["y"]]
			# point_center = [filtered_x,	filtered_y]
			point_center = [measured_x,	measured_y]

			# find width, length, height
			start = time.time()
			# shape = find_shape_by_type_and_yaw(
			# 	json_data_lookup_table = json_data_lookup_table,
			# 	object_type            = object_type,
			# 	point_center           = point_center,
			# 	yaw                    = current_yaw
			# )
			shape = find_shape_by_type_id(
				scene_id               = find_scene_id(scene_name),
				json_data_lookup_table = json_data_lookup_table,
				object_id              = object_id,
				object_type            = object_type,
				point_center           = point_center,
				yaw                    = current_yaw
			)
			end = time.time()
			fps = 1 / (end - start)
			pbar.set_description(f"{scene_name} -- {object_id} -- {object_type} -- {fps:.2f} fps")
			json_data_output[object_id]["frames"][frame_current_name]["z"] = shape["height"] / 2.0  # z is half of height
			json_data_output[object_id]["frames"][frame_current_name]["w"] = shape["width"]
			json_data_output[object_id]["frames"][frame_current_name]["h"] = shape["height"]
			# json_data_output[object_id]["frames"][frame_current_name]["l"] = shape["length"]
			json_data_output[object_id]["frames"][frame_current_name]["l"] = shape["height"] / 3.0

			if object_type in ["Forklift"]:
				json_data_output[object_id]["frames"][frame_current_name]["yaw"] = shape["yaw"]

			pbar.update(1)
		pbar.close()

	# write output JSON file
	with open(json_data_output_path, 'w') as f:
		json.dump(json_data_output, f, cls=json_serialize)


def main_create_trajectory_middle_result_postprocess_json(scene_name_specific_list = None):
	folder_data_version = "MTMC_Tracking_2025_20250614"
	folder_input  = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/{folder_data_version}/ExtractFrames/lookup_table/"
	folder_output = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/{folder_data_version}/ExtractFrames/lookup_table/"
	list_scene    = ["Warehouse_017", "Warehouse_018", "Warehouse_019", "Warehouse_020"]
	# list_scene    = ["Warehouse_019"]
	postfix        = "avg" # or "max", "min", etc. based on your requirement

	# check if scene_name_specific_list is None
	if scene_name_specific_list is not None:
		list_scene = scene_name_specific_list

	for scene_name in tqdm(list_scene):
		# DEBUG:
		# if scene_name not in ["Warehouse_017", "Warehouse_019"]:
		# 	continue

		json_path_lookup_table = os.path.join(folder_input, f"{scene_name}_lookup_table.json")
		json_path_preprosess   = os.path.join(folder_input, f"{scene_name}_preprocess.json")

		# load preprocessed JSON file
		with open(json_path_preprosess, 'r') as f:
			data_preprocess = json.load(f)

		create_trajectory_middle_result_postprocess(
			scene_name             = scene_name,
			json_path_lookup_table = json_path_lookup_table,
			data_preprocess        = data_preprocess,
			folder_output          = folder_output,
			postfix                = postfix
		)


def main_create_trajectory_middle_result_postprocess_txt(scene_name_specific_list = None):
	"""Structure of the preprocessed JSON file:
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
	"""
	folder_data_version = "MTMC_Tracking_2025_20250614"
	folder_input  = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/{folder_data_version}/ExtractFrames/lookup_table/"
	folder_output = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/{folder_data_version}/ExtractFrames/lookup_table/"
	list_scene    = ["Warehouse_017", "Warehouse_018", "Warehouse_019", "Warehouse_020"]
	# list_scene    = ["Warehouse_019"]

	# check if scene_name_specific_list is None
	if scene_name_specific_list is not None:
		list_scene = scene_name_specific_list

	for scene_name in tqdm(list_scene):
		# DEBUG:
		# if scene_name not in ["Warehouse_017"]:
		# 	continue

		txt_file_in  = os.path.join(folder_input, f"{scene_name}_250617.txt")
		# txt_file_in  = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/{folder_data_version}/ExtractFrames/lookup_table/{scene_name}_250617.txt"

		scene_id     = find_scene_id(scene_name)
		json_path_lookup_table = os.path.join(folder_input, f"{scene_name}_lookup_table.json")
		final_result = load_final_result(txt_file_in, scene_id)


		# <scene_id> <class_id> <object_id> <frame_id> <x> <y> <z> <width> <length> <height> <yaw>
		# Structure of the preprocessed JSON file:
		# 		{
		# 		    "object_id" : {
		# 		        "object_type_id": .,
		# 		        "frames" : {
		# 		            "frame_id" : {
		# 		                "x":,
		# 		                "y":,
		# 		            },
		# 		            ...
		# 		        }
		# 		    }
		# 		    ...
		# 		}
		# Create the strutucture as preprocessed JSON file
		data_preprocess = {}
		for result_index, result in enumerate(final_result):
			if int(result[0]) != scene_id:
				continue
			object_type = object_type_name[int(result[1])]
			object_id   = int(result[2])
			frame_id    = int(result[3])
			x           = float(result[4])
			y 		    = float(result[5])
			if object_id not in data_preprocess:
				data_preprocess[object_id] = {
					"object_type_id": int(result[1]),
					"frames": {}
				}
			# add new frame data
			data_preprocess[object_id]["frames"][str(int(frame_id))] = {
				"x": x,
				"y": y,
			}

		# Fill middle points between two appear
		for object_id in data_preprocess:
			data_preprocess[object_id] = fill_middle_point_between_two_appear(data_object_preprocess = data_preprocess[object_id])

		create_trajectory_middle_result_postprocess(
			scene_name             = scene_name,
			json_path_lookup_table = json_path_lookup_table,
			data_preprocess        = data_preprocess,
			folder_output          = folder_output,
			postfix                = "avg"  # or "max", "min", etc. based on your requirement
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
	folder_data_version = "MTMC_Tracking_2025_20250614"
	folder_input  = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/{folder_data_version}/ExtractFrames/lookup_table/"
	folder_output = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/{folder_data_version}/ExtractFrames/lookup_table/"
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
		Create the final result file for each scene.
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
					f_write.write(f"{scene_id} {object_type_index} {int(object_id)} {img_index} "
								  f"{frame_data['x']:.2f} {frame_data['y']:.2f} {frame_data['z']:.2f} "
								  f"{frame_data['w']:.2f} {frame_data['l']:.2f} {frame_data['h']:.2f} "
								  f"{frame_data['yaw']:.2f}\n")
					# f_write.write(f"{scene_id} {object_type_index} {int(object_id)} {img_index} "
					#               f"{frame_data['x']:f} {frame_data['y']:f} {frame_data['z']:f} "
					#               f"{frame_data['w']:f} {frame_data['l']:f} {frame_data['h']:f} "
					#               f"{frame_data['yaw']:f}\n")
				except KeyError as e:
					f_write.write(f"{scene_id} {object_type_index} {int(object_id)} {img_index} "
								  f"{frame_data['x']:.2f} {frame_data['y']:.2f} {frame_data['z']:.2f} "
								  f"{frame_data['w']:.2f} {frame_data['l']:.2f} {frame_data['h']:.2f} "
								  f"{frame_data['rotation']:.2f}\n")
					# f_write.write(f"{scene_id} {object_type_index} {int(object_id)} {img_index} "
					#               f"{frame_data['x']:f} {frame_data['y']:f} {frame_data['z']:f} "
					#               f"{frame_data['w']:f} {frame_data['l']:f} {frame_data['h']:f} "
					#               f"{frame_data['rotation']:f}\n")

				# DEBUG:
				# break


def main_create_final_result():
	"""
		Create the final result file for each scene.
		By combining all the postprocessed JSON files into a single text file.
	Returns:
		None
	"""
	folder_data_version = "MTMC_Tracking_2025_20250614"
	folder_input  = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/{folder_data_version}/ExtractFrames/lookup_table/"
	folder_output = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/{folder_data_version}/ExtractFrames/lookup_table/"
	list_scene    = ["Warehouse_017", "Warehouse_018", "Warehouse_019", "Warehouse_020"]
	# list_scene    = ["Warehouse_019"]

	# create final result file
	postfix           = "avg"  # or "max", "min", etc. based on your requirement
	# result_final_path = os.path.join(folder_output, f"final_result_{postfix}.txt")
	result_final_path = os.path.join(folder_output, f"final_result.txt")
	with open(result_final_path, 'w') as f:
		f.write("")

	for scene_name in tqdm(list_scene):
		# DEBUG:
		# if scene_name not in ["Warehouse_019"]:
		# 	continue

		# json_data_postprocess_path   = os.path.join(folder_input, f"{scene_name}_postprocess_{postfix}.json")
		json_data_postprocess_path   = os.path.join(folder_input, f"{scene_name}_postprocess.json")
		create_final_result(
			scene_name             = scene_name,
			json_path_postprosess  = json_data_postprocess_path,
			result_final_path      = result_final_path
		)


def main():
	# main_create_lookup_table()
	# main_create_trajectory_middle_result_postprocess_json()
	main_create_trajectory_middle_result_postprocess_txt()

	main_optimize_postprocess()

	main_create_final_result()

	# main_rotated_3d_bbox_object_txt()
	pass


if __name__ == "__main__":
	# Get the current script directory
	main()