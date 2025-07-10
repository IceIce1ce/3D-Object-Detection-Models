import os
import sys
import glob
import json
import shutil
from os.path import split

from typing import Optional

import cv2
from loguru import logger
from tqdm import tqdm

import numpy as np

from mtmc.core.objects.units import Camera, MapWorld


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
	0 : "Person", # red
	1 : "Forklift", # green
	2 : "NovaCarter", # blue
	3 : "Transporter", # yellow
	4 : "FourierGR1T2", # purple
	5 : "AgilityDigit", # pink
}

object_type_id = {
	"Person"       : 0, # red
	"Forklift"     : 1, # green
	"NovaCarter"   : 2, # blue
	"Transporter"  : 3, # yellow
	"FourierGR1T2" : 4, # purple
	"AgilityDigit" : 5, # pink
}

color_chart = {
	"Person"      : (162, 162, 245), # red
	"Forklift"    : (0  , 255, 0)  , # green
	"NovaCarter"  : (235, 229, 52) , # blue
	"Transporter" : (0  , 255, 255), # yellow
	"FourierGR1T2": (162, 245, 214), # purple
	"AgilityDigit": (162, 241, 245), # pink
}

categories = [
	{
		"supercategory": "Person",
		"id": 0,
		"name": "Person"
	},
	{
		"supercategory": "Forklift",
		"id": 1,
		"name": "Forklift"
	},
	{
		"supercategory": "NovaCarter",
		"id": 2,
		"name": "NovaCarter"
	},
	{
		"supercategory": "Transporter",
		"id": 3,
		"name": "Transporter"
	},
	{
		"supercategory": "FourierGR1T2",
		"id": 4,
		"name": "FourierGR1T2"
	},
	{
		"supercategory": "AgilityDigit",
		"id": 5,
		"name": "AgilityDigit"
	}
]

number_image_per_camera = 9000  # 9000 images per camera, each camera has 9000 frames
number_image_skip       = 9 # skip every 1 image, so we have 9000 images per camera
number_image_need_to_get= 8000 # 9 images for training, 1 image for validation
train_ratio             = 5
test_ratio              = 3

################################################################################
# REGION: Functions
################################################################################

def extract_frame():
	# Initialize hyperparameters
	folder_input_video = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/train/Warehouse_008/videos"
	folder_output_frame = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/images_extract_full/Warehouse_008/"

	list_video = glob.glob(os.path.join(folder_input_video, "*.mp4"))

	for video_path in tqdm(list_video):
		video_basename       = os.path.basename(video_path)
		video_basename_noext = os.path.splitext(video_basename)[0]
		video_name_correct   = Camera.adjust_camera_id(video_basename_noext)

		# create output folder for each video
		folder_output_frame_folder = os.path.join(folder_output_frame, video_name_correct)
		os.makedirs(folder_output_frame_folder, exist_ok=True)

		# extract frames using ffmpeg
		os.system(f"ffmpeg -i {video_path} -start_number 0 {folder_output_frame_folder}/%07d.jpg")

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

def find_category_id(category_name):
	"""
	Finds the category ID based on the given category name.
	"""
	for category in categories:
		if category["name"] == category_name:
			return category["id"]
	logger.error(f"Category name {category_name} not found.")
	return None

def get_image_id(camera_id, frame_id):
	"""
	Generates a unique image ID based on the camera ID and frame ID.
	"""
	return (camera_id * number_image_per_camera) + frame_id

def read_KITTI_json():
	"""
	Reads the KITTI JSON files and prints the contents.
	"""
	# Define the path to the KITTI JSON files
	kitti_json_path = '/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/KITTI_annotations_format/KITTI_train.json'

	# Check if the directory exists
	if not os.path.exists(kitti_json_path):
		print(f" {kitti_json_path} does not exist.")
		return

	# Read and print each JSON file
	with open(kitti_json_path, 'r') as file:
		data_json = json.load(file)
		# print(f"Contents of {kitti_json_path}:")
		# print(json.dumps(data, indent=4))
		# print("\n")

	# DEBUG: Print the keys in the JSON data
	for key, value in data_json.items():
		print(f"Key: {key}")

def create_info(scene_name, id: Optional[int] = 0, split: Optional[str] = "Train"):
	# info = {
	# 	"id"    : find_scene_id(scene_name),
	# 	"source": "AIC25_Track_1",
	# 	"name"  : f"AIC25_Track_1 {scene_name}",
	# 	"split" : split
	# }
	info = {
		"id"    : id,
		"source": "AIC25_Track_1",
		"name"  : f"AIC25_Track_1_{split}",
		"split" : split
	}
	return info

def create_images_json(scene_name):
	"""
		Creates a JSON file containing image information for the specified scene.
		image {
			"id"			: int,
			"dataset_id"	: int,
			"width"			: int,
			"height"		: int,
			"file_path"		: str,
			"K"			    : list (3x3),
			"src_90_rotate"	: int,					# im was rotated X times, 90 deg counterclockwise
			"src_flagged"	: bool,					# flagged as potentially inconsistent sky direction
		}
	"""
	# Initialize the data structure
	folder_input_frame = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/images_extract_full/"
	calibration_path   = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/train/Warehouse_008/calibration.json"
	groundtruth_path   = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/train/Warehouse_008/ground_truth.json"

	folder_output_json  = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/KITTI_annotations_format/Warehouse_008_KITTI/"
	folder_output_train = os.path.join(folder_output_json, "train")
	folder_output_test = os.path.join(folder_output_json, "test")

	info_train = create_info(scene_name, id = 0, split="Train")
	info_test = create_info(scene_name, id = 1, split="Train")

	# get scene information
	map_cfg = {
		"name"              : scene_name,
		"id"                : find_scene_id(scene_name),
		"type"              : "cartesian",
		"size"              : [1920, 1080],
		"map_image"         : None,
		"calibration_path"  : calibration_path,
		"groundtruth_path"  : None,
		"folder_videos_path": None,
	}
	map_world = MapWorld(map_cfg)

	list_camera = glob.glob(os.path.join(folder_input_frame, scene_name, "*"))

	data_json = {}
	images_train  = []
	images_test   = []
	img_count     = -1
	img_get       = 0
	for camera_path in tqdm(list_camera, desc=f"Processing {scene_name}"):
		# get camera name from the path
		camera_name  = os.path.basename(camera_path)
		camera_name  = Camera.adjust_camera_id(camera_name)
		camera_index = int(camera_name.split("_")[-1]) if "_" in camera_name else 0

		# Get list of images in the camera folder
		list_img = sorted(glob.glob(os.path.join(camera_path, "*.jpg")))

		# load image information

		for img_path in tqdm(list_img, desc=f"Processing {camera_name}"):

			# DEBUG: check skip images
			img_count += 1
			if img_count % number_image_skip != 0:
				continue
			img_get += 1
			if img_get > number_image_need_to_get:
				continue


			# Get image name and ID
			image_name     = os.path.basename(img_path)
			image_id       = int(os.path.splitext(image_name)[0])

			# copy image to the output folder
			ratio = get_image_id(camera_index, image_id) % (train_ratio + test_ratio)
			if ratio < train_ratio:
				img_path_new = os.path.join(folder_output_train, f"{scene_name}___{camera_name}___{image_id:07d}.jpg")
			else:
				img_path_new = os.path.join(folder_output_test, f"{scene_name}___{camera_name}___{image_id:07d}.jpg")
			img_path_short = img_path_new.replace(folder_output_json, "")
			shutil.copyfile(img_path, img_path_new)

			# get image width and height
			img = cv2.imread(img_path)
			if img is None:
				logger.warning(f"Failed to read image: {img_path}")
				continue
			img_h, img_w = img.shape[:2]

			# Create the image entry
			image = {
				"width"     : img_w,
				"height"    : img_h,
				"file_path" : img_path_short,
				"K"         : map_world.cameras[camera_name].intrinsic_matrix,
				"id"        : get_image_id(camera_index, image_id),  # Unique ID for the image
				"dataset_id": "",
			}
			ratio = get_image_id(camera_index, image_id) % (train_ratio + test_ratio)
			if ratio < train_ratio:
				image["dataset_id"] = info_train["id"]
				images_train.append(image)
			else:
				image["dataset_id"] = info_test["id"]
				images_test.append(image)

	# Create the final JSON structure
	data_json = {
		"images": images_train
	}
	with open(os.path.join(folder_output_json, f"{scene_name}_images_train.json"), 'w') as file:
		json.dump(data_json, file, indent=4)

	data_json = {
		"images": images_test
	}
	with open(os.path.join(folder_output_json, f"{scene_name}_images_test.json"), 'w') as file:
		json.dump(data_json, file, indent=4)


def create_annotations_json(scene_name):
	"""
	Creates a JSON file containing annotations for the specified scene in KITTI format.
	object {

			"id"           : int, # unique annotation identifier
			"image_id"     : int, # identifier for image
			"category_id"  : int, # identifier for the category
			"category_name": str, # plain name for the category

			# General 2D/3D Box Parameters.
			# Values are set to -1 when unavailable.
			"valid3D"		: bool,				        # flag for no reliable 3D box
			"bbox2D_tight"	: [x1, y1, x2, y2],			# 2D corners of annotated tight box
			"bbox2D_proj"	: [x1, y1, x2, y2],			# 2D corners projected from bbox3D
			"bbox2D_trunc"	: [x1, y1, x2, y2],			# 2D corners projected from bbox3D then truncated
			"bbox3D_cam"	: [[x1, y1, z1]...[x8, y8, z8]]		# 3D corners in meters and camera coordinates
			"center_cam"	: [x, y, z],				# 3D center in meters and camera coordinates
			"dimensions"	: [width, height, length],		# 3D attributes for object dimensions in meters
			"R_cam"			: list (3x3),				# 3D rotation matrix to the camera frame rotation

			# Optional dataset specific properties,
			# used mainly for evaluation and ignore.
			# Values are set to -1 when unavailable.
			"behind_camera"		: bool,					# a corner is behind camera
			"visibility"		: float, 				# annotated visibility 0 to 1
			"truncation"		: float, 				# computed truncation 0 to 1
			"segmentation_pts"	: int, 					# visible instance segmentation points
			"lidar_pts" 		: int, 					# visible LiDAR points in the object
			"depth_error"		: float,				# L1 of depth map and rendered object
		}
	"""
	# Initialize the data structure
	folder_input_frame = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/KITTI_annotations_format/Warehouse_008_KITTI/"
	calibration_path   = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/train/Warehouse_008/calibration.json"
	groundtruth_path   = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/train/Warehouse_008/ground_truth.json"

	folder_output_json  = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/KITTI_annotations_format/Warehouse_008_KITTI/"

	info = create_info(scene_name, split="Train")

	# get scene information
	map_cfg = {
		"name"              : scene_name,
		"id"                : find_scene_id(scene_name),
		"type"              : "cartesian",
		"size"              : [1920, 1080],
		"map_image"         : None,
		"calibration_path"  : calibration_path,
		"groundtruth_path"  : groundtruth_path,
		"folder_videos_path": None,
	}
	map_world = MapWorld(map_cfg)

	list_image = glob.glob(os.path.join(folder_input_frame, "*/*.jpg"))
	annotations_train = []
	annotations_test = []
	object_id   = 0
	# run through all images in the scene
	for img_path in tqdm(list_image, desc=f"Processing {scene_name}"):

		img_basename            = os.path.basename(img_path)
		img_basename_noext      = os.path.splitext(img_basename)[0]
		_, camera_id, img_index = img_basename_noext.split("___")
		split_dataset        	= os.path.basename(os.path.dirname(img_path))
		img_index               = int(img_index)

		# DEBUG: check if the image is in the training set or test set
		# print(f"Processing image: {img_basename} with camera ID: {camera_id} and index: {img_index} in split {split_dataset}")
		# continue

		# run through all instances in the map world
		# for instance_key in tqdm(map_world.instances, desc=f"Processing instances for image {img_index}"):
		for instance_key in map_world.instances:
			instance = map_world.instances[instance_key]
			# check object is visible in any camera
			if instance.frames is None or str(img_index) not in instance.frames:
				continue

			# check all cameras for this instance
			# for camera in tqdm(map_world.cameras, desc=f"Processing cameras for instance {instance_key}"):
			# for camera in map_world.cameras:
			# camera_id    = map_world.cameras[camera].id  # string like "Camera_0000"
			camera_index = int(camera_id.split("_")[-1]) if "_" in camera_id else 0  # convert to integer

			# check object is visible in specific camera
			if camera_id not in instance.frames[str(img_index)]["bbox_visible_2d"]:
				continue

			# get 3D bounding box in camera coordinates
			_, bbox3D_cam = instance.get_3d_bounding_box_on_2d_image_coordinate(
				location_3d      = instance.frames[str(img_index)]["location_3d"],
				scale_3d         = instance.frames[str(img_index)]["scale_3d"],
				rotation_3d      = instance.frames[str(img_index)]["rotation_3d"],
				intrinsic_matrix = map_world.cameras[camera_id].intrinsic_matrix,
				extrinsic_matrix = map_world.cameras[camera_id].extrinsic_matrix,
			)
			bbox3D_cam = bbox3D_cam.tolist()

			# get rotation matrix and translation vector from camera matrix
			# Example camera matrix P (3x4)
			camera_matrix = np.array(map_world.cameras[camera_id].camera_matrix)

			# Extract rotation matrix (first 3 columns)
			rotation_matrix = camera_matrix[:, :3].tolist()

			# Extract translation vector (last column)
			translation_vector = camera_matrix[:, 3].reshape((3, 1))
			# print("Rotation matrix R:\n", R)
			# print("Translation vector t:\n", t)

			# convert [w, l, h] to [width, height, length]
			dimensions = instance.frames[str(img_index)]["scale_3d"]  # [w, l, h]
			dimensions = [dimensions[0], dimensions[2], dimensions[1]]  # [width, height, length]

			annotation = {
				"id"           : object_id, # unique annotation identifier
				"image_id"     : get_image_id(camera_index, img_index), # identifier for image
				"category_id"  : find_category_id(instance.object_type), # identifier for the category
				"category_name": instance.object_type, # plain name for the category

				# General 2D/3D Box Parameters.
				# Values are set to -1 when unavailable.
				"valid3D"		: True,				        # flag for no reliable 3D box
				"bbox2D_tight"	: instance.frames[str(img_index)]["bbox_visible_2d"][camera_id],			# 2D corners of annotated tight box
				"bbox2D_proj"	: instance.frames[str(img_index)]["bbox_visible_2d"][camera_id],			# 2D corners projected from bbox3D
				"bbox2D_trunc"	: instance.frames[str(img_index)]["bbox_visible_2d"][camera_id],			# 2D corners projected from bbox3D then truncated
				"bbox3D_cam"	: bbox3D_cam,		                                              # 3D corners in meters and camera coordinates
				"center_cam"	: instance.frames[str(img_index)]["location_3d"],				# 3D center in meters and camera coordinates
				"dimensions"	: dimensions,	                                              # [width, height, length], 3D attributes for object dimensions in meters
				"R_cam"			: rotation_matrix,				                             # 3D rotation matrix to the camera frame rotation

				# Optional dataset specific properties,
				# used mainly for evaluation and ignore.
				# Values are set to -1 when unavailable.
				"behind_camera"		: False,				# a corner is behind camera
				"visibility"		: 1, 			     	# annotated visibility 0 to 1
				"truncation"		: -1, 				    # computed truncation 0 to 1
				"segmentation_pts"	: -1, 					# visible instance segmentation points
				"lidar_pts" 		: -1, 					# visible LiDAR points in the object
				"depth_error"		: -1,			    	# L1 of depth map and rendered object
			}
			if split_dataset == "train":
				annotations_train.append(annotation)
			else:
				annotations_test.append(annotation)
			object_id = object_id + 1

	# Create the final JSON structure
	data_json   = {
		"annotations" : annotations_train,
	}
	with open(os.path.join(folder_output_json, f"{scene_name}_annotations_train.json"), 'w') as file:
		json.dump(data_json, file)
	# Create the final JSON structure
	data_json   = {
		"annotations" : annotations_test,
	}
	with open(os.path.join(folder_output_json, f"{scene_name}_annotations_test.json"), 'w') as file:
		json.dump(data_json, file)


def combine_all_components(scene_name):
	"""
	Combines all components (images and annotations) into a single JSON file.
	"""
	folder_output_json = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/KITTI_annotations_format/Warehouse_008_KITTI/"
	images_train_path  = os.path.join(folder_output_json, f"{scene_name}_images_train.json")
	images_test_path   = os.path.join(folder_output_json, f"{scene_name}_images_test.json")
	annotations_train_path = os.path.join(folder_output_json, f"{scene_name}_annotations_train.json")
	annotations_test_path  = os.path.join(folder_output_json, f"{scene_name}_annotations_test.json")

	with open(images_train_path, 'r') as file:
		images_train = json.load(file)

	with open(images_test_path, 'r') as file:
		images_test = json.load(file)

	with open(annotations_train_path, 'r') as file:
		annotations_train = json.load(file)

	with open(annotations_test_path, 'r') as file:
		annotations_test = json.load(file)

	data_json = {
		"info"       : create_info(scene_name, id = 0, split="Train"),
		"categories ": categories,
		"images"     : images_train["images"],
		"annotations": annotations_train["annotations"],
	}
	output_file = os.path.join(folder_output_json, f"{scene_name}_train.json")
	with open(output_file, 'w') as file:
		json.dump(data_json, file)
	logger.info(f"Combined JSON saved to {output_file}")

	data_json = {
		"info"       : create_info(scene_name, id = 1, split="Test"),
		"categories ": categories,
		"images"     : images_test["images"],
		"annotations": annotations_test["annotations"],
	}
	output_file = os.path.join(folder_output_json, f"{scene_name}_test.json")
	with open(output_file, 'w') as file:
		json.dump(data_json, file)
	logger.info(f"Combined JSON saved to {output_file}")


def main():
	scene_name = "Warehouse_008"

	# read_KITTI_json()

	# extract_frame()

	create_images_json(scene_name)

	create_annotations_json(scene_name)

	combine_all_components(scene_name)

if __name__ == "__main__":
	main()