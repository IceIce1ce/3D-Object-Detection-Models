import os
import sys
import glob
import json
import shutil
from os.path import split

from typing import Optional

import cv2
import torch
from loguru import logger
from tqdm import tqdm

import numpy as np

from mtmc.core.objects.units import Camera, MapWorld

from ultilities.boxes import BoxMode, iou

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
		"supercategory": "person",
		"id": 0,
		"name": "person"
	},
	{
		"supercategory": "forklift",
		"id": 1,
		"name": "forklift"
	},
	{
		"supercategory": "novacarter",
		"id": 2,
		"name": "novacarter"
	},
	{
		"supercategory": "transporter",
		"id": 3,
		"name": "transporter"
	},
	{
		"supercategory": "fouriergr1t2",
		"id": 4,
		"name": "fouriergr1t2"
	},
	{
		"supercategory": "agilitydigit",
		"id": 5,
		"name": "agilitydigit"
	}
]

number_image_per_camera = 9000  # 9000 images per camera, each camera has 9000 frames
number_image_skip       = 9 # skip every 1 image, so we have 9000 images per camera
number_image_train      = 5000 # 9 images for training, 1 image for validation
number_image_test       = 3000 # 9 images for training, 1 image for validation
train_ratio             = 5
test_ratio              = 3

################################################################################
# REGION: Functions
################################################################################


class json_serialize(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.integer):
			return int(obj)
		if isinstance(obj, np.floating):
			return float(obj)
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return json.JSONEncoder.default(self, obj)

def estimate_truncation(K, box3d, R, imW, imH):

	box2d, out_of_bounds, fully_behind =  convert_3d_box_to_2d(K, box3d, R, imW, imH)

	if fully_behind:
		return 1.0

	box2d = box2d.detach().cpu().numpy().tolist()[0]
	box2d_XYXY = BoxMode.convert(box2d, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
	image_box = np.array([0, 0, imW-1, imH-1])

	truncation = 1 - iou(np.array(box2d_XYXY)[np.newaxis], image_box[np.newaxis], ign_area_b=True)

	return truncation.item(), box2d_XYXY, fully_behind.cpu().numpy()[0]

def to_float_tensor(input):

	data_type = type(input)

	if data_type != torch.Tensor:
		input = torch.tensor(input)
	return input.float()

def get_cuboid_verts(K, box3d, R=None, view_R=None, view_T=None):
	# make sure types are correct
	K = to_float_tensor(K)
	box3d = to_float_tensor(box3d)
	if R is not None:
		R = to_float_tensor(R)
	squeeze = len(box3d.shape) == 1
	if squeeze:
		box3d = box3d.unsqueeze(0)
		if R is not None:
			R = R.unsqueeze(0)
	n = len(box3d)
	if len(K.shape) == 2:
		K = K.unsqueeze(0).repeat([n, 1, 1])
	corners_3d, _ = get_cuboid_verts_faces(box3d, R)
	if view_T is not None:
		corners_3d -= view_T.view(1, 1, 3)
	if view_R is not None:
		corners_3d = (view_R @ corners_3d[0].T).T.unsqueeze(0)
	if view_T is not None:
		corners_3d[:, :, -1] += view_T.view(1, 1, 3)[:, :, -1]*1.25
	# project to 2D
	corners_2d = K @ corners_3d.transpose(1, 2)
	corners_2d[:, :2, :] = corners_2d[:, :2, :] / corners_2d[:, 2, :].unsqueeze(1)
	corners_2d = corners_2d.transpose(1, 2)
	if squeeze:
		corners_3d = corners_3d.squeeze()
		corners_2d = corners_2d.squeeze()
	return corners_2d, corners_3d

def get_cuboid_verts_faces(box3d=None, R=None):
	"""
	Computes vertices and faces from a 3D cuboid representation.
	Args:
		bbox3d (flexible): [[X Y Z W H L]]
		R (flexible): [np.array(3x3)]
	Returns:
		verts: the 3D vertices of the cuboid in camera space
		faces: the vertex indices per face
	"""

	if box3d is None:
		box3d = [0, 0, 0, 1, 1, 1]

	# make sure types are correct
	box3d = to_float_tensor(box3d)

	if R is not None:
		R = to_float_tensor(R)
	squeeze = len(box3d.shape) == 1

	if squeeze:
		box3d = box3d.unsqueeze(0)

		if R is not None:
			R = R.unsqueeze(0)

	n = len(box3d)
	x3d = box3d[:, 0].unsqueeze(1)
	y3d = box3d[:, 1].unsqueeze(1)
	z3d = box3d[:, 2].unsqueeze(1)
	w3d = box3d[:, 3].unsqueeze(1)
	h3d = box3d[:, 4].unsqueeze(1)
	l3d = box3d[:, 5].unsqueeze(1)
	'''
					v4_____________________v5
					/|                    /|
				   / |                   / |
				  /  |                  /  |
				 /___|_________________/   |
			  v0|    |                 |v1 |
				|    |                 |   |
				|    |                 |   |
				|    |                 |   |
				|    |_________________|___|
				|   / v7               |   /v6
				|  /                   |  /
				| /                    | /
				|/_____________________|/
				v3                     v2
	'''
	verts = to_float_tensor(torch.zeros([n, 3, 8], device=box3d.device))
	# setup X
	verts[:, 0, [0, 3, 4, 7]] = -l3d / 2
	verts[:, 0, [1, 2, 5, 6]] = l3d / 2
	# setup Y
	verts[:, 1, [0, 1, 4, 5]] = -h3d / 2
	verts[:, 1, [2, 3, 6, 7]] = h3d / 2
	# setup Z
	verts[:, 2, [0, 1, 2, 3]] = -w3d / 2
	verts[:, 2, [4, 5, 6, 7]] = w3d / 2
	if R is not None:
		# rotate
		verts = R @ verts

	# translate
	verts[:, 0, :] += x3d
	verts[:, 1, :] += y3d
	verts[:, 2, :] += z3d
	verts = verts.transpose(1, 2)
	faces = torch.tensor([
		[0, 1, 2], # front TR
		[2, 3, 0], # front BL
		[1, 5, 6], # right TR
		[6, 2, 1], # right BL
		[4, 0, 3], # left TR
		[3, 7, 4], # left BL
		[5, 4, 7], # back TR
		[7, 6, 5], # back BL
		[4, 5, 1], # top TR
		[1, 0, 4], # top BL
		[3, 2, 6], # bottom TR
		[6, 7, 3], # bottom BL
	]).float().unsqueeze(0).repeat([n, 1, 1])
	if squeeze:
		verts = verts.squeeze()
		faces = faces.squeeze()
	return verts, faces.to(verts.device)

def convert_3d_box_to_2d(K, box3d, R=None, clipw=0, cliph=0, XYWH=True, min_z=0.20):
	"""
	Converts a 3D box to a 2D box via projection.
	Args:
		K (np.array): intrinsics matrix 3x3
		box3d (flexible): [[X Y Z W H L]]
		R (flexible): [np.array(3x3)]
		clipw (int): clip invalid X to the image bounds. Image width is usually used here.
		cliph (int): clip invalid Y to the image bounds. Image height is usually used here.
		XYWH (bool): returns in XYWH if true, otherwise XYXY format.
		min_z: the threshold for how close a vertex is allowed to be before being
			considered as invalid for projection purposes.
	Returns:
		box2d (flexible): the 2D box results.
		behind_camera (bool): whether the projection has any points behind the camera plane.
		fully_behind (bool): all points are behind the camera plane.
	"""

	# bounds used for vertices behind image plane
	topL_bound = torch.tensor([[0, 0, 0]]).float()
	topR_bound = torch.tensor([[clipw-1, 0, 0]]).float()
	botL_bound = torch.tensor([[0, cliph-1, 0]]).float()
	botR_bound = torch.tensor([[clipw-1, cliph-1, 0]]).float()

	# make sure types are correct
	K = to_float_tensor(K)
	box3d = to_float_tensor(box3d)
	if R is not None:
		R = to_float_tensor(R)

	squeeze = len(box3d.shape) == 1
	if squeeze:
		box3d = box3d.unsqueeze(0)
		if R is not None:
			R = R.unsqueeze(0)
	n = len(box3d)
	verts2d, verts3d = get_cuboid_verts(K, box3d, R)

	# any boxes behind camera plane?
	verts_behind = verts2d[:, :, 2] <= min_z
	behind_camera = verts_behind.any(1)

	verts_signs = torch.sign(verts3d)

	# check for any boxes projected behind image plane corners
	topL = verts_behind & (verts_signs[:, :, 0] < 0) & (verts_signs[:, :, 1] < 0)
	topR = verts_behind & (verts_signs[:, :, 0] > 0) & (verts_signs[:, :, 1] < 0)
	botL = verts_behind & (verts_signs[:, :, 0] < 0) & (verts_signs[:, :, 1] > 0)
	botR = verts_behind & (verts_signs[:, :, 0] > 0) & (verts_signs[:, :, 1] > 0)
	# clip values to be in bounds for invalid points
	verts2d[topL] = topL_bound
	verts2d[topR] = topR_bound
	verts2d[botL] = botL_bound
	verts2d[botR] = botR_bound

	x, xi = verts2d[:, :, 0].min(1)
	y, yi = verts2d[:, :, 1].min(1)
	x2, x2i = verts2d[:, :, 0].max(1)
	y2, y2i = verts2d[:, :, 1].max(1)

	fully_behind = verts_behind.all(1)

	width = x2 - x
	height = y2 - y

	if XYWH:
		box2d = torch.cat((x.unsqueeze(1), y.unsqueeze(1), width.unsqueeze(1), height.unsqueeze(1)), dim=1)
	else:
		box2d = torch.cat((x.unsqueeze(1), y.unsqueeze(1), x2.unsqueeze(1), y2.unsqueeze(1)), dim=1)

	if squeeze:
		box2d = box2d.squeeze()
		behind_camera = behind_camera.squeeze()
		fully_behind = fully_behind.squeeze()

	return box2d, behind_camera, fully_behind

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
	kitti_json_path = '/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/annotations_format_KITTI/KITTI_train.json'

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

def create_images_annotations_json(scene_name):
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
	folder_input_frame = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/images_extract_full/"
	calibration_path   = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/train/Warehouse_008/calibration.json"
	groundtruth_path   = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/train/Warehouse_008/ground_truth.json"

	folder_output_json  = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/annotations_format_KITTI/Warehouse_008_KITTI/"
	folder_output_train = os.path.join(folder_output_json, "Warehouse_008/train")
	folder_output_test = os.path.join(folder_output_json, "Warehouse_008/test")

	os.makedirs(folder_output_train, exist_ok=True)
	os.makedirs(folder_output_test, exist_ok=True)

	info_train = create_info(scene_name, id = 0, split="Train")
	info_test = create_info(scene_name, id = 1, split="Train")

	# get scene information
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

	list_camera = glob.glob(os.path.join(folder_input_frame, scene_name, "*"))

	images_train      = []
	images_test       = []
	img_count         = -1
	img_get           = 0
	img_get_train     = 0
	img_get_test      = 0
	annotations_train = []
	annotations_test  = []
	object_id         = 0
	split_dataset     = "train"
	for camera_path in tqdm(list_camera, desc=f"Processing {scene_name}"):
		# get camera name from the path
		camera_name  = os.path.basename(camera_path)
		camera_name  = Camera.adjust_camera_id(camera_name)
		camera_id    = camera_name
		camera_index = int(camera_name.split("_")[-1]) if "_" in camera_name else 0


		# Get list of images in the camera folder
		list_img = sorted(glob.glob(os.path.join(camera_path, "*.jpg")))

		# load image information

		for img_path in tqdm(list_img, desc=f"Processing {camera_name}"):

			# NOTE: skip every 9 images to reduce the number of images
			img_count += 1
			if img_count % number_image_skip != 0:
				continue
			# check total number of images
			img_get += 1
			if img_get > (number_image_train + number_image_test):
				continue

			# NOTE: FOR IMAGES

			# Get image name and ID
			image_name     = os.path.basename(img_path)
			image_id       = int(os.path.splitext(image_name)[0])
			img_index               = int(image_id)  # index of the image in the camera

			# copy image to the output folder
			ratio = get_image_id(camera_index, image_id) % (train_ratio + test_ratio)
			if ratio < train_ratio:
				split_dataset       = "train"
				img_name_new_noext  = f"{img_get_train:07d}"
				img_path_new        = os.path.join(folder_output_train, f"{img_name_new_noext}.jpg")
				img_get_train      += 1
			else:
				split_dataset       = "test"
				img_name_new_noext  = f"{(number_image_train + img_get_test):07d}"
				img_path_new        = os.path.join(folder_output_test, f"{img_name_new_noext}.jpg")
				img_get_test       += 1
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
				"id"        : int(img_name_new_noext),  # Unique ID for the image
				"dataset_id": "",
			}
			ratio = get_image_id(camera_index, image_id) % (train_ratio + test_ratio)
			if ratio < train_ratio:
				image["dataset_id"] = info_train["id"]
				images_train.append(image)
			else:
				image["dataset_id"] = info_test["id"]
				images_test.append(image)


			# NOTE: FOR ANNOTATIONS

			# DEBUG: check if the image is in the training set or test set
			# print(f"Processing image: {img_basename} with camera ID: {camera_id} and index: {img_index} in split {split_dataset}")
			# continue

			# run through all instances in the map world
			for instance_key in map_world.instances:
				instance = map_world.instances[instance_key]
				# check object is visible in any camera
				if instance.frames is None or str(img_index) not in instance.frames:
					continue


				# check object is visible in specific camera
				if camera_id not in instance.frames[str(img_index)]["bbox_visible_2d"]:
					continue

				# get 3D bounding box in camera coordinates
				proj_box3d, bbox3D_cam = instance.get_3d_bounding_box_on_2d_image_coordinate(
					location_3d      = instance.frames[str(img_index)]["location_3d"],
					scale_3d         = instance.frames[str(img_index)]["scale_3d"],
					rotation_3d      = instance.frames[str(img_index)]["rotation_3d"],
					intrinsic_matrix = map_world.cameras[camera_id].intrinsic_matrix,
					extrinsic_matrix = map_world.cameras[camera_id].extrinsic_matrix,
				)
				bbox3D_cam = bbox3D_cam.tolist()

				# convert [w, l, h] to [width, height, length]
				dimensions = instance.frames[str(img_index)]["scale_3d"]  # [w, l, h]
				dimensions = [dimensions[0], dimensions[2], dimensions[1]]  # [width, height, length]

				box3d = [[instance.frames[str(img_index)]["location_3d"][0],
				          instance.frames[str(img_index)]["location_3d"][1],
				          instance.frames[str(img_index)]["location_3d"][2],
				          dimensions[0],
				          dimensions[1],
				          dimensions[2]
				          ]]

				# box2d, behind_camera, _ = convert_3d_box_to_2d(
				# 	K     = map_world.cameras[camera_id].intrinsic_matrix,
				# 	box3d = box3d,
				# 	R     = map_world.cameras[camera_id].rotation_matrix,
				# 	clipw = img_w,  # image width
				# 	cliph = img_h,  # image height
				# )
				# box2d = box2d.cpu().numpy()[0]
				# behind_camera = behind_camera.cpu().numpy()[0]

				truncation, box2d, behind_camera  = estimate_truncation(
					K     = map_world.cameras[camera_id].intrinsic_matrix,
					box3d = box3d,
					R     = map_world.cameras[camera_id].rotation_matrix,
					imW = img_w,  # image width
					imH = img_h,  # image height
				)

				# DEBUG: print the box2d and bbox3D_cam
				print(f"Box2D: {box2d}, Behind Camera: {behind_camera}, Truncation: {truncation}")

				annotation = {
					"id"           : object_id, # unique annotation identifier
					"image_id"     : int(img_name_new_noext), # identifier for image
					"category_id"  : find_category_id(str(instance.object_type).lower()), # identifier for the category
					"category_name": str(instance.object_type).lower(), # plain name for the category

					# General 2D/3D Box Parameters.
					# Values are set to -1 when unavailable.
					"valid3D"		: True,				        # flag for no reliable 3D box
					"bbox2D_tight"	: instance.frames[str(img_index)]["bbox_visible_2d"][camera_id],			# 2D corners of annotated tight box
					"bbox2D_proj"	: box2d,			# 2D corners projected from bbox3D
					"bbox2D_trunc"	: box2d,			# 2D corners projected from bbox3D then truncated
					"bbox3D_cam"	: bbox3D_cam,		                                              # 3D corners in meters and camera coordinates
					"center_cam"	: instance.frames[str(img_index)]["location_3d"],				# 3D center in meters and camera coordinates
					"dimensions"	: dimensions,	                                              # [width, height, length], 3D attributes for object dimensions in meters
					"R_cam"			: map_world.cameras[camera_id].rotation_matrix,	  # 3D rotation matrix to the camera frame rotation

					# Optional dataset specific properties,
					# used mainly for evaluation and ignore.
					# Values are set to -1 when unavailable.
					"behind_camera"		: behind_camera,				# a corner is behind camera
					"visibility"		: 1, 			     	# annotated visibility 0 to 1
					"truncation"		: truncation, 				    # computed truncation 0 to 1
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
	data_json = {
		"images": images_train
	}
	with open(os.path.join(folder_output_json, f"{scene_name}_images_train.json"), 'w') as file:
		json.dump(data_json, file, indent=4, cls=json_serialize)

	data_json = {
		"images": images_test
	}
	with open(os.path.join(folder_output_json, f"{scene_name}_images_test.json"), 'w') as file:
		json.dump(data_json, file, indent=4, cls=json_serialize)

	# Create the final JSON structure
	data_json   = {
		"annotations" : annotations_train,
	}
	with open(os.path.join(folder_output_json, f"{scene_name}_annotations_train.json"), 'w') as file:
		json.dump(data_json, file, indent=4, cls=json_serialize)
	# Create the final JSON structure
	data_json   = {
		"annotations" : annotations_test,
	}
	with open(os.path.join(folder_output_json, f"{scene_name}_annotations_test.json"), 'w') as file:
		json.dump(data_json, file, indent=4, cls=json_serialize)


def combine_all_components(scene_name):
	"""
	Combines all components (images and annotations) into a single JSON file.
	"""
	folder_output_json = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/annotations_format_KITTI/Warehouse_008_KITTI/"
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
		"categories" : categories,
		"images"     : images_train["images"],
		"annotations": annotations_train["annotations"],
	}
	output_file = os.path.join(folder_output_json, f"{scene_name}_train.json")
	with open(output_file, 'w') as file:
		json.dump(data_json, file)
	logger.info(f"Combined JSON saved to {output_file}")

	data_json = {
		"info"       : create_info(scene_name, id = 1, split="Test"),
		"categories" : categories,
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

	create_images_annotations_json(scene_name)

	combine_all_components(scene_name)

if __name__ == "__main__":
	main()