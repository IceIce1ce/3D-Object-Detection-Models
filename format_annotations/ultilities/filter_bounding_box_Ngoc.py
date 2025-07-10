import sys
import os
import glob
import shutil
import json

from tqdm import tqdm
from loguru import logger

import numpy as np
import cv2
import shapely

from mtmc.core.objects.units import MapWorld, Camera, Instance
# from mtmc.core.utils.bbox import bbox_xywh_to_cxcywh_norm

################################################################################
# REGION: Hyperparameter
################################################################################

color_chart = {
	"Person"      : (162, 162, 245), # red
	"Forklift"    : (0  , 255, 0)  , # green
	"NovaCarter"  : (235, 229, 52) , # blue
	"Transporter" : (0  , 255, 255), # yellow
	"FourierGR1T2": (162, 245, 214), # purple
	"AgilityDigit": (162, 241, 245), # pink
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

################################################################################
# REGION: Functions
################################################################################

def bbox_xywh_to_cxcywh(xywh: np.ndarray) -> np.ndarray:
	"""Convert the bounding box's format from
	[top_left_x, top_left_y, width, height] to
	[center_x, center_y, width, height].
	"""
	cxcywh = xywh.copy()
	cxcywh = cxcywh.astype(float)
	if cxcywh.ndim == 1:
		cxcywh[0]    = cxcywh[0] + (cxcywh[2] / 2.0)
		cxcywh[1]    = cxcywh[1] + (cxcywh[3] / 2.0)
	elif cxcywh.ndim == 2:
		cxcywh[:, 0] = cxcywh[:, 0] + (cxcywh[:, 2] / 2.0)
		cxcywh[:, 1] = cxcywh[:, 1] + (cxcywh[:, 3] / 2.0)
	else:
		raise ValueError(f"Farray dimensions {cxcywh.ndim} is not "
		                 f"supported.")
	return cxcywh

def bbox_xywh_to_cxcywh_norm(xywh: np.ndarray, height, width) -> np.ndarray:
	"""Convert the bounding box's format from
	[top_left_x, top_left_y, width, height] to
	[center_x_norm, center_y_norm, width_norm, height_norm].
	Where:
		- F[center_x_norm, center_y_norm, width_norm, height_norm] are
		  normalized in the range `[0.0, 1.0]`.
		  For example:
			  `x_norm = absolute_x / image_width`
			  `height_norm = absolute_height / image_height`.
	"""
	cxcywh_norm = bbox_xywh_to_cxcywh(xywh)
	if cxcywh_norm.ndim == 1:
		cxcywh_norm[0] /= width
		cxcywh_norm[1] /= height
		cxcywh_norm[2] /= width
		cxcywh_norm[3] /= height
	elif cxcywh_norm.ndim == 2:
		cxcywh_norm[:, 0] = cxcywh_norm[:, 0] / width
		cxcywh_norm[:, 1] = cxcywh_norm[:, 1] / height
		cxcywh_norm[:, 2] = cxcywh_norm[:, 2] / width
		cxcywh_norm[:, 3] = cxcywh_norm[:, 3] / height
	else:
		raise ValueError(f"Farray dimensions {cxcywh_norm.ndim} is not "
		                 f"supported.")
	return cxcywh_norm

def rename_files_folder():
	# init folder
	folder_input          = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames"
	folder_input_mot_lbl  = os.path.join(folder_input, "labels_mot")
	folder_input_yolo_lbl = os.path.join(folder_input, "labels_yolo")
	folder_input_img      = os.path.join(folder_input, "images")

	# camera_list_mot_lbl = glob.glob(os.path.join(folder_input_mot_lbl, "*/*"))
	# camera_list_yolo_lbl = glob.glob(os.path.join(folder_input_yolo_lbl, "*/*"))
	camera_list_img = glob.glob(os.path.join(folder_input_img, "*/*"))

	for camera_path in tqdm(camera_list_img):
		camera_path_old = camera_path
		camera_name_old = os.path.basename(camera_path)

		if os.path.isdir(camera_path):
			camera_name_new = Camera.adjust_camera_id(camera_name_old)
		else:
			camera_basename_old, ext = os.path.splitext(camera_name_old)
			camera_name_new          = f"{Camera.adjust_camera_id(camera_basename_old)}{ext}"

		camera_path_new = os.path.join(os.path.dirname(camera_path), camera_name_new)

		try:
			# Rename the folder
			os.rename(camera_path_old, camera_path_new)
			print(f"Folder '{camera_path_old}' successfully renamed to '{camera_path_new}'.")
		except FileNotFoundError:
			print(f"Error: Folder '{camera_path_old}' not found.")
		except FileExistsError:
			print(f"Error: Folder '{camera_path_new}' already exists.")
		except Exception as e:
			print(f"An unexpected error occurred: {e}")

def get_view_point_name(scene_name, camera_name):
	"""Get view point name from scene name and camera name.
	"""
	camera_name = os.path.basename(camera_name)
	scene_name  = os.path.basename(scene_name)
	view_name   = f"{scene_name}__{camera_name}"
	return view_name

def create_viewpoints_image():
	# init folder
	folder_input      = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames"
	folder_input_img  = os.path.join(folder_input, "images")
	folder_output_roi = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/view_points/"

	camera_list_img = glob.glob(os.path.join(folder_input_img, "*/*"))

	for camera_path in tqdm(camera_list_img):
		# get information of camera
		camera_name = os.path.basename(camera_path)
		scene_name  = os.path.basename(os.path.dirname(camera_path))
		# view_name   = f"{scene_name}__{camera_name}"
		view_name   = get_view_point_name(scene_name, camera_name)

		# get view point image
		view_image_path_in  = glob.glob(os.path.join(camera_path, "*.png"))[0]
		img_in              = cv2.imread(view_image_path_in)
		view_image_path_ou  = os.path.join(folder_output_roi, f"{view_name}.jpg")

		# copy image to the folder, and conver it to jpg
		# shutil.copy(view_image_path_in, view_image_path_ou)
		cv2.imwrite(view_image_path_ou, img_in, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

def load_json_filter_bounding_box_x_anylabeling(json_path):
	# Load the JSON file
	with open(json_path, "r") as f:
		data = json.load(f)

	# Extract shapes
	shapes = data.get("shapes", [])

	# Collect (label, points) pairs
	rois = []
	for shape in shapes:
		label   = shape.get("label", "undefined")
		points  = shape.get("points", [])
		polygon = shapely.geometry.Polygon(points)
		rois.append({
			"label"  : label,
			"points" : points,
			"polygon": polygon,
		})

	return rois, data.get('imageWidth'), data.get('imageHeight')

# TODO: FILTER HERE
def filter_bounding_box(scene_name, camera_name, rois, instance_data, img_size):
	"""

	Args:
		scene_name:
		camera_name:
		rois:
		instance_data: {
				"camera_id"      : camera_name,
				"frame_id"       : int(parts[0]),
				"object id"      : int(parts[1]), # track_id
				"x_tl"           : max(float(parts[2]), 1.0),
				"y_tl"           : max(float(parts[3]), 1.0),
				"x_br"           : min(float(parts[2]) + float(parts[4]), img_size[0] - 1),
				"y_br"           : min(float(parts[3]) + float(parts[5]), img_size[1] - 1),
				"w"              : float(parts[4]),
				"h"              : float(parts[5]),
				"not_ignored"    : int(parts[6]),
				"object type"    : object_type_name[int(parts[7])],
				"object_type_int": int(parts[7]),
				"visibility"     : float(parts[8]),
			}
		img_size:

	Returns:

	"""
	# get the center bottom of bounding box
	point_bottom_center = shapely.geometry.Point(
		float(instance_data["x_tl"] + (instance_data["w"] / 2.0)),
		min(float(instance_data["y_br"]), img_size[1] - 5),
	)

	# check ZERO size of bounding box
	if instance_data["w"] <= 0 or instance_data["h"] <= 0:
		return False

	# check inside the none ROI
	if rois is not None:
		for roi in rois:
			if roi["label"] == "none":
				if roi["polygon"].contains(point_bottom_center):
					return False
			if roi["label"] == "none-person" and instance_data["object type"] == "Person":
				if roi["polygon"].contains(point_bottom_center):
					return False
			if roi["label"] == "none-transporter" and instance_data["object type"] == "Transporter":
				if roi["polygon"].contains(point_bottom_center):
					return False
			if roi["label"] == "none-novacarter" and instance_data["object type"] == "NovaCarter":
				if roi["polygon"].contains(point_bottom_center):
					return False

	# check ratio of the bounding box
	# if instance_data["object type"] in ["Person"]:
	# 	if instance_data["h"] / instance_data["w"] < 1.5 / 1:
	# 		return False

	# if instance_data["object type"] in ["Person", "AgilityDigit"]:		
	if instance_data["h"] < 25 or instance_data["w"] < 25:
		if instance_data["object type"] in ["NovaCarter"]:
			if instance_data["h"] < 25 or instance_data["w"] < 25:
				return False
		else:
			return False

	# remove specific object id
	# if scene_name == scene_name and camera_name == camera_name:
	# 	if instance_data["object id"] in [15]:
	# 		return False

	# remove specific object id in specific frame id 
	# ngochdm

	if scene_name == "Warehouse_016":

		if camera_name == "Camera_0001":
			if instance_data["frame_id"] in [3225, 5475]:
				if instance_data["object id"] == 353:
					return False
			if instance_data["frame_id"] in [1155]:
				if instance_data["object id"] == 352:
					return False				
			if instance_data["frame_id"] in [1305, 1335, 1365, 1395, 1425, 1455, 1485, 1515, 1545, 1575, 1605, 4335, 4395, 4575, 5415, 6345, 7425]:
				if instance_data["object id"] == 354:
					return False
			if instance_data["frame_id"] in [255, 1215, 2235, 4155, 4515]:
				if instance_data["object id"] == 625:
					return False
			if instance_data["frame_id"] in [4515, 5865, 8595]:
				if instance_data["object id"] == 626:
					return False
			if instance_data["frame_id"] in [3885, 6255, 7665, 7815]:
				if instance_data["object id"] == 627:
					return False
			if instance_data["frame_id"] in [2955, 3495, 4455, 4695, 5055, 5235, 7185, 7215, 7245, 7275, 7305, 7335, 7365, 7395]:
				if instance_data["object id"] == 182:
					return False
		
		if camera_name == "Camera_0002":
			if instance_data["object id"] == 182:
				deleted_frames = [2955, 3855]
				for i in range(225, 440, 30): deleted_frames.append(i)
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 351:
				deleted_frames = [645]
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 352:
				deleted_frames = [1785]
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 354:
				deleted_frames = [6315, 7425, 7695]
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 625:
				deleted_frames = [8505]
				for i in range(3555,3680, 30): deleted_frames.append(i)
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 626:
				deleted_frames = [3195, 7905]
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 699:
				deleted_frames = [4065]
				for i in range(8385,9000, 30): deleted_frames.append(i)
				if instance_data["frame_id"] in deleted_frames:
					return False

		if camera_name == "Camera_0003":
			if instance_data["object id"] == 182:
				deleted_frames = [885, 2985, 7185, 8565]
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 350:
				deleted_frames = [2235, 2955, 2985, 7605, 8625]
				for i in range(15,170,30): deleted_frames.append(i)
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 351:
				deleted_frames = [195, 645, 1365, 3495, 4935, 7275]
				for i in range(1455,1640,30): deleted_frames.append(i)
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 352:
				deleted_frames = [525, 1155, 1185, 4845, 5655, 8685, 8805, 8865]
				for i in range(1215,1610,30): deleted_frames.append(i)
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 353:
				deleted_frames = [3225, 3975, 5925, 8775]
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 354:
				deleted_frames = [1485, 2595, 3735, 4125, 4215, 8505, 8535]
				for i in range(1515,1680,30): deleted_frames.append(i)
				for i in range(7515,7640,30): deleted_frames.append(i)
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 625:
				deleted_frames = [285, 495, 585, 615, 1545, 3975, 4455, 4605, 6765]
				for i in range(8895,9000,30): deleted_frames.append(i)
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 626:
				deleted_frames = [375, 405, 435, 465, 3525, 4065, 5175, 5205]
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 627:
				deleted_frames = [615, 2325, 2595, 3285, 3375, 4875, 4905, 5895, 6645, 6765, 8385]
				for i in range(645, 860, 30): deleted_frames.append(i)
				for i in range(1665,2150,30): deleted_frames.append(i)
				for i in range(2355,2420,30): deleted_frames.append(i)
				for i in range(2655,2930,30): deleted_frames.append(i)
				for i in range(3015,3110,30): deleted_frames.append(i)
				for i in range(3915,4010,30): deleted_frames.append(i)
				for i in range(4965,5090,30): deleted_frames.append(i)
				for i in range(5145,5270,30): deleted_frames.append(i)
				for i in range(5505,5690,30): deleted_frames.append(i)
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 699:
				deleted_frames = [3705, 3795]
				for i in range(0,9): deleted_frames.append(i * 30 + 15)
				for i in range(705, 1490,30): deleted_frames.append(i)
				for i in range(2865,2930,30): deleted_frames.append(i)
				if instance_data["frame_id"] in deleted_frames:
					return False

		if camera_name == "Camera_0004":
			if instance_data["object id"] == 182:
				deleted_frames = [5955, 6015]
				for i in range(555, 710, 30): deleted_frames.append(i)
				for i in range(2805,2900,30): deleted_frames.append(i)
				for i in range(5295,5600,30): deleted_frames.append(i)
				for i in range(8265,8330,30): deleted_frames.append(i)
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 352:
				deleted_frames = [975, 1755, 6195]
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 353:
				deleted_frames = [555, 585, 1905]
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 354:
				deleted_frames = [1725, 2415]
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 626:
				deleted_frames = [5625]
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 627:
				deleted_frames = [8985]
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 699:
				deleted_frames = []
				for i in range(105, 200, 30): deleted_frames.append(i)
				for i in range(6405,6740,30): deleted_frames.append(i)
				if instance_data["frame_id"] in deleted_frames:
					return False

		if camera_name == "Camera_0005":
			if instance_data["object id"] == 182:
				deleted_frames = [2475, 5085, 6345, 6405]
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 351:
				deleted_frames = [1155]
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 352:
				deleted_frames = [3015, 3645, 3795, 5385, 6045, 6615, 6855, 7365]
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 353:
				deleted_frames = [1275, 3615, 6375, 6435, 7725]
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 354:
				deleted_frames = [1245]
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 625:
				deleted_frames = [4935]
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 626:
				deleted_frames = [6075]
				for i in range(1035,1310,30): deleted_frames.append(i)
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 699:
				deleted_frames = [5355]
				if instance_data["frame_id"] in deleted_frames:
					return False
				
		if camera_name == "Camera_0006":
			if instance_data["object id"] == 182:
				if instance_data["frame_id"] in [2055, 2475, 3825, 3855, 4275, 4425]:
					return False
			if instance_data["object id"] == 350:
				if instance_data["frame_id"] in [7125, 7605, 7785, 8655, 8685]:
					return False
			if instance_data["object id"] == 352:
				if instance_data["frame_id"] in [345, 8835]:
					return False
			if instance_data["object id"] == 625:
				if instance_data["frame_id"] in [7065, 7515, 7725]:
					return False

		if camera_name == "Camera_0007":
			if instance_data["object id"] == 182:
				deleted_frames = [6225, 8235]
				for i in range(6465,6770,30): deleted_frames.append(i)
				for i in range(8625,9000,30): deleted_frames.append(i)
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 350:
				deleted_frames = [735, 1785, 1815, 3855, 6645, 6855]
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 351:
				deleted_frames = [1005, 1095, 1395, 5325, 5355, 5565, 5595, 5715, 5745, 5925, 5955, 6525, 6555, 6855]
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 352:
				deleted_frames = [735, 765, 1695, 2505, 2565, 2595, 2625, 2655, 2685, 2715, 2805, 3585, 3705, 3735, 4725, 4755, 4845, 4875, 6195, 7185, 7215, 7245, 7305, 7635, 8505]
				for i in range(885, 980, 30): deleted_frames.append(i)
				for i in range(3045,3530,30): deleted_frames.append(i)
				for i in range(5115,5270,30): deleted_frames.append(i)
				for i in range(5535,5660,30): deleted_frames.append(i)
				for i in range(8115,8330,30): deleted_frames.append(i)
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 353:
				deleted_frames = [495, 1335, 1995, 2025, 2055, 2865, 4305, 6465, 7305, 7485, 8505, 8535, 8685, 8715]
				for i in range(4425,5090,30): deleted_frames.append(i)
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 354:
				deleted_frames = [525, 1245, 1935, 1965, 3765, 7035, 7095, 7245, 7275, 7305, 7425, 8115]
				for i in range(1755,1910,30): deleted_frames.append(i)
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 625:
				deleted_frames = [495, 555, 4815, 4845, 4875, 4905]
				for i in range(5355,5780,30): deleted_frames.append(i)
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 626:
				deleted_frames = [615, 945, 2685, 3225, 3435, 3855, 3915, 3945, 4695, 4725, 5565, 5595, 6045, 6075]
				for i in range(2805,2930,30): deleted_frames.append(i)
				for i in range(4185,4340,30): deleted_frames.append(i)
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 627:
				deleted_frames = [3405, 6105, 6135, 6195, 6825, 6885, 6915]
				for i in range(615, 1700,30): deleted_frames.append(i)
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 699:
				deleted_frames = [7935, 8145]
				for i in range(5205,5330,30): deleted_frames.append(i)
				if instance_data["frame_id"] in deleted_frames:
					return False
		
		if camera_name == "Camera_0008":
			if instance_data["object id"] == 182:
				deleted_frames = [6375]
				for i in range(7485,7580,30): deleted_frames.append(i)
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 350:
				deleted_frames = [3585]
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 352:
				deleted_frames = [1095]
				for i in range(1305,1610,30): deleted_frames.append(i)
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 353:
				deleted_frames = [1215, 2055, 5085, 6345, 6465, 6705, 7305, 8385, 8475]
				for i in range(2835,3170,30): deleted_frames.append(i)
				deleted_frames.remove(2895)
				for i in range(4455,5000,30): deleted_frames.append(i)
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 625:
				deleted_frames = []
				for i in range(7785,8030,30): deleted_frames.append(i)
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 626:
				deleted_frames = [5625, 5835, 6015, 6075, 6105, 6915]
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 699:
				deleted_frames = []
				for i in range(5475,5600,30): deleted_frames.append(i)
				if instance_data["frame_id"] in deleted_frames:
					return False
				
		if camera_name == "Camera_0009":
			if instance_data["frame_id"] in [1275]:
				if instance_data["object id"] == 182:
					return False

		if camera_name == "Camera_0010":
			if instance_data["object id"] == 352:
				deleted_frames = [3585]
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 353:
				deleted_frames = [675, 795, 1215, 2085, 4185, 4245, 4365, 5475, 8535]
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 354:
				deleted_frames = [2415, 3585]
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 625:
				deleted_frames = [5025]
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 626:
				deleted_frames = []
				for i in range(2835,3170,30): deleted_frames.append(i)
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 627:
				deleted_frames = [3825, 3855]
				# for i in range(2835,3170,30): deleted_frames.append(i)
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 699:
				deleted_frames = [885]
				if instance_data["frame_id"] in deleted_frames:
					return False
				
		if camera_name == "Camera_0011":
			if instance_data["object id"] == 182:
				deleted_frames = [7425]
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 350:
				deleted_frames = [4335]
				for i in range(5235,6200,30): deleted_frames.append(i)
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 351:
				deleted_frames = [675, 1305, 4365, 8085]
				for i in range(4065,4250,30): deleted_frames.append(i)
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 352:
				deleted_frames = [1005, 4755, 5175, 5565]
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 353:
				deleted_frames = [3225, 3285, 3315, 6435, 8565]
				for i in range(3705,3890,30): deleted_frames.append(i)
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 354:
				deleted_frames = [315, 4245, 5415, 7755, 8115]
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 625:
				deleted_frames = [1185]
				for i in range(2115,2210,30): deleted_frames.append(i)
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 626:
				deleted_frames = [945, 4395, 5655]
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 627:
				deleted_frames = [2205, 3285, 6915]
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 699:
				deleted_frames = [495, 885]
				if instance_data["frame_id"] in deleted_frames:
					return False
			
	if scene_name == "Warehouse_012":

		if camera_name == "Camera_0007":
			if instance_data["object id"] == 350:
				deleted_frames = [1185, 2115]
				for i in range(6945,7010,30): deleted_frames.append(i)
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 452:
				deleted_frames = [285, 315, 2955, 3015]
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 453:
				deleted_frames = [4665]
				for i in range(345,440, 30): deleted_frames.append(i)
				for i in range(6585,6770,30): deleted_frames.append(i)
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 616:
				deleted_frames = [6375]
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 617:
				deleted_frames = [6765, 8445]
				for i in range(6675,6740,30): deleted_frames.append(i)
				for i in range(8115,8340,30): deleted_frames.append(i)
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 618:
				deleted_frames = []
				for i in range(2145,2390,30): deleted_frames.append(i)
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 685:
				deleted_frames = [225, 4215, 4395, 8865]
				for i in range(5385,5600,30): deleted_frames.append(i)
				for i in range(8175,8210,30): deleted_frames.append(i)
				if instance_data["frame_id"] in deleted_frames:
					return False

		if camera_name == "Camera_0008":
			if instance_data["object id"] == 183:
				deleted_frames = []
				for i in range(1515,1670,30): deleted_frames.append(i)
				for i in range(2325,2360,30): deleted_frames.append(i)
				for i in range(8805,9000,30): deleted_frames.append(i)
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 350:
				deleted_frames = [7455]
				for i in range(2685,2840,30): deleted_frames.append(i)
				for i in range(3105,3230,30): deleted_frames.append(i)
				for i in range(3315,3560,30): deleted_frames.append(i)
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 352:
				deleted_frames = [1575, 2535, 3615, 3645, 6045, 7785]
				for i in range(4245,4340,30): deleted_frames.append(i)
				for i in range(4995,5240,30): deleted_frames.append(i)
				for i in range(7335,7460,30): deleted_frames.append(i)
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 452:
				deleted_frames = [2055, 3405]
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 617:
				deleted_frames = [3315, 6165, 6285]
				if instance_data["frame_id"] in deleted_frames:
					return False
		
		if camera_name == "Camera_0009":
			if instance_data["object id"] == 182:
				deleted_frames = [6975]
				for i in range(6555,6620,30): deleted_frames.append(i)
				for i in range(7005,7130,30): deleted_frames.append(i)
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 452:
				deleted_frames = [2145]
				if instance_data["frame_id"] in deleted_frames:
					return False
		
		if camera_name == "Camera_0010":
			if instance_data["object id"] == 350:
				deleted_frames = [7275]
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 351:
				deleted_frames = [285]
				for i in range(825, 980, 30): deleted_frames.append(i)
				for i in range(8085,9000,30): deleted_frames.append(i)
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 453:
				deleted_frames = [3735]
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 616:
				deleted_frames = [1245, 3555, 4995, 5235]
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 617:
				deleted_frames = [525, 555]
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 618:
				deleted_frames = [3975, 7935]
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 685:
				deleted_frames = [615, 795, 1455, 4605, 4635]
				for i in range(7095,7130,30): deleted_frames.append(i)
				if instance_data["frame_id"] in deleted_frames:
					return False
				
		if camera_name == "Camera_0011":
			if instance_data["object id"] == 350:
				deleted_frames = [645, 795, 825, 915, 1545, 7305]
				for i in range(15,  80,  30): deleted_frames.append(i)
				for i in range(7485,7580,30): deleted_frames.append(i)
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 352:
				deleted_frames = [4755, 6555]
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 616:
				deleted_frames = [765, 945, 1215, 7665, 7785, 8205, 8775]
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 617:
				deleted_frames = [3135, 3795]
				for i in range(3885,4220,30): deleted_frames.append(i)
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 618:
				deleted_frames = [1695, 3225]
				if instance_data["frame_id"] in deleted_frames:
					return False
			if instance_data["object id"] == 685:
				deleted_frames = [615]
				if instance_data["frame_id"] in deleted_frames:
					return False
		
	return True


def load_and_filter_labels_mot(lbl_path, map_world, rois, scene_name, camera_name, img_size):
	"""

	Args:
		lbl_path:
		map_world:
		rois:
		scene_name:
		camera_name:
		img_size (img_w, img_h):

	Returns:
		map_world
	"""
	# Load the labels from the MOT format file
	# FORMAT: frame_id, track_id, x, y, w, h, "not ignored", class_id, visibility
	with open(lbl_path, "r") as f_read:
		for line in tqdm(f_read, desc= f"Loading and filtering labels mot {scene_name}__{camera_name}"):
			parts = line.strip().split(",")
			instance_data = {
				"camera_id"      : camera_name,
				"frame_id"       : int(parts[0]),
				"object id"      : int(parts[1]), # track_id
				"x_tl"           : max(float(parts[2]), 1.0),
				"y_tl"           : max(float(parts[3]), 1.0),
				"x_br"           : min(float(parts[2]) + float(parts[4]), img_size[0] - 1),
				"y_br"           : min(float(parts[3]) + float(parts[5]), img_size[1] - 1),
				"w"              : float(parts[4]),
				"h"              : float(parts[5]),
				"not_ignored"    : int(parts[6]),
				"object type"    : object_type_name[int(parts[7])],
				"object_type_int": int(parts[7]),
				"visibility"     : float(parts[8]),
			}
			instance_data["w"] = abs(instance_data["x_br"] - instance_data["x_tl"])
			instance_data["h"] = abs(instance_data["y_br"] - instance_data["y_tl"])

			if filter_bounding_box(scene_name, camera_name, rois, instance_data, img_size):
				if instance_data["object id"] not in map_world.instances:
					map_world.instances[instance_data["object id"]] = Instance(instance_data)
				map_world.instances[instance_data["object id"]].update_bbox(instance_data)

	return map_world


################################################################################
# REGION: Main
################################################################################


def main_filter_bounding_box():
	# init folder
	folder_input             = "/media/ngochdm/Projects/AutomationLab/AIC25/Dataset/ACI25-ExtractFrames/"
	folder_input_mot_img     = os.path.join(folder_input, "images")
	folder_input_mot_lbl     = os.path.join(folder_input, "labels_mot")
	folder_input_view_points = os.path.join(folder_input, "view_points")

	folder_output_mot_lbl    = os.path.join(folder_input, "labels_mot_filtered")
	folder_output_yolo_lbl   = os.path.join(folder_input, "labels_yolo_filtered")
	folder_output_draw_img   = os.path.join(folder_input, "images_draw")

	camera_list_img_in       = glob.glob(os.path.join(folder_input_mot_img, "*/*"))

	#ngochdm
	scene_name_spec  = "Warehouse_012"
	camera_name_spec = "Camera_0007"

	# create folder of scene in ROI folder
	for camera_path_in in tqdm(camera_list_img_in, desc="Processing camera"):
		# get information of camera
		camera_name = os.path.basename(camera_path_in)
		scene_name  = os.path.basename(os.path.dirname(camera_path_in))
		view_point_name = get_view_point_name(scene_name, camera_name)
		view_point_path = os.path.join(folder_input_view_points, f"{view_point_name}.json")
		lbl_mot_path    = os.path.join(folder_input_mot_lbl, scene_name, f"{camera_name}.txt")

		folder_output_draw_img_camera = os.path.join(folder_output_draw_img, scene_name, camera_name)

		# DEBUG: run on specific camera name in map world
		if scene_name_spec is not None and scene_name != scene_name_spec:
			continue
		if camera_name_spec is not None and camera_name != camera_name_spec:
			continue

		# check view point path is exist
		if not os.path.exists(view_point_path):
			logger.warning(f"View point path: {view_point_path} is not exist")
			continue

		# check label path is exist
		if not os.path.exists(lbl_mot_path):
			logger.warning(f"Label path: {lbl_mot_path} is not exist")
			continue

		# NOTE: load json filter file
		rois, img_w, img_h = load_json_filter_bounding_box_x_anylabeling(view_point_path)

		# create map world
		map_cfg = {
			"name"              : scene_name,
			"id"                : scene_name,
			"size"              : [img_w, img_h]
		}
		map_world = MapWorld(map_cfg)

		# NOTE: load and filter label mot files
		# frame_id, track_id, x, y, w, h, "not ignored", class_id, visibility
		map_world = load_and_filter_labels_mot(lbl_mot_path, map_world, rois, scene_name, camera_name, (img_w, img_h))

		# sort frames_id
		for instance in map_world.instances:
			map_world.instances[instance].sort_frames()

		# NOTE: write the mot labels to the folder
		list_img_came_in     = glob.glob(os.path.join(camera_path_in, "*.png"))
		lbl_path_mot_out     = os.path.join(folder_output_mot_lbl, scene_name, f"{camera_name}.txt")
		folder_camera_path_yolo_out = os.path.join(folder_output_yolo_lbl, scene_name, camera_name)

		os.makedirs(os.path.dirname(lbl_path_mot_out), exist_ok=True)
		os.makedirs(folder_camera_path_yolo_out, exist_ok=True)
		# run base on list image in camera
		with open(lbl_path_mot_out, "w") as f_mot_write:
			for img_path_in in tqdm(list_img_came_in, desc=f"Output mot {view_point_name}"):

				img_basename       = os.path.basename(img_path_in)
				img_basename_noext = os.path.splitext(img_basename)[0]
				frame_id     = str(int(img_basename_noext))
				lbl_path_yolo_out = os.path.join(folder_camera_path_yolo_out, f"{img_basename_noext}.txt")

				with open(lbl_path_yolo_out, "w") as f_yolo_write:
					# frame_id, track_id, x, y, w, h, "not ignored", class_id, visibility
					for instance in map_world.instances:
						# check instance is in the frame
						if frame_id in map_world.instances[instance].frames:
							instance_data = map_world.instances[instance].frames[frame_id]['bbox_visible_2d'][camera_name]
							f_mot_write.write(f"{frame_id},"
							              f"{map_world.instances[instance].object_id},"
							              f"{int(instance_data[0])},"
							              f"{int(instance_data[1])},"
							              f"{abs(int(instance_data[2]) - int(instance_data[0]))},"
							              f"{abs(int(instance_data[3]) - int(instance_data[1]))},"
							              f"1,"
							              f"{object_type_id[map_world.instances[instance].object_type]},"
							              f"1\n")

							bbox = np.array((
								instance_data[0],
								instance_data[1],
								abs(instance_data[2] -instance_data[0]),
								abs(instance_data[3] - instance_data[1])))

							bbox = bbox_xywh_to_cxcywh_norm(xywh=bbox, height=float(img_h), width=float(img_w))
							f_yolo_write.write(f"{object_type_id[map_world.instances[instance].object_type]} "
							              f"{bbox[0]} "
							              f"{bbox[1]} "
							              f"{bbox[2]} "
							              f"{bbox[3]}\n")

		# NOTE: draw bounding box on image
		os.makedirs(folder_output_draw_img_camera, exist_ok=True)
		# get list image in camera
		list_img_came_in = glob.glob(os.path.join(camera_path_in, "*.png"))
		# drawing
		for img_path_in in tqdm(list_img_came_in, desc=f"Drawing image {view_point_name}"):
			cam_img      = cv2.imread(img_path_in)
			img_basename = os.path.basename(img_path_in)
			frame_id     = int(os.path.splitext(img_basename)[0])

			cam_img = map_world.draw_information_on_map(cam_img, frame_id, color=color_chart)
			cam_img = map_world.draw_instances_2D_on_camera(cam_img, camera_name, frame_id, color=color_chart)

			img_path_ou = os.path.join(folder_output_draw_img_camera, os.path.basename(img_path_in.replace(".png", ".jpg")))
			cv2.imwrite(img_path_ou, cam_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])



def main():
	# rename folder of camera to become the format Camera_XXXX
	# rename_files_folder()

	# create view points from iamge
	# create_viewpoints_image()

	main_filter_bounding_box()
	pass


if __name__ == "__main__":
	main()