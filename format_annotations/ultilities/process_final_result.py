import json
import math
import os
import sys
import glob
import threading
from functools import cmp_to_key

import numpy as np
from loguru import logger
from tqdm import tqdm

import cv2

from mtmc.core.objects.units import put_text_with_border
from mtmc.core.utils.bbox import bbox_xywh_to_cxcywh_norm
from ultilities.filter_bounding_box import object_area_specific_dict


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


def draw_information_on_map(map_img, frame_id, color, size_multi=1):
	"""Show the map information on the map.

	Args:
		map_img: Image to draw on
		frame_id: Frame ID to draw the instance on
		color: Arrow and text color (BGR), or chart color

	Returns:
		map_img: Image with the camera drawn on it
	"""
	# Check if frame_id is string type
	if not isinstance(frame_id, str):
		frame_id = str(frame_id)

	# NOTE: draw frame_id, panel, and object class color chart on the map
	# init values
	font              = cv2.FONT_HERSHEY_SIMPLEX
	font_scale        = 3 * size_multi
	thickness         = 2 * size_multi
	frame_id_label_tl = (5, 5)

	# Get the text size
	text_size, _ = cv2.getTextSize(frame_id, font, font_scale, thickness)
	text_width, text_height = text_size

	# Calculate the background rectangle coordinates
	x, y              = frame_id_label_tl
	# y                 = y - 5 - text_height - 10 * size_multi  # Position at the bottom
	top_left          = (x, y)
	bottom_right      = (x + text_width + 10, y + text_height + 10)
	frame_id_label_bl = (x, y + text_height)

	# Draw the background rectangle
	cv2.rectangle(map_img, top_left, bottom_right, (214, 224, 166), -1)

	# Draw the text with a border
	cv2.putText(map_img, frame_id, frame_id_label_bl, font, font_scale, (0 , 0 , 0), thickness + 2, cv2.LINE_AA )
	cv2.putText(map_img, frame_id, frame_id_label_bl, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA )

	# NOTE: draw object class color chart
	if isinstance(color, dict):
		# Create a color chart
		for i, (object_type, object_color) in enumerate(color.items()):
			top_left = (frame_id_label_bl[0], frame_id_label_bl[1] + 50 + i * 30)
			bottom_right = (frame_id_label_bl[0] + 100 , frame_id_label_bl[1] + 80 + i * 30)
			bottom_left  = (frame_id_label_bl[0], frame_id_label_bl[1] + 70 + i * 30)
			cv2.rectangle(map_img, top_left, bottom_right, object_color, -1)
			cv2.putText(map_img, object_type, bottom_left, font, 0.5, color=(0, 0, 0), thickness=1)

	return map_img


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


def draw_panel_on_map(map_img, panel_text, point_tl=(5, 5), font_scale = 3, thickness = 2):
	# Check if frame_id is string type
	if not isinstance(panel_text, str):
		panel_text = str(panel_text)

	# NOTE: draw frame_id, panel, and object class color chart on the map
	# init values
	font              = cv2.FONT_HERSHEY_SIMPLEX
	# font_scale        = 3
	# thickness         = 2
	frame_id_label_tl = point_tl

	# Get the text size
	text_size, _ = cv2.getTextSize(panel_text, font, font_scale, thickness)
	text_width, text_height = text_size

	# Calculate the background rectangle coordinates
	x, y              = frame_id_label_tl
	top_left          = (x, y)
	bottom_right      = (x + text_width + 10, y + text_height + 10)
	frame_id_label_bl = (x, y + text_height)

	# Draw the background rectangle
	cv2.rectangle(map_img, top_left, bottom_right, (214, 224, 166), -1)

	# Draw the text with a border
	cv2.putText(map_img, panel_text, frame_id_label_bl, font, font_scale, (0 , 0 , 0), thickness + 2, cv2.LINE_AA)
	cv2.putText(map_img, panel_text, frame_id_label_bl, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

	return map_img


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


def load_final_result(final_result_path, scene_id):
	"""
	Load the final result from the given file path.
	Returns a list of results, each result is a numpy array.
	"""
	# <scene_id> <class_id> <object_id> <frame_id> <x> <y> <z> <width> <length> <height> <yaw>
	final_result = []
	with open(final_result_path, 'r') as f:
		lines = f.readlines()
		for line in lines:
			parts = np.array(line.split(), dtype=np.float32)
			if int(parts[0]) == scene_id:
				final_result.append(parts)

	return sorted(final_result, key=cmp_to_key(custom_result_sort))


def main_visualization_final_result_only_points():
	# initialize paths
	folder_data_version = "MTMC_Tracking_2025_20250614"
	folder_info_in    = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/{folder_data_version}/ExtractFrames/lookup_table/"
	folder_in         = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/{folder_data_version}/ExtractFrames/image_bev_track/bev_track_1_size_full/"
	final_result_path = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/{folder_data_version}/ExtractFrames/lookup_table/final_result_avg.txt"
	outline_color     = (0, 0, 0) # Outline color (B, G, R) - Blue
	outline_thickness = 2       # Thickness of the outline
	radius            = 10  # Radius for the circle

	folder_out  = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/{folder_data_version}/ExtractFrames/lookup_table/"
	list_scene  = ["Warehouse_017", "Warehouse_018", "Warehouse_019", "Warehouse_020"]


	for scene_name in tqdm(list_scene):
		# DEBUG:
		if scene_name not in ["Warehouse_017"]:
			continue

		folder_img_in = os.path.join(folder_in, scene_name)
		folder_img_ou = os.path.join(folder_out, scene_name)
		scene_id      = find_scene_id(scene_name)
		os.makedirs(folder_img_ou, exist_ok=True)

		# load calibration info
		json_calibration_path = os.path.join(folder_info_in, f"{scene_name}_calibration.json")
		with open(json_calibration_path, 'r') as f:
			json_data_calibration = json.load(f)

		scale_factor                      = float(json_data_calibration["sensors"][0]["scaleFactor"])
		translation_to_global_coordinates = json_data_calibration["sensors"][0]["translationToGlobalCoordinates"]

		# load final result
		final_result = load_final_result(final_result_path, scene_id)

		# <scene_id> <class_id> <object_id> <frame_id> <x> <y> <z> <width> <length> <height> <yaw>
		img_in            = None
		img_current_index = -1
		img_h             = 0
		img_w			  = 0
		pbar = tqdm(total=len(final_result), desc=f"Processing visualization {scene_name}")
		for result in final_result:
			scene_id    = int(result[0])
			object_type = object_type_name[int(result[1])]
			object_id   = int(result[2])
			frame_id    = int(result[3])
			x           = (float(result[4]) + float(translation_to_global_coordinates['x'])) * scale_factor
			y           = (float(result[5]) + float(translation_to_global_coordinates['y'])) * scale_factor
			z           = float(result[6]) * scale_factor
			width       = float(result[7]) * scale_factor
			length      = float(result[8]) * scale_factor
			height      = float(result[9]) * scale_factor
			yaw         = float(result[10])

			pbar.set_description(f"Processing visualization {scene_name} - Frame {frame_id}")

			# DEBUG:
			# if frame_id > 1:
			# 	return

			# load image
			if img_in is None or frame_id != img_current_index:
				if img_in is not None:
					# Draw information on the map
					img_in = draw_information_on_map(img_in, int(img_current_index), color_chart)
					# Save the modified image
					img_path_ou = os.path.join(folder_img_ou, f"{img_current_index:08d}.jpg")
					cv2.imwrite(img_path_ou, img_in)

				img_path_in       = os.path.join(folder_img_in, f"{frame_id:08d}.jpg")
				img_in            = cv2.imread(img_path_in)
				img_current_index = frame_id
				img_h, img_w      = img_in.shape[:2]

			point_center= np.array([x, img_h - y])  # Adjust y-coordinate for image height

			# choose color based on object type
			color = color_chart[object_type] if object_type in color_chart else (255, 255, 255)

			# Draw  circle of the instance and label
			img_in = cv2.circle(img_in, (int(point_center[0]), int(point_center[1])), radius, color, -1)
			img_in = cv2.circle(img_in, (int(point_center[0]), int(point_center[1])), radius, outline_color, outline_thickness)
			# draw label with border
			# img_in = put_text_with_border(img_in, f"{object_id}", (int(point_center[0]) + 5, int(point_center[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color=color, border_color=(0,0,0), thickness=1)

			pbar.update(1)
		pbar.close()

		# Draw information on the map
		img_in = draw_information_on_map(img_in, int(img_current_index), color_chart)
		# save final image
		img_path_ou = os.path.join(folder_img_ou, f"{img_current_index:08d}.jpg")
		cv2.imwrite(img_path_ou, img_in)


def main_visualization_final_result_only_point_with_bbox():
	# initialize paths
	folder_data_version = "MTMC_Tracking_2025_20250614"
	folder_info_in    = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/{folder_data_version}/ExtractFrames/lookup_table/"
	folder_in         = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/{folder_data_version}/ExtractFrames/image_bev_track/bev_track_1_size_full/"
	final_result_path = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/{folder_data_version}/ExtractFrames/lookup_table/final_result_max.txt"
	outline_color     = (0, 0, 0) # Outline color (B, G, R) - Blue
	outline_thickness = 2       # Thickness of the outline
	radius            = 5  # Radius for the circle

	folder_out  = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/{folder_data_version}/ExtractFrames/image_result/"
	list_scene  = ["Warehouse_017", "Warehouse_018", "Warehouse_019", "Warehouse_020"]


	for scene_name in tqdm(list_scene):
		# DEBUG:
		if scene_name not in ["Warehouse_017"]:
			continue

		folder_img_in = os.path.join(folder_in, scene_name)
		folder_img_ou = os.path.join(folder_out, scene_name)
		scene_id      = find_scene_id(scene_name)
		os.makedirs(folder_img_ou, exist_ok=True)

		# load calibration info
		json_calibration_path = os.path.join(folder_info_in, f"{scene_name}_calibration.json")
		with open(json_calibration_path, 'r') as f:
			json_data_calibration = json.load(f)

		scale_factor                      = float(json_data_calibration["sensors"][0]["scaleFactor"])
		translation_to_global_coordinates = json_data_calibration["sensors"][0]["translationToGlobalCoordinates"]

		# load final result
		final_result = load_final_result(final_result_path, scene_id)

		# <scene_id> <class_id> <object_id> <frame_id> <x> <y> <z> <width> <length> <height> <yaw>
		img_in            = None
		img_current_index = -1
		img_h             = 0
		img_w			  = 0
		pbar = tqdm(total=len(final_result), desc=f"Processing visualization {scene_name}")
		for result in final_result:
			scene_id    = int(result[0])
			object_type = object_type_name[int(result[1])]
			object_id   = int(result[2])
			frame_id    = int(result[3])
			x           = (float(result[4]) + float(translation_to_global_coordinates['x'])) * scale_factor
			y           = (float(result[5]) + float(translation_to_global_coordinates['y'])) * scale_factor
			z           = float(result[6]) * scale_factor
			width       = float(result[7]) * scale_factor
			length      = float(result[8]) * scale_factor
			height      = float(result[9]) * scale_factor
			yaw         = float(result[10])

			pbar.set_description(f"Processing visualization {scene_name} - Frame {frame_id}")

			# DEBUG:
			# if frame_id > 1:
			# 	return

			# load image
			if img_in is None or frame_id != img_current_index:
				if img_in is not None:
					# Draw information on the map
					img_in = draw_information_on_map(img_in, int(img_current_index), color_chart)
					# Save the modified image
					img_path_ou = os.path.join(folder_img_ou, f"{img_current_index:08d}.jpg")
					cv2.imwrite(img_path_ou, img_in)

				img_path_in       = os.path.join(folder_img_in, f"{frame_id:08d}.jpg")
				img_in            = cv2.imread(img_path_in)
				img_current_index = frame_id
				img_h, img_w      = img_in.shape[:2]

			point_center= np.array([x, img_h - y])  # Adjust y-coordinate for image height

			# choose color based on object type
			color = color_chart[object_type] if object_type in color_chart else (255, 255, 255)

			# Draw circle of the instance
			img_in = cv2.circle(img_in, (int(point_center[0]), int(point_center[1])), radius, color, -1)
			img_in = cv2.circle(img_in, (int(point_center[0]), int(point_center[1])), radius, outline_color, outline_thickness)

			# Draw rotated rectangle for the bounding box
			# if object_type in ["NovaCarter", "Transporter"]:
			# 	rect = (point_center, (width, length), ((-yaw) * (180 / math.pi) - 90))
			# else:
			rect = (point_center, (width, length), abs(yaw) * (180 / math.pi))
			bbox = cv2.boxPoints(rect)  # Get 4 corners of the rotated rect
			bbox = np.array(bbox, dtype=np.int32)
			cv2.polylines(img_in, [bbox], isClosed=True, color=color, thickness=outline_thickness)

			# draw label with border
			img_in = put_text_with_border(img_in, f"{object_id}", (int(point_center[0]) + 5, int(point_center[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color=color, border_color=(0,0,0), thickness=1)

			pbar.update(1)
		pbar.close()

		# Draw information on the map
		img_in = draw_information_on_map(img_in, int(img_current_index), color_chart)
		# save final image
		img_path_ou = os.path.join(folder_img_ou, f"{img_current_index:08d}.jpg")
		cv2.imwrite(img_path_ou, img_in)


def main_create_yolo_format_final_result_preprocess():
	# initialize paths
	folder_data_version = "MTMC_Tracking_2025_20250614"
	folder_info_in      = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/{folder_data_version}/ExtractFrames/lookup_table/"
	folder_in           = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/{folder_data_version}/ExtractFrames/image_bev_track/bev_track_1_size_full/"
	final_result_path   = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/{folder_data_version}/ExtractFrames/lookup_table/final_result_avg.txt"
	outline_color       = (0, 0, 0) # Outline color (B, G, R) - Blue
	outline_thickness   = 2       # Thickness of the outline

	folder_out          = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/{folder_data_version}/ExtractFrames/lookup_table/"
	list_scene          = ["Warehouse_017", "Warehouse_018", "Warehouse_019", "Warehouse_020"]


	for scene_name in tqdm(list_scene):
		# DEBUG:
		if scene_name not in ["Warehouse_017"]:
			continue

		folder_img_in = os.path.join(folder_in, scene_name)
		folder_lbl_ou = os.path.join(folder_out, f"{scene_name}_labels")
		scene_id      = find_scene_id(scene_name)
		os.makedirs(folder_lbl_ou, exist_ok=True)

		# load calibration info
		json_calibration_path = os.path.join(folder_info_in, f"{scene_name}_calibration.json")
		with open(json_calibration_path, 'r') as f:
			json_data_calibration = json.load(f)

		scale_factor                      = float(json_data_calibration["sensors"][0]["scaleFactor"])
		translation_to_global_coordinates = json_data_calibration["sensors"][0]["translationToGlobalCoordinates"]

		# load final result
		final_result = load_final_result(final_result_path, scene_id)

		# DEBUG:
		# for result in final_result:
		# 	print(f"{result[0]} {object_type_name[int(result[1])]} {int(result[2])} {int(result[3])} {result[4]} {result[5]} {result[6]} {result[7]} {result[8]} {result[9]} {result[10]}")

		# <scene_id> <class_id> <object_id> <frame_id> <x> <y> <z> <width> <length> <height> <yaw>
		img_in            = None
		img_current_index = -1
		img_h             = 0
		img_w			  = 0
		radius            = 10  # Radius for the circle
		bboxes            = []
		pbar = tqdm(total=len(final_result), desc=f"Processing visualization {scene_name}")
		for result in final_result:
			scene_id          = int(result[0])
			object_type_index = int(result[1])
			object_type       = object_type_name[int(result[1])]
			object_id         = int(result[2])
			frame_id          = int(result[3])
			x                 = (float(result[4]) + float(translation_to_global_coordinates['x'])) * scale_factor
			y                 = (float(result[5]) + float(translation_to_global_coordinates['y'])) * scale_factor
			z                 = float(result[6]) * scale_factor
			width             = float(result[7]) * scale_factor
			length            = float(result[8]) * scale_factor
			height            = float(result[9]) * scale_factor
			yaw               = float(result[10])

			pbar.set_description(f"Processing visualization {scene_name} - Frame {frame_id}")

			# DEBUG:
			# if frame_id > 1:
			# 	return

			# DEBUG:
			# print(f"{scene_name} {object_type_name[object_type]} {object_id} {frame_id} {x} {y} {z} {width} {length} {height} {yaw}")

			# load image
			if img_in is None or frame_id != img_current_index:
				if img_in is not None:
					# MAIN PROCESSING:


					# Save the modified image
					lbl_path_ou = os.path.join(folder_lbl_ou, f"{img_current_index:08d}.txt")
					with open(lbl_path_ou, 'w') as f:
						for bbox in bboxes:
							f.write(f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]}\n")

				img_path_in       = os.path.join(folder_img_in, f"{frame_id:08d}.jpg")
				img_in            = cv2.imread(img_path_in)
				img_current_index = frame_id
				img_h, img_w      = img_in.shape[:2]
				bboxes            = []

			# point_center= np.array([x, img_h - y])  # Adjust y-coordinate for image height

			# bbox to list bboxes
			bbox_xywh   = np.array([x - (radius + 2), img_h - y - (radius + 2), radius * 2 + 4, radius * 2 + 4])
			bbox_xywhn  = bbox_xywh_to_cxcywh_norm(bbox_xywh, float(img_h), float(img_w))
			bboxes.append([object_type_index, bbox_xywhn[0], bbox_xywhn[1], bbox_xywhn[2], bbox_xywhn[3]])

			pbar.update(1)
		pbar.close()

		# save final image
		lbl_path_ou = os.path.join(folder_lbl_ou, f"{img_current_index:08d}.txt")
		with open(lbl_path_ou, 'w') as f:
			for bbox in bboxes:
				f.write(f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]}\n")


def project_points_intrinsic_extrinsic(points_3d, intrinsic_matrix, extrinsic_matrix):
	# Convert points to homogeneous coordinates
	points_3d_h = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])  # (N, 4)
	# Transform to camera coordinates
	points_cam = (extrinsic_matrix @ points_3d_h.T).T  # (N, 4) -> (N, 3) if extrinsic is 3x4
	points_cam = points_cam[:, :3]
	# Project to image plane
	points_img = (intrinsic_matrix @ points_cam.T).T  # (N, 3)
	points_img = points_img[:, :2] / points_img[:, 2:3]
	return points_img.astype(int)


def euler_angles_to_rotation_matrix(yaw, pitch, roll):
	# Rotation matrices for yaw, pitch, roll (in radians)
	R_yaw = np.array([
		[np.cos(yaw), -np.sin(yaw), 0],
		[np.sin(yaw),  np.cos(yaw), 0],
		[0,            0,           1]
	])
	R_pitch = np.array([
		[np.cos(pitch), 0, np.sin(pitch)],
		[0,             1, 0],
		[-np.sin(pitch),0, np.cos(pitch)]
	])
	R_roll = np.array([
		[1, 0,            0],
		[0, np.cos(roll), -np.sin(roll)],
		[0, np.sin(roll),  np.cos(roll)]
	])
	return R_yaw @ R_pitch @ R_roll


def draw_3d_bbox_with_euler(img, center, width, length, height, yaw, pitch, roll, intrinsic_matrix, extrinsic_matrix, color=(0,255,0), thickness=2):
	# 8 corners in local box coordinates (centered at origin)
	x_c, y_c, z_c = width/2, length/2, height/2
	corners = np.array([
		[-x_c, -y_c, -z_c],
		[ x_c, -y_c, -z_c],
		[ x_c,  y_c, -z_c],
		[-x_c,  y_c, -z_c],
		[-x_c, -y_c,  z_c],
		[ x_c, -y_c,  z_c],
		[ x_c,  y_c,  z_c],
		[-x_c,  y_c,  z_c],
	])
	# Apply rotation
	R = euler_angles_to_rotation_matrix(yaw, pitch, roll)
	corners_rot = (R @ corners.T).T
	# Translate to center
	corners_world = corners_rot + np.array(center)
	# Project to 2D
	pts_2d = project_points_intrinsic_extrinsic(corners_world, intrinsic_matrix, extrinsic_matrix)
	# Draw edges
	edges = [
		(0,1), (1,2), (2,3), (3,0),
		(4,5), (5,6), (6,7), (7,4),
		(0,4), (1,5), (2,6), (3,7)
	]
	for i, j in edges:
		cv2.line(img, tuple(pts_2d[i]), tuple(pts_2d[j]), color, thickness)

	# --- Draw center plane (left-right, width axis) ---
	# Four corners of the plane in local coordinates (width axis, full length and height)
	plane_corners_local = np.array([
		[-x_c, 0, -z_c],
		[x_c,  0, -z_c],
		[x_c,  0,  z_c],
		[-x_c, 0,  z_c],
	])
	# Move to center (x=0 plane)
	plane_corners_local[:,1] = 0
	# Rotate and translate
	plane_corners_world = (R @ plane_corners_local.T).T + np.array(center)
	plane_corners_2d = project_points_intrinsic_extrinsic(plane_corners_world, intrinsic_matrix, extrinsic_matrix)
	# Draw filled polygon for the plane
	cv2.drawContours(img, [plane_corners_2d.astype(np.int32)], -1, color, thickness=1)

	return img


def draw_points_on_camera(img_world, camera_calibration, folder_img_in, img_index, object_id_panel, radius=10, size_multi=4.0):
	img_world_h, img_world_w = img_world.shape[:2]
	img_world_h_ori = float(img_world_h)
	img_world_h     = int(img_world_h * size_multi)
	img_world_w     = int(img_world_w * size_multi)
	img_world       = cv2.resize(img_world, (img_world_w, img_world_h))
	number_camera   = len(camera_calibration)
	ratio           = math.ceil(number_camera / 2) + 1  # Resize ratio for the images on the map
	outline_color   = (0, 0, 0)  # Outline color (B, G, R) - black

	for camera_index, camera in enumerate(camera_calibration):
		# Extract camera parameters
		camera_id                         = camera["id"]
		scale_factor                      = float(camera["scaleFactor"])
		translation_to_global_coordinates = camera["translationToGlobalCoordinates"]
		intrinsic_matrix                  = camera["intrinsicMatrix"]
		extrinsic_matrix                  = camera["extrinsicMatrix"]
		homography_matrix                 = camera["homography"]

		# Load the camera image
		img_path = os.path.join(folder_img_in, camera_id, f"{img_index:08d}.jpg")
		img      = cv2.imread(img_path)

		for obj_info in object_id_panel:
			object_id    = obj_info["object_id"]
			point_center = obj_info["point_center"]
			color        = obj_info["color"]
			yaw          = obj_info["yaw"]
			width        = obj_info["shape"]["width"]
			length       = obj_info["shape"]["length"]
			height       = obj_info["shape"]["height"]

			point_c    = [0.0, 0.0]  # point in camera coordinates
			point_c[0] = float(point_center[0]) / scale_factor  - translation_to_global_coordinates['x']
			point_c[1] = (img_world_h_ori - float(point_center[1])) / scale_factor - translation_to_global_coordinates['y']

			# Convert world point to homogeneous coordinates
			world_pt = np.array([point_c[0], point_c[1], 1.0])
			# Apply homography
			img_pt = homography_matrix @ world_pt
			img_pt /= img_pt[2]
			img_pt_x, img_pt_y = int(img_pt[0]), int(img_pt[1])

			if img_pt_x < -300 or img_pt_x >= img.shape[1] + 300 or img_pt_y < -300 or img_pt_y >= img.shape[0] + 300:
				# Skip points that are out of bounds
				continue

			# Draw the point on camera image
			try:
				# Draw circle of the instance
				img = cv2.circle(img, (img_pt_x, img_pt_y), int(radius), color, -1)
				img = cv2.circle(img, (img_pt_x, img_pt_y), int(radius), outline_color, 2)
				# draw label with border
				img = put_text_with_border(img, f"{object_id}", (img_pt_x + 10 * size_multi, img_pt_y - 10 * size_multi), cv2.FONT_HERSHEY_SIMPLEX, (1  * size_multi), text_color=obj_info["color"], border_color=(0,0,0), thickness=size_multi * 2)
			except cv2.error as e:
				print(f"{img_pt_x=} {img_pt_x=}")
				logger.error(f"OpenCV Error: {e}")
			# DEBUG:
			# print(f"Camera {camera_index} has {object_id} at point {(img_x, img_y)}")

			# draw 3D bounding box
			x = float(point_center[0]) / scale_factor - translation_to_global_coordinates['x']
			y = (img_world_h_ori - float(point_center[1])) / scale_factor - translation_to_global_coordinates['y']
			z = float(point_center[2]) / scale_factor  # Assuming z is already in camera coordinates
			img = draw_3d_bbox_with_euler(
				img,
				[x, y, z],  # Center in camera coordinates
				width / scale_factor,  # Width in camera coordinates
				length / scale_factor,  # Length in camera coordinates
				height / scale_factor,  # Height in camera coordinates
				yaw,  # Yaw in radians
				0.0,  # Pitch in radians
				0.0,  # Roll in radians
				intrinsic_matrix,  # Intrinsic matrix
				extrinsic_matrix,  # Extrinsic matrix
				color=color,  # Color for the bounding box
				thickness=int(2 * size_multi)  # Thickness of the bounding box lines
			)

	# draw information on the map
		# img = draw_panel_on_map(img, camera_index)
		img_h, img_w  = img.shape[:2]
		img_h_resized = int(img_h * size_multi // ratio)
		img_w_resized = int(img_w * size_multi // ratio)

		# Resize the image to fit into the map world
		img = cv2.resize(img, (img_w_resized, img_h_resized))

		# Calculate the position to place the image on the map
		if camera_index  < math.ceil(number_camera / 2):
			index = camera_index
			point = (150 * size_multi, index * img_h_resized + index * 20  * size_multi + 100 * size_multi)
		else:
			index = camera_index % (math.ceil(number_camera / 2))
			point = (img_world_w - 100 * size_multi - img_w_resized, index * img_h_resized + index * 20 * size_multi + 100 * size_multi)

		# DEBUG: check if the point is within the bounds of the map world
		if point[0] + img_w_resized > img_world.shape[1] or point[1] + img_h_resized > img_world.shape[0]:
			logger.warning(f"Image from camera {camera_index} exceeds map bounds. Skipping.")
			continue

		# Place the image on the map world
		img_world = cv2.rectangle(img_world, (int(point[0]), int(point[1])), (int(point[0] + img_w_resized), int(point[1] + img_h_resized)), (255, 255, 255), -1)
		img_world[point[1]:point[1] + img_h_resized, point[0]:point[0] + img_w_resized] = img

		# add panel camera
		img_world = draw_panel_on_map(
			img_world,
			camera_index,
			point_tl= (
				point[0] - 80 * size_multi,
				point[1] - 80 * size_multi + img_h_resized // 2
			),
			font_scale = 3 * size_multi,
			thickness = 2 * size_multi
		)

	return img_world


def visualization_final_result_with_camera_points_with_bboxes(scene_name, final_result_path, folder_info_in, folder_in, folder_out, radius=10, outline_color = (0, 0, 0), outline_thickness = 2, size_multi=4):
	final_result_path_basename_noext = os.path.splitext(os.path.basename(final_result_path))[0]
	scene_id      = find_scene_id(scene_name)
	folder_img_in = os.path.join(folder_in, scene_name)
	folder_img_ou = os.path.join(folder_out, f"{scene_name}_{final_result_path_basename_noext}")
	os.makedirs(folder_img_ou, exist_ok=True)

	# load camera information
	json_calibration_path = os.path.join(folder_info_in, f"{scene_name}_calibration.json")
	with open(json_calibration_path, 'r') as f:
		json_data_calibration = json.load(f)
	camera_calibration                = json_data_calibration["sensors"]
	scale_factor                      = float(camera_calibration[0]["scaleFactor"])
	translation_to_global_coordinates = camera_calibration[0]["translationToGlobalCoordinates"]

	# load final result
	final_result = load_final_result(final_result_path, scene_id)

	img_map_world_path = os.path.join(folder_in, scene_name, "map.png")
	img_map_world      = None
	img_current_index  = -1
	img_h              = 0
	img_w			   = 0
	object_id_panel    = []

	# <scene_id> <class_id> <object_id> <frame_id> <x> <y> <z> <width> <length> <height> <yaw>
	pbar = tqdm(total=len(final_result), desc=f"Processing visualization {os.path.basename(folder_img_ou)}")
	for result_index, result in enumerate(final_result):
		scene_id    = int(result[0])
		object_type = object_type_name[int(result[1])]
		object_id   = int(result[2])
		frame_id    = int(result[3])
		x           = (float(result[4]) + float(translation_to_global_coordinates['x'])) * scale_factor
		y           = (float(result[5]) + float(translation_to_global_coordinates['y'])) * scale_factor
		z           = float(result[6]) * scale_factor
		width       = float(result[7]) * scale_factor
		length      = float(result[8]) * scale_factor
		height      = float(result[9]) * scale_factor
		yaw         = float(result[10])

		pbar.set_description(f"Processing visualization {os.path.basename(folder_img_ou)} - Frame {frame_id}")

		# DEBUG:
		# if frame_id > 1:
		# 	return
		# if frame_id < 5940:
		# 	pbar.update(1)
		# 	continue


		# load or save image
		if img_map_world is None or frame_id != img_current_index:
			if img_map_world is not None:

				# add object_id_panel to the image
				for obj_info in object_id_panel:
					# draw label with border
					img_map_world = put_text_with_border(img_map_world, f"{obj_info['object_id']}", (obj_info["point_center"][0] + 5, obj_info["point_center"][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color=obj_info["color"], border_color=(0,0,0), thickness=1)

				# draw points on camera images
				img_map_world = draw_points_on_camera(
					img_map_world,
					camera_calibration,
					folder_img_in,
					img_current_index,
					object_id_panel,
					radius     = radius * 4,
					size_multi = size_multi
				)

				# Draw information on the map
				img_map_world = draw_information_on_map(img_map_world, int(img_current_index), color_chart)

				# Save the modified image
				img_path_ou = os.path.join(folder_img_ou, f"{img_current_index:08d}.jpg")
				cv2.imwrite(img_path_ou, img_map_world)

			# reset parameters
			img_map_world     = cv2.imread(img_map_world_path)
			img_current_index = frame_id
			img_h, img_w      = img_map_world.shape[:2]
			object_id_panel   = []

		point_center= np.array([x, img_h - y])  # Adjust y-coordinate for image height

		# choose color based on object type
		color = color_chart[object_type] if object_type in color_chart else (255, 255, 255)

		# Draw  circle of  the instance
		img_map_world   = cv2.circle(img_map_world, (int(point_center[0]), int(point_center[1])), radius, color, -1)
		img_map_world   = cv2.circle(img_map_world, (int(point_center[0]), int(point_center[1])), radius, outline_color, outline_thickness)

		# Draw rotated rectangle for the bounding box
		# if object_type in ["NovaCarter", "Transporter"]:
		# 	rect = (point_center, (width, length), ((-yaw) * (180 / math.pi) - 90))
		# else:
		rect = (point_center, (width, length), abs(yaw) * (180 / math.pi))
		bbox = cv2.boxPoints(rect)  # Get 4 corners of the rotated rect
		bbox = np.array(bbox, dtype=np.int32)
		cv2.polylines(img_map_world, [bbox], isClosed=True, color=color, thickness=outline_thickness)

		# save information to object_id_panel for later use
		object_id_panel.append({
			"object_id"    : object_id,
			"color"        : color,
			"point_center" : [int(point_center[0]), int(point_center[1]), z],
			"yaw"          : yaw,
			"shape"        : {
				"width" : float(width),
				"length": float(length),
				"height": float(height),
			}
		})

		pbar.update(1)
	pbar.close()

	# save the last image after process the last result
	# add object_id_panel to the image
	for obj_info in object_id_panel:
		# draw label with border
		img_map_world = put_text_with_border(img_map_world, f"{obj_info['object_id']}", (obj_info["point_center"][0] + 5, obj_info["point_center"][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color=obj_info["color"], border_color=(0,0,0), thickness=1)

	# draw points on camera images
	img_map_world = draw_points_on_camera(
		img_map_world,
		camera_calibration,
		folder_img_in,
		img_current_index,
		object_id_panel,
		radius     = radius * 4,
		size_multi = size_multi
	)

	# Draw information on the map
	img_map_world = draw_information_on_map(img_map_world, int(img_current_index), color_chart)

	# save final image
	img_path_ou = os.path.join(folder_img_ou, f"{img_current_index:08d}.jpg")
	cv2.imwrite(img_path_ou, img_map_world)

	# Create video from this images by using ffmpeg
	os.system(f"ffmpeg -y -framerate 30 -pattern_type glob "
	          f"-i '{folder_img_ou}/*.jpg' "
	          f" {folder_out}/{os.path.basename(folder_img_ou)}.mp4")


def main_visualization_final_result_with_camera_points_with_bboxes(list_scene=None):
	# initialize paths
	folder_data_version = "MTMC_Tracking_2025_20250614"
	folder_info_in      = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/{folder_data_version}/ExtractFrames/lookup_table/"
	final_result_path   = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/{folder_data_version}/ExtractFrames/lookup_table/final_result.txt"
	folder_in           = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/{folder_data_version}/ExtractFrames/images_extract_full/"

	outline_color       = (0, 0, 0) # Outline color (B, G, R) - Blue
	outline_thickness   = 2  # Thickness of the outline
	radius              = 5  # Radius for the circle
	size_multi          = 1  # Size multiplier for the camera images

	folder_out          = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/{folder_data_version}/ExtractFrames/image_result/"
	if list_scene is None:
		list_scene  = ["Warehouse_017", "Warehouse_018", "Warehouse_019", "Warehouse_020"]

	for scene_name in tqdm(list_scene):
		visualization_final_result_with_camera_points_with_bboxes(
			scene_name, final_result_path, folder_info_in, folder_in, folder_out, radius, outline_color, outline_thickness, size_multi
		)


def run_multi_threads():
	list_scene   = ["Warehouse_017", "Warehouse_018", "Warehouse_019", "Warehouse_020"]
	num_threads  = 4
	num_video_per_processes = math.ceil(len(list_scene) / num_threads)

	# Split the list into chunks for each process
	res = []
	for n, i in enumerate(list_scene):
		if n % num_video_per_processes == 0 and n + num_video_per_processes < len(list_scene):
			res.append(list_scene[n:n + num_video_per_processes])
		elif n + num_video_per_processes >= len(list_scene):
			res.append(list_scene[n:])

	logger.info(f"Number of processes: {num_threads}")
	logger.info(f"Number of maps: {len(list_scene)}")

	# creating processes
	threads = []
	for i in range(num_threads):
		p = threading.Thread(target=main_visualization_final_result_with_camera_points_with_bboxes, args=(res[i],))
		threads.append(p)

	# starting process
	for i in range(num_threads):
		threads[i].start()

	# wait until process is finished
	for i in range(num_threads):
		threads[i].join()


def run_single_thread():
	main_visualization_final_result_with_camera_points_with_bboxes(list_scene  = ["Warehouse_017", "Warehouse_018", "Warehouse_019", "Warehouse_020"])
	# main_visualization_final_result_with_camera_points_with_bboxes(list_scene  = ["Warehouse_017"])


def main():
	# HOTA combines three main components:
	# Detection (DetA): How well the tracker detects objects.
	# Association (AssA): How well the tracker maintains consistent identities over time.
	# Localization (LocA): How accurately the tracker localizes the objects.
	# main_visualization_final_result_only_points()

	# main_create_yolo_format_final_result_preprocess()

	# main_visualization_final_result_only_point_with_bbox()

	# main_visualization_final_result_with_camera_points_with_bboxes()


	# main_rotated_3d_bbox_object_txt()

	run_single_thread()
	# run_multi_threads()



if __name__ == "__main__":
	main()
