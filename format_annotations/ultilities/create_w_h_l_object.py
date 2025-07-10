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

# <scene_id> <class_id> <object_id> <frame_id> <x> <y> <z> <width> <length> <height> <yaw>
warehouse_017_objects = [
		[17.00, 0.00, 0.00 , 0.00, -5.87, 1.84  , 0.92, 0.59, 0.63, 1.84, -1.01],
		[17.00, 0.00, 1.00 , 0.00, -2.49, 0.57  , 0.92, 0.59, 0.63, 1.84, 1.22],
		[17.00, 0.00, 2.00 , 0.00, 0.65 , -6.58 , 0.77, 0.55, 0.63, 1.54, -0.72],
		[17.00, 0.00, 3.00 , 0.00, 5.40 , -5.25 , 0.78, 0.55, 0.63, 1.56, -0.63],
		[17.00, 0.00, 4.00 , 0.00, -2.04, -3.11 , 0.77, 0.50, 0.63, 1.54, -0.72],
		[17.00, 0.00, 5.00 , 0.00, 8.19 , -2.64 , 0.92, 0.73, 0.63, 1.84, -0.65],
		[17.00, 0.00, 6.00 , 0.00, -0.87, 5.15  , 0.85, 0.59, 0.63, 1.70, -0.82],
		[17.00, 0.00, 7.00 , 0.00, 6.11 , -8.93 , 0.92, 0.59, 0.63, 1.84, -0.78],
		[17.00, 0.00, 8.00 , 0.00, -1.06, -9.90 , 0.82, 0.62, 0.63, 1.64, -0.80],
		[17.00, 5.00, 9.00 , 0.00, -2.29, -5.36 , 0.86, 0.54, 0.80, 1.71, -0.31],
		[17.00, 4.00, 10.00, 0.00, -5.72, -5.86 , 0.80, 0.60, 0.45, 1.63, 1.20],
		[17.00, 4.00, 11.00, 0.00, -2.60, -7.55 , 0.81, 0.60, 0.45, 1.62, -0.45],
		[17.00, 5.00, 12.00, 0.00, 5.77 , 3.06  , 0.86, 0.54, 0.80, 1.71, 0.17],
		[17.00, 3.00, 13.00, 0.00, 5.27 , 8.11  , 0.10, 1.43, 0.65, 0.20, -0.19],
		[17.00, 3.00, 14.00, 0.00, -4.30, -2.76 , 0.10, 1.43, 0.65, 0.20, -0.41],
		[17.00, 2.00, 15.00, 0.00, 0.94 , 6.62  , 0.24, 0.71, 0.48, 0.48, -1.14],
		[17.00, 4.00, 16.00, 0.00, 0.00 , 16.25 , 0.82, 0.60, 0.45, 1.65, 0.01],
		[17.00, 5.00, 17.00, 0.00, -1.31, 15.85 , 0.86, 0.54, 0.80, 1.71, 0.07],
		[17.00, 3.00, 18.00, 0.00, 1.19 , 4.49  , 0.10, 1.43, 0.65, 0.20, 0.18],
		[17.00, 3.00, 19.00, 0.00, 1.95 , -10.12, 0.10, 1.43, 0.65, 0.20, 0.26]
]

# <scene_id> <class_id> <object_id> <frame_id> <x> <y> <z> <width> <length> <height> <yaw>
warehouse_018_objects = [
	[18.00, 0.00, 0.00 , 0.00, -3.16, -3.54, 0.92, 0.59, 0.53, 1.84, -0.94],
	[18.00, 0.00, 1.00 , 0.00, -3.62, 0.82 , 0.92, 0.45, 0.53, 1.45, -1.26],
	[18.00, 0.00, 2.00 , 0.00, 4.77 , -9.33, 0.92, 0.59, 0.53, 1.84, -0.97],
	[18.00, 0.00, 3.00 , 0.00, 6.07 , 13.06, 0.92, 0.59, 0.53, 1.35, 0.13],
	[18.00, 0.00, 4.00 , 0.00, 6.87 , 10.68, 0.92, 0.59, 0.53, 1.84, -0.63],
	[18.00, 0.00, 5.00 , 0.00, 3.02 , -5.34, 0.92, 0.50, 0.53, 1.75, 0.94],
	[18.00, 0.00, 6.00 , 0.00, -5.81, 4.51 , 0.92, 0.59, 0.53, 1.84, 0.90],
	[18.00, 0.00, 7.00 , 0.00, -6.48, 0.62 , 0.92, 0.59, 0.53, 1.45, -0.72],
	[18.00, 0.00, 8.00 , 0.00, 1.29 , 4.84 , 0.92, 0.59, 0.53, 1.84, -0.81],
	[18.00, 4.00, 9.00 , 0.00, -4.03, -1.42, 0.82, 0.60, 0.45, 1.65, -0.91],
	[18.00, 4.00, 10.00, 0.00, -7.48, -3.37, 0.82, 0.60, 0.45, 1.65, 1.51],
	[18.00, 5.00, 11.00, 0.00, -0.57, -6.31, 0.86, 0.54, 0.80, 1.71, 0.03],
	[18.00, 4.00, 12.00, 0.00, 0.25 , -8.88, 0.82, 0.60, 0.45, 1.65, -0.23],
	[18.00, 2.00, 13.00, 0.00, 6.67 , 16.65, 0.24, 0.71, 0.48, 0.48, 0.69],
	[18.00, 3.00, 14.00, 0.00, 1.93 , 11.27, 0.10, 1.43, 0.65, 0.20, 0.26],
	[18.00, 5.00, 15.00, 0.00, 6.31 , 3.96 , 0.86, 0.54, 0.80, 1.71, 0.37],
	[18.00, 3.00, 16.00, 0.00, 5.65 , -1.11, 0.10, 1.43, 0.65, 0.20, 1.06],
	[18.00, 4.00, 17.00, 0.00, 3.02 , 7.20 , 0.82, 0.60, 0.45, 1.65, -0.73],
	[18.00, 3.00, 18.00, 0.00, 2.92 , 16.01, 0.10, 1.43, 0.65, 0.20, -0.30],
	[18.00, 3.00, 19.00, 0.00, 1.59 , 13.33, 0.10, 1.43, 0.65, 0.20, -0.60],
	[18.00, 5.00, 20.00, 0.00, -5.18, 16.38, 0.86, 0.54, 0.80, 1.71, 0.62],
]

# <scene_id> <class_id> <object_id> <frame_id> <x> <y> <z> <width> <length> <height> <yaw>
warehouse_019_objects = [
	[19.00, 0.00, 0.00 , 0.00, -7.48, 6.88 , 0.92, 0.68, 0.61, 1.84, -0.67],
	[19.00, 0.00, 1.00 , 0.00, -3.52, 5.42 , 0.92, 0.59, 0.61, 1.84, -0.01],
	[19.00, 0.00, 2.00 , 0.00, 1.20 , -7.14, 0.92, 0.59, 0.61, 1.55, -0.60],
	[19.00, 0.00, 3.00 , 0.00, 5.82 , 15.24, 0.92, 0.78, 0.61, 1.84, -0.73],
	[19.00, 0.00, 4.00 , 0.00, 0.56 , 17.13, 0.92, 0.59, 0.61, 1.74, -1.12],
	[19.00, 0.00, 5.00 , 0.00, -5.49, 0.59 , 0.92, 0.59, 0.61, 1.78, 1.00],
	[19.00, 0.00, 6.00 , 0.00, -8.77, -9.73, 0.92, 0.59, 0.61, 1.84, 0.38],
	[19.00, 0.00, 7.00 , 0.00, 7.27 , -4.36, 0.92, 0.59, 0.61, 1.84, -0.36],
	[19.00, 0.00, 8.00 , 0.00, 7.74 , -4.50, 0.92, 0.59, 0.61, 1.84, -0.99],
	[19.00, 0.00, 9.00 , 0.00, 8.79 , -1.25, 0.92, 0.59, 0.61, 1.84, -0.86],
	[19.00, 4.00, 10.00, 0.00, -7.04, 11.62, 0.82, 0.60, 0.45, 1.65, 0.85],
	[19.00, 4.00, 11.00, 0.00, -2.50, 3.34 , 0.82, 0.60, 0.45, 1.65, -0.01],
	[19.00, 3.00, 12.00, 0.00, -4.98, 3.30 , 0.10, 1.43, 0.65, 0.20, -0.21],
	[19.00, 4.00, 13.00, 0.00, -2.86, 0.59 , 0.82, 0.60, 0.45, 1.65, 0.20],
	[19.00, 5.00, 14.00, 0.00, 2.78 , 10.22, 0.86, 0.54, 0.80, 1.71, -0.07],
	[19.00, 5.00, 15.00, 0.00, 4.64 , 7.40 , 0.86, 0.54, 0.80, 1.71, 1.35],
	[19.00, 2.00, 16.00, 0.00, 1.53 , 7.88 , 0.24, 0.71, 0.48, 0.48, 0.72],
	[19.00, 4.00, 17.00, 0.00, 3.13 , 10.15, 0.82, 0.60, 0.45, 1.65, -0.84],
	[19.00, 5.00, 18.00, 0.00, 1.47 , 4.90 , 0.86, 0.54, 0.80, 1.71, -0.35],
	[19.00, 3.00, 19.00, 0.00, 4.20 , 5.42 , 0.10, 1.43, 0.65, 0.20, -0.07],
	[19.00, 3.00, 20.00, 0.00, -3.00, -7.45, 0.10, 1.43, 0.65, 0.20, -0.34],
	[19.00, 5.00, 21.00, 0.00, -4.64, -6.55, 0.86, 0.54, 0.80, 1.71, 0.86],
	[19.00, 2.00, 22.00, 0.00, -3.49, -4.24, 0.24, 0.71, 0.48, 0.48, -0.41],
	[19.00, 3.00, 23.00, 0.00, 3.57 , 15.00, 0.10, 1.43, 0.65, 0.20, -0.38],
	[19.00, 5.00, 24.00, 0.00, 6.87 , 3.74 , 0.86, 0.54, 0.80, 1.71, 0.60],
	[19.00, 1.00, 25.00, 0.00, 4.14 , -1.68, 1.08, 1.21, 2.31, 2.15, -1.01],
	[19.00, 1.00, 26.00, 0.00, -7.25, -2.84, 1.08, 1.21, 2.31, 2.15, -1.57],
]

# <scene_id> <class_id> <object_id> <frame_id> <x> <y> <z> <width> <length> <height> <yaw>
warehouse_020_objects = [
	[20.00, 0.00, 0.00 , 0.00  , -14.96, -2.80 , 0.92, 0.59, 0.62, 1.96, -0.31],
	[20.00, 0.00, 1.00 , 0.00  , -8.24 , -17.81, 0.92, 0.59, 0.62, 1.75, -1.10],
	[20.00, 0.00, 2.00 , 0.00  , 5.56  , -10.06, 0.92, 0.59, 0.62, 1.83, -0.48],
	[20.00, 0.00, 3.00 , 0.00  , 1.50  , 0.94  , 0.92, 0.59, 0.62, 1.83, -0.78],
	[20.00, 0.00, 4.00 , 0.00  , -12.89, 5.27  , 0.92, 0.59, 0.62, 1.42, -0.30],
	[20.00, 0.00, 5.00 , 0.00  , 0.49  , 15.09 , 0.92, 0.59, 0.62, 1.83, -0.76],
	[20.00, 0.00, 6.00 , 0.00  , -4.25 , 22.15 , 0.92, 0.59, 0.62, 1.45, -0.92],
	[20.00, 0.00, 7.00 , 0.00  , -23.39, -8.36 , 0.92, 0.61, 0.62, 1.75, -0.85],
	[20.00, 3.00, 8.00 , 0.00  , -19.86, 4.30  , 0.10, 1.43, 0.65, 0.20, -0.40],
	[20.00, 5.00, 9.00 , 0.00  , -23.25, -1.37 , 0.86, 0.54, 0.80, 1.71, -0.92],
	[20.00, 2.00, 10.00, 0.00  , -18.30, 0.50  , 0.24, 0.71, 0.48, 0.48, 0.23],
	[20.00, 4.00, 11.00, 0.00  , -13.25, -0.53 , 0.82, 0.60, 0.45, 1.65, -0.20],
	# [20.00, 5.00, 12.00, 0.00  , -13.10, 5.24  , 0.86, 0.54, 0.80, 1.71, 0.81],
	[20.00, 4.00, 13.00, 0.00  , -8.75 , 15.57 , 0.82, 0.60, 0.45, 1.65, 0.06],
	[20.00, 4.00, 14.00, 0.00  , -10.08, 6.17  , 0.82, 0.60, 0.45, 1.65, -0.42],
	[20.00, 3.00, 15.00, 0.00  , 2.32  , -21.65, 0.10, 1.43, 0.65, 0.20, -0.09],
	[20.00, 3.00, 16.00, 0.00  , 1.42  , -20.33, 0.10, 1.43, 0.65, 0.20, -0.50],
	[20.00, 3.00, 17.00, 0.00  , 0.46  , -6.71 , 0.10, 1.43, 0.65, 0.20, -0.40],
	[20.00, 2.00, 18.00, 0.00  , 2.51  , -1.07 , 0.24, 0.71, 0.48, 0.48, 0.65],
	[20.00, 5.00, 19.00, 0.00  , 2.75  , 21.67 , 0.86, 0.54, 0.80, 1.71, -0.23],
	[20.00, 4.00, 20.00, 0.00  , -13.30, -11.95, 0.82, 0.60, 0.45, 1.65, -1.20],
	[20.00, 5.00, 21.00, 0.00  , -12.41, -9.25 , 0.86, 0.54, 0.80, 1.71, -1.33],
	[20.00, 2.00, 22.00, 0.00  , -12.66, 26.18 , 0.24, 0.71, 0.48, 0.48, -0.00],
	[20.00, 5.00, 23.00, 0.00  , -23.51, -13.06, 0.86, 0.54, 0.80, 1.71, -0.17],
	[20.00, 5.00, 24.00, 0.00  , -22.49, -13.95, 0.86, 0.54, 0.80, 1.71, 0.63],
	[20.00, 4.00, 25.00, 0.00  , -22.23, 19.41 , 0.82, 0.60, 0.45, 1.65, 0.01],
	[20.00, 1.00, 28.00, 0.00  , 2.59  , 10.80 , 1.08, 1.21, 2.31, 2.15, 1.57],
	[20.00, 1.00, 29.00, 0.00  , -25.22, -6.95 , 1.41, 1.13, 1.87, 2.77, -3.14],
	[20.00, 1.00, 30.00, 0.00  , 3.01  , 28.89 , 1.41, 1.13, 1.87, 2.77, 1.57],
	[20.00, 0.00, 26.00, 188.00, -23.14, -17.69, 0.92, 0.59, 0.62, 1.83, 0.37],
	[20.00, 0.00, 27.00, 548.00, -9.25 , -8.25 , 0.92, 0.59, 0.62, 1.83, 0.42],
]


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
	y                 = y - 5 + text_height + 10 * size_multi  # Position at the bottom
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
			text_size, _ = cv2.getTextSize(frame_id, font, fontScale=0.5 * size_multi, thickness=size_multi)
			text_width, text_height = text_size
			top_left = (frame_id_label_bl[0], frame_id_label_bl[1] + i * text_height * size_multi)
			bottom_right = (frame_id_label_bl[0] + 100 * size_multi , frame_id_label_bl[1] + (i + 1) * text_height * size_multi)
			cv2.rectangle(map_img, top_left, bottom_right, object_color, -1)
			bottom_left  = (frame_id_label_bl[0], frame_id_label_bl[1] + (i + 1) * text_height * size_multi)
			cv2.putText(map_img, object_type, bottom_left, font, 0.5 * size_multi, color=(0, 0, 0), thickness=size_multi)

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


def find_shape_by_type_id(scene_id, json_data_lookup_table, object_id , object_type, point_center, yaw):
	"""
	Find the shape of the object by its ID in the given scene.
	Returns a numpy array with the shape parameters.
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
	if object_type  in ["Forklift", "NovaCarter", "Transporter"]:
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
				shape["width"]      = shape_data["width"]
				shape["length"]     = shape_data["length"]
				shape["height"]     = shape_data["height"]
	else:
		if scene_id == 17:
			warehouse_objects = warehouse_017_objects
		elif scene_id == 18:
			warehouse_objects = warehouse_018_objects
		elif scene_id == 19:
			warehouse_objects = warehouse_019_objects
		elif scene_id == 20:
			warehouse_objects = warehouse_020_objects

		# <scene_id> <class_id> <object_id> <frame_id> <x> <y> <z> <width> <length> <height> <yaw>
		for index in range(len(warehouse_objects)):
			if int(warehouse_objects[index][2]) == object_id:
				shape = {
					"width" : warehouse_objects[index][7],  # width
					"length": warehouse_objects[index][8],  # length
					"height": warehouse_objects[index][9],  # height
					"pitch" : 0.0,
					"roll"  : 0.0,
					"yaw"   : 0.0,
				}
				return shape
	return shape


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
			if line.startswith("#") or line.strip() == "":
				continue
			parts = np.array(line.split(), dtype=np.float32)
			if int(parts[0]) == scene_id:
				final_result.append(parts)

	# DEBUG:
	# print("\n\n\n\n\n[")
	# for result in final_result:
	# 	print("[", end="")
	# 	for index, ele in enumerate(result):
	# 		if index < len(result) - 1:
	# 			print(f"{ele:.2f},", end="")
	# 	print(f"{result[-1]:.2f}],")
	# print("]\n\n\n\n\n")

	return sorted(final_result, key=cmp_to_key(custom_result_sort))


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
	yaw, pitch, roll = 0.0, 0.0, 0.0  # Ensure yaw, pitch, roll are in radians
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
		cv2.line(img, tuple(pts_2d[i]), tuple(pts_2d[j]), color, 1)

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
				img = put_text_with_border(img, f"{object_id}", (img_pt_x + 5 * size_multi, img_pt_y - 5 * size_multi), cv2.FONT_HERSHEY_SIMPLEX, (0.5  * size_multi), text_color=obj_info["color"], border_color=(0,0,0), thickness=size_multi)
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
				thickness=int(size_multi)  # Thickness of the bounding box lines
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
	# final_result = load_final_result(final_result_path, scene_id)
	if scene_id == 17:
		final_result = warehouse_017_objects
	elif scene_id == 18:
		final_result = warehouse_018_objects
	elif scene_id == 19:
		final_result = warehouse_019_objects
	elif scene_id == 20:
		final_result = warehouse_020_objects

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
				img_map_world = draw_information_on_map(img_map_world, int(img_current_index), color_chart, size_multi=size_multi)

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
	img_map_world = draw_information_on_map(img_map_world, int(img_current_index), color_chart, size_multi=size_multi)

	# save final image
	img_path_ou = os.path.join(folder_img_ou, f"{img_current_index:08d}.jpg")
	cv2.imwrite(img_path_ou, img_map_world)


def main_visualization_final_result_with_camera_points_with_bboxes(list_scene=None):
	# initialize paths
	folder_data_version = "MTMC_Tracking_2025_20250614"
	folder_info_in      = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/{folder_data_version}/ExtractFrames/lookup_table/"
	final_result_path   = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/{folder_data_version}/ExtractFrames/lookup_table/final_result.txt"
	folder_in           = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/{folder_data_version}/ExtractFrames/images_extract_full/"

	outline_color       = (0, 0, 0) # Outline color (B, G, R) - Blue
	outline_thickness   = 2  # Thickness of the outline
	radius              = 5  # Radius for the circle
	size_multi          = 8  # Size multiplier for the camera images

	folder_out          = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/{folder_data_version}/ExtractFrames/image_result/"
	if list_scene is None:
		list_scene  = ["Warehouse_017", "Warehouse_018", "Warehouse_019", "Warehouse_020"]

	for scene_name in tqdm(list_scene):
		visualization_final_result_with_camera_points_with_bboxes(
			scene_name, final_result_path, folder_info_in, folder_in, folder_out, radius, outline_color, outline_thickness, size_multi
		)


def run_single_thread():
	# main_visualization_final_result_with_camera_points_with_bboxes(list_scene  = ["Warehouse_017", "Warehouse_018", "Warehouse_019", "Warehouse_020"])
	main_visualization_final_result_with_camera_points_with_bboxes(list_scene  = ["Warehouse_020"])


def main():


	run_single_thread()



if __name__ == "__main__":
	main()
