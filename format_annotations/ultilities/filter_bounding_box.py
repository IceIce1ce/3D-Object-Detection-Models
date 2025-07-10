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

# NOTE: remove specific object id base on frame_id -> list object_id
frame_id_to_object_id_specific_dict = {
	"Warehouse_014" :{
		"Camera_0003" :{
			# frame id : list object id
			3045 : [171]
		}
	}
}

# NOTE: remove specific object id base on object_id -> list frame_id
object_id_to_frame_id_specific_dict = {
	"Warehouse_012" :{
		"Camera_0000" :{
			# object id : list frame id
			182 : [[7155, 7155]],
			183 : [[7755, 7755]],
			233 : [8925],
			352 : [[5925, 6015], [8355, 8715]],
			685 : [[5925, 6915]],
			616 : [[8205, 8205]],
		},
		"Camera_0001" :{
			# object id : list frame id
			351 : [[1155, 1155]],
			352 : [[5685, 5685], [8025, 8025]],
			452 : [[15, 555], [1275, 1365],[1785, 1785], [2535, 2595], [2775, 3795]],
			616 : [[6855, 6855]],
			618 : [[225, 315]],
		},
		"Camera_0002" :{
			# object id : list frame id
			182 : [[675, 675], [1785, 1785]],
			183 : [[4335, 4335], [5085, 5085]],
			453 : [[2595, 2775]],
			617 : [[1065, 1065]],
			618 : [[5205, 5205], [5865, 5895]],
			685 : [[2505, 2505]],
		},
		"Camera_0003" :{
			# object id : list frame id
			452 : [[2415, 2415]],
			183 : [[2535, 2865], [4155,4155]],
		},
		"Camera_0004" :{
			# object id : list frame id
			182 : [[5715, 5775], [8295, 8985]],
			350 : [[2895, 2895], [4245, 4815], [7275, 7275]],
			452 : [[345, 615], [765, 945], [1155, 1815], [2115, 3255],
			       [4095, 4185], [4665, 5535]],
			453 : [[3735, 3735]],
			617 : [[4605, 4995], [6345, 6435], [6795, 6945], [8385, 8385]]
		},
		"Camera_0005" :{
			# object id : list frame id
			350 : [[1995, 2085], [4995, 4995], [5265, 5265], [5925, 5925],
			       [7095, 7125]],
			351 : [[1155, 1155]],
			352 : [[2415, 2505], [4635, 4635], [4785, 4935]],
			452 : [[1635, 1785], [3855, 3855]],
			453 : [[315, 435], [4365, 4365]],
			616 : [[8445, 8445]],
			617 : [[2595, 3075]],
			684 : [[3165, 3165],[3225, 3195]],
			685 : [[4605, 4635], [7245, 7245]],
		},
		"Camera_0006" :{
			# object id : list frame id
			182 : [[1215, 1215]],
			350 : [[4995, 4995], [5055, 5265], [6885, 6885]],
			452 : [[1425, 1425]],
			684 : [[8325, 8415]],
		},
	},
	"Warehouse_013" :{
		"Camera_0000" :{
			# object id : list frame id
			233  : [8925],
		},
		"Camera_0001" :{
			# object id : list frame id
			233  : [4905],
			336  : [285],
		},
		"Camera_0002" :{
			# object id : list frame id
			128  : [675],
			233  : [4455],
			234  : [2025],
			235  : [3585, 3615, 3645, 3675],
		},
		"Camera_0003" :{
			# object id : list frame id
			129  : [[6465, 6675]],
			336  : [7125, 7155, [7635, 8985]],
		},
		"Camera_0004" :{
			# object id : list frame id
			0   : [[7035, 8985]],
			336 : [345, 645]
		},
		"Camera_0005" :{
			# object id : list frame id
			0   : [[3915, 3945]],
		},
		"Camera_0006" :{
			# object id : list frame id
			128 : [[6615, 8985]],
		},
		"Camera_0007" :{
			# object id : list frame id
			336 : [[15, 795]],
		},
		"Camera_0011" :{
			# object id : list frame id
			0   : [[15, 75], [825, 885]],
			2   : [4515, [6375, 6465]],
			235 : [1125, [7665, 7725]],
		},
	},
	"Warehouse_014" :{
		"Camera_0004" :{
			# object id : list frame id
			632  : [1185],
		},
		"Camera_0006" :{
			# object id : list frame id
			171  : [105, 1455, 1965, 3855, 5205, 5745, 7095, 7605, 8445],
			173  : [2115],
		},
		"Camera_0007" :{
			# object id : list frame id
			564  : "all",
			632  : [345, 3645, 3735, 3795, 3915, 3945, 7395],
			172  : [8655],
		},
		"Camera_0011" :{
			# object id : list frame id
			631  : [4005],
			# 632  : [675, 4275, 6075],
		},


	}
}

# NOTE: remove specific object type base on area of bounding box
object_area_specific_dict = {
	"Warehouse_012" :{
		"Camera_0000" :{
			# object id : list frame id
			"Person"  : 3675.0
		},
		"Camera_0001" :{
			# object id : list frame id
			"Person"  : 1120.0
		},
		"Camera_0002" :{
			# object id : list frame id
			"Person"  : 3675.0
		},
		"Camera_0003" :{
			# object id : list frame id
			"NovaCarter"  : 1120.0
		},
		"Camera_0004" :{
			# object id : list frame id
			"Person"  : 1120.0
		},
		"Camera_0005" :{
			# object id : list frame id
			"Person"  : 3675.0
		},
		"Camera_0006" :{
			# object id : list frame id
			"Person"      : 3675.0,
			"NovaCarter"  : 1120.0
		},
	},
	"Warehouse_013" :{
		"Camera_0002" :{
			# object id : list frame id
			"Person"  : 3675.0
		},
		"Camera_0004" :{
			# object id : list frame id
			"Person"  : 1120.0
		},
		"Camera_0005" :{
			# object id : list frame id
			"Person"  : 3675.0
		},
		"Camera_0006" :{
			# object id : list frame id
			"Person"      : 3675.0,
			"NovaCarter"  : 3675.0
		},
		"Camera_0007" :{
			# object id : list frame id
			"Person"      : 3275.0,
			"NovaCarter"  : 1120.0
		},
		"Camera_0008" :{
			# object id : list frame id
			"Person"      : 3275.0,
		},
		"Camera_0009" :{
			# object id : list frame id
			"Person"      : 2120.0,
		},
		"Camera_0010" :{
			# object id : list frame id
			"Person"      : 3275.0,
		},
		"Camera_0011" :{
			# object id : list frame id
			"Person"      : 3275.0,
		},
	},
	"Warehouse_014" :{
		"Camera_0007" :{
			# object id : list frame id
			"Transporter"  : 1120.0
		},
		"Camera_0011" :{
			# object id : list frame id
			"Transporter"  : 1120.0
		},
	}
}


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

def get_view_point_name(scene_name, camera_name):
	"""Get view point name from scene name and camera name.
	"""
	camera_name = os.path.basename(camera_name)
	scene_name  = os.path.basename(scene_name)
	view_name   = f"{scene_name}__{camera_name}"
	return view_name

def load_json_filter_bounding_box_x_anylabeling(json_path):
	# Load the JSON file
	with open(json_path, "r") as f:
		data = json.load(f)

	# Extract shapes
	shapes = data.get("shapes", [])

	# Collect (label, points) pairs
	rois = []
	for shape in shapes:
		label       = shape.get("label", "undefined")
		points      = shape.get("points", [])
		description = shape.get("description")
		polygon     = shapely.geometry.Polygon(points)
		rois.append({
			"label"      : label,
			"points"     : points,
			"description": description,
			"polygon"    : polygon,
		})

	return rois, data.get('imageWidth'), data.get('imageHeight')

# TODO: FILTER HERE
def filter_bounding_box(scene_name, camera_name, rois, instance_data, img_size):
	"""Check if the bounding box is valid or not

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
	# DEBUG:
	# return True


	# get the center bottom of bounding box
	point_bottom_center = shapely.geometry.Point(
		float(instance_data["x_tl"] + (instance_data["w"] / 2.0)),
		min(float(instance_data["y_br"]), img_size[1] - 5),
	)

	object_area = instance_data["w"] * instance_data["h"]

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

	# check ratio of the bounding box (NOT NEEDED)
	# if instance_data["object type"] in ["Person"]:
	# 	if instance_data["h"] / instance_data["w"] < 1.5 / 1:
	# 		return False

	# DEBUG: check ratio of the bounding box
	# if instance_data["object type"] in ["Forklift", "NovaCarter", "Transporter"]:
	# if instance_data["frame_id"] == 7935:
	# 	if instance_data["object id"] == 235:
	# 		print(f"{instance_data['frame_id']=} -- {instance_data['object id']=} -- {instance_data['object type']=}")
	# 		print(object_area)

	# remove specific object type base on area of bounding box
	# if scene_name in object_area_specific_dict:
	# 	if camera_name in object_area_specific_dict[scene_name]:
	# 		if instance_data["object type"] in object_area_specific_dict[scene_name][camera_name]:
	# 			if object_area < object_area_specific_dict[scene_name][camera_name][instance_data["object type"]]:
	# 				return False

	# remove specific object id base on frame_id -> list object_id
	# if scene_name in frame_id_to_object_id_specific_dict:
	# 	if camera_name in frame_id_to_object_id_specific_dict[scene_name]:
	# 		if instance_data["frame_id"] in frame_id_to_object_id_specific_dict[scene_name][camera_name]:
	# 			if instance_data["object id"] in frame_id_to_object_id_specific_dict[scene_name][camera_name][instance_data["frame_id"]]:
	# 				return False

	# remove specific object id base on object_id -> list frame_id
	# if scene_name in object_id_to_frame_id_specific_dict:
	# 	if camera_name in object_id_to_frame_id_specific_dict[scene_name]:
	# 		if instance_data["object id"] in object_id_to_frame_id_specific_dict[scene_name][camera_name]:
	# 			if "all" in object_id_to_frame_id_specific_dict[scene_name][camera_name][instance_data["object id"]]:
	# 				return False
	# 			# check if the frame_id is in the list of frame_id
	# 			for frame_id in object_id_to_frame_id_specific_dict[scene_name][camera_name][instance_data["object id"]]:
	# 				if frame_id == "all":
	# 					continue
	# 				# check if the frame_id is in the list of frame_id
	# 				if isinstance(frame_id, int) and instance_data["frame_id"] == frame_id:
	# 					return False
	# 				elif isinstance(frame_id, list):
	# 					if frame_id[0] <= instance_data["frame_id"] <= frame_id[1]:
	# 						return False

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


def main_filter_bounding_box_based_on_MOT():
	# init folder
	folder_input             = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames"
	folder_input_mot_img     = os.path.join(folder_input, "images")
	folder_input_mot_lbl     = os.path.join(folder_input, "labels_mot")
	folder_input_view_points = os.path.join(folder_input, "view_points")

	folder_output_mot_lbl    = os.path.join(folder_input, "labels_mot_filtered")
	folder_output_yolo_lbl   = os.path.join(folder_input, "labels_yolo_filtered")
	folder_output_draw_img   = os.path.join(folder_input, "images_draw")

	camera_list_img_in       = glob.glob(os.path.join(folder_input_mot_img, "*/*"))

	scene_name_spec  = "Warehouse_004"
	camera_name_spec = None

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


def main_filter_bounding_box_based_on_groundtruth():
	# init folder
	folder_data_version = "MTMC_Tracking_2025_20250614"
	folder_input        = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/{folder_data_version}/ExtractFrames/calibration_camera/modified/"

	scene_name_spec  = "Warehouse_000"
	camera_name_spec = ["Camera_0002", "Camera_0003"]
	# scene_name_spec  = "Warehouse_002"
	# camera_name_spec = ["Camera_0045"]

	groundtruth_path_in = os.path.join(folder_input, f"{scene_name_spec}_ground_truth.json")
	calibration_path_in = os.path.join(folder_input, f"{scene_name_spec}_calibration.json")
	groundtruth_path_ou = os.path.join(folder_input, f"{scene_name_spec}_ground_truth.json")

	rois  = {}
	img_ws = {}
	img_hs = {}
	for camera_name in tqdm(camera_name_spec, desc=f"Loading information of {scene_name_spec}"):
		view_point_name = get_view_point_name(scene_name_spec, Camera.adjust_camera_id(camera_name))
		view_point_path = os.path.join(folder_input, f"{view_point_name}.json")

		# NOTE: load json filter file
		roi, img_w, img_h   = load_json_filter_bounding_box_x_anylabeling(view_point_path)
		rois[camera_name]   = roi
		img_ws[camera_name] = img_w
		img_hs[camera_name] = img_h

	# Load the ground truth data
	with open(groundtruth_path_in, 'r') as f:
		json_data_groundtruth = json.load(f)

	# Iterate through the frame_id keys (e.g., "0", "1", "2")
	for frame_id in tqdm(json_data_groundtruth, desc=f"Processing frame_id in {scene_name_spec}__{camera_name}"):

		# Each key contains a list of instances (objects)
		frame_id_data_temp = []
		for instance in json_data_groundtruth[frame_id]:
			# Check if the key to remove exists in the nested dictionary
			bounding_box_visible_2d = {}
			for camera_id in instance["2d bounding box visible"]:
				camera_name = Camera.adjust_camera_id(camera_id)
				if camera_name not in camera_name_spec:
					continue
				bbox = np.array(instance["2d bounding box visible"][camera_id])

				instance_data = {
					"camera_id"      : camera_name,
					"frame_id"       : int(frame_id),
					"object id"      : instance["object id"],
					"x_tl"           : max(float(bbox[0]), 1.0),
					"y_tl"           : max(float(bbox[1]), 1.0),
					"x_br"           : min(float(bbox[2]), img_ws[camera_name] - 1),
					"y_br"           : min(float(bbox[3]), img_hs[camera_name] - 1),
					"w"              : abs(float(bbox[2]) - float(bbox[0])),
					"h"              : abs(float(bbox[3]) - float(bbox[1])),
				}
				if filter_bounding_box(scene_name_spec, camera_name, rois[camera_name], instance_data, (img_ws[camera_name], img_hs[camera_name])):
					bounding_box_visible_2d[camera_name] = instance["2d bounding box visible"][camera_id]
			instance["2d bounding box visible"] = bounding_box_visible_2d

			if bool(instance["2d bounding box visible"]):
				frame_id_data_temp.append(instance)

		json_data_groundtruth[frame_id] = frame_id_data_temp

	# Write the modified data to a new JSON file
	with open(groundtruth_path_ou, 'w') as f:
		json.dump(json_data_groundtruth, f)

	logger.info(f"All camera IDs have been replaced and saved to {groundtruth_path_ou}.")



def main():
	# rename folder of camera to become the format Camera_XXXX
	# rename_files_folder()

	# create view points from iamge
	# create_viewpoints_image()

	# main_filter_bounding_box_based_on_MOT()

	main_filter_bounding_box_based_on_groundtruth()
	pass


if __name__ == "__main__":
	main()