# import os
# import glob
# import json
# import shutil
# from typing import Optional
# import cv2
# from loguru import logger
# from tqdm import tqdm
# import numpy as np
# from mtmc.core.objects.units import Camera, MapWorld
#
# scene_id_table ={"Train" : {0 : "Warehouse_000", 1 : "Warehouse_001", 2 : "Warehouse_002", 3 : "Warehouse_003", 4 : "Warehouse_004", 5 : "Warehouse_005", 6 : "Warehouse_006",
# 						    7 : "Warehouse_007", 8 : "Warehouse_008", 9 : "Warehouse_009", 10: "Warehouse_010", 11: "Warehouse_011", 12: "Warehouse_012", 13: "Warehouse_013", 14: "Warehouse_014"},
# 				 "Val" : {15: "Warehouse_015", 16: "Warehouse_016", 22: "Lab_000", 23: "Hospital_000"},
# 				 "Test" : {17: "Warehouse_017", 18: "Warehouse_018", 19: "Warehouse_019", 20: "Warehouse_020"}}
# categories = [{"supercategory": "person", "id": 0, "name": "person"}, {"supercategory": "forklift", "id": 1, "name": "forklift"}, {"supercategory": "novacarter", "id": 2, "name": "novacarter"},
# 			  {"supercategory": "transporter", "id": 3, "name": "transporter"}, {"supercategory": "fouriergr1t2", "id": 4, "name": "fouriergr1t2"}, {"supercategory": "agilitydigit", "id": 5, "name": "agilitydigit"}]
# number_image_per_camera = 9000 # total frame each video
# number_image_skip = 9 # skip each one image
# number_image_train = 6000 # total training images
# number_image_test = 2000 # total testing image
# train_ratio = 6
# test_ratio = 2
#
# class json_serialize(json.JSONEncoder):
# 	def default(self, obj):
# 		if isinstance(obj, np.integer):
# 			return int(obj)
# 		if isinstance(obj, np.floating):
# 			return float(obj)
# 		if isinstance(obj, np.ndarray):
# 			return obj.tolist()
# 		return json.JSONEncoder.default(self, obj)
#
# def find_scene_id(scene_name):
# 	for split in scene_id_table:
# 		for scene_id, name in scene_id_table[split].items():
# 			if name == scene_name:
# 				return scene_id
# 	logger.error(f"Scene name {scene_name} not found in any dataset split.")
# 	return None
#
# def find_category_id(category_name):
# 	for category in categories:
# 		if category["name"] == category_name:
# 			return category["id"]
# 	logger.error(f"Category name {category_name} not found.")
# 	return None
#
# def get_image_id(camera_id, frame_id):
# 	return (camera_id * number_image_per_camera) + frame_id
#
# def create_info(id: Optional[int]=0, split: Optional[str]="Train"):
# 	info = {"id": id, "source": "Warehouse_008", "name": f"Warehouse_008_{split}", "split": split}
# 	return info
#
# ### compute bbox_3D ###
# def get_yaw_rotation_matrix(yaw):
# 	yaw = np.radians(yaw)
# 	return np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
#
# def process_world_coordinates(bbox_wpos, bbox_wscale, bbox_yaw):
# 	x, y, z = bbox_wpos
# 	w, h, d = bbox_wscale
# 	yaw = bbox_yaw
# 	corners = np.array([[w/2, h/2, d/2], [-w/2, h/2, d/2], [-w/2, -h/2, d/2], [w/2, -h/2, d/2],
# 						[w/2, h/2, -d/2], [-w/2, h/2, -d/2], [-w/2, -h/2, -d/2], [w/2, -h/2, -d/2]])
# 	rot_mat = get_yaw_rotation_matrix(yaw)
# 	rotated_corners = (rot_mat @ corners.T).T + np.array([x, y, z])
# 	return rotated_corners
#
# def convertWorldToCamera(points, extrinsic_mat):
# 	points = np.asarray(points)
# 	if points.ndim == 1 and points.shape[0] == 3:
# 		points = points.reshape(1, 3)
# 	elif points.ndim != 2 or points.shape[1] != 3:
# 		raise ValueError("Input must be shape (3,) or (N, 3)")
# 	homogeneous_corners = np.hstack([points, np.ones((points.shape[0], 1))])
# 	camera_coords = (np.dot(extrinsic_mat, homogeneous_corners.T)).T
# 	return camera_coords
#
# def get_coco_bbox(projected_2d):
# 	x_coords = projected_2d[:, 0]
# 	y_coords = projected_2d[:, 1]
# 	x1 = float(np.min(x_coords))
# 	y1 = float(np.min(y_coords))
# 	x2 = float(np.max(x_coords))
# 	y2 = float(np.max(y_coords))
# 	return [x1, y1, x2, y2]
#
# def projectCamera(camera_coords, intrinsic_mat):
# 	projected = (intrinsic_mat @ camera_coords.T).T
# 	projected_2d = projected[:, :2] / projected[:, 2:]
# 	return projected_2d
#
# def process_3D_bbox(bbox_wpos, bbox_wscale, bbox_yaw, extrinsicMatrix, intrinsicMatrix):
# 	intrinsicMat = np.array(intrinsicMatrix)
# 	extrinsicMat = np.array(extrinsicMatrix)
# 	R_cam = extrinsicMat[:, :3]
# 	world_coords = process_world_coordinates(bbox_wpos, bbox_wscale, bbox_yaw)
# 	camera_coords = convertWorldToCamera(world_coords, extrinsicMat)
# 	# get 2D COCO bbox from 3D bbox
# 	projected_coords = projectCamera(camera_coords, intrinsicMat)
# 	bbox2D_proj = get_coco_bbox(projected_coords)
# 	return camera_coords, R_cam, bbox2D_proj
# ### compute bbox_3D ###
#
# def create_images_annotations_json(scene_name):
# 	folder_input_frame = "extracted_AIC"
# 	calibration_path   = "extracted_AIC/calibration.json"
# 	groundtruth_path   = "extracted_AIC/ground_truth.json"
# 	folder_output_json  = "extracted_AIC/Warehouse_008_KITTI/"
# 	folder_output_train = os.path.join(folder_output_json, "Warehouse_008/train")
# 	folder_output_test = os.path.join(folder_output_json, "Warehouse_008/test")
# 	os.makedirs(folder_output_train, exist_ok=True)
# 	os.makedirs(folder_output_test, exist_ok=True)
# 	info_train = create_info(id=0, split="Train")
# 	info_test = create_info(id=1, split="Test")
# 	map_cfg = {"name": scene_name, "id": find_scene_id(scene_name), "type": "cartesian", "size": [1920, 1080], "map_image": None, "calibration_path": calibration_path,
# 			   "groundtruth_path": groundtruth_path, "folder_videos_path": None}
# 	map_world = MapWorld(map_cfg)
# 	list_camera = glob.glob(os.path.join(folder_input_frame, scene_name, "*"))
# 	images_train = []
# 	images_test = []
# 	img_count = -1
# 	img_get = 0
# 	img_get_train = 0
# 	img_get_test = 0
# 	annotations_train = []
# 	annotations_test  = []
# 	object_id = 0
# 	for camera_path in tqdm(list_camera, desc=f"Processing {scene_name}"):
# 		camera_name = os.path.basename(camera_path) # Camera_003
# 		camera_name = Camera.adjust_camera_id(camera_name) # Camera_003
# 		camera_id = camera_name
# 		camera_index = int(camera_name.split("_")[-1]) if "_" in camera_name else 0 # 3
# 		list_img = sorted(glob.glob(os.path.join(camera_path, "*.jpg")))
# 		for img_path in tqdm(list_img, desc=f"Processing {camera_name}"):
# 			img_count += 1
# 			if img_count % number_image_skip != 0:
# 				continue
# 			img_get += 1
# 			if img_get > (number_image_train + number_image_test):
# 				continue
# 			image_name = os.path.basename(img_path)
# 			image_id = int(os.path.splitext(image_name)[0])
# 			img_index = int(image_id)
# 			ratio = get_image_id(camera_index, image_id) % (train_ratio + test_ratio)
# 			if ratio < train_ratio:
# 				split_dataset = "train"
# 				img_name_new_noext  = f"{img_get_train:07d}"
# 				img_path_new = os.path.join(folder_output_train, f"{img_name_new_noext}.jpg")
# 				img_get_train += 1
# 			else:
# 				split_dataset = "test"
# 				img_name_new_noext = f"{(number_image_train + img_get_test):07d}"
# 				img_path_new = os.path.join(folder_output_test, f"{img_name_new_noext}.jpg")
# 				img_get_test += 1
# 			img_path_short = img_path_new.replace(folder_output_json, "")
# 			shutil.copyfile(img_path, img_path_new)
# 			img = cv2.imread(img_path)
# 			if img is None:
# 				logger.warning(f"Failed to read image: {img_path}")
# 				continue
# 			img_h, img_w = img.shape[:2]
# 			image = {"width": img_w, "height": img_h, "file_path": img_path_short, "K": map_world.cameras[camera_name].intrinsic_matrix, "id": int(img_name_new_noext), "dataset_id": ""}
# 			ratio = get_image_id(camera_index, image_id) % (train_ratio + test_ratio)
# 			if ratio < train_ratio:
# 				image["dataset_id"] = info_train["id"]
# 				images_train.append(image)
# 			else:
# 				image["dataset_id"] = info_test["id"]
# 				images_test.append(image)
# 			for instance_key in map_world.instances:
# 				instance = map_world.instances[instance_key]
# 				if instance.frames is None or str(img_index) not in instance.frames:
# 					continue
# 				if camera_id not in instance.frames[str(img_index)]["bbox_visible_2d"]:
# 					continue
# 				bbox3D_cam, R_cam, bbox2D_proj = process_3D_bbox(instance.frames[str(img_index)]["location_3d"], instance.frames[str(img_index)]["scale_3d"],
# 											 instance.frames[str(img_index)]["rotation_3d"][2], map_world.cameras[camera_id].extrinsic_matrix,
# 											 map_world.cameras[camera_id].intrinsic_matrix)
# 				# proj_box3d, bbox3D_cam = instance.get_3d_bounding_box_on_2d_image_coordinate(location_3d=instance.frames[str(img_index)]["location_3d"],
# 				# 																			 scale_3d=instance.frames[str(img_index)]["scale_3d"],
# 				# 																			 rotation_3d=instance.frames[str(img_index)]["rotation_3d"],
# 				# 																			 intrinsic_matrix=map_world.cameras[camera_id].intrinsic_matrix,
# 				# 																			 extrinsic_matrix=map_world.cameras[camera_id].extrinsic_matrix) # [8, 2, 1], [8, 3]
# 				dimensions = instance.frames[str(img_index)]["scale_3d"] # [w, l, h]
# 				dimensions = [dimensions[0], dimensions[2], dimensions[1]] # [w, h, l]
# 				# annotation = {"id": object_id, "image_id": int(img_name_new_noext), "category_id": find_category_id(instance.object_type.lower()), "category_name": str(instance.object_type).lower(),
# 				# 			  "valid3D": True, "bbox2D_tight": instance.frames[str(img_index)]["bbox_visible_2d"][camera_id], "bbox2D_proj": instance.frames[str(img_index)]["bbox_visible_2d"][camera_id],
# 				# 			  "bbox2D_trunc": instance.frames[str(img_index)]["bbox_visible_2d"][camera_id], "bbox3D_cam": bbox3D_cam, "center_cam": instance.frames[str(img_index)]["location_3d"],
# 				# 			  "dimensions": dimensions, "R_cam": map_world.cameras[camera_id].rotation_matrix, "behind_camera": False, "visibility": 1, "truncation": 0,
# 				# 			  "segmentation_pts": -1, "lidar_pts": -1, "depth_error": -1}
# 				annotation = {"id": object_id, "image_id": int(img_name_new_noext), "category_id": find_category_id(instance.object_type.lower()),
# 							  "category_name": str(instance.object_type).lower(), "valid3D": True, "bbox2D_tight": instance.frames[str(img_index)]["bbox_visible_2d"][camera_id],
# 							  "bbox2D_proj": bbox2D_proj, "bbox2D_trunc": bbox2D_proj, "bbox3D_cam": bbox3D_cam, "center_cam": instance.frames[str(img_index)]["location_3d"],
# 							  "dimensions": dimensions, "R_cam": R_cam, "behind_camera": False, "visibility": 1, "truncation": 0, "segmentation_pts": -1, "lidar_pts": -1, "depth_error": -1}
# 				if split_dataset == "train":
# 					annotations_train.append(annotation)
# 				else:
# 					annotations_test.append(annotation)
# 				object_id = object_id + 1
# 	data_json = {"images": images_train}
# 	with open(os.path.join(folder_output_json, f"{scene_name}_images_train.json"), 'w') as file:
# 		json.dump(data_json, file, indent=4, cls=json_serialize)
# 	data_json = {"images": images_test}
# 	with open(os.path.join(folder_output_json, f"{scene_name}_images_test.json"), 'w') as file:
# 		json.dump(data_json, file, indent=4, cls=json_serialize)
# 	data_json = {"annotations" : annotations_train,}
# 	with open(os.path.join(folder_output_json, f"{scene_name}_annotations_train.json"), 'w') as file:
# 		json.dump(data_json, file, indent=4, cls=json_serialize)
# 	data_json = {"annotations" : annotations_test}
# 	with open(os.path.join(folder_output_json, f"{scene_name}_annotations_test.json"), 'w') as file:
# 		json.dump(data_json, file, indent=4, cls=json_serialize)
#
# def combine_all_components(scene_name):
# 	folder_output_json = "extracted_AIC/Warehouse_008_KITTI/"
# 	images_train_path = os.path.join(folder_output_json, f"{scene_name}_images_train.json")
# 	images_test_path = os.path.join(folder_output_json, f"{scene_name}_images_test.json")
# 	annotations_train_path = os.path.join(folder_output_json, f"{scene_name}_annotations_train.json")
# 	annotations_test_path = os.path.join(folder_output_json, f"{scene_name}_annotations_test.json")
# 	with open(images_train_path, 'r') as file:
# 		images_train = json.load(file)
# 	with open(images_test_path, 'r') as file:
# 		images_test = json.load(file)
# 	with open(annotations_train_path, 'r') as file:
# 		annotations_train = json.load(file)
# 	with open(annotations_test_path, 'r') as file:
# 		annotations_test = json.load(file)
# 	data_json = {"info": create_info(id=0, split="Train"), "categories": categories, "images": images_train["images"], "annotations": annotations_train["annotations"]}
# 	output_file = os.path.join(folder_output_json, f"{scene_name}_train.json")
# 	with open(output_file, 'w') as file:
# 		json.dump(data_json, file)
# 	logger.info(f"Combined JSON saved to {output_file}")
# 	data_json = {"info": create_info(id=1, split="Test"), "categories": categories, "images": images_test["images"], "annotations": annotations_test["annotations"]}
# 	output_file = os.path.join(folder_output_json, f"{scene_name}_test.json")
# 	with open(output_file, 'w') as file:
# 		json.dump(data_json, file)
# 	logger.info(f"Combined JSON saved to {output_file}")
#
# if __name__ == "__main__":
# 	scene_name = "Warehouse_008"
# 	create_images_annotations_json(scene_name)
# 	combine_all_components(scene_name)

# import os
# import glob
# import json
# import shutil
# from typing import Optional
# import cv2
# from loguru import logger
# from tqdm import tqdm
# import numpy as np
# from mtmc.core.objects.units import Camera, MapWorld
# import math
#
# scene_id_table ={"Train" : {0 : "Warehouse_000", 1 : "Warehouse_001", 2 : "Warehouse_002", 3 : "Warehouse_003", 4 : "Warehouse_004", 5 : "Warehouse_005", 6 : "Warehouse_006",
# 						    7 : "Warehouse_007", 8 : "Warehouse_008", 9 : "Warehouse_009", 10: "Warehouse_010", 11: "Warehouse_011", 12: "Warehouse_012", 13: "Warehouse_013", 14: "Warehouse_014"},
# 				 "Val" : {15: "Warehouse_015", 16: "Warehouse_016", 22: "Lab_000", 23: "Hospital_000"},
# 				 "Test" : {17: "Warehouse_017", 18: "Warehouse_018", 19: "Warehouse_019", 20: "Warehouse_020"}}
# categories = [{"supercategory": "person", "id": 0, "name": "person"}, {"supercategory": "vehicle & road", "id": 1, "name": "forklift"},
# 			  {"supercategory": "vehicle & road", "id": 2, "name": "novacarter"}, {"supercategory": "vehicle & road", "id": 3, "name": "transporter"},
# 			  {"supercategory": "person", "id": 4, "name": "fouriergr1t2"}, {"supercategory": "person", "id": 5, "name": "agilitydigit"}]
# # categories = [{"supercategory": "person", "id": 0, "name": "person"}, {"supercategory": "vehicle & road", "id": 1, "name": "novacarter"},
# # 			  {"supercategory": "vehicle & road", "id": 2, "name": "transporter"}]
# number_image_per_camera = 9000 # total frame each video
# number_image_skip = 9 # skip each one image
# number_image_train = 6000 # total training images
# number_image_val = 1000 # total testing image
# number_image_test = 1000
# train_ratio = 6
# val_ratio = 1
# test_ratio = 1
#
# class json_serialize(json.JSONEncoder):
# 	def default(self, obj):
# 		if isinstance(obj, np.integer):
# 			return int(obj)
# 		if isinstance(obj, np.floating):
# 			return float(obj)
# 		if isinstance(obj, np.ndarray):
# 			return obj.tolist()
# 		return json.JSONEncoder.default(self, obj)
#
# def find_scene_id(scene_name):
# 	for split in scene_id_table:
# 		for scene_id, name in scene_id_table[split].items():
# 			if name == scene_name:
# 				return scene_id
# 	logger.error(f"Scene name {scene_name} not found in any dataset split.")
# 	return None
#
# def find_category_id(category_name):
# 	for category in categories:
# 		if category["name"] == category_name:
# 			return category["id"]
# 	logger.error(f"Category name {category_name} not found.")
# 	return None
#
# def get_image_id(camera_id, frame_id):
# 	return (camera_id * number_image_per_camera) + frame_id
#
# def create_info(id: Optional[int]=0, split: Optional[str]="Train", scene_name: Optional[str]='Warehouse_008'):
# 	info = {"id": id, "source": scene_name, "name": f"{scene_name}_{split}", "split": split}
# 	return info
#
# # def get_yaw_rotation_matrix(pitch, roll, yaw):
# # 	pitch = np.radians(pitch)
# # 	roll = np.radians(roll)
# # 	yaw = np.radians(yaw)
# # 	R_x = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(pitch), -np.sin(pitch)], [0.0, np.sin(pitch), np.cos(pitch)]])
# # 	R_y = np.array([[np.cos(roll), 0.0, np.sin(roll)], [0.0, 1.0, 0.0], [-np.sin(roll), 0.0, np.cos(roll)]])
# # 	R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
# # 	R = R_z @ R_y @ R_x
# # 	return R, R_y
#
# # def process_world_coordinates(bbox_wpos, bbox_wscale, bbox_pitch, bbox_roll, bbox_yaw):
# # 	x, y, z = bbox_wpos
# # 	w, h, d = bbox_wscale
# # 	pitch, roll, yaw = bbox_pitch, bbox_roll, bbox_yaw
# # 	corners = np.array([[-w / 2, -h / 2, -d / 2], [w / 2, -h / 2, -d / 2], [w / 2, h / 2, -d / 2], [-w / 2, h / 2, -d / 2],
# # 		 				[-w / 2, -h / 2, d / 2], [w / 2, -h / 2, d / 2], [w / 2, h / 2, d / 2], [-w / 2, h / 2, d / 2]])
# # 	rot_mat, R_y = get_yaw_rotation_matrix(pitch, roll, yaw)
# # 	rotated_corners = (rot_mat @ corners.T).T + np.array([x, y, z])
# # 	return rotated_corners, R_y
#
# def process_world_coordinates(bbox_wpos, bbox_wscale, pitch, roll, yaw):
# 	center_x, center_y, center_z = bbox_wpos
# 	w, l, h = bbox_wscale
# 	yaw = np.radians(yaw)
# 	local_corners = np.array([[-l / 2, -w / 2, -h / 2], [l / 2, -w / 2, -h / 2], [l / 2, w / 2, -h / 2], [-l / 2, w / 2, -h / 2],
# 							  [-l / 2, -w / 2, h / 2], [l / 2, -w / 2, h / 2], [l / 2, w / 2, h / 2], [-l / 2, w / 2, h / 2]])
# 	rotation_matrix = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]])
# 	rotated_corners = np.dot(local_corners, rotation_matrix.T)
# 	global_corners = rotated_corners + np.array([center_x, center_y, center_z])
# 	return global_corners, rotation_matrix
#
# def convertWorldToCamera(points, extrinsic_mat):
# 	points = np.asarray(points)
# 	if points.ndim == 1 and points.shape[0] == 3:
# 		points = points.reshape(1, 3)
# 	elif points.ndim != 2 or points.shape[1] != 3:
# 		raise ValueError("Input must be shape (3,) or (N, 3)")
# 	homogeneous_corners = np.hstack([points, np.ones((points.shape[0], 1))])
# 	camera_coords = (np.dot(extrinsic_mat, homogeneous_corners.T)).T
# 	return camera_coords
#
# def get_coco_bbox(projected_2d):
# 	x_coords = projected_2d[:, 0]
# 	y_coords = projected_2d[:, 1]
# 	x1 = float(np.min(x_coords))
# 	y1 = float(np.min(y_coords))
# 	x2 = float(np.max(x_coords))
# 	y2 = float(np.max(y_coords))
# 	return [x1, y1, x2, y2]
#
# def projectCamera(camera_coords, intrinsic_mat):
# 	projected = (intrinsic_mat @ camera_coords.T).T
# 	projected_2d = projected[:, :2] / projected[:, 2:]
# 	return projected_2d
#
# def process_3D_bbox(bbox_wpos, bbox_wscale, bbox_pitch, bbox_roll, bbox_yaw, extrinsicMatrix, intrinsicMatrix):
# 	intrinsicMat = np.array(intrinsicMatrix)
# 	extrinsicMat = np.array(extrinsicMatrix)
# 	world_coords, R_cam = process_world_coordinates(bbox_wpos, bbox_wscale, bbox_pitch, bbox_roll, bbox_yaw)
# 	camera_coords = convertWorldToCamera(world_coords, extrinsicMat)
# 	projected_coords = projectCamera(camera_coords, intrinsicMat)
# 	bbox2D_proj = get_coco_bbox(projected_coords)
# 	return camera_coords, R_cam, bbox2D_proj
#
# def create_images_annotations_json(scene_name):
# 	folder_input_frame = "extracted_AIC"
# 	calibration_path = f"extracted_AIC/{scene_name}_calibration.json"
# 	groundtruth_path = f"extracted_AIC/{scene_name}_ground_truth.json"
# 	folder_output_json = f"extracted_AIC/{scene_name}_KITTI/"
# 	folder_output_train = os.path.join(folder_output_json, f"{scene_name}/train")
# 	folder_output_val = os.path.join(folder_output_json, f"{scene_name}/val")
# 	folder_output_test = os.path.join(folder_output_json, f"{scene_name}/test")
# 	os.makedirs(folder_output_train, exist_ok=True)
# 	os.makedirs(folder_output_val, exist_ok=True)
# 	os.makedirs(folder_output_test, exist_ok=True)
# 	info_train = create_info(id=0, split="Train", scene_name=scene_name)
# 	info_val = create_info(id=1, split="Validation", scene_name=scene_name)
# 	info_test = create_info(id=2, split="Test", scene_name=scene_name)
# 	map_cfg = {"name": scene_name, "id": find_scene_id(scene_name), "type": "cartesian", "size": [1920, 1080], "map_image": None, "calibration_path": calibration_path,
# 			   "groundtruth_path": groundtruth_path, "folder_videos_path": None}
# 	map_world = MapWorld(map_cfg)
# 	list_camera = glob.glob(os.path.join(folder_input_frame, scene_name, "*"))
# 	images_train = []
# 	images_val = []
# 	images_test = []
# 	img_count = -1
# 	img_get = 0
# 	img_get_train = 0
# 	img_get_val = 0
# 	img_get_test = 0
# 	annotations_train = []
# 	annotations_val = []
# 	annotations_test  = []
# 	object_id = 1
# 	for camera_path in tqdm(list_camera, desc=f"Processing {scene_name}"):
# 		camera_name = os.path.basename(camera_path)
# 		camera_name = Camera.adjust_camera_id(camera_name)
# 		camera_id = camera_name
# 		camera_index = int(camera_name.split("_")[-1]) if "_" in camera_name else 0
# 		list_img = sorted(glob.glob(os.path.join(camera_path, "*.jpg")))
# 		for img_path in tqdm(list_img, desc=f"Processing {camera_name}"):
# 			img_count += 1
# 			if img_count % number_image_skip != 0:
# 				continue
# 			img_get += 1
# 			if img_get > (number_image_train + number_image_val + number_image_test):
# 				continue
# 			image_name = os.path.basename(img_path)
# 			image_id = int(os.path.splitext(image_name)[0])
# 			img_index = int(image_id)
# 			ratio = get_image_id(camera_index, image_id) % (train_ratio + val_ratio + test_ratio)
# 			if ratio < train_ratio:
# 				split_dataset = "train"
# 				img_name_new_noext = f"{img_get_train:07d}"
# 				img_path_new = os.path.join(folder_output_train, f"{img_name_new_noext}.jpg")
# 				img_get_train += 1
# 			elif ratio < (train_ratio + val_ratio):
# 				split_dataset = "val"
# 				img_name_new_noext = f"{(number_image_train + img_get_val):07d}"
# 				img_path_new = os.path.join(folder_output_val, f"{img_name_new_noext}.jpg")
# 				img_get_val += 1
# 			else:
# 				split_dataset = "test"
# 				img_name_new_noext = f"{(number_image_train + number_image_val + img_get_test):07d}"
# 				img_path_new = os.path.join(folder_output_test, f"{img_name_new_noext}.jpg")
# 				img_get_test += 1
# 			img_path_short = img_path_new.replace(folder_output_json, "")
# 			shutil.copyfile(img_path, img_path_new)
# 			img = cv2.imread(img_path)
# 			if img is None:
# 				logger.warning(f"Failed to read image: {img_path}")
# 				continue
# 			img_h, img_w = img.shape[:2]
# 			image = {"width": img_w, "height": img_h, "file_path": img_path_short, "K": map_world.cameras[camera_name].intrinsic_matrix, "id": int(img_name_new_noext), "dataset_id": ""}
# 			if split_dataset == "train":
# 				image["dataset_id"] = info_train["id"]
# 				images_train.append(image)
# 			elif split_dataset == "val":
# 				image["dataset_id"] = info_val["id"]
# 				images_val.append(image)
# 			else:
# 				image["dataset_id"] = info_test["id"]
# 				images_test.append(image)
# 			for instance_key in map_world.instances:
# 				instance = map_world.instances[instance_key]
# 				if instance.frames is None or str(img_index) not in instance.frames:
# 					continue
# 				if camera_id not in instance.frames[str(img_index)]["bbox_visible_2d"]:
# 					continue
# 				bbox3D_cam, R_cam, bbox2D_proj = process_3D_bbox(instance.frames[str(img_index)]["location_3d"], instance.frames[str(img_index)]["scale_3d"],
# 																 instance.frames[str(img_index)]["rotation_3d"][0], instance.frames[str(img_index)]["rotation_3d"][1],
# 											 					 instance.frames[str(img_index)]["rotation_3d"][2], map_world.cameras[camera_id].extrinsic_matrix,
# 											 					 map_world.cameras[camera_id].intrinsic_matrix)
# 				dimensions = instance.frames[str(img_index)]["scale_3d"] # [w, l, h]
# 				dimensions = [dimensions[0], dimensions[2], dimensions[1]] # [w, h, l]
# 				convert_center_cam = np.array(bbox3D_cam)
# 				convert_center_cam = np.mean(convert_center_cam, axis=0).tolist()
# 				annotation = {"id": object_id, "image_id": int(img_name_new_noext), "category_id": find_category_id(instance.object_type.lower()),
# 							  "category_name": str(instance.object_type).lower(), "valid3D": True, "bbox2D_tight": instance.frames[str(img_index)]["bbox_visible_2d"][camera_id],
# 							  "bbox2D_proj": bbox2D_proj, "bbox2D_trunc": bbox2D_proj, "bbox3D_cam": bbox3D_cam, "center_cam": convert_center_cam,
# 							  "dimensions": dimensions, "R_cam": R_cam, "behind_camera": False, "visibility": 1.0, "truncation": 0.0, "segmentation_pts": -1, "lidar_pts": -1,
# 							  "depth_error": -1, 'dataset_id': image["dataset_id"]}
# 				if split_dataset == "train":
# 					annotations_train.append(annotation)
# 				elif split_dataset == 'val':
# 					annotations_val.append(annotation)
# 				else:
# 					annotations_test.append(annotation)
# 				object_id += 1
# 	data_json = {"images": images_train}
# 	with open(os.path.join(folder_output_json, f"{scene_name}_images_train.json"), 'w') as file:
# 		json.dump(data_json, file, indent=4, cls=json_serialize)
# 	data_json = {"images": images_val}
# 	with open(os.path.join(folder_output_json, f"{scene_name}_images_val.json"), 'w') as file:
# 		json.dump(data_json, file, indent=4, cls=json_serialize)
# 	data_json = {"images": images_test}
# 	with open(os.path.join(folder_output_json, f"{scene_name}_images_test.json"), 'w') as file:
# 		json.dump(data_json, file, indent=4, cls=json_serialize)
# 	data_json = {"annotations" : annotations_train}
# 	with open(os.path.join(folder_output_json, f"{scene_name}_annotations_train.json"), 'w') as file:
# 		json.dump(data_json, file, indent=4, cls=json_serialize)
# 	data_json = {"annotations": annotations_val}
# 	with open(os.path.join(folder_output_json, f"{scene_name}_annotations_val.json"), 'w') as file:
# 		json.dump(data_json, file, indent=4, cls=json_serialize)
# 	data_json = {"annotations" : annotations_test}
# 	with open(os.path.join(folder_output_json, f"{scene_name}_annotations_test.json"), 'w') as file:
# 		json.dump(data_json, file, indent=4, cls=json_serialize)
#
# def combine_all_components(scene_name):
# 	folder_output_json = f"extracted_AIC/{scene_name}_KITTI/"
# 	images_train_path = os.path.join(folder_output_json, f"{scene_name}_images_train.json")
# 	images_val_path = os.path.join(folder_output_json, f"{scene_name}_images_val.json")
# 	images_test_path = os.path.join(folder_output_json, f"{scene_name}_images_test.json")
# 	annotations_train_path = os.path.join(folder_output_json, f"{scene_name}_annotations_train.json")
# 	annotations_val_path = os.path.join(folder_output_json, f"{scene_name}_annotations_val.json")
# 	annotations_test_path = os.path.join(folder_output_json, f"{scene_name}_annotations_test.json")
# 	with open(images_train_path, 'r') as file:
# 		images_train = json.load(file)
# 	with open(images_val_path, 'r') as file:
# 		images_val = json.load(file)
# 	with open(images_test_path, 'r') as file:
# 		images_test = json.load(file)
# 	with open(annotations_train_path, 'r') as file:
# 		annotations_train = json.load(file)
# 	with open(annotations_val_path, 'r') as file:
# 		annotations_val = json.load(file)
# 	with open(annotations_test_path, 'r') as file:
# 		annotations_test = json.load(file)
# 	data_json = {"info": create_info(id=0, split="Train"), "categories": categories, "images": images_train["images"], "annotations": annotations_train["annotations"]}
# 	output_file = os.path.join(folder_output_json, f"{scene_name}_train.json")
# 	with open(output_file, 'w') as file:
# 		json.dump(data_json, file)
# 	logger.info(f"Combined JSON saved to {output_file}")
# 	data_json = {"info": create_info(id=1, split="Validation"), "categories": categories, "images": images_val["images"], "annotations": annotations_val["annotations"]}
# 	output_file = os.path.join(folder_output_json, f"{scene_name}_val.json")
# 	with open(output_file, 'w') as file:
# 		json.dump(data_json, file)
# 	logger.info(f"Combined JSON saved to {output_file}")
# 	data_json = {"info": create_info(id=2, split="Test"), "categories": categories, "images": images_test["images"], "annotations": annotations_test["annotations"]}
# 	output_file = os.path.join(folder_output_json, f"{scene_name}_test.json")
# 	with open(output_file, 'w') as file:
# 		json.dump(data_json, file)
# 	logger.info(f"Combined JSON saved to {output_file}")
#
# if __name__ == "__main__":
# 	scene_name = "Warehouse_008"
# 	create_images_annotations_json(scene_name)
# 	combine_all_components(scene_name)

import os
import glob
import json
import shutil
from typing import Optional
import cv2
from loguru import logger
from tqdm import tqdm
import numpy as np
from mtmc.core.objects.units import Camera, MapWorld
import math
from natsort import natsorted

class json_serialize(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.integer):
			return int(obj)
		if isinstance(obj, np.floating):
			return float(obj)
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return json.JSONEncoder.default(self, obj)

def find_scene_id(scene_name):
	for split in scene_id_table:
		for scene_id, name in scene_id_table[split].items():
			if name == scene_name:
				return scene_id
	logger.error(f"Scene name {scene_name} not found in any dataset split.")
	return None

def find_category_id(category_name):
	for category in categories:
		if category["name"] == category_name:
			return category["id"]
	logger.error(f"Category name {category_name} not found.")
	return None

def get_image_id(camera_id, frame_id):
	return (camera_id * number_image_per_camera) + frame_id

def create_info(id: Optional[int]=0, split: Optional[str]="Train", scene_name: Optional[str]='Warehouse_008'):
	info = {"id": id, "source": scene_name, "name": f"{scene_name}_{split}", "split": split}
	return info

def process_world_coordinates(bbox_wpos, bbox_wscale, pitch, roll, yaw):
	center_x, center_y, center_z = bbox_wpos
	h, w, l = bbox_wscale
	# local_corners = np.array([[-l / 2, -w / 2, -h / 2], [l / 2, -w / 2, -h / 2], [l / 2, w / 2, -h / 2], [-l / 2, w / 2, -h / 2],
	# 						  [-l / 2, -w / 2, h / 2], [l / 2, -w / 2, h / 2], [l / 2, w / 2, h / 2], [-l / 2, w / 2, h / 2]])
	local_corners = np.array([[-h / 2, -l / 2, -w / 2], [h / 2, -l / 2, -w / 2], [h / 2, -l / 2, w / 2], [-h / 2, -l / 2, w / 2],
		 					  [-h / 2, l / 2, -w / 2], [h / 2, l / 2, -w / 2], [h / 2, l / 2, w / 2], [-h / 2, l / 2, w / 2]])
	rotation_matrix = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]])
	rotated_corners = np.dot(local_corners, rotation_matrix.T)
	global_corners = rotated_corners + np.array([center_x, center_y, center_z])
	return global_corners, rotation_matrix

# def euler2mat(euler):
# 	R_x = np.array([[1, 0, 0], [0, math.cos(euler[0]), -math.sin(euler[0])], [0, math.sin(euler[0]), math.cos(euler[0])]])
# 	R_y = np.array([[math.cos(euler[1]), 0, math.sin(euler[1])], [0, 1, 0], [-math.sin(euler[1]), 0, math.cos(euler[1])]])
# 	R_z = np.array([[math.cos(euler[2]), -math.sin(euler[2]), 0], [math.sin(euler[2]), math.cos(euler[2]), 0], [0, 0, 1]])
# 	R = np.dot(R_z, np.dot(R_y, R_x))
# 	return R
#
# def process_world_coordinates(bbox_wpos, bbox_wscale, pitch, roll, yaw):
# 	center_x, center_y, center_z = bbox_wpos
# 	# w, l, h = bbox_wscale[0] / 2, bbox_wscale[1] / 2, bbox_wscale[2] / 2
# 	l, w, h = bbox_wscale[0] / 2, bbox_wscale[1] / 2, bbox_wscale[2] / 2
# 	local_corners = np.array([[-l, -w, -h], [l, -w, -h], [l, w, -h], [-l, w, -h], [-l, -w, h], [l, -w, h], [l, w, h], [-l, w, h]])
# 	# Rx = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
# 	# Ry = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
# 	# Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
# 	# rotation_matrix = Rz @ Ry @ Rx
# 	rotation_matrix = euler2mat([roll, pitch, yaw])
# 	rotated_corners = (rotation_matrix @ local_corners.T).T
# 	# rotated_corners = np.dot(local_corners, rotation_matrix.T)
# 	global_corners = rotated_corners + np.array([center_x, center_y, center_z])
# 	return global_corners, rotation_matrix

def convertWorldToCamera(points, extrinsic_mat):
	points = np.asarray(points)
	if points.ndim == 1 and points.shape[0] == 3:
		points = points.reshape(1, 3)
	elif points.ndim != 2 or points.shape[1] != 3:
		raise ValueError("Input must be shape (3,) or (N, 3)")
	homogeneous_corners = np.hstack([points, np.ones((points.shape[0], 1))])
	camera_coords = (np.dot(extrinsic_mat, homogeneous_corners.T)).T
	return camera_coords

def get_coco_bbox(projected_2d):
	x_coords = projected_2d[:, 0]
	y_coords = projected_2d[:, 1]
	x1 = float(np.min(x_coords))
	y1 = float(np.min(y_coords))
	x2 = float(np.max(x_coords))
	y2 = float(np.max(y_coords))
	return [x1, y1, x2, y2]

def projectCamera(camera_coords, intrinsic_mat):
	projected = (intrinsic_mat @ camera_coords.T).T
	projected_2d = projected[:, :2] / projected[:, 2:]
	return projected_2d

def process_3D_bbox(bbox_wpos, bbox_wscale, bbox_pitch, bbox_roll, bbox_yaw, extrinsicMatrix, intrinsicMatrix):
	intrinsicMat = np.array(intrinsicMatrix)
	extrinsicMat = np.array(extrinsicMatrix)
	world_coords, R_cam = process_world_coordinates(bbox_wpos, bbox_wscale, bbox_pitch, bbox_roll, bbox_yaw)
	camera_coords = convertWorldToCamera(world_coords, extrinsicMat)
	projected_coords = projectCamera(camera_coords, intrinsicMat)
	bbox2D_proj = get_coco_bbox(projected_coords)
	return camera_coords, R_cam, bbox2D_proj

def create_images_annotations_json(images_train, images_val, images_test, img_get, img_get_train, img_get_val, img_get_test, annotations_train, annotations_val,
								   annotations_test, object_id, root_folder, scene_name, folder_output_json):
	calibration_path = f"{root_folder}/{scene_name}_calibration.json"
	groundtruth_path = f"{root_folder}/{scene_name}_ground_truth.json"
	folder_output_train = os.path.join(folder_output_json, f"{scene_name}/train")
	folder_output_val = os.path.join(folder_output_json, f"{scene_name}/val")
	folder_output_test = os.path.join(folder_output_json, f"{scene_name}/test")
	os.makedirs(folder_output_train, exist_ok=True)
	os.makedirs(folder_output_val, exist_ok=True)
	os.makedirs(folder_output_test, exist_ok=True)
	info_train = create_info(id=0, split="Train", scene_name=scene_name)
	info_val = create_info(id=1, split="Validation", scene_name=scene_name)
	info_test = create_info(id=2, split="Test", scene_name=scene_name)
	map_cfg = {"name": scene_name, "id": find_scene_id(scene_name), "type": "cartesian", "size": [1920, 1080], "map_image": None, "calibration_path": calibration_path,
			   "groundtruth_path": groundtruth_path, "folder_videos_path": None}
	map_world = MapWorld(map_cfg)
	list_camera = glob.glob(os.path.join(root_folder, scene_name, "*"))
	for camera_path in tqdm(list_camera, desc=f"Processing {scene_name}"):
		camera_name = os.path.basename(camera_path)
		camera_name = Camera.adjust_camera_id(camera_name)
		camera_id = camera_name
		camera_index = int(camera_name.split("_")[-1]) if "_" in camera_name else 0
		list_img = sorted(glob.glob(os.path.join(camera_path, "*.jpg")))
		for img_path in tqdm(list_img, desc=f"Processing {camera_name}"):
			img_get += 1
			if img_get > (number_image_train + number_image_val + number_image_test):
				continue
			image_name = os.path.basename(img_path)
			image_id = int(os.path.splitext(image_name)[0])
			img_index = int(image_id)
			ratio = get_image_id(camera_index, image_id) % (train_ratio + val_ratio + test_ratio)
			if ratio < train_ratio:
				split_dataset = "train"
				img_name_new_noext = f"{img_get_train:07d}"
				img_path_new = os.path.join(folder_output_train, f"{img_name_new_noext}.jpg")
				img_get_train += 1
			elif ratio < (train_ratio + val_ratio):
				split_dataset = "val"
				img_name_new_noext = f"{(number_image_train + img_get_val):07d}"
				img_path_new = os.path.join(folder_output_val, f"{img_name_new_noext}.jpg")
				img_get_val += 1
			else:
				split_dataset = "test"
				img_name_new_noext = f"{(number_image_train + number_image_val + img_get_test):07d}"
				img_path_new = os.path.join(folder_output_test, f"{img_name_new_noext}.jpg")
				img_get_test += 1
			img_path_short = img_path_new.replace(folder_output_json, "")
			shutil.copyfile(img_path, img_path_new)
			img = cv2.imread(img_path)
			if img is None:
				logger.warning(f"Failed to read image: {img_path}")
				continue
			img_h, img_w = img.shape[:2]
			image = {"width": img_w, "height": img_h, "file_path": img_path_short, "K": map_world.cameras[camera_name].intrinsic_matrix, "id": int(img_name_new_noext), "dataset_id": ""}
			if split_dataset == "train":
				image["dataset_id"] = info_train["id"]
				images_train.append(image)
			elif split_dataset == "val":
				image["dataset_id"] = info_val["id"]
				images_val.append(image)
			else:
				image["dataset_id"] = info_test["id"]
				images_test.append(image)
			for instance_key in map_world.instances:
				instance = map_world.instances[instance_key]
				if instance.frames is None or str(img_index) not in instance.frames:
					continue
				if camera_id not in instance.frames[str(img_index)]["bbox_visible_2d"]:
					continue
				bbox3D_cam, R_cam, bbox2D_proj = process_3D_bbox(instance.frames[str(img_index)]["location_3d"], instance.frames[str(img_index)]["scale_3d"],
																 instance.frames[str(img_index)]["rotation_3d"][0], instance.frames[str(img_index)]["rotation_3d"][1],
											 					 instance.frames[str(img_index)]["rotation_3d"][2], map_world.cameras[camera_id].extrinsic_matrix,
											 					 map_world.cameras[camera_id].intrinsic_matrix)
				dimensions = instance.frames[str(img_index)]["scale_3d"] # [w, l, h]
				dimensions = [dimensions[0], dimensions[2], dimensions[1]] # [w, h, l]
				convert_center_cam = np.array(bbox3D_cam)
				convert_center_cam = np.mean(convert_center_cam, axis=0).tolist()
				annotation = {"id": object_id, "image_id": int(img_name_new_noext), "category_id": find_category_id(instance.object_type.lower()),
							  "category_name": str(instance.object_type).lower(), "valid3D": True, "bbox2D_tight": instance.frames[str(img_index)]["bbox_visible_2d"][camera_id],
							  "bbox2D_proj": bbox2D_proj, "bbox2D_trunc": bbox2D_proj, "bbox3D_cam": bbox3D_cam, "center_cam": convert_center_cam,
							  "dimensions": dimensions, "R_cam": R_cam, "behind_camera": False, "visibility": 1.0, "truncation": 0.0, "segmentation_pts": -1, "lidar_pts": -1,
							  "depth_error": -1, 'dataset_id': image["dataset_id"]}
				if split_dataset == "train":
					annotations_train.append(annotation)
				elif split_dataset == 'val':
					annotations_val.append(annotation)
				else:
					annotations_test.append(annotation)
				object_id += 1
	return images_train, images_val, images_test, annotations_train, annotations_val, annotations_test, img_get, img_get_train, img_get_val, img_get_test, object_id

def combine_all_components(folder_output_json):
	images_train_path = os.path.join(folder_output_json, f"Warehouse_images_train.json")
	images_val_path = os.path.join(folder_output_json, f"Warehouse_images_val.json")
	images_test_path = os.path.join(folder_output_json, f"Warehouse_images_test.json")
	annotations_train_path = os.path.join(folder_output_json, f"Warehouse_annotations_train.json")
	annotations_val_path = os.path.join(folder_output_json, f"Warehouse_annotations_val.json")
	annotations_test_path = os.path.join(folder_output_json, f"Warehouse_annotations_test.json")
	with open(images_train_path, 'r') as file:
		images_train = json.load(file)
	with open(images_val_path, 'r') as file:
		images_val = json.load(file)
	with open(images_test_path, 'r') as file:
		images_test = json.load(file)
	with open(annotations_train_path, 'r') as file:
		annotations_train = json.load(file)
	with open(annotations_val_path, 'r') as file:
		annotations_val = json.load(file)
	with open(annotations_test_path, 'r') as file:
		annotations_test = json.load(file)
	data_json = {"info": create_info(id=0, split="Train"), "categories": categories, "images": images_train["images"], "annotations": annotations_train["annotations"]}
	output_file = os.path.join(folder_output_json, f"Warehouse_train.json")
	with open(output_file, 'w') as file:
		json.dump(data_json, file)
	logger.info(f"Combined JSON saved to {output_file}")
	data_json = {"info": create_info(id=1, split="Validation"), "categories": categories, "images": images_val["images"], "annotations": annotations_val["annotations"]}
	output_file = os.path.join(folder_output_json, f"Warehouse_val.json")
	with open(output_file, 'w') as file:
		json.dump(data_json, file)
	logger.info(f"Combined JSON saved to {output_file}")
	data_json = {"info": create_info(id=2, split="Test"), "categories": categories, "images": images_test["images"], "annotations": annotations_test["annotations"]}
	output_file = os.path.join(folder_output_json, f"Warehouse_test.json")
	with open(output_file, 'w') as file:
		json.dump(data_json, file)
	logger.info(f"Combined JSON saved to {output_file}")

if __name__ == "__main__":
	root_folder = '3d_dataset'
	# warehouse_folders = natsorted([f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))])
	warehouse_folders = ['Warehouse_014']
	scene_id_table = {"Train": {0: "Warehouse_000", 1: "Warehouse_001", 2: "Warehouse_002", 3: "Warehouse_003", 4: "Warehouse_004", 5: "Warehouse_005", 6: "Warehouse_006",
				  				7: "Warehouse_007", 8: "Warehouse_008", 9: "Warehouse_009", 10: "Warehouse_010", 11: "Warehouse_011", 12: "Warehouse_012",
				 	 		    13: "Warehouse_013", 14: "Warehouse_014"},
					  "Val": {15: "Warehouse_015", 16: "Warehouse_016", 22: "Lab_000", 23: "Hospital_000"},
					  "Test": {17: "Warehouse_017", 18: "Warehouse_018", 19: "Warehouse_019", 20: "Warehouse_020"}}
	categories = [{"supercategory": "person", "id": 0, "name": "person"}, {"supercategory": "vehicle & road", "id": 1, "name": "forklift"},
				  {"supercategory": "vehicle & road", "id": 2, "name": "novacarter"}, {"supercategory": "vehicle & road", "id": 3, "name": "transporter"},
				  {"supercategory": "person", "id": 4, "name": "fouriergr1t2"}, {"supercategory": "person", "id": 5, "name": "agilitydigit"}]
	list_img = []
	for scene_name in warehouse_folders:
		number_image_per_camera = len(glob.glob(f'{root_folder}/{scene_name}/*/*.jpg'))
		list_img.append(number_image_per_camera)
	total_list_img = sum(list_img)
	number_image_train = int(total_list_img * 80 / 100) # 180000
	number_image_val = int(total_list_img * 10 / 100) # 22500
	number_image_test = int(total_list_img * 10 / 100) # 22500
	train_ratio = 8
	val_ratio = 1
	test_ratio = 1
	folder_output_json = f"{root_folder}/processed_Warehouse/"
	images_train = []
	images_val = []
	images_test = []
	img_get = 0
	img_get_train = 0
	img_get_val = 0
	img_get_test = 0
	annotations_train = []
	annotations_val = []
	annotations_test = []
	object_id = 1
	list_out_images_train, list_out_images_val, list_out_images_test = [], [], []
	list_out_annotations_train, list_out_annotations_val, list_out_annotations_test = [], [], []
	for scene_name in warehouse_folders:
		out_images_train, out_images_val, out_images_test, out_annotations_train, out_annotations_val, out_annotations_test, out_img_get, out_img_get_train,\
		out_img_get_val, out_img_get_test, out_object_id = create_images_annotations_json(images_train, images_val, images_test, img_get, img_get_train, img_get_val, img_get_test,
																						  annotations_train, annotations_val, annotations_test, object_id, root_folder,
																						  scene_name, folder_output_json)
		list_out_images_train.extend(out_images_train)
		list_out_images_val.extend(out_images_val)
		list_out_images_test.extend(out_images_test)
		list_out_annotations_train.extend(out_annotations_train)
		list_out_annotations_val.extend(out_annotations_val)
		list_out_annotations_test.extend(out_annotations_test)
		img_get = out_img_get
		img_get_train = out_img_get_train
		img_get_val = out_img_get_val
		img_get_test = out_img_get_test
		object_id = out_object_id
	data_json = {"images": list_out_images_train}
	with open(os.path.join(folder_output_json, f"Warehouse_images_train.json"), 'w') as file:
		json.dump(data_json, file, indent=4, cls=json_serialize)
	data_json = {"images": list_out_images_val}
	with open(os.path.join(folder_output_json, f"Warehouse_images_val.json"), 'w') as file:
		json.dump(data_json, file, indent=4, cls=json_serialize)
	data_json = {"images": list_out_images_test}
	with open(os.path.join(folder_output_json, f"Warehouse_images_test.json"), 'w') as file:
		json.dump(data_json, file, indent=4, cls=json_serialize)
	data_json = {"annotations": list_out_annotations_train}
	with open(os.path.join(folder_output_json, f"Warehouse_annotations_train.json"), 'w') as file:
		json.dump(data_json, file, indent=4, cls=json_serialize)
	data_json = {"annotations": list_out_annotations_val}
	with open(os.path.join(folder_output_json, f"Warehouse_annotations_val.json"), 'w') as file:
		json.dump(data_json, file, indent=4, cls=json_serialize)
	data_json = {"annotations": list_out_annotations_test}
	with open(os.path.join(folder_output_json, f"Warehouse_annotations_test.json"), 'w') as file:
		json.dump(data_json, file, indent=4, cls=json_serialize)
	combine_all_components(folder_output_json)