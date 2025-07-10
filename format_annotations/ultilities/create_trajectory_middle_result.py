import sys
import os
import json
import glob

from tqdm import tqdm

import cv2

from ultilities.create_trajectory_middle_list import (
	object_type_id,
	object_type_name,
	color_chart
)

def draw_information_on_map(map_img, frame_id, color):
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
	font_scale        = 3
	thickness         = 2
	frame_id_label_tl = (5, 5)

	# Get the text size
	text_size, _ = cv2.getTextSize(frame_id, font, font_scale, thickness)
	text_width, text_height = text_size

	# Calculate the background rectangle coordinates
	x, y              = frame_id_label_tl
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

def put_text_with_border(image, text, org, font, font_scale, text_color, border_color, thickness):
	"""
	Adds text with a border to an image.

	Args:
		image: The image to draw on.
		text: The text string to write.
		org: The bottom-left corner coordinates of the text string.
		font: Font type, e.g., cv2.FONT_HERSHEY_SIMPLEX.
		font_scale: Font scale factor.
		text_color: Text color in BGR format (e.g., (255, 255, 255) for white).
		border_color: Border color in BGR format (e.g., (0, 0, 0) for black).
		thickness: Thickness of the text and border.
	"""
	# Draw border by drawing the text first with the border color and a larger thickness
	cv2.putText(image, text, org, font, font_scale, border_color, thickness + 2, cv2.LINE_AA)
	# Draw the actual text on top of the border
	cv2.putText(image, text, org, font, font_scale, text_color, thickness, cv2.LINE_AA)
	return image

def create_trajectory_middle_video(scene_name, folder_input, folder_output):
	"""Create trajectory middle video for a specific scene.
	"""
	# get list of JSON files in the input folder
	list_json = glob.glob(os.path.join(folder_input, "*.json"))

	for json_path_in in tqdm(list_json, desc=f"Processing JSON files in {scene_name}"):
		# Load the JSON file
		with open(json_path_in, 'r') as f:
			json_data = json.load(f)

		# load image
		img_path_in = os.path.join(folder_input, json_data["imagePath"])
		img_in      = cv2.imread(img_path_in)
		img_basename = os.path.basename(img_path_in)
		img_basename_noext = os.path.splitext(img_basename)[0]

		for shape in json_data["shapes"]:
			if shape["shape_type"] in ["circle"]:
				object_id         = shape["group_id"]
				object_type       = shape["label"]
				object_type_index = object_type_id[object_type]
				point_center      = shape["points"][0]

				if object_type in color_chart:
					color = color_chart[object_type]
				else:
					color = (255, 255, 255)

				# Draw the instance and label
				# cv2.arrowedLine(img_in, pt1, pt2, color, thickness=3, tipLength=0.5)
				cv2.circle(img_in, (int(point_center[0]), int(point_center[1])), 10, color, -1)
				# cv2.putText(map_img, f"{self.object_id}", (x_px + 5, y_px - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
				img_in = put_text_with_border(img_in, f"{object_id}", (int(point_center[0]) + 5, int(point_center[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color=color, border_color=(0,0,0), thickness=1)

		# Save the modified image
		img_in = draw_information_on_map(img_in, int(img_basename_noext), color_chart)
		cv2.imwrite(os.path.join(folder_output, json_data["imagePath"]), img_in)

		# DEBUG:
		# break

def main_create_trajectory_middle_video():
	# Example usage
	folder_root  = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/image_bev_track/bev_track_1_size_full/"
	list_scene = ["Warehouse_017", "Warehouse_018", "Warehouse_019", "Warehouse_020"]

	for scene_name in tqdm(list_scene):
		# DEBUG:
		if scene_name not in ["Warehouse_017"]:
			continue

		folder_input  = os.path.join(folder_root, scene_name)
		folder_output = os.path.join(folder_root, f"{scene_name}_draw")
		os.makedirs(folder_output, exist_ok=True)

		create_trajectory_middle_video(scene_name, folder_input, folder_output)


def create_trajectory_middle_result_preprocess(scene_name, folder_input, folder_output, scale_factor, translation_to_global_coordinates):
	"""Create trajectory middle video for a specific scene.
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
	"""
	# get list of JSON files in the input folder
	list_json = glob.glob(os.path.join(folder_input, "*.json"))
	# translation_to_global_coordinates['x'] = translation_to_global_coordinates['x'] * scale_factor
	# translation_to_global_coordinates['y'] = translation_to_global_coordinates['y'] * scale_factor

	print(f"Processing {scene_name} with scale factor {scale_factor} and translation {translation_to_global_coordinates}")

	json_data_re = {}
	for json_path_in in tqdm(list_json, desc=f"Processing preprocess in {scene_name}"):
		# Load the JSON file
		with open(json_path_in, 'r') as f:
			json_data = json.load(f)

		# load image
		img_path_in  = os.path.join(folder_input, json_data["imagePath"])
		img_in       = cv2.imread(img_path_in)
		img_basename = os.path.basename(img_path_in)
		img_basename_noext = os.path.splitext(img_basename)[0]
		img_width    = json_data["imageWidth"]
		img_height   = json_data["imageHeight"]

		for shape in json_data["shapes"]:
			if shape["shape_type"] in ["circle"]:
				object_id         = shape["group_id"]
				object_type       = shape["label"]
				object_type_index = object_type_id[object_type]
				point_center      = shape["points"][0]

				# check exist object id
				if object_id not in json_data_re:
					json_data_re[object_id] = {
						"object_type_id": object_type_index,
						"frames": {}
					}

				# add new frame data
				json_data_re[object_id]["frames"][str(int(img_basename_noext))] =\
					{
						"x": (point_center[0] / scale_factor - translation_to_global_coordinates['x']),  # B: because (0,0) is top_left
						"y": ((img_height - point_center[1]) / scale_factor - translation_to_global_coordinates['y']) ,  # B: because (0,0) is bottom_left
					}

	# write result
	json_path_ou = os.path.join(folder_output, f"{scene_name}_preprocess.json")
	with open(json_path_ou, 'w') as f:
		json.dump(json_data_re, f)


def main_create_trajectory_middle_result_preprocess():
	# Example usage
	folder_root   = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/image_bev_track/bev_track_1_size_full/"
	folder_output = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/lookup_table/"
	list_scene    = ["Warehouse_017", "Warehouse_018", "Warehouse_019", "Warehouse_020"]
	# list_scene    = ["Warehouse_019"]

	for scene_name in tqdm(list_scene):
		# DEBUG:
		# if scene_name not in ["Warehouse_017", "Warehouse_019"]:
		# 	continue

		json_path_calibration = os.path.join(folder_output, f"{scene_name}_calibration.json")
		with open(json_path_calibration, 'r') as f:
			json_data_calibration = json.load(f)
		scale_factor                      =  json_data_calibration["sensors"][0]["scaleFactor"]
		translation_to_global_coordinates = json_data_calibration["sensors"][0]["translationToGlobalCoordinates"]

		folder_input  = os.path.join(folder_root, scene_name)
		create_trajectory_middle_result_preprocess(scene_name, folder_input, folder_output, scale_factor, translation_to_global_coordinates)


def main():
	# main_create_trajectory_middle_video()
	main_create_trajectory_middle_result_preprocess()
	pass

if __name__ == "__main__":
	main()