import os
import shutil
import sys
import glob
import json

from loguru import logger
from tqdm import tqdm

from mtmc.core.objects.units import MapWorld, Camera, Instance

import numpy as np
import cv2


def get_view_point_name(scene_name, camera_name):
	"""Get view point name from scene name and camera name.
	"""
	camera_name = os.path.basename(camera_name)
	scene_name  = os.path.basename(scene_name)
	view_name   = f"{scene_name}__{camera_name}"
	return view_name

def draw_polygons_on_image(image_path, json_path, output_path, fill_color=(0, 0, 255), outline_color=(0, 0, 255), line_width=2, alpha=0.4):
	"""
	Draws polygons from a JSON file onto an image using OpenCV.

	Args:
		image_path (str): Path to the input image.
		json_path (str): Path to the JSON file containing polygon data.
		output_path (str): Directory where the output image will be saved.
		fill_color (tuple): Color to fill the polygons (BGR format).
		outline_color (tuple): Color for the polygon outlines (BGR format).
		line_width (int): Width of the polygon outlines.
		alpha (float): Transparency level for the overlay.
	"""
	try:
		# 1. Load the image using OpenCV
		img = cv2.imread(image_path)
		img_mask = np.zeros_like(img)
		if img is None:
			raise FileNotFoundError(f"Image not found or could not be loaded: {image_path}")

		# Create a blank overlay image for drawing with transparency
		overlay = img.copy()
		output = img.copy()

		# 2. Load the JSON data
		with open(json_path, 'r') as f:
			data = json.load(f)

		shapes = data.get('shapes', [])

		# 3. Draw each polygon
		for shape in shapes:
			if shape.get('shape_type') == 'polygon' and shape.get('points'):
				# Convert points to NumPy array of integers for OpenCV
				# Points should be in (x, y) format
				polygon_points = np.array(shape['points'], dtype=np.int32)

				# Reshape for fillPoly (expects an array of contours)
				# For a single polygon, it's (1, N, 2)
				contours = [polygon_points]

				# Draw filled polygon on the overlay
				cv2.fillPoly(overlay, contours, fill_color)
				# Draw outline on the overlay
				cv2.polylines(overlay, contours, isClosed=True, color=outline_color, thickness=line_width)

				# Draw filled polygon on the mask
				cv2.fillPoly(img_mask, contours, (255, 255, 255))

		# 4. Blend the overlay with the original image for transparency
		# new_image = alpha * overlay + (1 - alpha) * original_image
		cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, output)

		img_output_path      = os.path.join(output_path, os.path.basename(image_path))
		img_output_path_draw = os.path.join(output_path, os.path.basename(image_path).replace('.jpg', '_draw.jpg'))

		# 5. Save the resulting image
		cv2.imwrite(img_output_path, img_mask)
		cv2.imwrite(img_output_path_draw, output)
		print(f"Image with polygons saved successfully to: {output_path}")

	except FileNotFoundError as e:
		print(f"Error: {e}")
	except json.JSONDecodeError:
		print(f"Error: Could not decode JSON from '{json_path}'. Check if it's valid JSON.")
	except Exception as e:
		print(f"An unexpected error occurred: {e}")


def rename_files():
	# init folder
	folder_input  = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/images_extract_full/"
	list_txt      = glob.glob(os.path.join(folder_input, "*/*/*.jpg"))

	for txt_path in tqdm(list_txt):
		txt_path_old     = txt_path
		txt_name_old     = os.path.basename(txt_path)
		txt_name_old_ext = os.path.splitext(txt_name_old)[0]

		txt_name_new = f'{int(txt_name_old.split("_")[-1].split(".")[0]):08d}.jpg'
		txt_path_new = os.path.join(os.path.dirname(txt_path), txt_name_new)

		# DEBUG:
		# print(f"{txt_name_old} -- {txt_name_new}")

		try:
			# Rename the folder
			os.rename(txt_path_old, txt_path_new)
			logger.info(f"Folder '{txt_path_old}' successfully renamed to '{txt_path_new}'.")
		except FileNotFoundError:
			logger.error(f"Error: Folder '{txt_path_old}' not found.")
		except FileExistsError:
			logger.error(f"Error: Folder '{txt_path_new}' already exists.")
		except Exception as e:
			logger.error(f"An unexpected error occurred: {e}")


def rename_full_video_txts():
	def custom_file_sort(file_path):
		basename       = os.path.basename(file_path)
		basename_noext = os.path.splitext(basename)[0]
		file_index     = basename_noext.split("_")[-1]
		return int(file_index)

	# init folder
	folder_input     = "/media/vsw/SSD_2/1_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/image_result/Warehouse_020/cycle_2_dect/"
	folder_input_lbl = os.path.join(folder_input, "labels")
	folder_input_img = os.path.join(folder_input, "images")
	list_txt      = sorted(glob.glob(os.path.join(folder_input_lbl, "*.txt")), key=custom_file_sort)

	for txt_path_old in tqdm(list_txt, desc=f"Processing Warehouse_020"):
		basename       = os.path.basename(txt_path_old)
		basename_noext = os.path.splitext(basename)[0]
		file_index     = basename_noext.split("_")[-1]

		txt_path_new   = os.path.join(folder_input_lbl, f"{int(file_index):08d}.txt")

		# rename
		# os.rename(txt_path_old, txt_path_new)

		# DEBUG:
		print(f"{basename} -- {int(file_index):08d}")

def rename_folders():
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


def create_viewpoints_image():
	# init folder
	folder_input      = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames"
	folder_input_img  = os.path.join(folder_input, "images")
	folder_output_roi = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/view_points/"

	camera_list_img = glob.glob(os.path.join(folder_input_img, "*/*"))

	for camera_path in tqdm(camera_list_img):
		if not os.path.isdir(camera_path):
			continue
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


def copy_yolo_dataset():
	folder_input_lbl   = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/labels_yolo_filtered/"
	folder_input_img   = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/images"

	folder_output_lbl   = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/yolo_format/cycle_1_Warehouse_017_only_box/train/labels"
	folder_output_img   = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/yolo_format/cycle_1_Warehouse_017_only_box/train/images"

	os.makedirs(folder_output_lbl, exist_ok=True)
	os.makedirs(folder_output_img, exist_ok=True)

	# scene_name_spec  = ["Warehouse_008", "Warehouse_012", "Warehouse_013", "Warehouse_014", "Warehouse_016"]
	scene_name_spec  = ["Warehouse_017_only_box"]
	camera_name_spec = None

	for scene_name in tqdm(scene_name_spec):

		camera_list = glob.glob(os.path.join(folder_input_lbl, scene_name, "*"))

		for camera_path in tqdm(camera_list, desc=f"Processing  {scene_name}"):
			camera_name = os.path.basename(camera_path)
			camera_name = Camera.adjust_camera_id(camera_name)
			camera_path_img = os.path.join(folder_input_img, scene_name, camera_name)

			# get list of image
			image_list = glob.glob(os.path.join(camera_path_img, "*.png"))
			for image_index, image_path_in in enumerate(tqdm(image_list, desc=f"Processing {camera_name}")):
				if image_index % 1 != 0:
					continue

				image_name       = os.path.basename(image_path_in)
				image_name_noext = os.path.splitext(image_name)[0]

				image_name_ou    = f"{get_view_point_name(scene_name, camera_name)}_{image_name_noext}.jpg"
				image_path_ou = os.path.join(folder_output_img, image_name_ou)

				# copy label
				label_name_ou    = f"{get_view_point_name(scene_name, camera_name)}_{image_name_noext}.txt"
				label_path_in    = os.path.join(folder_input_lbl, scene_name, camera_name, f"{image_name_noext}.txt")
				label_path_ou    = os.path.join(folder_output_lbl, label_name_ou)

				# check if label file exists
				# if not os.path.exists(label_path_in):
				# 	continue

				# copy
				shutil.copy(label_path_in, label_path_ou)
				cv2.imwrite(image_path_ou, cv2.imread(image_path_in), [int(cv2.IMWRITE_JPEG_QUALITY), 100])

		# DEBUG:
		# sys.exit()


def transfer_yolo_dataset():
	folder_input_lbl   = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/lookup_table/Warehouse_017_labels"
	folder_input_img   = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/lookup_table/Warehouse_017"

	folder_output_lbl   = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/yolo_format/cycle_1_Warehouse_017_map_world/val/labels"
	folder_output_img   = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/yolo_format/cycle_1_Warehouse_017_map_world/val/images"

	os.makedirs(folder_output_lbl, exist_ok=True)
	os.makedirs(folder_output_img, exist_ok=True)

	# get list of image
	image_list = glob.glob(os.path.join(folder_input_img, "*.jpg"))
	for image_index, image_path_in in enumerate(tqdm(image_list, desc=f"Processing tranfer yolo dataset")):
		if image_index % 9 != 0:
			continue

		image_name       = os.path.basename(image_path_in)
		image_name_noext = os.path.splitext(image_name)[0]

		image_path_ou    = os.path.join(folder_output_img, image_name)

		# copy label
		label_name_ou    = f"{image_name_noext}.txt"
		label_path_in    = os.path.join(folder_input_lbl, f"{image_name_noext}.txt")
		label_path_ou    = os.path.join(folder_output_lbl, label_name_ou)

		# check if label file exists
		# if not os.path.exists(label_path_in):
		# 	continue

		# copy
		shutil.copy(label_path_in, label_path_ou)
		cv2.imwrite(image_path_ou, cv2.imread(image_path_in), [int(cv2.IMWRITE_JPEG_QUALITY), 100])



if __name__ == "__main__":
	# rename_files()
	# rename_folders()
	# rename_full_video_txts()
	# create_viewpoints_image()
	# copy_yolo_dataset()
	transfer_yolo_dataset()
	# draw_polygons_on_image(
	# 	image_path= "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/view_points/Warehouse_017__Camera_0002.jpg",
	# 	json_path="/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/view_points/Warehouse_017__Camera_0002.json",
	# 	output_path="/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/view_points_json/view_points_draw/",
	# 	fill_color=(255, 255, 255),  # Blue fill color
	# 	outline_color=(0, 0, 255),  # Red outline color
	# )
	pass