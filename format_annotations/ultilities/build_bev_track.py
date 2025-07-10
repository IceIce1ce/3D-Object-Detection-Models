import math
import os
import sys
import glob

import cv2
from tqdm import tqdm
from loguru import logger

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

def draw_bev_track_on_map(scene_name = "Warehouse_017", number_camera = 8):
	# Initialize hyperparameters
	number_image_per_camera = 9000  # 9000 images per camera, each camera has 9000 frames
	img_step                = 30  # Step for processing images, can be adjusted as needed
	# scene_name		      = "Warehouse_017"
	# number_camera           = 8  # Number of cameras
	folder_intput           = f"/media/sugarmini/Data1/2_dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/image_result/{scene_name}/cycle_2_track/"
	folder_intput_img       = os.path.join(folder_intput, "images/")
	ratio                   = math.ceil(number_camera / 2) + 1  # Resize ratio for the images on the map
	size_multi              = 4

	folder_output_bev = os.path.join(folder_intput, f"bev_track_{size_multi}_size/")
	os.makedirs(folder_output_bev, exist_ok=True)


	for frame_id in tqdm(range(0, number_image_per_camera, img_step), desc=f"Processing scene  {scene_name}"):

		img_world = cv2.imread(os.path.join(folder_intput, f"map_in_{scene_name}.jpg"))
		img_world = draw_panel_on_map(img_world, frame_id)
		img_world_h, img_world_w = img_world.shape[:2]


		img_world_h = img_world_h * size_multi
		img_world_w = img_world_w * size_multi
		img_world = cv2.resize(img_world, (img_world_w, img_world_h))

		for camera_index in range(number_camera):

			frame_name = camera_index * number_image_per_camera + frame_id

			# Construct the image path
			img_path_in = os.path.join(folder_intput_img, f"{frame_name:08d}.jpg")

			# Get the image by camera index and frame id
			img = cv2.imread(img_path_in)

			# Check if the image was successfully retrieved
			if img is None:
				continue

			# draw information on the map
			# img = draw_panel_on_map(img, camera_index)
			img_h, img_w  = img.shape[:2]
			img_h_resized = img_h * size_multi // ratio
			img_w_resized = img_w * size_multi // ratio

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
				logger.warning(f"Image {frame_name:08d} from camera {camera_index} exceeds map bounds. Skipping.")
				continue

			# Place the image on the map world
			img_world = cv2.rectangle(img_world, point, (point[0] + img_w_resized, point[1] + img_h_resized), (255, 255, 255), -1)
			img_world[point[1]:point[1] + img_h_resized, point[0]:point[0] + img_w_resized] = img

			# add panel camera
			img_world = draw_panel_on_map(
				img_world,
				camera_index,
				point_tl= (
					point[0] - 80 * size_multi,
					point[1] - 80 * size_multi + img_h_resized // 2
				),
				font_scale = 12,
				thickness = 8
			)

		# Save the map world image to the output folder
		img_path_ou = os.path.join(folder_output_bev, f"{frame_id:08d}.jpg")
		cv2.imwrite(img_path_ou, img_world)

	# DEBUG:
	# break

def draw_bev_raw_on_map(scene_name = "Warehouse_017", number_camera = 8):
	# Initialize hyperparameters
	number_image_per_camera = 9000  # 9000 images per camera, each camera has 9000 frames
	img_step                = 15  # Step for processing images, can be adjusted as needed
	# scene_name		      = "Warehouse_017"
	# number_camera           = 8  # Number of cameras
	folder_intput           = f"/media/sugarmini/Data1/2_dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/image_result/{scene_name}/cycle_2_track/"
	folder_intput_img       = f"/media/sugarmini/Data1/2_dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/images_extract_full/{scene_name}/"
	ratio                   = math.ceil(number_camera / 2) + 1  # Resize ratio for the images on the map
	size_multi              = 1
	list_camera             = glob.glob(os.path.join(folder_intput_img, "*"))
	number_camera           = len(list_camera)

	folder_output_bev = os.path.join(folder_intput, f"bev_track_{size_multi}_size/")
	os.makedirs(folder_output_bev, exist_ok=True)


	for frame_id in tqdm(range(number_image_per_camera), desc=f"Processing scene  {scene_name}"):

		if frame_id > 1:
			if frame_id % img_step != 0:
				continue


		img_world = cv2.imread(os.path.join(folder_intput, f"map_in_{scene_name}.jpg"))
		img_world = draw_panel_on_map(img_world, frame_id)
		img_world_h, img_world_w = img_world.shape[:2]


		img_world_h = img_world_h * size_multi
		img_world_w = img_world_w * size_multi
		img_world = cv2.resize(img_world, (img_world_w, img_world_h))

		for camera_index, camera_path in enumerate(list_camera):

			frame_name = frame_id

			# Construct the image path
			img_path_in = os.path.join(camera_path, f"{frame_name:07d}.jpg")
			if not os.path.exists(img_path_in):
				img_path_in = os.path.join(camera_path, f"{frame_name:08d}.jpg")



			# Get the image by camera index and frame id
			img = cv2.imread(img_path_in)

			# Check if the image was successfully retrieved
			if img is None:
				logger.warning(f"Can not find {img_path_in}")
				continue

			# draw information on the map
			# img = draw_panel_on_map(img, camera_index)
			img_h, img_w  = img.shape[:2]
			img_h_resized = img_h * size_multi // ratio
			img_w_resized = img_w * size_multi // ratio

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
				logger.warning(f"Image {frame_name:08d} from camera {camera_index} exceeds map bounds. Skipping.")
				continue

			# Place the image on the map world
			img_world = cv2.rectangle(img_world, point, (point[0] + img_w_resized, point[1] + img_h_resized), (255, 255, 255), -1)
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
				thickness = 2  * size_multi,
			)

		# Save the map world image to the output folder
		img_path_ou = os.path.join(folder_output_bev, f"{frame_id:08d}.jpg")
		cv2.imwrite(img_path_ou, img_world)

	# DEBUG:
	# break


def main():
	list_scene = ["Warehouse_017", "Warehouse_019"]
	for scene_name in tqdm(list_scene):
		# draw_bev_track_on_map(scene_name, number_camera=8)
		draw_bev_raw_on_map(scene_name, number_camera=8)
	list_scene = ["Warehouse_018", "Warehouse_018-enhance", "Warehouse_020"]
	# list_scene = ["Warehouse_018-enhance"]
	for scene_name in tqdm(list_scene):
		# draw_bev_track_on_map(scene_name, number_camera=9)
		draw_bev_raw_on_map(scene_name, number_camera=9)


if __name__ == "__main__":
	main()
