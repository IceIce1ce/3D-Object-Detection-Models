import os
import sys
import glob
import json
import math

import multiprocessing
import threading

from tqdm import tqdm
from loguru import logger
import cv2

import numpy as np

from mtmc.core.objects.units import MapWorld

################################################################################
# REGION: Functions
################################################################################

def draw_map_world_with_cameras_into_the_map_image(
		map_cfg,
		map_world,
		folder_input
	):
	"""Draw map world with cameras into the map image.
	"""
	map_image = cv2.imread(map_world.map_image_path)
	map_image = map_world.draw_cameras_on_map(map_image, length=30, color=(0, 255, 255))

	# output result
	output_path  = os.path.join(folder_input, f"map_in_{map_cfg['name']}.jpg")
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	cv2.imwrite(output_path, map_image)

	# DEBUG: collect all map images into one folder
	# output_path  = os.path.join("/media/sugarubuntu/DataSKKU3/ResilioSync/Work/AIC25/dataset/all_map/", f"map_in_{map_cfg['name']}.jpg")
	# os.makedirs(os.path.dirname(output_path), exist_ok=True)
	# cv2.imwrite(output_path, map_image)
	pass


def draw_map_world_with_instance_into_the_map_image(
		map_cfg,
		map_world,
		color_chart,
		folder_input,
		folder_output_map_instance
	):
	"""Draw map world with instances into the map image.
	"""
	# get frame id range
	frame_id_min    = sys.maxsize
	frame_id_max    = 0
	for instance in map_world.instances:
		frame_id_min = min(frame_id_min, int(map_world.instances[instance].frame_id_min))
		frame_id_max = max(frame_id_max, int(map_world.instances[instance].frame_id_max))

	# DEBUG: run on specific range of frame_id
	# frame_id_min = 0
	# frame_id_max = 100

	# output from creating images by drawing map world with cameras into the map image
	os.makedirs(folder_output_map_instance, exist_ok=True)
	for frame_id in tqdm(range(frame_id_min, frame_id_max + 1), desc=f"Drawing map {map_cfg['name']} world with instances"):
		map_image = cv2.imread(map_world.map_image_path)
		map_image = map_world.draw_instances_on_map(map_image, frame_id, length=20, color=color_chart)

		map_image = map_world.draw_information_on_map(map_image, frame_id, color=color_chart)

		output_path  = os.path.join(folder_output_map_instance, f"{map_cfg['name']}_{int(frame_id):05d}.jpg")
		cv2.imwrite(output_path, map_image)

	# Create video from this images by using ffmpeg
	os.system(f"ffmpeg -y -framerate 30 -pattern_type glob "
	          f"-i '{folder_output_map_instance}/*.jpg' "
	          f" {folder_input}/{map_cfg['name']}.mp4")

	# remove folder
	os.system(f"rm -rf {folder_output_map_instance}/")
	pass


def draw_3d_bounding_boxes_on_cameras_image_with_map_world(
		map_world,
		color_chart,
		folder_input,
		folder_input_camera_videos
	):
	"""Draw 3d bounding boxes on cameras image with map world.
	"""
	folder_output_videos_cameras = os.path.join(folder_input, "videos_cameras")
	os.makedirs(folder_output_videos_cameras, exist_ok=True)

	for camera_idx in map_world.cameras:
		# init folder
		folder_output_videos_one_camera = os.path.join(folder_output_videos_cameras, f"{camera_idx}")

		# get camera object
		camera            = map_world.cameras[camera_idx]

		# get camera video path
		camera_video_path = os.path.join(folder_input_camera_videos, f"{camera.camera_file_name}.mp4")
		if not os.path.exists(camera_video_path):
			camera_name_file     = camera.camera_file_name
			camera_name_file     = camera_name_file.replace('_','').replace('Camera','')
			if camera_name_file == "":
				camera_name_file = "Camera_0000"
			else:
				camera_name_file = str(f"Camera_{int(camera_name_file):04d}")
			camera_video_path = os.path.join(folder_input_camera_videos, f"{camera_name_file}.mp4")

		# DEBUG: run on specific camera
		if camera.id not in ["Camera_0002"]:
			continue

		# create output folder for this camera
		os.makedirs(folder_output_videos_one_camera, exist_ok=True)

		# get video information
		cap         = cv2.VideoCapture(camera_video_path)
		frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) )
		if not cap.isOpened():
			logger.error(f"Cannot open video: {camera_video_path}")
			continue

		# DEBUG: run on specific frame
		frame_id_min      = 1
		frame_id_max      = 200
		# frame_id_specific = [483]
		# frame_id_specific = [231, 293,  436]
		frame_id_specific = None

		# drawing
		pbar      = tqdm(total=frame_count, desc=f"Draw 3d bboxes on {camera.id}")
		frame_idx = 0
		while True:
			ret, cam_img = cap.read()
			if not ret:
				break

			# DEBUG: running on specific frame
			if frame_id_specific is not None:
				if frame_idx not in frame_id_specific:
					frame_idx += 1
					pbar.update(1)
					continue
			elif frame_idx < frame_id_min:
				frame_idx += 1
				pbar.update(1)
				continue
			elif frame_idx > frame_id_max:
				break

			# get map image
			map_img   = cv2.imread(map_world.map_image_path)

			# drawing 3d bounding boxes on camera image
			cam_img, map_img = map_world.draw_instances_on_camera_and_map(cam_img, map_img, camera_idx, frame_idx, length=20, color=color_chart)
			map_img          = map_world.draw_information_on_map(map_img, frame_idx, color=color_chart)

			# combine image
			combined_image = np.hstack((cam_img, map_img))

			# output result
			output_path = os.path.join(folder_output_videos_one_camera, f"{int(frame_idx):05d}.jpg")
			cv2.imwrite(output_path, combined_image)

			frame_idx += 1
			pbar.update(1)
		pbar.close()
		cap.release()

		# Create video from this images by using ffmpeg
		# os.system(f"ffmpeg -y -framerate 30 -pattern_type glob "
		#           f"-i '{folder_output_videos_one_camera}/*.jpg' "
		#           f" {folder_output_videos_cameras}/{camera.id}.mp4")

		# remove folder
		# os.system(f"rm -rf {folder_output_videos_one_camera}/")

		# DEBUG: out of this function
		# sys.exit()
		pass


################################################################################
# REGION: Main
################################################################################

def draw_groundtruth_calibration(map_list):

	for map_path in tqdm(map_list):
		# init
		folder_input               = os.path.dirname(map_path) # parent folder of map.png
		folder_input_camera_videos = os.path.join(folder_input, "videos")
		map_name                   = os.path.basename(folder_input)
		calibration_path           = os.path.join(folder_input, "calibration.json")
		groundtruth_path           = os.path.join(folder_input, "ground_truth.json")
		folder_output_map_instance = os.path.join(folder_input, "video_map_instance")

		# DEBUG: run on specific map name
		if map_name not in ["Warehouse_014"]:
			continue

		logger.info(f"Processing map: {map_name} with path: {map_path}")

		color_chart = {
			"Person"      : (162, 162, 245), # red
			"Forklift"    : (0  , 255, 0)  , # green
			"NovaCarter"  : (235, 229, 52) , # blue
			"Transporter" : (0  , 255, 255), # yellow
			"FourierGR1T2": (162, 245, 214), # purple
			"AgilityDigit": (162, 241, 245), # pink
		}

		# load map world
		map_cfg = {
			"name"              : map_name,
			"id"                : map_name,
			"type"              : "cartesian",
			"size"              : [1920, 1080],
			"map_image"         : map_path,
			"calibration_path"  : calibration_path,
			"groundtruth_path"  : groundtruth_path,
			"folder_videos_path": folder_input_camera_videos,
		}
		map_world = MapWorld(map_cfg)
		# sort frames_id
		for instance in map_world.instances:
			map_world.instances[instance].sort_frames()

		# NOTE: draw map world with cameras into the map image
		# draw_map_world_with_cameras_into_the_map_image(map_cfg, map_world, folder_input)

		# NOTE: draw map world with instance into the map image
		# draw_map_world_with_instance_into_the_map_image(map_cfg, map_world, color_chart, folder_input, folder_output_map_instance)

		# NOTE: draw 3d bounding boxes on cameras image with map world
		if os.path.exists(groundtruth_path):
			draw_3d_bounding_boxes_on_cameras_image_with_map_world(map_world, color_chart, folder_input, folder_input_camera_videos)


def run_multi_threads():

	# Init folder
	folder_input = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/"
	map_list     = glob.glob(os.path.join(folder_input,"*/*/map.png"))

	num_threads           = 4
	num_video_per_processes = math.ceil(len(map_list) / num_threads)

	# Split the list into chunks for each process
	res = []
	for n, i in enumerate(map_list):
		if n % num_video_per_processes == 0 and n + num_video_per_processes < len(map_list):
			res.append(map_list[n:n + num_video_per_processes])
		elif n + num_video_per_processes >= len(map_list):
			res.append(map_list[n:])
			break

	logger.info(f"Number of processes: {num_threads}")
	logger.info(f"Number of maps: {len(map_list)}")

	# creating processes
	threads = []
	for i in range(num_threads):
		p = threading.Thread(target=draw_groundtruth_calibration, args=(res[i],))
		threads.append(p)

	# starting process
	for i in range(num_threads):
		threads[i].start()

	# wait until process is finished
	for i in range(num_threads):
		threads[i].join()


def run_single_thread():
	# Init folder
	folder_input = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/"
	# folder_input = "/media/vsw/SSD_2/1_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/"
	# folder_input = "/media/sugarmini/Data1/2_dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/"

	# get list map
	map_list = glob.glob(os.path.join(folder_input,"*/*/map.png"))
	# map_list = glob.glob(os.path.join(folder_input,"val/Lab_000/map.png"))
	# map_list = glob.glob(os.path.join(folder_input,"train/Warehouse_008/map.png"))

	draw_groundtruth_calibration(map_list)


def main():
	# run only one process per time
	run_single_process()

	# run multi processes per time
	# run_multi_process()
	pass


if __name__ == "__main__":
	main()
