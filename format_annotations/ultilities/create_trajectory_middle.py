import json
import os
import glob
from copy import deepcopy

import numpy as np
from tqdm import tqdm

from ultilities.create_trajectory_middle_list import (
	object_type_id,
	object_type_name,
	object_warehouse_017,
	object_warehouse_018,
	object_warehouse_019,
	object_warehouse_020,
)

class json_serialize(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.integer):
			return int(obj)
		if isinstance(obj, np.floating):
			return float(obj)
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return json.JSONEncoder.default(self, obj)


def adjust_image_name(image_name):
	return f"{int(image_name):08d}"


def creat_between(period, shape_start, shape_end, frame_id):
	# DEBUG:
	# print("*********")
	# print(period)
	# print(shape_start)
	# print(shape_end)
	# print("*********")


	period_type  = period["type"]
	period_start = period["start"]
	period_end   = period["end"]

	shape_middle  = deepcopy(shape_start)
	points_start  = np.array(shape_start["points"])
	points_end    = np.array(shape_end["points"])
	points_middle = (points_start + (float(frame_id - period_start) / float(period_end - period_start)) * (points_end - points_start))

	if period_type in ["clock", "counter_clock"]:
		# Average direction if available
		direction_start = shape_start.get("direction")
		direction_end   = shape_end.get("direction")
		if direction_start is not None and direction_end is not None:
			if period_type == "clock":
				direction_middle = (points_start + (float(frame_id - period_start) / float(period_end - period_start)) * (direction_end - direction_start))
			else:  # counter_clock
				direction_middle = (points_start + (float(frame_id - period_start) / float(period_end - period_start)) * (direction_start - direction_end))
			shape_middle["direction"] = direction_middle
		else:
			shape_middle["direction"] = direction_start or direction_end # If only one exists, use that one

	shape_middle["description"] = ""
	shape_middle["points"]      = points_middle
	return shape_middle


def create_trajectory_by_period(object_instance, list_img):
	object_id      = object_instance["id"]
	object_type    = object_instance["type"]

	# surf all periods
	for period in object_instance["periods"]:
		period_type  = period["type"]
		period_start = period["start"]
		period_end   = period["end"]

		# get the start lbl information
		lbl_path_start = os.path.join(
			os.path.dirname(list_img[0]),
			f"{adjust_image_name(period_start)}.json"
		)
		with open(lbl_path_start, "r") as f:
			json_data_start = json.load(f)
			for shape in json_data_start["shapes"]:
				if shape["label"] == object_type and shape["group_id"] == object_id:
					shape_start = shape

		# get the end lbl information
		lbl_path_end   = os.path.join(
			os.path.dirname(list_img[0]),
			f"{adjust_image_name(period_end)}.json"
		)
		with open(lbl_path_end, "r") as f:
			json_data_end = json.load(f)
			for shape in json_data_end["shapes"]:
				if shape["label"] == object_type and shape["group_id"] == object_id:
					shape_end = shape

		# create middle
		for img_path in list_img:
			img_name = os.path.basename(img_path)
			frame_id = int(os.path.splitext(img_name)[0])

			# get middle lbl information
			if period_start < frame_id < period_end:

				# get middle lbl path
				lbl_path_middle = os.path.join(
					os.path.dirname(img_path),
					f"{adjust_image_name(frame_id)}.json"
				)

				# check existence of lbl file
				if os.path.exists(lbl_path_middle):
					with open(lbl_path_middle, "r") as f:
						json_data_middle = json.load(f)
				else:
					# if lbl file does not exist, create it
					json_data_middle = deepcopy(json_data_start)
					json_data_middle["imagePath"] = img_name
					json_data_middle["shapes"]    = []

				# create the trajectory for this frame
				shape_middle = creat_between(period, shape_start, shape_end, frame_id)

				# add to middle json
				is_exist = False
				for shape in json_data_middle["shapes"]:
					if shape["label"] == object_type and shape["group_id"] == object_id:
						is_exist = True
						shape["points"] = shape_middle["points"]
						break
				if not is_exist:
					json_data_middle["shapes"].append(shape_middle)

				# write to json file
				with open(lbl_path_middle, 'w') as file:
					json.dump(json_data_middle, file, indent=4, cls=json_serialize)

			# stop if frame_id is out of range
			if frame_id > period_end:
				break

		# DEBUG: run on specific frame_id
		# if frame_id > period_start:
		# 	break

def create_trajectory_by_nodes(object_instance, list_img, img_name_stop=9000):
	object_id      = object_instance["id"]
	object_type    = object_instance["type"]
	period         = {"type": "line" , "start": None, "end": None}
	shape_start    = None
	shape_end      = None
	is_middle      = False
	img_list_middle= []

	for img_index, img_path_in in enumerate(tqdm(list_img)):

		# Pass 2 begin images
		if img_index < 1:
			continue

		img_basename       = os.path.basename(img_path_in)
		img_basename_noext = os.path.splitext(img_basename)[0]


		if int(img_basename_noext) >= img_name_stop:
			return

		lbl_path = os.path.join(
			os.path.dirname(img_path_in),
			f"{img_basename_noext}.json"
		)

		if is_middle:
			img_list_middle.append(img_path_in)

		if os.path.exists(lbl_path):
			# load current json

			with open(lbl_path, "r") as f:
				json_data = json.load(f)
				for shape in json_data["shapes"]:
					if shape["label"] == object_type and shape["group_id"] == object_id:

						if "node" in shape["description"].lower():
							if shape_start is None:
								shape_start     = shape
								json_data_start = deepcopy(json_data)
								period["start"] = int(img_basename_noext)
								is_middle       = True
							else:
								shape_end       = shape
								period["end"]   = int(img_basename_noext)
								is_middle       = False

							# remove last imgage, which belongs shape_start, or shape end
							if len(img_list_middle) > 0:
								img_list_middle.pop()

			# DEBUG:
			# print("*********")
			# print(is_middle)
			# print(img_basename_noext)
			# print(period)
			# print(shape_start)
			# print(shape_end)
			# print(img_list_middle)
			# print("*********")


			# create middle
			if shape_end is not None:
				# create middle
				for img_path in img_list_middle:
					img_name = os.path.basename(img_path)
					frame_id = int(os.path.splitext(img_name)[0])

					# get middle lbl path
					lbl_path_middle = os.path.join(
						os.path.dirname(img_path),
						f"{adjust_image_name(frame_id)}.json"
					)

					# check existence of lbl file
					if os.path.exists(lbl_path_middle):
						with open(lbl_path_middle, "r") as f:
							json_data_middle = json.load(f)
					else:
						# if lbl file does not exist, create it
						json_data_middle = deepcopy(json_data_start)
						json_data_middle["imagePath"] = img_name
						json_data_middle["shapes"]    = []

					# create the trajectory for this frame
					shape_middle = creat_between(period, shape_start, shape_end, frame_id)

					# add to middle json
					is_exist = False
					for shape in json_data_middle["shapes"]:
						if shape["label"] == object_type and shape["group_id"] == object_id:
							is_exist = True
							shape["points"] = shape_middle["points"]
							break
					if not is_exist:
						json_data_middle["shapes"].append(shape_middle)

					# DEBUG:
					# print("")
					# print(json_data_middle)
					# print(lbl_path_middle)

					# write to json file
					with open(lbl_path_middle, 'w') as file:
						json.dump(json_data_middle, file, indent=4, cls=json_serialize)


				# reset middle
				shape_start     = deepcopy(shape_end)
				shape_end       = None
				img_list_middle = []
				period["start"] = period["end"]
				period["end"]   = None
				is_middle       = True

def main():
	# Init hyperparameters
	object_scene  = object_warehouse_017
	scene_name    = "Warehouse_017"
	folder_intput = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/image_bev_track/bev_track_1_size_full"
	folder_lbl    = os.path.join(folder_intput, scene_name)

	list_img      =  sorted(glob.glob(os.path.join(folder_lbl, "*.jpg")))

	pbar = tqdm(total=len(object_scene["objects"]))
	for object_instance in object_scene["objects"]:
		# NOTE: specific object id
		# if object_instance['id']  != 1:
		# 	continue

		pbar.set_description(f"Processing {object_instance['id']} -- {object_instance['type']}")
		# create_trajectory_by_period(object_instance, list_img)
		create_trajectory_by_nodes(object_instance, list_img, img_name_stop=9000)

		pbar.update(1)
	pbar.close()


if __name__ == "__main__":
	main()