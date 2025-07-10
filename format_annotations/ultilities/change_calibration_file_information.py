import glob
import json
import os
import shutil
from copy import deepcopy

from tqdm import tqdm

from mtmc.core.objects.units import Camera


def adjust_camera_id_calibration(json_file):

	# Load the file
	with open(json_file, "r") as f_read:
		data = json.load(f_read)

	# Replace id for each sensor of type "camera"
	for sensor in data.get("sensors", []):
		if sensor.get("type") == "camera":
			sensor["id"] = Camera.adjust_camera_id(sensor["id"])

	# Optionally, save to a new file
	with open(json_file, "w") as f_write:
		json.dump(data, f_write, indent=4)

	print(f"All camera IDs have been replaced and saved to {json_file}.")


def remove_camera_key_calibration(input_filename="calibration.json", output_filename="calibration_modified.json", camera_id_to_keep=["Camera_0000"]):

	# Load the file
	with open(input_filename, "r") as f_read:
		data = json.load(f_read)

	# Replace id for each sensor of type "camera"
	sensors = []
	for sensor in data.get("sensors", []):
		if sensor.get("type") == "camera":
			sensor["id"] = Camera.adjust_camera_id(sensor["id"])
			if sensor["id"] in camera_id_to_keep:
				sensors.append(sensor)
	data["sensors"] = sensors

	# Optionally, save to a new file
	with open(output_filename, "w") as f_write:
		json.dump(data, f_write, indent=4)

	print(f"All camera IDs have been replaced and saved to {input_filename}.")


def adjust_camera_id_groundtruth(json_file):
	# Load the file
	try:
		with open(json_file, 'r') as f:
			data = json.load(f)
	except FileNotFoundError:
		print(f"Error: Input file '{json_file}' not found.")
		return
	except json.JSONDecodeError:
		print(f"Error: Could not decode JSON from '{json_file}'.")
		return

	# Iterate through the frame_id keys (e.g., "0", "1", "2")
	for frame_id in data:
		# Each key contains a list of instances (objects)
		for instance in data[frame_id]:
			# Check if the key to remove exists in the nested dictionary
			bounding_box_visible_2d = {}
			for camera_id in instance["2d bounding box visible"]:
				camera_name = Camera.adjust_camera_id(camera_id)
				bounding_box_visible_2d[camera_name] = instance["2d bounding box visible"][camera_id]
			instance["2d bounding box visible"] = bounding_box_visible_2d

	# Write the modified data to a new JSON file
	with open(json_file, 'w') as f:
		json.dump(data, f)

	print(f"All camera IDs have been replaced and saved to {json_file}.")


def remove_camera_key_groundtruth(input_filename="ground_truth.json", output_filename="ground_truth_modified.json", camera_id_to_keep=["Camera_0000"]):
	"""
	Reads a JSON file, removes a specified key from a nested dictionary,
	and writes the modified data to a new JSON file.

	Args:
		input_filename (str): Path to the input JSON file.
		output_filename (str): Path to the output JSON file.
		camera_id_to_keep (list): List of camera IDs to keep in the "2d bounding box visible" dictionary.
	"""
	try:
		with open(input_filename, 'r') as f:
			data = json.load(f)
	except FileNotFoundError:
		print(f"Error: Input file '{input_filename}' not found.")
		return
	except json.JSONDecodeError:
		print(f"Error: Could not decode JSON from '{input_filename}'.")
		return

	# Iterate through the frame_id keys (e.g., "0", "1", "2")
	for frame_id in data:
		# Each key contains a list of instances (objects)
		for instance in data[frame_id]:
			# Check if the key to remove exists in the nested dictionary
			bounding_box_visible_2d = {}
			for camera_id in instance["2d bounding box visible"]:
				camera_name = Camera.adjust_camera_id(camera_id)
				if camera_id_to_keep is not None and camera_name in camera_id_to_keep:
					bounding_box_visible_2d[camera_name] = instance["2d bounding box visible"][camera_id]
			instance["2d bounding box visible"] = bounding_box_visible_2d

	# Write the modified data to a new JSON file
	with open(output_filename, 'w') as f:
		json.dump(data, f)

	# print(f"Successfully removed '{key_to_remove}' and saved the modified data to '{output_filename}'.")

def main():
	folder_data_version = "MTMC_Tracking_2025_20250614"
	folder_data_root    = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/{folder_data_version}/"
	folder_intput       = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/{folder_data_version}/ExtractFrames/calibration_camera/"
	list_scene          = [
		"Warehouse_000",
		"Warehouse_001", "Warehouse_002", "Warehouse_003", "Warehouse_004", "Warehouse_005", "Warehouse_006",
		"Warehouse_007", "Warehouse_008", "Warehouse_009", "Warehouse_010", "Warehouse_011", "Warehouse_012",
		"Warehouse_013", "Warehouse_014", "Warehouse_015", "Warehouse_016", "Warehouse_017", "Warehouse_018",
		"Warehouse_019", "Warehouse_020", "Lab_000"]
	# list_scene = ["Warehouse_000", "Warehouse_002", "Warehouse_012", "Warehouse_013", "Warehouse_014", "Warehouse_016"]
	# list_scene = ["Warehouse_012", "Warehouse_013", "Warehouse_014", "Warehouse_016"]
	# list_scene = ["Warehouse_017", "Warehouse_018", "Warehouse_019", "Warehouse_020"]
	# list_scene = ["Warehouse_000"]
	# list_scene = ["Warehouse_002"]
	# list_scene = ["Lab_000"]
	list_scene = ["Warehouse_008"]

	# NOTE: copy file
	# for scene_name in tqdm(list_scene):
	# 	for type_json in ["calibration", "ground_truth"]:
	# 		if len(glob.glob(os.path.join(folder_data_root, f"*/{scene_name}/{type_json}.json"))) > 0:
	# 			json_in = glob.glob(os.path.join(folder_data_root, f"*/{scene_name}/{type_json}.json"))[0]
	# 			json_ou = os.path.join(folder_intput, f"{scene_name}_{type_json}.json")
	# 			shutil.copy(json_in, json_ou)

	# NOTE: Adjust camera IDs in each JSON file
	for scene_name in tqdm(list_scene):
		adjust_camera_id_calibration(os.path.join(folder_intput, f"{scene_name}_calibration.json"))
		adjust_camera_id_groundtruth(os.path.join(folder_intput, f"{scene_name}_ground_truth.json"))

	# NOTE: remove camera_id
	# list_scene = ["Warehouse_012", "Warehouse_013", "Warehouse_014", "Warehouse_016"]
	# camera_id_to_keep= ["Camera_0000", "Camera_0001", "Camera_0002"]
	# list_scene = ["Warehouse_000"]
	# camera_id_to_keep= ["Camera_0002", "Camera_0003"]
	# list_scene = ["Warehouse_002"]
	# camera_id_to_keep= ["Camera_0045"]
	# for scene_name in tqdm(list_scene):
	# 	json_in = os.path.join(folder_intput, f"{scene_name}_calibration.json")
	# 	json_ou = os.path.join(folder_intput, f"modified/{scene_name}_calibration.json")
	# 	remove_camera_key_calibration(input_filename=json_in, output_filename=json_ou, camera_id_to_keep=camera_id_to_keep)
	# 	json_in = os.path.join(folder_intput, f"{scene_name}_ground_truth.json")
	# 	json_ou = os.path.join(folder_intput, f"modified/{scene_name}_ground_truth.json")
	# 	remove_camera_key_groundtruth(input_filename=json_in, output_filename=json_ou, camera_id_to_keep=camera_id_to_keep)

if __name__ == "__main__":
	main()
