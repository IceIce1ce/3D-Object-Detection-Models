import glob
import json
import os.path

from tqdm import tqdm


def upsize(scene_name, folder_input, folder_output, ratio = 4):
	# get list of JSON files in the input folder
	list_json = glob.glob(os.path.join(folder_input, "*.json"))

	for json_path_in in tqdm(list_json, desc=f"Processing JSON files in {scene_name}"):
		# Load the JSON file
		with open(json_path_in, 'r') as f:
			data = json.load(f)

		# ratio = 4

		# Scale all points in all shapes
		for shape in data.get("shapes", []):
			shape["points"] = [
				[x * ratio, y * ratio] for x, y in shape.get("points", [])
			]

		data["imageWidth"]  = int(data.get("imageWidth", 0) / ratio)
		data["imageHeight"]  = int(data.get("imageHeight", 0) / ratio)

		# Save results to a new JSON file
		json_path_ou = os.path.join(folder_output, os.path.basename(json_path_in))
		with open(json_path_ou, 'w') as f:
			json.dump(data, f, indent=4)


def downsize(scene_name, folder_intput, folder_output, ratio = 4):
	# get list of JSON files in the input folder
	list_json = glob.glob(os.path.join(folder_intput, "*.json"))

	for json_path_in in tqdm(list_json, desc=f"Processing JSON files in {scene_name}"):
		# Load the JSON file
		with open(json_path_in, 'r') as f:
			data = json.load(f)

		# ratio = 4

		# Scale all points in all shapes
		for shape in data.get("shapes", []):
			shape["points"] = [
				[x / ratio, y / ratio] for x, y in shape.get("points", [])
			]

		data["imageWidth"]  = int(data.get("imageWidth", 0) / ratio)
		data["imageHeight"]  = int(data.get("imageHeight", 0) / ratio)

		# Save results to a new JSON file
		json_path_ou = os.path.join(folder_output, os.path.basename(json_path_in))
		with open(json_path_ou, 'w') as f:
			json.dump(data, f, indent=4)

if __name__ == "__main__":
	list_scene = ["Warehouse_017", "Warehouse_018", "Warehouse_019", "Warehouse_020"]
	for scene_name in list_scene:
		folder_intput = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/image_bev_track/bev_track_1_size/{scene_name}/"
		folder_output = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/image_bev_track/bev_track_1_size_full/{scene_name}/"
		upsize(scene_name, folder_intput, folder_output, ratio=1)