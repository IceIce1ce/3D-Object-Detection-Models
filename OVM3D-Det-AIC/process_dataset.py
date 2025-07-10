import json

def process_kitti_json():
    modes = ['test', 'train', 'val']
    for mode in modes:
        input_file = f'datasets/Omni3D/KITTI_{mode}.json'
        output_file = f'datasets/Omni3D/processed_KITTI_{mode}.json'
        with open(input_file, 'r') as f:
            data = json.load(f)
        processed_data = {"info": {"id": data["info"]["id"], "source": data["info"]["source"], "name": data["info"]["name"], "split": data["info"]["split"]},
                          "images": [{"width": img["width"], "height": img["height"],"file_path": img["file_path"], "K": img["K"], "id": img["id"], "dataset_id": img["dataset_id"]} for img in data["images"]],
                          "categories": data["categories"], "annotations": data["annotations"]}
        with open(output_file, 'w') as f:
            json.dump(processed_data, f, indent=4)

if __name__ == "__main__":
    process_kitti_json()