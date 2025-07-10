import json
import os
import cv2

image_dir = "datasets/KITTI_object/training/image_2"
json_file = "omni_instances_results.json"
vis_2d_dir = "vis_2d"
os.makedirs(vis_2d_dir, exist_ok=True)
with open(json_file, 'r') as f:
    annotations = json.load(f)
annotations_by_image = {}
for ann in annotations:
    img_id = ann['image_id']
    if img_id not in annotations_by_image:
        annotations_by_image[img_id] = []
    annotations_by_image[img_id].append(ann)
category_colors = {1: (0, 255, 0), 5: (0, 0, 255)}

def visualize_2d(image_id, anns):
    img_path = os.path.join(image_dir, f"{image_id:06d}.png")
    if not os.path.exists(img_path):
        print(f"Image {img_path} not found, skipping.")
        return
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image {img_path}, skipping.")
        return
    for ann in anns:
        bbox = ann['bbox']
        category_id = ann['category_id']
        x_min, y_min, w, h = [int(v) for v in bbox]
        color = category_colors.get(category_id, (255, 255, 255))
        cv2.rectangle(img, (x_min, y_min), (x_min + w, y_min + h), color, 2)
        label = f"Cat:{category_id} Score:{ann['score']:.2f}"
        cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    output_path = os.path.join(vis_2d_dir, f"{image_id:06d}.png")
    cv2.imwrite(output_path, img)
    print(f"Saved 2D visualization to {output_path}")

for image_id in annotations_by_image:
    anns = annotations_by_image[image_id]
    visualize_2d(image_id, anns)
print("2D visualization completed.")