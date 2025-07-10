import json
import os
import numpy as np
import cv2
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

vis_2d_dir = "vis_bbox/vis_2d"
vis_3d_dir = "vis_bbox/vis_3d"
os.makedirs(vis_2d_dir, exist_ok=True)
os.makedirs(vis_3d_dir, exist_ok=True)
category_colors = {"person": (0, 0, 255), "forklift": (255, 0, 0), "novacarter": (0, 255, 0), "transporter": (0, 255, 255), "fouriergr1t2": (255, 0, 255), "agilitydigit": (0, 128, 255)}
ignore_categories = {"dontcare", "ignore", "void"}

def project_3d_to_2d(points_3d, K, w, h):
    K = np.array(K)
    points_3d = np.array(points_3d)
    proj = (K @ points_3d.T).T
    valid = proj[:, 2] > 0
    u = np.clip(proj[valid, 0] / proj[valid, 2], 0, w - 1)
    v = np.clip(proj[valid, 1] / proj[valid, 2], 0, h - 1)
    out = np.full((len(points_3d), 2), np.nan)
    out[valid] = np.stack((u, v), axis=1)
    return out

def draw_3d_bbox(img, pts2d, color):
    if np.any(np.isnan(pts2d)):
        return
    pts2d = pts2d.astype(int)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for s, e in edges:
        cv2.line(img, tuple(pts2d[s]), tuple(pts2d[e]), color, 1)

def process_image_task(img_data):
    img_id, img_path, K, W, H, anns = img_data
    if not os.path.exists(img_path):
        return f"[{img_id}] missing"
    image = cv2.imread(img_path)
    if image is None:
        return f"[{img_id}] failed to read"
    image_2d = image.copy()
    image_3d = image.copy()
    for ann in anns:
        cat = ann['category_name']
        if cat in ignore_categories:
            continue
        color = category_colors.get(cat, (255, 255, 255))
        # draw 2D bbox
        x_min, y_min, x_max, y_max = map(int, ann['bbox2D_tight'])
        cv2.rectangle(image_2d, (x_min, y_min), (x_max, y_max), color, 1)
        cv2.putText(image_2d, cat, (x_min, max(0, y_min - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        # draw 3D bbox
        if 'bbox3D_cam' in ann:
            pts2d = project_3d_to_2d(ann['bbox3D_cam'], K, W, H)
            draw_3d_bbox(image_3d, pts2d, color)
            if not np.any(np.isnan(pts2d[:4])):
                top_x = int(np.mean(pts2d[:4, 0]))
                top_y = int(np.min(pts2d[:4, 1]))
                cv2.putText(image_3d, cat, (top_x, max(0, top_y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    cv2.imwrite(f"{vis_2d_dir}/image_{img_id}_2d.jpg", image_2d)
    cv2.imwrite(f"{vis_3d_dir}/image_{img_id}_3d.jpg", image_3d)
    return f"[{img_id}] done"

with open("Warehouse_test.json", "r") as f:
    data = json.load(f)
images = data["images"]
annotations = data["annotations"]
image_to_anns = defaultdict(list)
for ann in annotations:
    image_to_anns[ann["image_id"]].append(ann)
tasks = []
for img in images:
    img_id = img["id"]
    tasks.append((img_id, img["file_path"], img["K"], img["width"], img["height"], image_to_anns[img_id]))
with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
    for result in tqdm(executor.map(process_image_task, tasks), total=len(tasks)):
        if result.endswith("missing") or result.endswith("failed to read"):
            print(result)