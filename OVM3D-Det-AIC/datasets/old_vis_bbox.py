import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os

vis_2d_dir = "vis_bbox/vis_2d"
vis_3d_dir = "vis_bbox/vis_3d"
os.makedirs(vis_2d_dir, exist_ok=True)
os.makedirs(vis_3d_dir, exist_ok=True)
category_colors = {"person": "red", "forklift": "blue", "novacarter": "green", "transporter": "yellow", "fouriergr1t2": "purple", "agilitydigit": "orange"}
ignore_categories = {"dontcare", "ignore", "void"}

def project_3d_to_2d(points_3d, K, img_width, img_height):
    points_2d = []
    K = np.array(K)
    for point in points_3d:
        point = np.array(point)
        proj = K @ point
        if proj[2] <= 0:
            points_2d.append([None, None])
        else:
            u, v = proj[0] / proj[2], proj[1] / proj[2]
            u = max(0, min(img_width - 1, u))
            v = max(0, min(img_height - 1, v))
            points_2d.append([u, v])
    return points_2d

def draw_3d_bbox(ax, corners_2d, color):
    if any(c[0] is None or c[1] is None for c in corners_2d):
        return None, None
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    top_y = min(c[1] for c in corners_2d[:4])
    top_x = (min(c[0] for c in corners_2d[:4]) + max(c[0] for c in corners_2d[:4])) / 2
    for start, end in edges:
        if corners_2d[start][0] is None or corners_2d[end][0] is None:
            continue
        x = [corners_2d[start][0], corners_2d[end][0]]
        y = [corners_2d[start][1], corners_2d[end][1]]
        ax.plot(x, y, color=color, linewidth=1.0)
    return top_x, top_y

json_file = "Warehouse_008_test.json"
with open(json_file, 'r') as f:
    data = json.load(f)
images = data['images']
annotations = data['annotations']
for img in images:
    img_id = img['id']
    img_path = img['file_path']
    K = img['K']
    img_width = img['width']
    img_height = img['height']
    if not os.path.exists(img_path):
        print(f"Image file {img_path} not found, skipping...")
        continue
    image = Image.open(img_path)
    aspect_ratio = img_width / img_height
    fig_width = 10
    fig_height = fig_width / aspect_ratio
    # 2D bbox visualization
    fig_2d, ax_2d = plt.subplots(figsize=(fig_width, fig_height))
    ax_2d.imshow(image)
    ax_2d.set_position([0, 0, 1, 1])
    ax_2d.axis('off')
    img_annotations = [ann for ann in annotations if ann['image_id'] == img_id and ann['category_name'] not in ignore_categories]
    for ann in img_annotations:
        bbox = ann['bbox2D_tight']
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        category_name = ann['category_name']
        color = category_colors.get(category_name, "white")
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=0.6, edgecolor=color, facecolor='none')
        ax_2d.add_patch(rect)
        plt.text(x_min + width / 2, y_min - 10, category_name, color=color, fontsize=8, ha='center')
    output_2d_filename = os.path.join(vis_2d_dir, f"image_{img_id}_2d_bbox.jpg")
    plt.tight_layout(pad=0)
    plt.savefig(output_2d_filename, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig_2d)
    print(f"Saved 2D visualization to {output_2d_filename}")
    # 3D bbox visualization
    fig_3d, ax_3d = plt.subplots(figsize=(fig_width, fig_height))
    ax_3d.imshow(image)
    ax_3d.set_position([0, 0, 1, 1])
    ax_3d.axis('off')
    ax_3d.set_xlim(0, img_width)
    ax_3d.set_ylim(img_height, 0)
    for ann in img_annotations:
        if 'bbox3D_cam' not in ann:
            continue
        category_name = ann['category_name']
        color = category_colors.get(category_name, "white")
        corners_3d = ann['bbox3D_cam']
        corners_2d = project_3d_to_2d(corners_3d, K, img_width, img_height)
        top_x, top_y = draw_3d_bbox(ax_3d, corners_2d, color)
        if top_x is not None and top_y is not None:
            plt.text(top_x, top_y - 10, category_name, color=color, fontsize=8, ha='center')
    output_3d_filename = os.path.join(vis_3d_dir, f"image_{img_id}_3d_bbox.jpg")
    plt.tight_layout(pad=0)
    plt.savefig(output_3d_filename, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig_3d)
    print(f"Saved 3D visualization to {output_3d_filename}")