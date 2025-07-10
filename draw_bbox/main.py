# import os
# import numpy as np
# import cv2
# import json
# import random
#
# WAREHOUSE_ID = "Warehouse_000"
# BASE_URL = f"datasets/train/{WAREHOUSE_ID}"
# gt_path = f"{BASE_URL}/ground_truth.json"
# cali_path = f"{BASE_URL}/calibration.json"
# map_path = f"{BASE_URL}/map.png"
#
# def get_color():
#     return (random.randint(64, 255), random.randint(64, 255), random.randint(64, 255))
#
# def get_camera_calibration(camera_id):
#     with open(cali_path, 'r') as f:
#         cali_data = json.load(f)
#     sensors = cali_data["sensors"]
#     for sensor in sensors:
#         if sensor.get("type") == "camera" and sensor.get("id") == camera_id:
#             return sensor
#     print(f"Camera with ID {camera_id} not found.")
#     return None
#
# def get_gt():
#     with open(gt_path, 'r') as f:
#         gt_data = json.load(f)
#     gt_keys = list(gt_data.keys())
#     num_frames = len(gt_keys)
#     bbox_data = []
#     for idx in range(num_frames):
#         frame_data = gt_data[str(idx)]
#         for objdata in frame_data:
#             object_id = objdata['object id']
#             wx, wy, wz = objdata['3d location']
#             w, h, d = objdata['3d bounding box scale']
#             pitch, roll, yaw = objdata['3d bounding box rotation']
#             sample = {"obj_id": object_id, "obj_pos": [wx ,wy ,wz], "obj_scale": [w ,h ,d], "yaw": yaw}
#             bbox_data.append(sample)
#         break
#     return bbox_data
#
# def get_yaw_rotation_matrix(yaw):
#     return np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
#
# def process(bbox_data, camera_data, image, connected):
#     intrinsicMat = np.array(camera_data['intrinsicMatrix'])
#     extrinsicMat = np.array(camera_data['extrinsicMatrix'])
#     for bbox in bbox_data:
#         bbox_wpos = bbox['obj_pos']
#         bbox_wscale = bbox['obj_scale']
#         bbox_yaw = bbox['yaw']
#         center_world = np.array(bbox_wpos + [1])
#         center_camera = np.dot(extrinsicMat, center_world)
#         center_2d = np.dot(intrinsicMat, center_camera)
#         center_2d /= center_2d[2]
#         world_coords = process_world_coordinates(bbox_wpos, bbox_wscale, bbox_yaw)
#         camera_coords = convertWorldToCamera(world_coords, extrinsicMat)
#         projected_coords = projectCamera(camera_coords, intrinsicMat)
#         image = draw_3d_bbox(image, projected_coords, connected)
#         text = f"({bbox_wpos[0]:.2f}, {bbox_wpos[1]:.2f}, {bbox_wpos[2]:.2f})"
#         text_position = tuple(center_2d[:2].astype(int))
#         cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 255), 1, cv2.LINE_AA)
#
# def process_world_coordinates(bbox_wpos, bbox_wscale, bbox_yaw):
#     x, y, z = bbox_wpos
#     w, h, d = bbox_wscale
#     yaw = bbox_yaw
#     corners = np.array([[w/ 2, h / 2, d / 2], [-w / 2, h / 2, d / 2], [-w / 2, -h / 2, d / 2], [w / 2, -h / 2, d / 2], [w / 2, h / 2, -d / 2],
#                         [-w / 2, h / 2, -d / 2], [-w / 2, -h / 2, -d / 2], [w / 2, -h / 2, -d / 2]])
#     rot_mat = get_yaw_rotation_matrix(yaw)
#     rotated_corners = (rot_mat @ corners.T).T + np.array([x, y, z])
#     return rotated_corners
#
# def convertWorldToCamera(points, extrinsic_mat):
#     points = np.asarray(points)
#     if points.ndim == 1 and points.shape[0] == 3:
#         points = points.reshape(1, 3)
#     elif points.ndim != 2 or points.shape[1] != 3:
#         raise ValueError("Input must be shape (3,) or (N, 3)")
#     homogeneous_corners = np.hstack([points, np.ones((points.shape[0], 1))])
#     camera_coords = (np.dot(extrinsic_mat, homogeneous_corners.T)).T
#     return camera_coords
#
# def projectCamera(camera_coords, intrinsic_mat):  # using Intrinsic matrix
#     projected = (intrinsic_mat @ camera_coords.T).T
#     projected_2d = projected[:, :2] / projected[:, 2:]
#     return projected_2d
#
# def draw_3d_bbox(image, projected_2d, connected):
#     for i, pt in enumerate(projected_2d.astype(int)):
#         cv2.circle(image, tuple(pt), 3, (0, 255, 0), -1)
#     if connected:
#         color = get_color()
#         edges = [(0, 1), (1, 2), (2, 3), (3, 0), # bottom square
#                  (4, 5), (5, 6), (6, 7), (7, 4), # top square
#                  (0, 4), (1, 5), (2, 6), (3, 7)] # vertical lines
#         for start, end in edges:
#             pt1 = tuple(map(int, projected_2d[start]))
#             pt2 = tuple(map(int, projected_2d[end]))
#             cv2.line(image, pt1, pt2, color, 1)
#     return image
#
# def main():
#     bbox_data = get_gt()
#     camera_id = "Camera_0000"
#     video_file = f"{BASE_URL}/videos/{camera_id}.mp4"
#     vidcap = cv2.VideoCapture(video_file)
#     success, frame = vidcap.read()
#     if not success:
#         print(f"[ERROR] Failed to read from video: {video_file}")
#         return
#     camera_num = camera_id.split('_')
#     if len(camera_num) > 1:
#         num = camera_num[1]
#         num = int(num)
#         if num == 0:
#             camera_id = "Camera"
#         else:
#             camera_id = f"Camera_{str(num)}"
#     camera_data = get_camera_calibration(camera_id)
#     if not camera_data:
#         print(f"[ERROR] Camera calibration not found for ID: {camera_id}")
#         return
#     frame_copy = frame.copy()
#     connected = True
#     process(bbox_data, camera_data, frame_copy, connected)
#     output_path = f"output/{camera_id}_frame0_with_bbox.jpg"
#     if not os.path.exists('output'):
#         os.makedirs('output')
#     cv2.imwrite(output_path, frame_copy)
#     print(f"[INFO] Processed frame saved to: {output_path}")
#
# if __name__ == "__main__":
#     main()

# import numpy as np
# import cv2
# import json
# import os
#
# WAREHOUSE_ID = "Warehouse_014"
# BASE_URL = f"datasets/train/{WAREHOUSE_ID}"
# gt_path = f"{BASE_URL}/ground_truth.json"
# cali_path = f"{BASE_URL}/calibration.json"
# map_path = f"{BASE_URL}/map.png"
#
# def get_color(obj_type):
#     class_colors = {"Person": (255, 0, 0), "Forklift": (0, 255, 0), "NovaCarter": (0, 0, 255), "Transporter": (255, 255, 0), "FourierGR1T2": (255, 0, 255), "AgilityDigit": (0, 255, 255)}
#     return class_colors.get(obj_type, (128, 128, 128))
#
# def get_camera_calibration(camera_id):
#     with open(cali_path, 'r') as f:
#         cali_data = json.load(f)
#     sensors = cali_data["sensors"]
#     for sensor in sensors:
#         if sensor.get("type") == "camera" and sensor.get("id") == camera_id:
#             return sensor
#     print(f"Camera with ID {camera_id} not found.")
#     return None
#
# def get_gt():
#     with open(gt_path, 'r') as f:
#         gt_data = json.load(f)
#     gt_keys = list(gt_data.keys())
#     num_frames = len(gt_keys)
#     bbox_data_per_frame = []
#     for idx in range(num_frames):
#         frame_data = gt_data[str(idx)]
#         bbox_data = []
#         for objdata in frame_data:
#             object_id = objdata['object id']
#             wx, wy, wz = objdata['3d location']
#             w, h, d = objdata['3d bounding box scale']
#             pitch, roll, yaw = objdata['3d bounding box rotation']
#             obj_type = objdata['object type']
#             sample = {"obj_id": object_id, "obj_pos": [wx, wy, wz], "obj_scale": [w, h, d], "yaw": yaw, "obj_type": obj_type}
#             bbox_data.append(sample)
#         bbox_data_per_frame.append(bbox_data)
#     return bbox_data_per_frame
#
# def get_yaw_rotation_matrix(yaw):
#     yaw = np.radians(yaw)
#     return np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
#
# ###
# def get_coco_bbox(projected_2d):
#     x_coords = projected_2d[:, 0]
#     y_coords = projected_2d[:, 1]
#     x1 = float(np.min(x_coords))
#     y1 = float(np.min(y_coords))
#     x2 = float(np.max(x_coords))
#     y2 = float(np.max(y_coords))
#     return [x1, y1, x2, y2]
#
# def process(bbox_data, camera_data, image, connected):
#     intrinsicMat = np.array(camera_data['intrinsicMatrix'])
#     extrinsicMat = np.array(camera_data['extrinsicMatrix'])
#     for bbox in bbox_data:
#         bbox_wpos = bbox['obj_pos']
#         bbox_wscale = bbox['obj_scale']
#         bbox_yaw = bbox['yaw']
#         obj_type = bbox['obj_type']
#         color = get_color(obj_type)
#         world_coords = process_world_coordinates(bbox_wpos, bbox_wscale, bbox_yaw)
#         camera_coords = convertWorldToCamera(world_coords, extrinsicMat)
#         projected_coords = projectCamera(camera_coords, intrinsicMat)
#         coco_bbox = get_coco_bbox(projected_coords)
#         x1, y1, x2, y2 = coco_bbox
#         pt1 = (int(x1), int(y1))
#         pt2 = (int(x2), int(y2))
#         cv2.rectangle(image, pt1, pt2, color, 1)
#         if connected:
#             image = draw_3d_bbox(image, projected_coords, connected, color)
#         top_midpoint_2d = np.array([x1 + (x2 - x1) / 2, y1])
#         text_position = tuple(top_midpoint_2d.astype(int))
#         cv2.putText(image, obj_type, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
#     return image
# ###
#
# # def process(bbox_data, camera_data, image, connected):
# #     intrinsicMat = np.array(camera_data['intrinsicMatrix'])
# #     extrinsicMat = np.array(camera_data['extrinsicMatrix'])
# #     for bbox in bbox_data:
# #         bbox_wpos = bbox['obj_pos']
# #         bbox_wscale = bbox['obj_scale']
# #         bbox_yaw = bbox['yaw']
# #         obj_type = bbox['obj_type']
# #         color = get_color(obj_type)
# #         world_coords = process_world_coordinates(bbox_wpos, bbox_wscale, bbox_yaw)
# #         camera_coords = convertWorldToCamera(world_coords, extrinsicMat)
# #         projected_coords = projectCamera(camera_coords, intrinsicMat)
# #         image = draw_3d_bbox(image, projected_coords, connected, color)
# #         top_midpoint_2d = (projected_coords[0] + projected_coords[1]) / 2
# #         text_position = tuple(top_midpoint_2d.astype(int))
# #         cv2.putText(image, obj_type, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
# #     return image
#
# def process_world_coordinates(bbox_wpos, bbox_wscale, bbox_yaw):
#     x, y, z = bbox_wpos
#     w, h, d = bbox_wscale
#     yaw = bbox_yaw
#     corners = np.array([[w/2, h/2, d/2], [-w/2, h/2, d/2], [-w/2, -h/2, d/2], [w/2, -h/2, d/2],
#                         [w/2, h/2, -d/2], [-w/2, h/2, -d/2], [-w/2, -h/2, -d/2], [w/2, -h/2, -d/2]])
#     rot_mat = get_yaw_rotation_matrix(yaw)
#     rotated_corners = (rot_mat @ corners.T).T + np.array([x, y, z])
#     return rotated_corners
#
# def convertWorldToCamera(points, extrinsic_mat):
#     points = np.asarray(points)
#     if points.ndim == 1 and points.shape[0] == 3:
#         points = points.reshape(1, 3)
#     elif points.ndim != 2 or points.shape[1] != 3:
#         raise ValueError("Input must be shape (3,) or (N, 3)")
#     homogeneous_corners = np.hstack([points, np.ones((points.shape[0], 1))])
#     camera_coords = (np.dot(extrinsic_mat, homogeneous_corners.T)).T
#     return camera_coords
#
# def projectCamera(camera_coords, intrinsic_mat):
#     projected = (intrinsic_mat @ camera_coords.T).T
#     projected_2d = projected[:, :2] / projected[:, 2:]
#     return projected_2d
#
# def draw_3d_bbox(image, projected_2d, connected, color):
#     for i, pt in enumerate(projected_2d.astype(int)):
#         cv2.circle(image, tuple(pt), 3, (0, 255, 0), -1)
#     if connected:
#         edges = [(0, 1), (1, 2), (2, 3), (3, 0), # bottom square
#                  (4, 5), (5, 6), (6, 7), (7, 4), # top square
#                  (0, 4), (1, 5), (2, 6), (3, 7)] # vertical lines
#         for start, end in edges:
#             pt1 = tuple(map(int, projected_2d[start]))
#             pt2 = tuple(map(int, projected_2d[end]))
#             cv2.line(image, pt1, pt2, color, 1)
#     return image
#
# def main():
#     bbox_data_per_frame = get_gt()
#     camera_id = "Camera"
#     video_file = f"{BASE_URL}/videos/{camera_id}.mp4"
#     vidcap = cv2.VideoCapture(video_file)
#     if not vidcap.isOpened():
#         print(f"[ERROR] Failed to open video: {video_file}")
#         return
#     camera_num = camera_id.split('_')
#     if len(camera_num) > 1:
#         num = camera_num[1]
#         num = int(num)
#         if num == 0:
#             camera_id = "Camera"
#         else:
#             camera_id = f"Camera_{str(num)}"
#     camera_data = get_camera_calibration(camera_id)
#     if not camera_data:
#         print(f"[ERROR] Camera calibration not found for ID: {camera_id}")
#         return
#     os.makedirs(f"output", exist_ok=True)
#     frame_idx = 0
#     while vidcap.isOpened():
#         success, frame = vidcap.read()
#         if not success:
#             print(f"[INFO] Processed {frame_idx} frames. End of video.")
#             break
#         if frame_idx >= len(bbox_data_per_frame):
#             print(f"[WARNING] No ground truth data for frame {frame_idx}. Stopping.")
#             break
#         frame_copy = frame.copy()
#         connected = True
#         bbox_data = bbox_data_per_frame[frame_idx]
#         process(bbox_data, camera_data, frame_copy, connected)
#         output_path = f"output/{camera_id}_frame{frame_idx:04d}_with_bbox.jpg"
#         cv2.imwrite(output_path, frame_copy)
#         print(f"[INFO] Saved frame {frame_idx} to: {output_path}")
#         frame_idx += 1
#     vidcap.release()
#
# if __name__ == "__main__":
#     main()

import numpy as np
import cv2
import json
import os

WAREHOUSE_ID = "Warehouse_014"
BASE_URL = f"datasets/train/{WAREHOUSE_ID}"
gt_path = f"{BASE_URL}/ground_truth.json"
cali_path = f"{BASE_URL}/calibration.json"
map_path = f"{BASE_URL}/map.png"

def get_color(obj_type):
    class_colors = {"Person": (255, 0, 0), "Forklift": (0, 255, 0), "NovaCarter": (0, 0, 255), "Transporter": (255, 255, 0), "FourierGR1T2": (255, 0, 255), "AgilityDigit": (0, 255, 255)}
    return class_colors.get(obj_type, (128, 128, 128))

def get_camera_calibration(camera_id):
    with open(cali_path, 'r') as f:
        cali_data = json.load(f)
    sensors = cali_data["sensors"]
    for sensor in sensors:
        if sensor.get("type") == "camera" and sensor.get("id") == camera_id:
            return sensor
    print(f"Camera with ID {camera_id} not found.")
    return None

def get_gt():
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    gt_keys = list(gt_data.keys())
    num_frames = len(gt_keys)
    bbox_data_per_frame = []
    for idx in range(num_frames):
        frame_data = gt_data[str(idx)]
        bbox_data = []
        for objdata in frame_data:
            object_id = objdata['object id']
            wx, wy, wz = objdata['3d location']
            w, h, d = objdata['3d bounding box scale']
            pitch, roll, yaw = objdata['3d bounding box rotation']
            obj_type = objdata['object type']
            sample = {"obj_id": object_id, "obj_pos": [wx, wy, wz], "obj_scale": [w, h, d], "pitch": pitch, "roll": roll, "yaw": yaw, "obj_type": obj_type}
            bbox_data.append(sample)
        bbox_data_per_frame.append(bbox_data)
    return bbox_data_per_frame

def get_yaw_rotation_matrix(pitch, roll, yaw):
    pitch = np.radians(pitch)
    roll = np.radians(roll)
    yaw = np.radians(yaw)
    # R_x = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(roll), -np.sin(roll)], [0.0, np.sin(roll), np.cos(roll)]])
    # R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    # R_y = np.array([[np.cos(pitch), 0.0, np.sin(pitch)], [0.0, 1.0, 0.0], [-np.sin(pitch), 0.0, np.cos(pitch)]])
    R_x = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(pitch), -np.sin(pitch)], [0.0, np.sin(pitch), np.cos(pitch)]])
    R_y = np.array([[np.cos(roll), 0.0, np.sin(roll)], [0.0, 1.0, 0.0], [-np.sin(roll), 0.0, np.cos(roll)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    R = R_z @ R_y @ R_x
    return R

def process(bbox_data, camera_data, image, connected):
    intrinsicMat = np.array(camera_data['intrinsicMatrix'])
    extrinsicMat = np.array(camera_data['extrinsicMatrix'])
    for bbox in bbox_data:
        bbox_wpos = bbox['obj_pos']
        bbox_wscale = bbox['obj_scale']
        bbox_pitch = bbox['pitch']
        bbox_roll = bbox['roll']
        bbox_yaw = bbox['yaw']
        obj_type = bbox['obj_type']
        color = get_color(obj_type)
        world_coords = process_world_coordinates(bbox_wpos, bbox_wscale, bbox_pitch, bbox_roll, bbox_yaw)
        camera_coords = convertWorldToCamera(world_coords, extrinsicMat)
        projected_coords = projectCamera(camera_coords, intrinsicMat)
        image = draw_3d_bbox(image, projected_coords, connected, color)
        top_midpoint_2d = (projected_coords[0] + projected_coords[1]) / 2
        text_position = tuple(top_midpoint_2d.astype(int))
        cv2.putText(image, obj_type, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return image

def process_world_coordinates(bbox_wpos, bbox_wscale, bbox_pitch, bbox_roll, bbox_yaw):
    x, y, z = bbox_wpos
    w, h, d = bbox_wscale
    pitch, roll, yaw = bbox_pitch, bbox_roll, bbox_yaw
    # corners = np.array([[w/2, h/2, d/2], [-w/2, h/2, d/2], [-w/2, -h/2, d/2], [w/2, -h/2, d/2],
    #                     [w/2, h/2, -d/2], [-w/2, h/2, -d/2], [-w/2, -h/2, -d/2], [w/2, -h/2, -d/2]])
    corners = np.array([[-w/2, -h/2, -d/2], [w/2, -h/2, -d/2], [w/2, h/2, -d/2], [-w/2, h/2, -d/2],
                        [-w/2, -h/2, d/2], [w/2, -h/2, d/2], [w/2, h/2, d/2], [-w/2, h/2, d/2]])
    rot_mat = get_yaw_rotation_matrix(pitch, roll, yaw)
    rotated_corners = (rot_mat @ corners.T).T + np.array([x, y, z])
    return rotated_corners

def convertWorldToCamera(points, extrinsic_mat):
    points = np.asarray(points)
    if points.ndim == 1 and points.shape[0] == 3:
        points = points.reshape(1, 3)
    elif points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Input must be shape (3,) or (N, 3)")
    homogeneous_corners = np.hstack([points, np.ones((points.shape[0], 1))])
    camera_coords = (np.dot(extrinsic_mat, homogeneous_corners.T)).T
    return camera_coords

def projectCamera(camera_coords, intrinsic_mat):
    projected = (intrinsic_mat @ camera_coords.T).T
    projected_2d = projected[:, :2] / projected[:, 2:]
    return projected_2d

def draw_3d_bbox(image, projected_2d, connected, color):
    for i, pt in enumerate(projected_2d.astype(int)):
        cv2.circle(image, tuple(pt), 3, (0, 255, 0), -1)
    if connected:
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), # bottom square
                 (4, 5), (5, 6), (6, 7), (7, 4), # top square
                 (0, 4), (1, 5), (2, 6), (3, 7)] # vertical lines
        for start, end in edges:
            pt1 = tuple(map(int, projected_2d[start]))
            pt2 = tuple(map(int, projected_2d[end]))
            cv2.line(image, pt1, pt2, color, 1)
    return image

def main():
    bbox_data_per_frame = get_gt()
    camera_id = "Camera"
    video_file = f"{BASE_URL}/videos/{camera_id}.mp4"
    vidcap = cv2.VideoCapture(video_file)
    if not vidcap.isOpened():
        print(f"[ERROR] Failed to open video: {video_file}")
        return
    camera_num = camera_id.split('_')
    if len(camera_num) > 1:
        num = camera_num[1]
        num = int(num)
        if num == 0:
            camera_id = "Camera"
        else:
            camera_id = f"Camera_{str(num)}"
    camera_data = get_camera_calibration(camera_id)
    if not camera_data:
        print(f"[ERROR] Camera calibration not found for ID: {camera_id}")
        return
    os.makedirs(f"output", exist_ok=True)
    frame_idx = 0
    while vidcap.isOpened():
        success, frame = vidcap.read()
        if not success:
            print(f"[INFO] Processed {frame_idx} frames. End of video.")
            break
        if frame_idx >= len(bbox_data_per_frame):
            print(f"[WARNING] No ground truth data for frame {frame_idx}. Stopping.")
            break
        frame_copy = frame.copy()
        connected = True
        bbox_data = bbox_data_per_frame[frame_idx]
        process(bbox_data, camera_data, frame_copy, connected)
        output_path = f"output/{camera_id}_frame{frame_idx:04d}_with_bbox.jpg"
        cv2.imwrite(output_path, frame_copy)
        print(f"[INFO] Saved frame {frame_idx} to: {output_path}")
        frame_idx += 1
    vidcap.release()

if __name__ == "__main__":
    main()
