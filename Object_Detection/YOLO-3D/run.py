import os
import time
import cv2
import numpy as np
import torch
if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
from detection_model import ObjectDetector
from depth_model import DepthEstimator
from bbox3d_utils import BBox3DEstimator
import h5py
from PIL import Image

def draw_3d_bbox(image, projected_2d, color):
    for i, pt in enumerate(projected_2d.astype(int)):
        cv2.circle(image, tuple(pt), 3, (0, 255, 0), -1)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), # bottom square
             (4, 5), (5, 6), (6, 7), (7, 4), # top square
             (0, 4), (1, 5), (2, 6), (3, 7)] # vertical lines
    for start, end in edges:
        pt1 = tuple(map(int, projected_2d[start]))
        pt2 = tuple(map(int, projected_2d[end]))
        cv2.line(image, pt1, pt2, color, 1)
    return image

def main():
    source_depth = 'Camera.h5'
    source = 'Camera.mp4'
    output_path = "output.mp4"
    depth_model_size = "large" # "small", "base", "large"
    device = 'cuda'
    conf_threshold = 0.25
    iou_threshold = 0.45
    classes = None # filter by class [0, 1, 2]
    enable_tracking = True
    try:
        detector = ObjectDetector(conf_thres=conf_threshold, iou_thres=iou_threshold, classes=classes, device=device)
    except Exception as e:
        print(f"Error initializing object detector: {e}")
        detector = ObjectDetector(conf_thres=conf_threshold, iou_thres=iou_threshold, classes=classes, device='cpu')
    try:
        depth_estimator = DepthEstimator(model_size=depth_model_size, device=device)
    except Exception as e:
        print(f"Error initializing depth estimator: {e}")
        depth_estimator = DepthEstimator(model_size=depth_model_size, device='cpu')
    bbox3d_estimator = BBox3DEstimator()
    try:
        if isinstance(source, str) and source.isdigit():
            source = int(source)
    except ValueError:
        pass
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    frame_count = 0
    start_time = time.time()
    fps_display = "FPS: --"
    while True:
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27 or (key & 0xFF) == ord('q') or (key & 0xFF) == 27:
            break
        try:
            ret, frame = cap.read()
            if not ret:
                break
            original_frame = frame.copy()
            detection_frame = frame.copy()
            result_frame = frame.copy()
            # step 1: object detection
            try:
                detection_frame, detections = detector.detect(detection_frame, track=enable_tracking)
            except Exception as e:
                print(f"Error during object detection: {e}")
                detections = []
                cv2.putText(detection_frame, "Detection Error", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # step 2: depth estimation
            # try:
            #     depth_map = depth_estimator.estimate_depth(original_frame)
            #     depth_colored = depth_estimator.colorize_depth(depth_map)
            # except Exception as e:
            #     print(f"Error during depth estimation: {e}")
            #     depth_map = np.zeros((height, width), dtype=np.float32)
            #     depth_colored = np.zeros((height, width, 3), dtype=np.uint8)
            #     cv2.putText(depth_colored, "Depth Error", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            try:
                with h5py.File(source_depth, "r") as f:
                    dataset_name = f'distance_to_image_plane_{frame_count:05d}.png'
                    depth_map = np.asarray(f[dataset_name])
                    depth_min = depth_map.min()
                    depth_max = depth_map.max()
                    if depth_max > depth_min:
                        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
                    depth_colored = depth_estimator.colorize_depth(depth_map)
            except FileNotFoundError:
                print(f"Error during depth estimation: {e}")
                depth_map = np.zeros((height, width), dtype=np.float32)
                depth_colored = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.putText(depth_colored, "Depth Error", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # step 3: 3D bbox estimation
            boxes_3d = []
            active_ids = []
            for detection in detections:
                try:
                    bbox, score, class_id, obj_id = detection
                    class_name = detector.get_class_names()[class_id]
                    if class_name.lower() in ['person', 'cat', 'dog', 'fouriergr1t2', 'agilitydigit']:
                        center_x = int((bbox[0] + bbox[2]) / 2)
                        center_y = int((bbox[1] + bbox[3]) / 2)
                        depth_value = depth_estimator.get_depth_at_point(depth_map, center_x, center_y)
                        depth_method = 'center'
                    else:
                        depth_value = depth_estimator.get_depth_in_region(depth_map, bbox, method='median')
                        depth_method = 'median'
                    box_3d = {'bbox_2d': bbox, 'depth_value': depth_value, 'depth_method': depth_method, 'class_name': class_name, 'object_id': obj_id, 'score': score}
                    # box_3d = bbox3d_estimator.estimate_3d_box(bbox, depth_value, class_name, obj_id)
                    boxes_3d.append(box_3d)
                    if obj_id is not None:
                        active_ids.append(obj_id)
                except Exception as e:
                    print(f"Error processing detection: {e}")
                    continue
            bbox3d_estimator.cleanup_trackers(active_ids)
            # Step 4: visualization
            for box_3d in boxes_3d:
                try:
                    class_name = box_3d['class_name'].lower()
                    if 'person' in class_name:
                        color = (255, 0, 0)
                    elif 'forklift' in class_name:
                        color = (0, 255, 0)
                    elif 'novacarter' in class_name:
                        color = (0, 0, 255)
                    elif 'transporter' in class_name:
                        color = (255, 255, 0)
                    elif 'fouriergr1t2' in class_name:
                        color = (255, 0, 255)
                    elif 'agilitydigit' in class_name:
                        color = (0, 255, 255)
                    else:
                        color = (128, 128, 128)
                    result_frame = bbox3d_estimator.draw_box_3d(result_frame, box_3d, color=color)
                    # result_frame = draw_3d_bbox(result_frame, bbox3d_estimator.project_box_3d_to_2d(box_3d), color)
                except Exception as e:
                    print(f"Error drawing box: {e}")
                    continue
            frame_count += 1
            if frame_count % 10 == 0:
                end_time = time.time()
                elapsed_time = end_time - start_time
                fps_value = frame_count / elapsed_time
                fps_display = f"FPS: {fps_value:.1f}"
            cv2.putText(result_frame, f"{fps_display} | Device: {device}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            try:
                depth_height = height // 4
                depth_width = depth_height * width // height
                depth_resized = cv2.resize(depth_colored, (depth_width, depth_height))
                result_frame[0:depth_height, 0:depth_width] = depth_resized
            except Exception as e:
                print(f"Error adding depth map to result: {e}")
            out.write(result_frame)
            cv2.imshow("3D Object Detection", result_frame)
            # cv2.imshow("Depth Map", depth_colored)
            # cv2.imshow("Object Detection", detection_frame)
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27 or (key & 0xFF) == ord('q') or (key & 0xFF) == 27:
                break
        except Exception as e:
            print(f"Error processing frame: {e}")
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27 or (key & 0xFF) == ord('q') or (key & 0xFF) == 27:
                break
            continue
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processing complete. Output saved to {output_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()