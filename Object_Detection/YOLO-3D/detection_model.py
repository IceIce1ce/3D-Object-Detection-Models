import os
import torch
import cv2
from ultralytics import YOLO
from collections import deque

class ObjectDetector:
    def __init__(self, conf_thres=0.25, iou_thres=0.45, classes=None, device=None):
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = device
        if self.device == 'mps':
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        try:
            self.model = YOLO('best.pt')
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = YOLO('best.pt')
        self.model.overrides['conf'] = conf_thres
        self.model.overrides['iou'] = iou_thres
        self.model.overrides['agnostic_nms'] = False
        self.model.overrides['max_det'] = 1000
        if classes is not None:
            self.model.overrides['classes'] = classes
        self.tracking_trajectories = {}
    
    def detect(self, image, track=True):
        detections = []
        annotated_image = image.copy()
        try:
            if track:
                results = self.model.track(image, verbose=False, device=self.device, persist=True)
            else:
                results = self.model.predict(image, verbose=False, device=self.device)
        except RuntimeError as e:
            if self.device == 'mps' and "not currently implemented for the MPS device" in str(e):
                print(f"MPS error during detection: {e}")
                if track:
                    results = self.model.track(image, verbose=False, device='cpu', persist=True)
                else:
                    results = self.model.predict(image, verbose=False, device='cpu')
            else:
                raise
        if track:
            for id_ in list(self.tracking_trajectories.keys()):
                if id_ not in [int(bbox.id) for predictions in results if predictions is not None for bbox in predictions.boxes if bbox.id is not None]:
                    del self.tracking_trajectories[id_]
            for predictions in results:
                if predictions is None:
                    continue
                if predictions.boxes is None:
                    continue
                for bbox in predictions.boxes:
                    scores = bbox.conf
                    classes = bbox.cls
                    bbox_coords = bbox.xyxy
                    if hasattr(bbox, 'id') and bbox.id is not None:
                        ids = bbox.id
                    else:
                        ids = [None] * len(scores)
                    for score, class_id, bbox_coord, id_ in zip(scores, classes, bbox_coords, ids):
                        xmin, ymin, xmax, ymax = bbox_coord.cpu().numpy()
                        detections.append([[xmin, ymin, xmax, ymax], float(score), int(class_id), int(id_) if id_ is not None else None])
                        cv2.rectangle(annotated_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 225), 2)
                        label = f"ID: {int(id_) if id_ is not None else 'N/A'} {predictions.names[int(class_id)]} {float(score):.2f}"
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        dim, baseline = text_size[0], text_size[1]
                        cv2.rectangle(annotated_image, (int(xmin), int(ymin)), (int(xmin) + dim[0], int(ymin) - dim[1] - baseline), (30, 30, 30), cv2.FILLED)
                        cv2.putText(annotated_image, label, (int(xmin), int(ymin) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        if id_ is not None:
                            centroid_x = (xmin + xmax) / 2
                            centroid_y = (ymin + ymax) / 2
                            if int(id_) not in self.tracking_trajectories:
                                self.tracking_trajectories[int(id_)] = deque(maxlen=10)
                            self.tracking_trajectories[int(id_)].append((centroid_x, centroid_y))
            for id_, trajectory in self.tracking_trajectories.items():
                for i in range(1, len(trajectory)):
                    thickness = int(2 * (i / len(trajectory)) + 1)
                    cv2.line(annotated_image, (int(trajectory[i-1][0]), int(trajectory[i-1][1])), (int(trajectory[i][0]), int(trajectory[i][1])), (255, 255, 255), thickness)
        else:
            for predictions in results:
                if predictions is None:
                    continue
                if predictions.boxes is None:
                    continue
                for bbox in predictions.boxes:
                    scores = bbox.conf
                    classes = bbox.cls
                    bbox_coords = bbox.xyxy
                    for score, class_id, bbox_coord in zip(scores, classes, bbox_coords):
                        xmin, ymin, xmax, ymax = bbox_coord.cpu().numpy()
                        detections.append([[xmin, ymin, xmax, ymax], float(score), int(class_id), None])
                        cv2.rectangle(annotated_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 225), 2)
                        label = f"{predictions.names[int(class_id)]} {float(score):.2f}"
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        dim, baseline = text_size[0], text_size[1]
                        cv2.rectangle(annotated_image, (int(xmin), int(ymin)), (int(xmin) + dim[0], int(ymin) - dim[1] - baseline), (30, 30, 30), cv2.FILLED)
                        cv2.putText(annotated_image, label, (int(xmin), int(ymin) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return annotated_image, detections
    
    def get_class_names(self):
        return self.model.names 