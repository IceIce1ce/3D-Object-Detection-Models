import numpy as np
import cv2
from filterpy.kalman import KalmanFilter
from collections import defaultdict

# camera intrinsic matrix
DEFAULT_K = np.array([[1662.7688185133422, 0.0, 960.0], [0.0, 1662.7688185133425, 540.0], [0.0, 0.0, 1.0]])
# camera projection matrix
DEFAULT_P = np.array([[114.15868151893618, -36.4757569661007, -29.620962538958054, 852.2776918751567],
                      [-14.75530703556646, -14.975095610860377, -110.43227348869844, 320.87896584648564],
                      [0.03956055774610825, 0.040244270330583164, -0.030835618258946876, 1.0]])
# average dimensions for common objects (height, width, length) in meters
DEFAULT_DIMS = {'Person': np.array([1.84, 0.59, 0.63]), 'Forklift': np.array([2.15, 1.21, 2.31]), 'NovaCarter': np.array([0.56, 0.71, 0.48]),
                'Transporter': np.array([0.23, 1.43, 0.65]), 'FourierGR1T2': np.array([1.66, 0.60, 0.45]), 'AgilityDigit': np.array([1.76, 0.54, 0.80]),
                'Car': np.array([1.52, 1.64, 3.85])}

class BBox3DEstimator:
    def __init__(self, camera_matrix=None, projection_matrix=None, class_dims=None):
        self.K = camera_matrix if camera_matrix is not None else DEFAULT_K
        self.P = projection_matrix if projection_matrix is not None else DEFAULT_P
        self.dims = class_dims if class_dims is not None else DEFAULT_DIMS
        self.kf_trackers = {}
        self.box_history = defaultdict(list)
        self.max_history = 5
    
    def estimate_3d_box(self, bbox_2d, depth_value, class_name, object_id=None):
        x1, y1, x2, y2 = bbox_2d
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        height_2d = y2 - y1
        if class_name in self.dims:
            dimensions = self.dims[class_name].copy()
        else:
            dimensions = self.dims['Car'].copy()
        if class_name.lower() in ['plant', 'potted plant']:
            dimensions[0] = height_2d / 120
            dimensions[1] = dimensions[0] * 0.6
            dimensions[2] = dimensions[0] * 0.6
        elif class_name.lower() in ['person', 'fouriergr1t2', 'agilitydigit']:
            dimensions[0] = height_2d / 100
            dimensions[1] = dimensions[0] * 0.3
            dimensions[2] = dimensions[0] * 0.3
        distance = 1.0 + depth_value * 9.0
        location = self._backproject_point(center_x, center_y, distance)
        if class_name.lower() in ['plant', 'potted plant']:
            bottom_y = y2
            location[1] = self._backproject_point(center_x, bottom_y, distance)[1]
        orientation = self._estimate_orientation(bbox_2d, location, class_name)
        box_3d = {'dimensions': dimensions, 'location': location, 'orientation': orientation, 'bbox_2d': bbox_2d, 'object_id': object_id, 'class_name': class_name}
        if object_id is not None:
            box_3d = self._apply_kalman_filter(box_3d, object_id)
            self.box_history[object_id].append(box_3d)
            if len(self.box_history[object_id]) > self.max_history:
                self.box_history[object_id].pop(0)
            box_3d = self._apply_temporal_filter(object_id)
        return box_3d
    
    def _backproject_point(self, x, y, depth):
        point_2d = np.array([x, y, 1.0])
        point_3d = np.linalg.inv(self.K) @ point_2d * depth
        point_3d[1] = point_3d[1] * 0.5
        return point_3d
    
    def _estimate_orientation(self, bbox_2d, location, class_name):
        theta_ray = np.arctan2(location[0], location[2])
        if class_name.lower() in ['plant', 'potted plant']:
            return theta_ray
        if class_name.lower() in ['person', 'fouriergr1t2', 'agilitydigit']:
            alpha = 0.0
        else:
            x1, y1, x2, y2 = bbox_2d
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / height if height > 0 else 1.0
            if aspect_ratio > 1.5:
                image_center_x = self.K[0, 2]
                if (x1 + x2) / 2 < image_center_x:
                    alpha = np.pi / 2
                else:
                    alpha = -np.pi / 2
            else:
                alpha = 0.0
        rot_y = alpha + theta_ray
        return rot_y
    
    def _init_kalman_filter(self, box_3d):
        # State: [x, y, z, width, height, length, yaw, vx, vy, vz, vyaw]
        kf = KalmanFilter(dim_x=11, dim_z=7)
        kf.x = np.array([box_3d['location'][0], box_3d['location'][1], box_3d['location'][2], box_3d['dimensions'][1], box_3d['dimensions'][0], box_3d['dimensions'][2],
                         box_3d['orientation'], 0, 0, 0, 0]) # dim[1]: width, dim[0]: height, dim[2]: length
        dt = 1.0
        kf.F = np.eye(11)
        kf.F[0, 7] = dt
        kf.F[1, 8] = dt
        kf.F[2, 9] = dt
        kf.F[6, 10] = dt
        kf.H = np.zeros((7, 11))
        kf.H[0, 0] = 1 # x
        kf.H[1, 1] = 1 # y
        kf.H[2, 2] = 1 # z
        kf.H[3, 3] = 1 # width
        kf.H[4, 4] = 1 # height
        kf.H[5, 5] = 1 # length
        kf.H[6, 6] = 1 # yaw
        kf.R = np.eye(7) * 0.1
        kf.R[0:3, 0:3] *= 1.0
        kf.R[3:6, 3:6] *= 0.1
        kf.R[6, 6] = 0.3
        kf.Q = np.eye(11) * 0.1
        kf.Q[7:11, 7:11] *= 0.5
        kf.P = np.eye(11) * 1.0
        kf.P[7:11, 7:11] *= 10.0
        return kf
    
    def _apply_kalman_filter(self, box_3d, object_id):
        if object_id not in self.kf_trackers:
            self.kf_trackers[object_id] = self._init_kalman_filter(box_3d)
        kf = self.kf_trackers[object_id]
        kf.predict()
        measurement = np.array([box_3d['location'][0], box_3d['location'][1], box_3d['location'][2], box_3d['dimensions'][1], box_3d['dimensions'][0],
                                box_3d['dimensions'][2], box_3d['orientation']])
        kf.update(measurement)
        filtered_box = box_3d.copy()
        filtered_box['location'] = np.array([kf.x[0], kf.x[1], kf.x[2]])
        filtered_box['dimensions'] = np.array([kf.x[4], kf.x[3], kf.x[5]]) # height, width, length
        filtered_box['orientation'] = kf.x[6]
        return filtered_box
    
    def _apply_temporal_filter(self, object_id):
        history = self.box_history[object_id]
        if len(history) < 2:
            return history[-1]
        current_box = history[-1]
        alpha = 0.7
        filtered_box = current_box.copy()
        for i in range(len(history) - 2, -1, -1):
            weight = alpha * (1 - alpha) ** (len(history) - i - 2)
            filtered_box['location'] = filtered_box['location'] * (1 - weight) + history[i]['location'] * weight
            angle_diff = history[i]['orientation'] - filtered_box['orientation']
            if angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            elif angle_diff < -np.pi:
                angle_diff += 2 * np.pi
            filtered_box['orientation'] += angle_diff * weight
        return filtered_box
    
    def project_box_3d_to_2d(self, box_3d):
        h, w, l = box_3d['dimensions']
        x, y, z = box_3d['location']
        rot_y = box_3d['orientation']
        class_name = box_3d['class_name'].lower()
        x1, y1, x2, y2 = box_3d['bbox_2d']
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width_2d = x2 - x1
        height_2d = y2 - y1
        R_mat = np.array([[np.cos(rot_y), 0, np.sin(rot_y)], [0, 1, 0], [-np.sin(rot_y), 0, np.cos(rot_y)]])
        if class_name in ['plant', 'potted plant']:
            x_corners = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
            y_corners = np.array([h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2])
            z_corners = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])
        else:
            x_corners = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
            y_corners = np.array([0, 0, 0, 0, -h, -h, -h, -h])
            z_corners = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])
        corners_3d = np.vstack([x_corners, y_corners, z_corners])
        corners_3d = R_mat @ corners_3d
        corners_3d[0, :] += x
        corners_3d[1, :] += y
        corners_3d[2, :] += z
        corners_3d_homo = np.vstack([corners_3d, np.ones((1, 8))])
        corners_2d_homo = self.P @ corners_3d_homo
        corners_2d = corners_2d_homo[:2, :] / corners_2d_homo[2, :]
        mean_x = np.mean(corners_2d[0, :])
        mean_y = np.mean(corners_2d[1, :])
        if abs(mean_x - center_x) > width_2d or abs(mean_y - center_y) > height_2d:
            shift_x = center_x - mean_x
            shift_y = center_y - mean_y
            corners_2d[0, :] += shift_x
            corners_2d[1, :] += shift_y
        return corners_2d.T
    
    def draw_box_3d(self, image, box_3d, color=(0, 255, 0), thickness=2):
        x1, y1, x2, y2 = [int(coord) for coord in box_3d['bbox_2d']]
        depth_value = box_3d.get('depth_value', 0.5)
        width = x2 - x1
        height = y2 - y1
        offset_factor = 1.0 - depth_value
        offset_x = int(width * 0.3 * offset_factor)
        offset_y = int(height * 0.3 * offset_factor)
        offset_x = max(15, min(offset_x, 50))
        offset_y = max(15, min(offset_y, 50))
        front_tl = (x1, y1)
        front_tr = (x2, y1)
        front_br = (x2, y2)
        front_bl = (x1, y2)
        back_tl = (x1 + offset_x, y1 - offset_y)
        back_tr = (x2 + offset_x, y1 - offset_y)
        back_br = (x2 + offset_x, y2 - offset_y)
        back_bl = (x1 + offset_x, y2 - offset_y)
        overlay = image.copy()
        cv2.rectangle(image, front_tl, front_br, color, thickness)
        cv2.line(image, front_tl, back_tl, color, thickness)
        cv2.line(image, front_tr, back_tr, color, thickness)
        cv2.line(image, front_br, back_br, color, thickness)
        cv2.line(image, front_bl, back_bl, color, thickness)
        cv2.line(image, back_tl, back_tr, color, thickness)
        cv2.line(image, back_tr, back_br, color, thickness)
        cv2.line(image, back_br, back_bl, color, thickness)
        cv2.line(image, back_bl, back_tl, color, thickness)
        pts_top = np.array([front_tl, front_tr, back_tr, back_tl], np.int32)
        pts_top = pts_top.reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts_top], color)
        pts_right = np.array([front_tr, front_br, back_br, back_tr], np.int32)
        pts_right = pts_right.reshape((-1, 1, 2))
        right_color = (int(color[0] * 0.7), int(color[1] * 0.7), int(color[2] * 0.7))
        cv2.fillPoly(overlay, [pts_right], right_color)
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        class_name = box_3d['class_name']
        obj_id = box_3d['object_id'] if 'object_id' in box_3d else None
        text_y = y1 - 10
        if obj_id is not None:
            cv2.putText(image, f"ID:{obj_id}", (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            text_y -= 15
        cv2.putText(image, class_name, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        text_y -= 15
        if 'depth_value' in box_3d:
            depth_value = box_3d['depth_value']
            depth_method = box_3d.get('depth_method', 'unknown')
            depth_text = f"D:{depth_value:.2f} ({depth_method})"
            cv2.putText(image, depth_text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            text_y -= 15
        if 'score' in box_3d:
            score = box_3d['score']
            score_text = f"S:{score:.2f}"
            cv2.putText(image, score_text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        ground_y = y2 + int(height * 0.2)
        cv2.line(image, (int((x1 + x2) / 2), y2), (int((x1 + x2) / 2), ground_y), color, thickness)
        cv2.circle(image, (int((x1 + x2) / 2), ground_y), thickness * 2, color, -1)
        return image
    
    def cleanup_trackers(self, active_ids):
        active_ids_set = set(active_ids)
        for obj_id in list(self.kf_trackers.keys()):
            if obj_id not in active_ids_set:
                del self.kf_trackers[obj_id]
        for obj_id in list(self.box_history.keys()):
            if obj_id not in active_ids_set:
                del self.box_history[obj_id]