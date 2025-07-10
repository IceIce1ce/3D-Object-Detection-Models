import numpy as np
import torch
import torch.nn as nn
from lib.datasets.utils import class2angle
from utils import box_ops

def decode_detections(dets, info, calibs, cls_mean_size, threshold):
    results = {}
    for i in range(dets.shape[0]):
        preds = []
        for j in range(dets.shape[1]):
            cls_id = int(dets[i, j, 0])
            score = dets[i, j, 1]
            if score < threshold:
                continue
            x = dets[i, j, 2] * info['img_size'][i][0]
            y = dets[i, j, 3] * info['img_size'][i][1]
            w = dets[i, j, 4] * info['img_size'][i][0]
            h = dets[i, j, 5] * info['img_size'][i][1]
            bbox = [x-w/2, y-h/2, x+w/2, y+h/2]
            depth = dets[i, j, 6]
            dimensions = dets[i, j, 31:34]
            dimensions += cls_mean_size[int(cls_id)]
            x3d = dets[i, j, 34] * info['img_size'][i][0]
            y3d = dets[i, j, 35] * info['img_size'][i][1]
            locations = calibs[i].img_to_rect(x3d, y3d, depth).reshape(-1)
            locations[1] += dimensions[0] / 2
            alpha = get_heading_angle(dets[i, j, 7:31])
            ry = calibs[i].alpha2ry(alpha, x)
            score = score * dets[i, j, -1]
            preds.append([cls_id, alpha] + bbox + dimensions.tolist() + locations.tolist() + [ry, score])
        results[info['img_id'][i]] = preds
    return results

def extract_dets_from_outputs(outputs, K=50, topk=50):
    out_logits = outputs['pred_logits']
    out_bbox = outputs['pred_boxes']
    prob = out_logits.sigmoid()
    topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), topk, dim=1)
    scores = topk_values
    topk_boxes = (topk_indexes // out_logits.shape[2]).unsqueeze(-1)
    labels = topk_indexes % out_logits.shape[2]
    heading = outputs['pred_angle']
    size_3d = outputs['pred_3d_dim']
    depth = outputs['pred_depth'][:, :, 0: 1]
    sigma = outputs['pred_depth'][:, :, 1: 2]
    sigma = torch.exp(-sigma)
    boxes = torch.gather(out_bbox, 1, topk_boxes.repeat(1, 1, 6))
    xs3d = boxes[:, :, 0: 1] 
    ys3d = boxes[:, :, 1: 2]
    heading = torch.gather(heading, 1, topk_boxes.repeat(1, 1, 24))
    depth = torch.gather(depth, 1, topk_boxes)
    sigma = torch.gather(sigma, 1, topk_boxes) 
    size_3d = torch.gather(size_3d, 1, topk_boxes.repeat(1, 1, 3))
    corner_2d = box_ops.box_cxcylrtb_to_xyxy(boxes)
    xywh_2d = box_ops.box_xyxy_to_cxcywh(corner_2d)
    size_2d = xywh_2d[:, :, 2: 4]
    xs2d = xywh_2d[:, :, 0: 1]
    ys2d = xywh_2d[:, :, 1: 2]
    batch = out_logits.shape[0]
    labels = labels.view(batch, -1, 1)
    scores = scores.view(batch, -1, 1)
    xs2d = xs2d.view(batch, -1, 1)
    ys2d = ys2d.view(batch, -1, 1)
    xs3d = xs3d.view(batch, -1, 1)
    ys3d = ys3d.view(batch, -1, 1)
    detections = torch.cat([labels, scores, xs2d, ys2d, size_2d, depth, heading, size_3d, xs3d, ys3d, sigma], dim=2)
    return detections

def _nms(heatmap, kernel=3):
    padding = (kernel - 1) // 2
    heatmapmax = nn.functional.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=padding)
    keep = (heatmapmax == heatmap).float()
    return heatmap * keep

def _topk(heatmap, K=50):
    batch, cat, height, width = heatmap.size()
    topk_scores, topk_inds = torch.topk(heatmap.view(batch, cat, -1), K)
    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_cls_ids = (topk_ind / K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)
    return topk_score, topk_inds, topk_cls_ids, topk_xs, topk_ys

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def get_heading_angle(heading):
    heading_bin, heading_res = heading[0:12], heading[12:24]
    cls = np.argmax(heading_bin)
    res = heading_res[cls]
    return class2angle(cls, res, to_label_format=True)