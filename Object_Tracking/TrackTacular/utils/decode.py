import torch
import torch.nn as nn

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def get_box_from_corners(corners):
    xmin = torch.min(corners[:, 0], dim=0, keepdim=True).values
    xmax = torch.max(corners[:, 0], dim=0, keepdim=True).values
    ymin = torch.min(corners[:, 1], dim=0, keepdim=True).values
    ymax = torch.max(corners[:, 1], dim=0, keepdim=True).values
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)

def get_alpha(rot):
    idx = (rot[:, 1] > rot[:, 5]).float()
    alpha1 = torch.arctan2(rot[:, 2], rot[:, 3]) + (-0.5 * torch.pi)
    alpha2 = torch.arctan2(rot[:, 6], rot[:, 7]) + (0.5 * torch.pi)
    return alpha1 * idx + alpha2 * (1 - idx)

def decoder(center_e, offset_e, size_e, rz_e=None, K=60):
    batch, cat, height, width = center_e.size()
    center_e = _nms(center_e)
    topk_scores, topk_inds = torch.topk(center_e.view(batch, cat, -1), K)
    topk_inds = topk_inds % (height * width)
    ys = (topk_inds / width).int().float()
    xs = (topk_inds % width).int().float()
    scores, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    clses = (topk_ind / K).int()
    offset = _transpose_and_gather_feat(offset_e, topk_ind)
    size = _transpose_and_gather_feat(size_e, topk_ind)
    if rz_e is not None:
        rz = _transpose_and_gather_feat(rz_e, topk_ind)
        rz = torch.stack([get_alpha(r) for r in rz])
    else:
        rz = torch.zeros_like(scores)
    ys = _gather_feat(ys.view(batch, -1, 1), topk_ind).view(batch, K)
    xs = _gather_feat(xs.view(batch, -1, 1), topk_ind).view(batch, K)
    xs = xs.view(batch, K, 1) + offset[:, :, 0:1]
    ys = ys.view(batch, K, 1) + offset[:, :, 1:2]
    xy = torch.cat((xs, ys), dim=2)
    xs_prev = xs.view(batch, K, 1) + offset[:, :, 2:3]
    ys_prev = ys.view(batch, K, 1) + offset[:, :, 3:4]
    xy_prev = torch.cat((xs_prev, ys_prev), dim=2)
    return xy.detach(), xy_prev.detach(), scores.detach(), clses.detach(), size.detach(), rz.detach()

def _topk(scores, K=40):
    batch, cat, length, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)
    return topk_score, topk_inds, topk_ys, topk_xs

def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
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