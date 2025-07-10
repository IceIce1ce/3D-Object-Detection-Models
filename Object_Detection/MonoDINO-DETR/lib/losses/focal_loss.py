import torch
import torch.nn.functional as F

def focal_loss(input, target, alpha=0.25, gamma=2.):
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()
    loss = 0
    pos_loss = torch.log(input) * torch.pow(1 - input, gamma) * pos_inds * alpha
    neg_loss = torch.log(1 - input) * torch.pow(input, gamma) * neg_inds * (1 - alpha)
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss.mean()

def focal_loss_cornernet(input, target, gamma=2.):
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()
    neg_weights = torch.pow(1 - target, 4)
    loss = 0
    pos_loss = torch.log(input) * torch.pow(1 - input, gamma) * pos_inds
    neg_loss = torch.log(1 - input) * torch.pow(input, gamma) * neg_inds * neg_weights
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss.mean()

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(1).sum() / num_boxes