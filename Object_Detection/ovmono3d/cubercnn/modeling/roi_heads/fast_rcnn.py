import torch
from typing import List, Tuple
from fvcore.nn import giou_loss, smooth_l1_loss
from detectron2.layers import cat, cross_entropy, nonzero_tuple, batched_nms
from detectron2.structures import Instances, Boxes
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, _log_classification_stats

def fast_rcnn_inference(boxes: List[torch.Tensor], scores: List[torch.Tensor], image_shapes: List[Tuple[int, int]], score_thresh: float, nms_thresh: float, topk_per_image: int):
    result_per_image = [fast_rcnn_inference_single_image(boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image)
                        for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes)]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]

def fast_rcnn_inference_single_image(boxes, scores, image_shape: Tuple[int, int], score_thresh: float, nms_thresh: float, topk_per_image: int):
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)
    filter_mask = scores > score_thresh
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores_full = scores[filter_inds[:, 0]]
    scores = scores[filter_mask]
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds, scores_full = boxes[keep], scores[keep], filter_inds[keep], scores_full[keep]
    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.scores_full = scores_full
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]

class FastRCNNOutputs(FastRCNNOutputLayers):
    def inference(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]):
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference(boxes, scores, image_shapes, self.test_score_thresh, self.test_nms_thresh, self.test_topk_per_image)

    def losses(self, predictions, proposals):
        scores, proposal_deltas = predictions
        gt_classes = cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            gt_boxes = cat([(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals], dim=0)
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)
        normalize_factor = max(gt_classes.numel(), 1.0)
        _log_classification_stats(scores, gt_classes)
        loss_cls = cross_entropy(scores, gt_classes, reduction="mean")
        loss_box_reg = self.box_reg_loss(proposal_boxes, gt_boxes, proposal_deltas, gt_classes, reduction="none")
        loss_box_reg = (loss_box_reg).sum() / normalize_factor
        losses = {"BoxHead/loss_cls": loss_cls, "BoxHead/loss_box_reg": loss_box_reg}
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
    
    def box_reg_loss(self, proposal_boxes, gt_boxes, pred_deltas, gt_classes, reduction='mean'):
        box_dim = proposal_boxes.shape[1]
        fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]
        if pred_deltas.shape[1] == box_dim:
            fg_pred_deltas = pred_deltas[fg_inds]
        else:
            fg_pred_deltas = pred_deltas.view(-1, self.num_classes, box_dim)[fg_inds, gt_classes[fg_inds]]
        if reduction == 'mean':
            if self.box_reg_loss_type == "smooth_l1":
                gt_pred_deltas = self.box2box_transform.get_deltas(proposal_boxes[fg_inds], gt_boxes[fg_inds])
                loss_box_reg = smooth_l1_loss(fg_pred_deltas, gt_pred_deltas, self.smooth_l1_beta, reduction="sum")
            elif self.box_reg_loss_type == "giou":
                fg_pred_boxes = self.box2box_transform.apply_deltas(fg_pred_deltas, proposal_boxes[fg_inds])
                loss_box_reg = giou_loss(fg_pred_boxes, gt_boxes[fg_inds], reduction="sum")
            else:
                raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")
            return loss_box_reg / max(gt_classes.numel(), 1.0)
        elif reduction == 'none':
            if self.box_reg_loss_type == "smooth_l1":
                gt_pred_deltas = self.box2box_transform.get_deltas(proposal_boxes[fg_inds], gt_boxes[fg_inds])
                loss_box_reg = smooth_l1_loss(fg_pred_deltas, gt_pred_deltas, self.smooth_l1_beta, reduction="none")
            else:
                raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")
            return loss_box_reg
        else:
            raise ValueError(f"Invalid bbox reg reduction type '{reduction}'")