import logging
import os
import argparse
import sys
import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_setup, launch
from detectron2.data import transforms as T
logger = logging.getLogger("detectron2")
sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)
from cubercnn.config import get_cfg_defaults
from cubercnn.modeling.proposal_generator import RPNWithIgnore
from cubercnn.modeling.roi_heads import ROIHeads3D_Text
from cubercnn.modeling.meta_arch import RCNN3D_text, build_model
from cubercnn.modeling.backbone import build_dla_from_vision_fpn_backbone
from cubercnn import util, vis
from transformers import BertTokenizer, BertModel
import cv2

def get_color(obj_type):
    class_colors = {"person": (255, 0, 0), "forklift": (0, 255, 0), "novacarter": (0, 0, 255), "transporter": (255, 255, 0), "fouriergr1t2": (255, 0, 255), "agilitydigit": (0, 255, 255)}
    return class_colors.get(obj_type, (128, 128, 128))

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

def do_test(args, cfg, model, text_embeddings):
    list_of_ims = util.list_files(os.path.join(args.input_folder, ''), '*')
    model.eval()
    focal_length = args.focal_length
    principal_point = args.principal_point
    thres = args.threshold
    output_dir = cfg.OUTPUT_DIR
    min_size = cfg.INPUT.MIN_SIZE_TEST
    max_size = cfg.INPUT.MAX_SIZE_TEST
    augmentations = T.AugmentationList([T.ResizeShortestEdge(min_size, max_size, "choice")])
    util.mkdir_if_missing(output_dir)
    category_path = os.path.join(util.file_parts(args.config_file)[0], 'category_meta.json')
    if category_path.startswith(util.CubeRCNNHandler.PREFIX):
        category_path = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, category_path)
    metadata = util.load_json(category_path)
    cats = metadata['thing_classes']
    for path in list_of_ims:
        im_name = util.file_parts(path)[1]
        im = util.imread(path)
        if im is None:
            continue
        image_shape = im.shape[:2]
        h, w = image_shape
        if focal_length == 0:
            focal_length_ndc = 4.0
            focal_length = focal_length_ndc * h / 2
        if len(principal_point) == 0:
            px, py = w / 2, h / 2
        else:
            px, py = principal_point
        K = np.array([[focal_length, 0.0, px], [0.0, focal_length, py], [0.0, 0.0, 1.0]])
        aug_input = T.AugInput(im)
        _ = augmentations(aug_input)
        image = aug_input.image
        batched = [{'image': torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))).cuda(), 'height': image_shape[0], 'width': image_shape[1], 'K': K}]
        dets = model(batched, text_embeddings)[0]['instances']
        n_det = len(dets)
        meshes = []
        meshes_text = []
        # if n_det > 0:
        #     for idx, (corners3D, center_cam, center_2D, dimensions, pose, score, cat_idx) in enumerate(zip(dets.pred_bbox3D, dets.pred_center_cam, dets.pred_center_2D,
        #               dets.pred_dimensions, dets.pred_pose, dets.scores, dets.pred_classes)):
        #         if score < thres:
        #             continue
        #         cat = cats[cat_idx]
        #         bbox3D = center_cam.tolist() + dimensions.tolist()
        #         meshes_text.append('{} {:.2f}'.format(cat, score))
        #         color = [c / 255.0 for c in util.get_color(idx)]
        #         box_mesh = util.mesh_cuboid(bbox3D, pose.tolist(), color=color)
        #         meshes.append(box_mesh)
        # print('File: {} with {} dets'.format(im_name, len(meshes)))
        # if len(meshes) > 0:
        #     im_drawn_rgb, im_topdown, _ = vis.draw_scene_view(im, K, meshes, text=meshes_text, scale=im.shape[0], blend_weight=0.5, blend_weight_overlay=0.85)
        #     if args.display:
        #         im_concat = np.concatenate((im_drawn_rgb, im_topdown), axis=1)
        #         vis.imshow(im_concat)
        #     util.imwrite(im_drawn_rgb, os.path.join(output_dir, im_name + '_boxes.jpg'))
        #     util.imwrite(im_topdown, os.path.join(output_dir, im_name + '_novel.jpg'))
        # else:
        #     util.imwrite(im, os.path.join(output_dir, im_name + '_boxes.jpg'))
        if n_det > 0:
            for idx, (corners3D, center_cam, center_2D, dimensions, pose, score, cat_idx) in enumerate(
                    zip(dets.pred_bbox3D, dets.pred_center_cam, dets.pred_center_2D,
                        dets.pred_dimensions, dets.pred_pose, dets.scores, dets.pred_classes)):
                if score < thres:
                    continue
                cat = cats[cat_idx]
                meshes_text.append(cat)
                corners3D = corners3D.cpu().detach().numpy()
                meshes.append(corners3D)
        print('File: {} with {} dets'.format(im_name, len(meshes)))
        if len(meshes) > 0:
            for idx, (category_name, bbox_3D) in enumerate(zip(meshes_text, meshes)):
                color = get_color(category_name)
                projected_coords = projectCamera(bbox_3D, K)
                im = draw_3d_bbox(im, projected_coords, True, color)
                top_midpoint_2d = (projected_coords[0] + projected_coords[1]) / 2
                text_position = tuple(top_midpoint_2d.astype(int))
                cv2.putText(im, category_name, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            output_path = os.path.join(output_dir, im_name + '_boxes.jpg')
            cv2.imwrite(output_path, im)
        else:
            util.imwrite(im, os.path.join(output_dir, im_name + '_boxes.jpg'))

def setup(args):
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    get_cfg_defaults(cfg)
    config_file = args.config_file
    if config_file.startswith(util.CubeRCNNHandler.PREFIX):    
        config_file = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, config_file)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)
    ### register the categories ###
    category_path = os.path.join(util.file_parts(args.config_file)[0], 'category_meta.json')
    if category_path.startswith(util.CubeRCNNHandler.PREFIX):
        category_path = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, category_path)
    metadata = util.load_json(category_path)
    thing_classes = metadata['thing_classes']
    ### register the categories ###
    ### load the BERT model and obtain the text embeddings of the categories ###
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    thing_classes.append('None')
    texts = thing_classes
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    encoded_texts = outputs.last_hidden_state
    text_embeddings = encoded_texts[:, 1:-1, :].mean(dim=1).cuda().detach()
    ### load the BERT model and obtain the text embeddings of the categories ###
    ### start demo ###
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=True)
    with torch.no_grad():
        do_test(args, cfg, model, text_embeddings)
    ### start demo ###

if __name__ == "__main__":
    parser = argparse.ArgumentParser(epilog=None, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument('--input-folder',  type=str, help='list of image folders to process', required=True)
    parser.add_argument("--focal-length", type=float, default=0, help="focal length for image inputs (in px)")
    parser.add_argument("--principal-point", type=float, default=[], nargs=2, help="principal point for image inputs (in px)")
    parser.add_argument("--threshold", type=float, default=0.25, help="threshold on score for visualizing")
    parser.add_argument("--display", default=False, action="store_true", help="Whether to show the images in matplotlib",)
    parser.add_argument("--eval-only", default=True, action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument("--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)")
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:{}".format(port), help="initialization URL for pytorch distributed backend. See https://pytorch.org/docs/stable/distributed.html for details.")
    parser.add_argument("opts", help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. See config references at https://detectron2.readthedocs.io/modules/config.html#config-references", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    launch(main, args.num_gpus, num_machines=args.num_machines, machine_rank=args.machine_rank, dist_url=args.dist_url, args=(args,))