import logging
import os
import sys
import numpy as np
import torch
import torch.distributed as dist
import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import default_argument_parser, default_setup
from detectron2.utils.logger import setup_logger
logger = logging.getLogger("cubercnn")
sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)
from cubercnn.solver import build_optimizer, freeze_bn, PeriodicCheckpointerOnlyOne
from cubercnn.config import get_cfg_defaults
from cubercnn.data import load_omni3d_json, DatasetMapper3D, build_detection_train_loader, build_detection_test_loader, get_omni3d_categories, simple_register
from cubercnn.evaluation import Omni3DEvaluator, Omni3Deval, Omni3DEvaluationHelper, inference_on_dataset
from cubercnn.modeling.proposal_generator import RPNWithIgnore
from cubercnn.modeling.roi_heads import ROIHeads3D_Text
from cubercnn.modeling.meta_arch import RCNN3D_text, build_model
from cubercnn.modeling.backbone import build_dla_from_vision_fpn_backbone
from cubercnn import util, data, generate_label
from cubercnn.generate_label import llm_generated_prior
import cubercnn.vis.logperf as utils_logperf

MAX_TRAINING_ATTEMPTS = 10

def generate_pseudo_label(cfg):
    dataset_names = cfg.DATASETS.TRAIN
    for dataset_name in dataset_names:
        dataset, mode = dataset_name.split('_')
        input_folder = f'pseudo_label/{dataset}/{mode}'
        output_folder = os.path.join(cfg.OUTPUT_DIR, dataset, mode)
        data_loader = build_detection_test_loader(cfg, dataset_name)
        cat_prior = llm_generated_prior[dataset]
        if dataset in ['SUNRGBD', 'ARKitScenes']:
            generate_label.process_indoor(data_loader.dataset, cat_prior, input_folder, output_folder)
        else:
            generate_label.process_outdoor(data_loader.dataset, cat_prior, input_folder, output_folder)

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
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="cubercnn")
    filter_settings = data.get_filter_settings_from_cfg(cfg)
    for dataset_name in cfg.DATASETS.TRAIN:
        simple_register(dataset_name, filter_settings, filter_empty=True)
    dataset_names_test = cfg.DATASETS.TEST
    for dataset_name in dataset_names_test:
        if not(dataset_name in cfg.DATASETS.TRAIN):
            simple_register(dataset_name, filter_settings, filter_empty=False)
    return cfg

def main(args):
    cfg = setup(args)
    logger.info('Preprocessing Training Datasets')
    filter_settings = data.get_filter_settings_from_cfg(cfg)
    dataset_paths = [os.path.join('datasets', 'Omni3D', name + '.json') for name in cfg.DATASETS.TRAIN]
    datasets = data.Omni3D(dataset_paths, filter_settings=filter_settings)
    data.register_and_store_model_metadata(datasets, cfg.OUTPUT_DIR, filter_settings)
    thing_classes = MetadataCatalog.get('omni3d_model').thing_classes
    dataset_id_to_contiguous_id = MetadataCatalog.get('omni3d_model').thing_dataset_id_to_contiguous_id
    infos = datasets.dataset['info']
    if type(infos) == dict:
        infos = [datasets.dataset['info']]
    dataset_id_to_unknown_cats = {}
    possible_categories = set(i for i in range(cfg.MODEL.ROI_HEADS.NUM_CLASSES + 1))
    dataset_id_to_src = {}
    for info in infos:
        dataset_id = info['id']
        known_category_training_ids = set()
        if not dataset_id in dataset_id_to_src:
            dataset_id_to_src[dataset_id] = info['source']
        for id in info['known_category_ids']:
            if id in dataset_id_to_contiguous_id:
                known_category_training_ids.add(dataset_id_to_contiguous_id[id])
        unknown_categories = possible_categories - known_category_training_ids
        dataset_id_to_unknown_cats[dataset_id] = unknown_categories
        logger.info('Available categories for {}'.format(info['name']))
        logger.info([thing_classes[i] for i in (possible_categories & known_category_training_ids)])
    generate_pseudo_label(cfg)

def allreduce_dict(input_dict, average=True):
    world_size = comm.get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)