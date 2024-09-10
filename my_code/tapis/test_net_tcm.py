#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import traceback
import numpy as np
import torch
from tqdm import tqdm

import utils.checkpoint as cu
import utils.logging as logging
import utils.misc as misc
from datasets import loader
from models import build_model
from config.defaults import assert_and_infer_cfg
from utils.misc import launch_job
from utils.parser import load_config, parse_args
from utils.compute_all_metrics import evaluate
from copy import copy
import os
import pycocotools.mask as m
import cv2

import matplotlib.pyplot as plt

logger = logging.get_logger(__name__)

@torch.no_grad()
def eval_epoch(val_loader, model, cfg):
    """
    Evaluate the model on the test set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """

    map_categories = {0: 'Preparation',
                      1: 'CalotTriangleDissection',
                      2: 'ClippingCutting',
                      3: 'GallbladderDissection',
                      4: 'GallbladderRetraction',
                      5: 'CleaningCoagulation',
                      6: 'GallbladderPackaging'}

    # Evaluation mode enabled.
    preds_dict = {}
    model.eval()

    for (inputs, labels, data, image_names) in tqdm(val_loader, desc='Temporal Consistency Module Predictions [PHASES RECOGNITION] ...'):

        video_name = image_names[0].split('/')[0]
        image_name = int(image_names[0].split('/')[1].split('.')[0].split('_')[-1]) 
        
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            if isinstance(inputs[0][0], (list,)):
                for i in range(len(inputs)):
                    for j in range(len(inputs[i])):
                        inputs[i][j] = inputs[i][j].cuda(non_blocking=True)
            else:
                for i in range(len(inputs)):
                    for j in range(len(inputs[i])):
                        inputs[i][j] = inputs[i][j].cuda(non_blocking=True)

        #Predictions for the "gestures" (action recognition) and "tools" (instrument segmentation) tasks
        preds = model(inputs)['phases'].cpu().numpy()
        phase = map_categories[np.argmax(preds, axis=1).item()]

        if video_name not in preds_dict:
            preds_dict[video_name] = {}
            preds_dict[video_name][image_name] = phase

        else:
            preds_dict[video_name][image_name] = phase
        
    # Save predictions in .csv for each video
    for video_name in preds_dict.keys():
        # Create directory for each video
        if not os.path.exists(os.path.join(cfg.OUTPUT_DIR, video_name)):
            os.makedirs(os.path.join(cfg.OUTPUT_DIR, video_name))
        with open(os.path.join(cfg.OUTPUT_DIR, video_name, video_name + '_Phases.csv'), 'w') as f:
            f.write('Frame,Phase\n')
            for frame in sorted(preds_dict[video_name].keys()):
                f.write(f'{frame},{preds_dict[video_name][frame]}\n')

def test(cfg):
    """
    Perform testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            config/defaults.py
    """
    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Build the video model and print model statistics.
    model = build_model(cfg)

    print(cfg.MODEL.ARCH)
    print(model)

    if cfg.NUM_GPUS:
        model = model.cuda()

    # Load checkpoints
    cu.load_test_checkpoint(cfg, model)

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "val")
    logger.info("Testing model for {} iterations".format(len(test_loader)))
    logger.info("Data taken from {}".format(os.getenv('TEST_DIR')))

    # # Perform test on the entire dataset.
    with torch.no_grad():
        eval_epoch(test_loader, model, cfg)

def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)

    # Perform multi-clip testing.
    if cfg.TEST.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=test)

if __name__ == "__main__":
    main()