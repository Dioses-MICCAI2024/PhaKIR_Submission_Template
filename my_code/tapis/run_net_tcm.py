#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test a video classification model."""
from config.defaults import assert_and_infer_cfg
from utils.misc import launch_job
from utils.parser import load_config, parse_args

from test_net_tcm import test
import os

def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)
    cfg.OUTPUT_DIR = os.getenv('OUTPUT_DIR')
    cfg.ENDOVIS_DATASET.FRAME_DIR = os.getenv('TEST_DIR')
    cfg.ENDOVIS_DATASET.FRAME_LIST_DIR = os.path.join(os.getenv('OUTPUT_DIR'),'stuff_P')
    cfg.ENDOVIS_DATASET.ANNOTATION_DIR = os.path.join(os.getenv('OUTPUT_DIR'), 'stuff_P')
    cfg.FEATURES.TEST_FEATURES_PATH = os.path.join(os.getenv('OUTPUT_DIR'),'stuff_P','features.pth')

    # Perform multi-clip testing.
    launch_job(cfg=cfg, init_method=args.init_method, func=test)

if __name__ == "__main__":
    main()