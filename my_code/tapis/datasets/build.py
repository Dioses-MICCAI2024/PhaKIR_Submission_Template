#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from datasets.phakir import Phakirms, Phakir_transformer

def build_dataset(dataset_name, cfg, split):
    """
    Build a dataset, defined by `dataset_name`.
    Args:
        dataset_name (str): the name of the dataset to be constructed.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    Returns:
        Dataset: a constructed dataset specified by dataset_name.
    """
    # Capitalize the the first letter of the dataset_name since the dataset_name
    # in configs may be in lowercase but the name of dataset class should always
    # start with an uppercase letter.
    # Sar_rarp(cfg, split)
    # name = dataset_name.capitalize()
    if dataset_name == "phakirms":
        return Phakirms(cfg, split) #DATASET_REGISTRY.get(name)(cfg, split)
    
    elif dataset_name == "Phakir_transformer":
        return Phakir_transformer(cfg, split)
    
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported.")
