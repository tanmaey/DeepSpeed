# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from deepspeed.utils import logger, log_dist
from deepspeed.runtime.checkpoint_engine.checkpoint_engine import \
    CheckpointEngine

def _to_cpu(ele, snapshot=None):
	#while True:
    if snapshot is None:
            snapshot = {}

    if hasattr(ele, 'cpu'):
            snapshot = ele.cpu()
    elif isinstance(ele, tuple):
        ele=[0 if x is None else x for x in ele]
        _to_cpu(ele)
        # print(type(ele))
        # # print(ele)
        # tensortemp=torch.tensor(ele)
        # snapshot=tensortemp.cpu()
    elif isinstance(ele, dict):
        snapshot = {}
        for k,v in ele.items():
                snapshot[k] = None
                snapshot[k] = _to_cpu(v, snapshot[k])
    elif isinstance(ele, list):
        snapshot  = [None for _ in range(len(ele))]
        for idx, v in enumerate(ele):
                snapshot[idx] = _to_cpu(v, snapshot[idx])

class TorchCheckpointEngine(CheckpointEngine):

    def __init__(self, config_params=None):
        super().__init__(config_params)

    def create(self, tag):
        log_dist(f"[Torch] Checkpoint {tag} is about to be saved!", ranks=[0])

    def save(self, state_dict, path: str):
        logger.info(f"[Torch] Saving {path}...")
        snapshot={}
        for name, ref in state_dict.items():

            snapshot[name] = _to_cpu(ref)
        # torch.save(state_dict, path)
        logger.info(f"[Torch] Saved {path}.")
        return None

    def load(self, path: str, map_location=None):
        logger.info(f"[Torch] Loading checkpoint from {path}...")
        partition = torch.load(path, map_location=map_location)
        logger.info(f"[Torch] Loaded checkpoint from {path}.")
        return partition

    def commit(self, tag):
        logger.info(f"[Torch] Checkpoint {tag} is ready now!")
        return True
