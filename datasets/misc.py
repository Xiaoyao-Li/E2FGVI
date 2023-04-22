from typing import Dict, List
import torch


def collate_fn_general(batch: List) -> Dict:
    """ General collate function used for dataloader.
    """
    batch_data = batch
    return batch_data


def collate_fn_epicsub(batch: List) -> Dict:
    """ Collate function used for dataloader.
    """
    batch_data = {"masked_imgs": [b['masked_imgs'] for b in batch], "outpaths": [b['outpaths'] for b in batch]}
    batch_data['masked_imgs'] = torch.cat(batch_data['masked_imgs'], dim=0)
    return batch_data
