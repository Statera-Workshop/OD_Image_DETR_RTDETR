"""Minimal stand-ins for detr/util/misc.py (single process, fixed-size batches without padding masks)."""

import torch


def nested_tensor_from_tensor_list(tensor_list):
    raise NotImplementedError("padding masks are not used in this framework (fixed 608 x 608 input)")


def interpolate(*args, **kwargs):
    return torch.nn.functional.interpolate(*args, **kwargs)
