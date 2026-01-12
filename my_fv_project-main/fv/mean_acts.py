"""Mean activation utilities for slot-based accumulation."""

from typing import Tuple


def extract_slot_activation(activation, slot_index: int) -> Tuple[object, object]:
    if activation is None or not hasattr(activation, "shape"):
        raise ValueError("Activation is not a tensor")
    if activation.dim() != 4:
        raise ValueError(f"Expected 4D activation, got {tuple(activation.shape)}")
    if activation.shape[0] != 1:
        raise ValueError(f"Expected batch size 1, got {activation.shape[0]}")
    if slot_index < 0 or slot_index >= activation.shape[1]:
        raise ValueError("slot_index out of range")
    captured = activation[:, slot_index, :, :].squeeze(0)
    return captured, activation.shape[1]
