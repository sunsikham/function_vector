"""Slot computation utilities."""

from typing import Dict


def compute_query_predictive_slot(
    prefix_str: str,
    full_str: str,
    tokenizer,
    add_special_tokens: bool = False,
) -> Dict[str, object]:
    prefix_ids = tokenizer.encode(prefix_str, add_special_tokens=add_special_tokens)
    full_ids = tokenizer.encode(full_str, add_special_tokens=add_special_tokens)

    s = len(prefix_ids)
    if s >= len(full_ids):
        trimmed_prefix = prefix_str.rstrip(" ")
        if trimmed_prefix != prefix_str:
            prefix_ids = tokenizer.encode(
                trimmed_prefix, add_special_tokens=add_special_tokens
            )
            s = len(prefix_ids)
    if s <= 0 or s >= len(full_ids):
        raise ValueError(
            f"Invalid prefix length for slot computation (s={s}, seq_len={len(full_ids)})"
        )

    slot_index = s - 1
    seq_len = len(full_ids)
    if not (0 <= slot_index < seq_len):
        raise ValueError("slot_index out of range")

    target_id = full_ids[s]
    target_token = tokenizer.convert_ids_to_tokens(target_id)
    answer = full_str[len(prefix_str) :]
    answer_with_space = answer if answer.startswith(" ") else f" {answer}"
    expected_ids = tokenizer.encode(answer_with_space, add_special_tokens=False)
    if expected_ids and target_id != expected_ids[0]:
        expected_token = tokenizer.convert_ids_to_tokens(expected_ids[0])
        raise ValueError(
            "Target id mismatch for first answer token: "
            f"expected_id={expected_ids[0]} expected_token={expected_token} "
            f"got_id={target_id} got_token={target_token}"
        )

    return {
        "s": s,
        "slot_index": slot_index,
        "seq_len": seq_len,
        "target_id": target_id,
        "target_token": target_token,
        "input_ids": full_ids,
    }
