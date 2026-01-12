"""Slot computation utilities."""

from typing import Dict, List, Optional, Tuple


def get_target_first_token_id_from_boundary(
    prefix_str: str,
    answer_str: str,
    tokenizer,
    tokenize_kwargs: Optional[Dict[str, object]] = None,
) -> int:
    if tokenize_kwargs is None:
        tokenize_kwargs = {}
    prefix_ids = tokenizer(prefix_str, **tokenize_kwargs)["input_ids"]
    full_ids = tokenizer(prefix_str + answer_str, **tokenize_kwargs)["input_ids"]
    s = len(prefix_ids)
    if len(full_ids) <= s:
        raise ValueError("Full tokenization does not extend prefix tokens")
    if full_ids[:s] != prefix_ids:
        raise ValueError("Prefix tokens mismatch full tokens at boundary")
    return full_ids[s]


def get_dummy_token_labels_and_slot_map(
    n_icl_examples: int,
    special_prefix: Optional[List[str]] = None,
    special_suffix: Optional[List[str]] = None,
) -> Tuple[List[str], Dict[str, int]]:
    if n_icl_examples < 0:
        raise ValueError("n_icl_examples must be >= 0")
    dummy_labels: List[str] = []
    slot_index_map: Dict[str, int] = {}

    if special_prefix:
        for label in special_prefix:
            slot_index_map[label] = len(dummy_labels)
            dummy_labels.append(label)

    for i in range(n_icl_examples):
        dummy_labels.append("STATIC_Q")
        slot_index_map[f"DEMO_{i}_IN"] = len(dummy_labels)
        dummy_labels.append(f"DEMO_{i}_IN")
        dummy_labels.append("STATIC_A")
        slot_index_map[f"DEMO_{i}_OUT"] = len(dummy_labels)
        dummy_labels.append(f"DEMO_{i}_OUT")
        if i < n_icl_examples - 1:
            dummy_labels.append("STATIC_SEP")

    dummy_labels.append("STATIC_Q")
    slot_index_map["QUERY_IN"] = len(dummy_labels)
    dummy_labels.append("QUERY_IN")

    # QUERY_PRED aligns with the last input token (end of "A: " prefix).
    slot_index_map["QUERY_PRED"] = len(dummy_labels)
    dummy_labels.append("QUERY_PRED")

    slot_index_map["QUERY_OUT"] = len(dummy_labels)
    dummy_labels.append("QUERY_OUT")

    if special_suffix:
        for label in special_suffix:
            slot_index_map[label] = len(dummy_labels)
            dummy_labels.append(label)

    return dummy_labels, slot_index_map


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

    answer = full_str[len(prefix_str) :]
    boundary_prefix = prefix_str
    boundary_answer = answer
    if boundary_prefix.endswith(" ") and not boundary_answer.startswith(" "):
        boundary_prefix = boundary_prefix[:-1]
        boundary_answer = f" {boundary_answer}"
    target_id = get_target_first_token_id_from_boundary(
        boundary_prefix,
        boundary_answer,
        tokenizer,
        tokenize_kwargs={"add_special_tokens": add_special_tokens},
    )
    target_token = tokenizer.convert_ids_to_tokens(target_id)
    full_target_id = full_ids[s]
    full_target_token = tokenizer.convert_ids_to_tokens(full_target_id)

    return {
        "s": s,
        "slot_index": slot_index,
        "seq_len": seq_len,
        "target_id": target_id,
        "full_target_id": full_target_id,
        "full_target_token": full_target_token,
        "target_token": target_token,
        "input_ids": full_ids,
    }
