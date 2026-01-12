"""Mean activation utilities with slot alignment."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from .prompting import build_prompt_qa
from .slots import get_dummy_token_labels_and_slot_map


def _build_prompt_segments(
    demos: Sequence[Tuple[str, str]],
    query: Tuple[str, str],
) -> List[Tuple[str, str]]:
    segments: List[Tuple[str, str]] = []
    for i, (demo_in, demo_out) in enumerate(demos):
        segments.append(("STATIC_Q", "Q:"))
        segments.append((f"DEMO_{i}_IN", f" {demo_in}"))
        segments.append(("STATIC_A", "\nA:"))
        segments.append((f"DEMO_{i}_OUT", f" {demo_out}"))
        segments.append(("STATIC_SEP", "\n\n"))
    segments.append(("STATIC_Q", "Q:"))
    segments.append(("QUERY_IN", f" {query[0]}"))
    segments.append(("QUERY_PRED", "\nA:"))
    segments.append(("QUERY_OUT", f" {query[1]}"))
    return segments


def _segments_to_text_and_spans(
    segments: Sequence[Tuple[str, str]],
) -> Tuple[str, List[Tuple[str, int, int]]]:
    parts: List[str] = []
    spans: List[Tuple[str, int, int]] = []
    cursor = 0
    for label, text in segments:
        start = cursor
        end = start + len(text)
        parts.append(text)
        spans.append((label, start, end))
        cursor = end
    return "".join(parts), spans


def _infer_special_token_labels(tokenizer, input_ids, offsets):
    special_labels: Dict[int, str] = {}
    for idx, (start, end) in enumerate(offsets):
        if start != end:
            continue
        token_id = input_ids[idx]
        if tokenizer.bos_token_id is not None and token_id == tokenizer.bos_token_id:
            special_labels[idx] = "SPECIAL_BOS"
        elif tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            special_labels[idx] = "SPECIAL_EOS"
        else:
            special_labels[idx] = "SPECIAL"
    return special_labels


def get_token_meta_labels(
    tokenizer,
    full_str: str,
    segments: Sequence[Tuple[str, str]],
    tok_add_special: bool,
    special_index_labels: Dict[int, str],
) -> Tuple[List[str], List[int], List[Tuple[int, int]]]:
    encoded = tokenizer(
        full_str,
        return_offsets_mapping=True,
        add_special_tokens=tok_add_special,
    )
    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]

    _, spans = _segments_to_text_and_spans(segments)
    token_labels: List[str] = []
    span_idx = 0
    for idx, (start, end) in enumerate(offsets):
        if idx in special_index_labels:
            token_labels.append(special_index_labels[idx])
            continue
        while span_idx < len(spans) and start >= spans[span_idx][2]:
            span_idx += 1
        if span_idx >= len(spans):
            raise ValueError("Token offset exceeds prompt span length")
        label, span_start, span_end = spans[span_idx]
        if start < span_start or start >= span_end:
            raise ValueError("Token offset not aligned to prompt spans")
        token_labels.append(label)

    return token_labels, input_ids, offsets


def compute_duplicated_labels(
    token_meta_labels: Sequence[str],
    dummy_labels: Sequence[str],
    allow_empty_labels: Optional[Sequence[str]] = None,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    idx_map: Dict[int, List[int]] = {}
    idx_avg: Dict[int, List[int]] = {}
    cursor = 0
    n_tokens = len(token_meta_labels)
    allow_empty = set(allow_empty_labels or [])
    for slot_idx, slot_label in enumerate(dummy_labels):
        if cursor >= n_tokens or token_meta_labels[cursor] != slot_label:
            if slot_label in allow_empty and slot_label not in token_meta_labels[cursor:]:
                idx_map[slot_idx] = []
                continue
            while cursor < n_tokens and token_meta_labels[cursor] != slot_label:
                cursor += 1
            if cursor >= n_tokens:
                raise ValueError(f"Failed to align slot label: {slot_label}")
        start = cursor
        while cursor < n_tokens and token_meta_labels[cursor] == slot_label:
            cursor += 1
        span = list(range(start, cursor))
        idx_map[slot_idx] = span
        if len(span) > 1:
            idx_avg[slot_idx] = span
    return idx_map, idx_avg


def apply_idx_map_to_activations(
    acts,
    idx_map: Dict[int, List[int]],
    n_slots: int,
):
    import torch

    if acts.dim() != 4:
        raise ValueError("Expected activations with shape (L,T,H,D)")
    n_layers, _seq_len, n_heads, head_dim = acts.shape
    slot_acts = torch.zeros(
        (n_layers, n_heads, n_slots, head_dim),
        device=acts.device,
        dtype=acts.dtype,
    )
    for slot_idx, token_indices in idx_map.items():
        if not token_indices:
            continue
        token_act = acts[:, token_indices, :, :]
        if len(token_indices) > 1:
            slot_act = token_act.mean(dim=1)
        else:
            slot_act = token_act[:, 0, :, :]
        slot_acts[:, :, slot_idx, :] = slot_act
    return slot_acts


def compute_mean_activations_ns(
    model,
    tokenizer,
    layer_modules: Dict[int, object],
    pairs,
    n_icl_examples: int,
    n_mean_trials: int,
    model_cfg: Dict[str, int],
    seed: int,
    tok_add_special: bool,
    device,
    shuffle_labels: bool = False,
    allow_alignment_skip: bool = False,
    logger=None,
):
    import torch

    if n_mean_trials < 1:
        raise ValueError("n_mean_trials must be >= 1")

    n_heads = int(model_cfg["n_heads"])
    head_dim = int(model_cfg["head_dim"])
    resid_dim = int(model_cfg["resid_dim"])
    n_layers = int(model_cfg["n_layers"])

    layers = sorted(layer_modules.keys())
    if not layers:
        raise ValueError("No layers provided for mean activation computation")

    def log(message: str) -> None:
        if logger is None:
            return
        if callable(logger):
            logger(message)
        elif hasattr(logger, "info"):
            logger.info(message)

    demo_seed_base = seed

    dummy_labels = None
    slot_index_map = None
    special_prefix = None
    special_suffix = None
    special_index_labels_ref = None
    avg_logged = False
    align_logged = False
    check_limit = 5
    skipped = 0

    mean = torch.zeros(
        (n_layers, n_heads, 0, head_dim),
        dtype=torch.float32,
        device="cpu",
    )
    count = 0

    for trial_idx in range(n_mean_trials):
        from .dataset_loader import sample_demos_and_query

        demos_orig, query = sample_demos_and_query(
            pairs, n_icl_examples, seed=demo_seed_base + trial_idx
        )
        if shuffle_labels:
            import random

            rng = random.Random(seed + trial_idx)
            perm = list(range(len(demos_orig)))
            rng.shuffle(perm)
            outputs = [y for _x, y in demos_orig]
            shuffled = [outputs[i] for i in perm]
            demos = [(demos_orig[i][0], shuffled[i]) for i in range(len(demos_orig))]
        else:
            demos = demos_orig
        prefix_str, full_str = build_prompt_qa(demos, query)

        segments = _build_prompt_segments(demos, query)
        full_str_segments, _spans = _segments_to_text_and_spans(segments)
        if full_str_segments != full_str:
            raise ValueError("Prompt builder mismatch with SSOT")

        encoded = tokenizer(
            full_str,
            return_offsets_mapping=True,
            add_special_tokens=tok_add_special,
        )
        input_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        if trial_idx < check_limit:
            ids_check = tokenizer(
                full_str,
                add_special_tokens=tok_add_special,
            )["input_ids"]
            if input_ids != ids_check:
                raise ValueError("Tokenizer output mismatch for prompt")

        special_index_labels = _infer_special_token_labels(
            tokenizer, input_ids, offsets
        )
        special_positions = sorted(special_index_labels.keys())
        if special_positions:
            if special_positions[0] != 0:
                raise ValueError("Special token in unexpected position")
            if special_positions[-1] != len(input_ids) - 1 and len(special_positions) > 1:
                raise ValueError("Special token not in prefix/suffix position")

        if dummy_labels is None:
            special_prefix = []
            special_suffix = []
            if special_positions and special_positions[0] == 0:
                special_prefix.append(special_index_labels[special_positions[0]])
            if special_positions and special_positions[-1] == len(input_ids) - 1:
                if special_positions[-1] != special_positions[0]:
                    special_suffix.append(special_index_labels[special_positions[-1]])
            dummy_labels, slot_index_map = get_dummy_token_labels_and_slot_map(
                n_icl_examples,
                special_prefix=special_prefix,
                special_suffix=special_suffix,
            )
            n_slots = len(dummy_labels)
            mean = torch.zeros(
                (n_layers, n_heads, n_slots, head_dim),
                dtype=torch.float32,
                device="cpu",
            )
            special_index_labels_ref = special_index_labels
        else:
            if special_index_labels != special_index_labels_ref:
                raise ValueError("Special token alignment mismatch across trials")

        token_meta_labels, _input_ids, _offsets = get_token_meta_labels(
            tokenizer,
            full_str,
            segments,
            tok_add_special,
            special_index_labels,
        )
        idx_map, idx_avg = compute_duplicated_labels(
            token_meta_labels,
            dummy_labels,
            allow_empty_labels=["QUERY_OUT"],
        )
        if len(idx_map) != len(dummy_labels):
            raise ValueError("idx_map missing slots")
        if sorted(idx_map.keys()) != list(range(len(dummy_labels))):
            raise ValueError("idx_map keys are not contiguous")
        if idx_avg and not avg_logged:
            log("multi-token slot detected; applying mean over span")
            avg_logged = True
        for slot_idx, token_indices in idx_map.items():
            if dummy_labels[slot_idx] == "QUERY_OUT":
                continue
            if not token_indices:
                raise ValueError("Empty token span for slot")
        prefix_ids = tokenizer.encode(prefix_str, add_special_tokens=tok_add_special)
        if not prefix_ids:
            raise ValueError("Prefix tokenization failed")
        prefix_last_idx = len(prefix_ids) - 1
        slot_q = slot_index_map["QUERY_PRED"]
        if prefix_last_idx not in idx_map.get(slot_q, []):
            if allow_alignment_skip:
                skipped += 1
                log(
                    "alignment_skip: "
                    f"trial={trial_idx} prefix_last_idx={prefix_last_idx} "
                    f"slot_q={slot_q} span={idx_map.get(slot_q, [])}"
                )
                continue
            raise ValueError("QUERY_PRED does not align with last token")
        if not align_logged:
            log(
                "QUERY_PRED aligns with last token: True "
                f"(last_idx={prefix_last_idx} span={idx_map.get(slot_q, [])})"
            )
            align_logged = True

        hook_state: Dict[int, torch.Tensor] = {}
        errors: List[str] = []

        def make_pre_hook(layer_idx: int):
            def pre_hook(_module, inputs):
                tensor = inputs[0] if inputs else None
                if tensor is None:
                    errors.append("Missing out_proj input")
                    return
                hook_state[layer_idx] = tensor.detach()

            return pre_hook

        handles = []
        for layer_idx, module in layer_modules.items():
            handles.append(module.register_forward_pre_hook(make_pre_hook(layer_idx)))

        inputs = tokenizer(
            full_str,
            return_tensors="pt",
            add_special_tokens=tok_add_special,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.inference_mode():
            _ = model(**inputs)

        for handle in handles:
            handle.remove()

        if errors:
            raise ValueError("; ".join(errors))

        layer_acts: List[torch.Tensor] = []
        for layer_idx in layers:
            if layer_idx not in hook_state:
                raise ValueError("Missing activation for layer")
            x = hook_state[layer_idx]
            if x.dim() != 3 or x.shape[-1] != resid_dim:
                raise ValueError("Activation shape mismatch")
            batch_size, seq_len, _hidden = x.shape
            if batch_size != 1:
                raise ValueError("Expected batch size 1 during mean computation")
            x_heads = x.reshape(batch_size, seq_len, n_heads, head_dim)
            layer_acts.append(x_heads.squeeze(0))

        acts = torch.stack(layer_acts, dim=0)
        slot_acts = apply_idx_map_to_activations(acts, idx_map, len(dummy_labels))
        slot_acts_cpu = slot_acts.to(dtype=torch.float32, device="cpu")

        mean_layers = mean[layers]
        mean[layers] = mean_layers + (slot_acts_cpu - mean_layers) / (count + 1)
        count += 1

    if skipped:
        log(f"alignment_skipped_trials={skipped}")
    if count == 0:
        raise ValueError("No trials counted for mean_activations")

    return mean, dummy_labels, slot_index_map
