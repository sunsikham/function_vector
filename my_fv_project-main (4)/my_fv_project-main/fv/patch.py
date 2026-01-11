"""Patch utilities for head-level replacement in c_proj hooks."""

from typing import Optional, Sequence


def _log_once(logger, state, message: str) -> None:
    if state.get("logged"):
        return
    state["logged"] = True
    if logger is None:
        return
    if callable(logger):
        logger(message)
        return
    if hasattr(logger, "info"):
        logger.info(message)
        return
    if hasattr(logger, "write"):
        logger.write(message + "\n")
        if hasattr(logger, "flush"):
            logger.flush()


def _validate_model_config(model_config: dict) -> None:
    if model_config is None:
        raise ValueError("model_config is required")
    for key in ("n_heads", "head_dim", "resid_dim"):
        if key not in model_config:
            raise ValueError(f"model_config missing '{key}'")
    if model_config["n_heads"] * model_config["head_dim"] != model_config["resid_dim"]:
        raise ValueError("model_config resid_dim must equal n_heads * head_dim")


def _normalize_token_index(token_idx: int, seq_len: int) -> int:
    if token_idx < 0:
        token_idx = seq_len + token_idx
    return token_idx


def _normalize_token_indices(token_indices: Sequence[int], seq_len: int) -> list[int]:
    if not token_indices:
        raise ValueError("token_indices cannot be empty")
    normalized = []
    for token_idx in token_indices:
        t_idx = _normalize_token_index(int(token_idx), seq_len)
        if t_idx < 0 or t_idx >= seq_len:
            raise ValueError("token_idx out of range")
        normalized.append(t_idx)
    return normalized


def _normalize_replace_vec(replace_vec, batch_size: int, head_dim: int, ref_tensor):
    import torch

    vec = torch.as_tensor(replace_vec, device=ref_tensor.device, dtype=ref_tensor.dtype)
    if vec.dim() == 1:
        if vec.shape[0] != head_dim:
            raise ValueError("replace_vec has wrong head_dim")
        vec = vec.view(1, head_dim).expand(batch_size, head_dim)
        return vec
    if vec.dim() == 2:
        if vec.shape[1] != head_dim:
            raise ValueError("replace_vec has wrong head_dim")
        if vec.shape[0] == 1:
            return vec.expand(batch_size, head_dim)
        if vec.shape[0] == batch_size:
            return vec
        raise ValueError("replace_vec batch mismatch")
    raise ValueError("replace_vec must be shape (D,) or (B,D)")


def _normalize_replace_vecs(
    replace_vecs,
    token_count: int,
    batch_size: int,
    head_dim: int,
    ref_tensor,
):
    if isinstance(replace_vecs, (list, tuple)):
        if len(replace_vecs) != token_count:
            raise ValueError("replace_vecs length mismatch")
        return [
            _normalize_replace_vec(vec, batch_size, head_dim, ref_tensor)
            for vec in replace_vecs
        ]
    return [
        _normalize_replace_vec(replace_vecs, batch_size, head_dim, ref_tensor)
        for _ in range(token_count)
    ]


def make_cproj_head_replacer(
    layer_idx: int,
    head_idx: int,
    token_idx: int,
    mode: str,
    replace_vec,
    model_config: dict,
    logger=None,
):
    """Create a forward_pre_hook to replace a single head vector.

    Signature:
        make_cproj_head_replacer(
            layer_idx, head_idx, token_idx, mode, replace_vec, model_config, logger=None
        )

    Example:
        hook = make_cproj_head_replacer(
            layer_idx=0,
            head_idx=3,
            token_idx=-1,
            mode="replace",
            replace_vec=my_vec,
            model_config=get_model_config("gpt2"),
            logger=print,
        )
    """

    _validate_model_config(model_config)
    n_heads = int(model_config["n_heads"])
    head_dim = int(model_config["head_dim"])
    resid_dim = int(model_config["resid_dim"])

    if mode not in ("replace", "self"):
        raise ValueError("mode must be 'replace' or 'self'")
    if head_idx < 0 or head_idx >= n_heads:
        raise ValueError("head_idx out of range")

    log_state = {"logged": False}

    def hook_fn(_module, inputs):
        if not inputs:
            return None
        x = inputs[0]
        if x is None or not hasattr(x, "shape"):
            return None
        if x.dim() != 3:
            raise ValueError("Expected input shape (B,T,resid_dim)")
        batch_size, seq_len, hidden = x.shape
        if hidden != resid_dim:
            raise ValueError("Input resid_dim mismatch")

        t_idx = _normalize_token_index(token_idx, seq_len)
        if t_idx < 0 or t_idx >= seq_len:
            raise ValueError("token_idx out of range")

        x_heads = x.reshape(batch_size, seq_len, n_heads, head_dim)
        if mode == "self":
            vec = x_heads[:, t_idx, head_idx, :]
        else:
            if replace_vec is None:
                raise ValueError("replace_vec required for mode='replace'")
            vec = _normalize_replace_vec(replace_vec, batch_size, head_dim, x)

        x_heads[:, t_idx, head_idx, :] = vec
        x_patched = x_heads.reshape(batch_size, seq_len, resid_dim)

        _log_once(
            logger,
            log_state,
            "hook fired "
            f"layer={layer_idx} head={head_idx} token_idx={token_idx} mode={mode}",
        )

        if isinstance(inputs, tuple):
            return (x_patched,) + inputs[1:]
        return (x_patched,)

    return hook_fn


def make_cproj_head_output_replacer(
    layer_idx: int,
    head_idx: int,
    token_idx: int,
    mode: str,
    replace_vec,
    model_config: dict,
    logger=None,
):
    """Create a forward hook to replace a single head vector and output.

    The hook uses the module input to build a patched input, then recomputes
    the module output using its weight (and bias, if present).
    """

    _validate_model_config(model_config)
    n_heads = int(model_config["n_heads"])
    head_dim = int(model_config["head_dim"])
    resid_dim = int(model_config["resid_dim"])

    if mode not in ("replace", "self"):
        raise ValueError("mode must be 'replace' or 'self'")
    if head_idx < 0 or head_idx >= n_heads:
        raise ValueError("head_idx out of range")

    log_state = {"logged": False}
    diag_state = {"logged": False}

    def _select_matmul(weight, resid_dim: int, module) -> str:
        if weight.dim() != 2:
            raise ValueError("module weight must be 2D for output recompute")
        if weight.shape[0] == resid_dim and weight.shape[1] != resid_dim:
            return "no_transpose"
        if weight.shape[1] == resid_dim and weight.shape[0] != resid_dim:
            return "transpose"
        module_name = module.__class__.__name__
        if module_name == "Conv1D":
            return "no_transpose"
        return "transpose"

    def _manual_out(x_patched, weight, bias, matmul_kind: str):
        if matmul_kind == "no_transpose":
            out = x_patched.matmul(weight)
        else:
            out = x_patched.matmul(weight.T)
        if bias is not None:
            out = out + bias
        return out

    def hook_fn(module, inputs, output):
        if not inputs:
            return output
        x = inputs[0]
        if x is None or not hasattr(x, "shape"):
            return output
        if x.dim() != 3:
            raise ValueError("Expected input shape (B,T,resid_dim)")
        batch_size, seq_len, hidden = x.shape
        if hidden != resid_dim:
            raise ValueError("Input resid_dim mismatch")

        t_idx = _normalize_token_index(token_idx, seq_len)
        if t_idx < 0 or t_idx >= seq_len:
            raise ValueError("token_idx out of range")

        x_heads = x.reshape(batch_size, seq_len, n_heads, head_dim)
        if mode == "self":
            vec = x_heads[:, t_idx, head_idx, :]
        else:
            if replace_vec is None:
                raise ValueError("replace_vec required for mode='replace'")
            vec = _normalize_replace_vec(replace_vec, batch_size, head_dim, x)

        x_heads[:, t_idx, head_idx, :] = vec
        x_patched = x_heads.reshape(batch_size, seq_len, resid_dim)

        weight = getattr(module, "weight", None)
        if weight is None:
            raise ValueError("module missing weight for output recompute")
        bias = getattr(module, "bias", None)

        matmul_kind = _select_matmul(weight, resid_dim, module)
        out = _manual_out(x_patched, weight, bias, matmul_kind)
        if output is not None and hasattr(output, "dtype"):
            if out.dtype != output.dtype:
                out = out.to(dtype=output.dtype)
        if output is not None and hasattr(output, "device"):
            if out.device != output.device:
                out = out.to(device=output.device)

        expected_hidden = weight.shape[1] if matmul_kind == "no_transpose" else weight.shape[0]
        if out.shape != (batch_size, seq_len, expected_hidden):
            raise ValueError("Output shape mismatch after recompute")

        _log_once(
            logger,
            log_state,
            "hook fired "
            f"layer={layer_idx} head={head_idx} token_idx={token_idx} mode={mode}",
        )

        if not diag_state.get("logged") and hasattr(module, "forward"):
            try:
                forward_out = module.forward(x_patched)
            except Exception:
                forward_out = None
            if forward_out is not None and hasattr(forward_out, "shape"):
                diff = (forward_out - out).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                _log_once(
                    logger,
                    diag_state,
                    "hook diagnostic "
                    f"layer={layer_idx} head={head_idx} token_idx={token_idx} "
                    f"mode={mode} matmul={matmul_kind} "
                    f"weight_shape={tuple(weight.shape)} "
                    f"max_abs_diff={max_diff:.6g} mean_abs_diff={mean_diff:.6g}",
                )

        return out

    return hook_fn


def make_cproj_head_output_replacer_multi(
    layer_idx: int,
    head_idx: int,
    token_indices: Sequence[int],
    mode: str,
    replace_vecs,
    model_config: dict,
    logger=None,
):
    """Create a forward hook to replace multiple head vectors and output.

    The hook uses the module input to build a patched input, then recomputes
    the module output using its weight (and bias, if present).
    """

    _validate_model_config(model_config)
    n_heads = int(model_config["n_heads"])
    head_dim = int(model_config["head_dim"])
    resid_dim = int(model_config["resid_dim"])

    if mode not in ("replace", "self"):
        raise ValueError("mode must be 'replace' or 'self'")
    if head_idx < 0 or head_idx >= n_heads:
        raise ValueError("head_idx out of range")

    log_state = {"logged": False}
    diag_state = {"logged": False}

    def _select_matmul(weight, resid_dim: int, module) -> str:
        if weight.dim() != 2:
            raise ValueError("module weight must be 2D for output recompute")
        if weight.shape[0] == resid_dim and weight.shape[1] != resid_dim:
            return "no_transpose"
        if weight.shape[1] == resid_dim and weight.shape[0] != resid_dim:
            return "transpose"
        module_name = module.__class__.__name__
        if module_name == "Conv1D":
            return "no_transpose"
        return "transpose"

    def _manual_out(x_patched, weight, bias, matmul_kind: str):
        if matmul_kind == "no_transpose":
            out = x_patched.matmul(weight)
        else:
            out = x_patched.matmul(weight.T)
        if bias is not None:
            out = out + bias
        return out

    def hook_fn(module, inputs, output):
        if not inputs:
            return output
        x = inputs[0]
        if x is None or not hasattr(x, "shape"):
            return output
        if x.dim() != 3:
            raise ValueError("Expected input shape (B,T,resid_dim)")
        batch_size, seq_len, hidden = x.shape
        if hidden != resid_dim:
            raise ValueError("Input resid_dim mismatch")

        t_indices = _normalize_token_indices(token_indices, seq_len)

        x_heads = x.reshape(batch_size, seq_len, n_heads, head_dim)
        if mode == "self":
            replace_values = [x_heads[:, t_idx, head_idx, :] for t_idx in t_indices]
        else:
            if replace_vecs is None:
                raise ValueError("replace_vecs required for mode='replace'")
            replace_values = _normalize_replace_vecs(
                replace_vecs, len(t_indices), batch_size, head_dim, x
            )

        for t_idx, vec in zip(t_indices, replace_values):
            x_heads[:, t_idx, head_idx, :] = vec
        x_patched = x_heads.reshape(batch_size, seq_len, resid_dim)

        weight = getattr(module, "weight", None)
        if weight is None:
            raise ValueError("module missing weight for output recompute")
        bias = getattr(module, "bias", None)

        matmul_kind = _select_matmul(weight, resid_dim, module)
        out = _manual_out(x_patched, weight, bias, matmul_kind)
        if output is not None and hasattr(output, "dtype"):
            if out.dtype != output.dtype:
                out = out.to(dtype=output.dtype)
        if output is not None and hasattr(output, "device"):
            if out.device != output.device:
                out = out.to(device=output.device)

        expected_hidden = (
            weight.shape[1] if matmul_kind == "no_transpose" else weight.shape[0]
        )
        if out.shape != (batch_size, seq_len, expected_hidden):
            raise ValueError("Output shape mismatch after recompute")

        _log_once(
            logger,
            log_state,
            "hook fired "
            f"layer={layer_idx} head={head_idx} token_indices={list(token_indices)} "
            f"mode={mode}",
        )

        if not diag_state.get("logged") and hasattr(module, "forward"):
            try:
                forward_out = module.forward(x_patched)
            except Exception:
                forward_out = None
            if forward_out is not None and hasattr(forward_out, "shape"):
                diff = (forward_out - out).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                _log_once(
                    logger,
                    diag_state,
                    "hook diagnostic "
                    f"layer={layer_idx} head={head_idx} "
                    f"token_indices={list(token_indices)} "
                    f"mode={mode} matmul={matmul_kind} "
                    f"weight_shape={tuple(weight.shape)} "
                    f"max_abs_diff={max_diff:.6g} mean_abs_diff={mean_diff:.6g}",
                )

        return out

    return hook_fn


def _self_test() -> None:
    """Minimal smoke test for shape handling."""

    import torch

    model_cfg = {"n_heads": 2, "head_dim": 3, "resid_dim": 6}
    x = torch.zeros((1, 4, 6))
    vec = torch.ones((3,))
    hook = make_cproj_head_replacer(
        layer_idx=0,
        head_idx=1,
        token_idx=-1,
        mode="replace",
        replace_vec=vec,
        model_config=model_cfg,
    )
    out = hook(None, (x,))
    assert out is not None
    patched = out[0]
    assert patched.shape == x.shape
