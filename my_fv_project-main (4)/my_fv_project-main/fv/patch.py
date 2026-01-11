"""Patch utilities for head-level replacement in c_proj hooks."""

from typing import Optional


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

        out = x_patched.matmul(weight.T)
        if bias is not None:
            out = out + bias
        if output is not None and hasattr(output, "dtype"):
            if out.dtype != output.dtype:
                out = out.to(dtype=output.dtype)
        if output is not None and hasattr(output, "device"):
            if out.device != output.device:
                out = out.to(device=output.device)

        expected_hidden = weight.shape[0]
        if out.shape != (batch_size, seq_len, expected_hidden):
            raise ValueError("Output shape mismatch after recompute")

        _log_once(
            logger,
            log_state,
            "hook fired "
            f"layer={layer_idx} head={head_idx} token_idx={token_idx} mode={mode}",
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
