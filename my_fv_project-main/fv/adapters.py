"""Model adapter helpers for locating blocks and projections."""

from .model_spec import get_model_spec


def _log(logger, message: str) -> None:
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


def resolve_by_path(root, dotted_path: str, spec_name: str, label: str):
    """Resolve attribute path strictly from the spec (no fallback)."""
    current = root
    parts = dotted_path.split(".")
    for idx, part in enumerate(parts):
        if not hasattr(current, part):
            prefix = ".".join(parts[: idx + 1])
            raise ValueError(
                f"Spec '{spec_name}' failed to resolve {label} '{dotted_path}': "
                f"missing '{part}' at '{prefix}'"
            )
        current = getattr(current, part)
    return current


def resolve_blocks(model, spec, logger=None):
    """Resolve transformer blocks strictly via spec.blocks_path."""
    blocks = resolve_by_path(
        model, spec.blocks_path, spec_name=spec.name, label="blocks_path"
    )
    blocks_list = list(blocks)
    if spec.n_layers is not None and len(blocks_list) != spec.n_layers:
        raise ValueError(
            "Model block count mismatch with spec: "
            f"spec='{spec.name}' expected={spec.n_layers} got={len(blocks_list)} "
            f"path='{spec.blocks_path}'"
        )
    if spec.n_layers is None:
        _log(
            logger,
            "resolve blocks: "
            f"model.{spec.blocks_path} (n_layers={len(blocks_list)})",
        )
    else:
        _log(
            logger,
            "resolve blocks: "
            f"model.{spec.blocks_path} (n_layers={spec.n_layers})",
        )
    return blocks_list


def resolve_attn(block, spec, logger=None):
    """Resolve attention module strictly via spec.attn_path_in_block."""
    attn = resolve_by_path(
        block, spec.attn_path_in_block, spec_name=spec.name, label="attn_path_in_block"
    )
    _log(
        logger,
        f"resolve attn: {spec.attn_path_in_block} (spec={spec.name})",
    )
    return attn


def resolve_out_proj(attn_module, spec, logger=None):
    """Resolve attention out_proj strictly via spec.out_proj_path_in_attn."""
    out_proj = resolve_by_path(
        attn_module,
        spec.out_proj_path_in_attn,
        spec_name=spec.name,
        label="out_proj_path_in_attn",
    )
    _log(
        logger,
        f"resolve out_proj: {spec.out_proj_path_in_attn} (spec={spec.name})",
    )
    return out_proj


def infer_head_dims(model, spec_name: str = "gpt2"):
    """Infer hidden_size, n_heads, and head_dim from model config."""
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("Model config not found")

    spec = get_model_spec(spec_name)

    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(config, "n_embd", None)
    n_heads = getattr(config, "num_attention_heads", None)
    if n_heads is None:
        n_heads = getattr(config, "n_head", None)
    head_dim = getattr(config, "head_dim", None)

    if spec.hidden_size is not None:
        if hidden_size is not None and hidden_size != spec.hidden_size:
            raise ValueError(
                f"hidden_size mismatch with spec '{spec.name}': "
                f"expected={spec.hidden_size} got={hidden_size}"
            )
        hidden_size = spec.hidden_size
    if spec.n_heads is not None:
        if n_heads is not None and n_heads != spec.n_heads:
            raise ValueError(
                f"n_heads mismatch with spec '{spec.name}': "
                f"expected={spec.n_heads} got={n_heads}"
            )
        n_heads = spec.n_heads
    if spec.head_dim is not None:
        if head_dim is not None and head_dim != spec.head_dim:
            raise ValueError(
                f"head_dim mismatch with spec '{spec.name}': "
                f"expected={spec.head_dim} got={head_dim}"
            )
        head_dim = spec.head_dim

    if hidden_size is None or n_heads is None:
        raise ValueError("Model config missing hidden_size or n_heads")
    if head_dim is None:
        if hidden_size % n_heads != 0:
            raise ValueError("hidden_size must be divisible by n_heads")
        head_dim = hidden_size // n_heads
    if hidden_size != n_heads * head_dim:
        raise ValueError("hidden_size does not match n_heads * head_dim")

    return {"hidden_size": hidden_size, "n_heads": n_heads, "head_dim": head_dim}
