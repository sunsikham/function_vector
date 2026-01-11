"""fv package for prompt/slot/hook/fv utilities."""

from .data_provider import DummyDataProvider
from .fv import build_fv, parse_heads
from .hooks import get_c_proj_pre_hook, reshape_resid_to_heads
from .intervene import make_residual_injection_hook
from .io import (
    default_run_id,
    infer_step5_metadata_path,
    load_json,
    load_csv,
    prepare_run_dirs,
    resolve_run_dir,
    save_json,
    save_csv,
    save_step5_artifacts,
    save_step6_results,
    step5_paths,
    step6_paths,
)
from .mean_acts import extract_slot_activation
from .model_config import get_model_config
from .patch import make_cproj_head_replacer
from .prompting import (
    ANTONYM_PAIRS,
    build_two_shot_prompt,
    build_zero_shot_prompt,
)
from .slots import compute_query_predictive_slot

__all__ = [
    "ANTONYM_PAIRS",
    "DummyDataProvider",
    "build_fv",
    "build_two_shot_prompt",
    "build_zero_shot_prompt",
    "compute_query_predictive_slot",
    "default_run_id",
    "extract_slot_activation",
    "get_c_proj_pre_hook",
    "get_model_config",
    "infer_step5_metadata_path",
    "load_csv",
    "load_json",
    "make_residual_injection_hook",
    "make_cproj_head_replacer",
    "parse_heads",
    "prepare_run_dirs",
    "resolve_run_dir",
    "reshape_resid_to_heads",
    "save_csv",
    "save_json",
    "save_step5_artifacts",
    "save_step6_results",
    "step5_paths",
    "step6_paths",
]
