# my_fv_project

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run (examples)
```bash
python scripts/smoke_test.py --model gpt2
python scripts/hook_smoke_test.py --model gpt2 --block-index 0
python scripts/run_step5_build_universal_fv.py --model gpt2 --layer 0 --n_trials 2 --heads 0:0
python scripts/run_step6_fv_injection_eval.py --model gpt2 --layer 0 --fv_path runs/<run_id>/artifacts/step5/fv_gpt2_layer0_n2.pt
python scripts/run_stepA_plumbing_inject_sanity.py --model gpt2 --layer 0 --head 0 --n_trials 3 --alpha 1.0 --run_id plumbing_001
python scripts/run_stepB_make_corrupted_and_baseline.py --model gpt2 --n_trials 50 --n_icl_examples 2 --run_id corrupted_001
python scripts/run_stepC_patch_sanity.py --model gpt2 --layer 0 --head 0 --n_trials 5 --mode self --run_id patch_sanity_001
python scripts/run_stepD_aie_head_sweep.py --model gpt2 --layers 0 --heads all --n_trials 10 --n_icl_examples 3 --run_id aie_smoke_001
python scripts/run_stepD_aie_head_sweep.py --model gpt2 --layers 0-11 --heads all --n_trials 20 --n_icl_examples 3 --run_id aie_gpt2_001
python scripts/run_stepE_topk_fv_and_eval.py --run_id_stepD aie_smoke_001 --k 20 --model gpt2 --alpha 1.0 --n_eval_trials 20 --run_id stepE_smoke_001
```

## Phase0 smoke (local CPU friendly)
Use tiny LLaMA to pass Phase0 without LLaMA3 8B/70B weights.
python scripts/smoke_resolve_spec_outproj.py --model gpt2 --model_spec gpt2 --layer 0 --device cpu --dtype fp32
python scripts/smoke_resolve_spec_outproj.py --model hf-internal-testing/tiny-random-LlamaForCausalLM --model_spec llama3 --layer 0 --device cpu --dtype fp32
(if resolve fails)
python scripts/smoke_resolve_spec_outproj.py --model hf-internal-testing/tiny-random-LlamaForCausalLM --model_spec llama3_wrapped --layer 0 --device cpu --dtype fp32
python scripts/smoke_resolve_spec_outproj.py --model gpt2 --model_spec gpt2 --layer 0 --device cpu --dtype fp32 --negative
Success criteria:
- resolve blocks/attn/out_proj logs are printed.
- out_proj pre-hook runs; captured tensor shape is (B,S,H)-like and last dim H matches hidden_size.
- In --negative mode, resolution must fail (script exits 0).

Runs default to `runs/<run_id>/artifacts` and `runs/<run_id>/logs` (override with `--out_dir`).
To inspect outputs in a run: `python scripts/inspect_artifacts.py --run_id <run_id> step5/`
