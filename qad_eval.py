%%writefile kaggle_eval_qad.py
"""
Usage (GSM8K only):
    python kaggle_eval_qad.py \
        --baseline_model Qwen/Qwen2.5-1.5B-Instruct \
        --condition condition2_zero=/kaggle/working/configs/condition2_zero_exemption_quantized \
        --condition condition3_heuristic=/kaggle/working/configs/condition3_fixed_heuristic_quantized \
        --condition condition4_autoquantize=/kaggle/working/configs/condition4_autoquantize_quantized \
        --num_fewshot 5 \
        --eval_dir /kaggle/working/Model-Optimizer/examples/llm_eval/
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

TASK = ["gsm8k"]


# --------------------------------------------------------------------------
# Pre-flight checkpoint verification
# --------------------------------------------------------------------------

def preflight_check_condition(tag: str, model_path: str) -> dict:
    """Load a ModelOpt fake-quant checkpoint and confirm quantizer state
    round-trips, before any lm_eval subprocess touches it. Mirrors the
    verification logic in inspect_quant_checkpoints.py (load_and_count_quantizers)
    so a checkpoint that fails inspection also fails here, rather than the
    two checks silently drifting apart over time.

    Returns a dict: {"ok": bool, "reason": str, "pct_linear_quantized": float|None}
    """
    p = Path(model_path)
    if not p.exists():
        return {"ok": False, "reason": f"path does not exist: {model_path}", "pct_linear_quantized": None}

    files = {x.name for x in p.iterdir()}
    if "hf_quant_config.json" in files:
        return {
            "ok": False,
            "reason": (
                "This is a bit-packed export_hf_checkpoint deployment checkpoint "
                "(has hf_quant_config.json), not a fake-quant checkpoint. Plain "
                "from_pretrained (used by lm_eval_hf.py's HF wrapper) cannot unpack "
                "NVFP4 weights and will silently reinit shape-mismatched Linear "
                "layers to random weights, exactly like the earlier condition2/3 bug. "
                "Rebuild this checkpoint with mto.enable_huggingface_checkpointing() "
                "+ save_pretrained() instead."
            ),
            "pct_linear_quantized": None,
        }
    if "modelopt_state.pth" not in files:
        return {
            "ok": False,
            "reason": f"no modelopt_state.pth found in {model_path} -- not a recognized ModelOpt checkpoint",
            "pct_linear_quantized": None,
        }

    try:
        import torch
        import modelopt.torch.opt as mto
        import modelopt.torch.quantization as mtq
        from transformers import AutoModelForCausalLM
    except Exception as e:
        return {"ok": False, "reason": f"import failure during preflight: {e}", "pct_linear_quantized": None}

    try:
        mto.enable_huggingface_checkpointing()
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu", trust_remote_code=True)
    except Exception as e:
        return {"ok": False, "reason": f"from_pretrained failed: {e}", "pct_linear_quantized": None}

    total = active = 0
    lin_total = lin_quantized = 0
    for name, module in model.named_modules():
        if isinstance(module, mtq.nn.TensorQuantizer):
            total += 1
            if bool(getattr(module, "is_enabled", False)):
                active += 1
        if hasattr(module, "weight_quantizer") and isinstance(
            getattr(module, "weight_quantizer"), mtq.nn.TensorQuantizer
        ):
            lin_total += 1
            if bool(getattr(module.weight_quantizer, "is_enabled", False)):
                lin_quantized += 1

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    pct = round(100.0 * lin_quantized / lin_total, 1) if lin_total else None

    if total == 0:
        return {
            "ok": False,
            "reason": "0 TensorQuantizers found after load -- checkpoint is effectively "
                      "unquantized (silent BF16 fallback).",
            "pct_linear_quantized": pct,
        }
    if active == 0:
        return {
            "ok": False,
            "reason": "all TensorQuantizers disabled after load -- model will run as plain BF16.",
            "pct_linear_quantized": pct,
        }

    return {"ok": True, "reason": f"{active}/{total} quantizers active, {pct}% of Linear modules quantized", "pct_linear_quantized": pct}


def run_preflight(conditions: dict, expected_pct: dict[str, float] | None = None) -> bool:
    """Run preflight_check_condition on every condition. Prints a report and
    returns True only if every condition passed. expected_pct, if given, maps
    tag -> expected pct_linear_quantized (from config-build time) for an
    additional consistency check."""
    print(">>> Pre-flight: verifying quantizer state for all conditions before running eval...")
    all_ok = True
    for tag, path in conditions.items():
        result = preflight_check_condition(tag, path)
        status = "OK" if result["ok"] else "FAIL"
        print(f"  [{status}] {tag}: {result['reason']}")
        if not result["ok"]:
            all_ok = False
            continue
        if expected_pct and tag in expected_pct and result["pct_linear_quantized"] is not None:
            delta = abs(result["pct_linear_quantized"] - expected_pct[tag])
            if delta > 1.0:
                print(f"    WARN: loaded pct={result['pct_linear_quantized']} differs from expected "
                      f"{expected_pct[tag]} by {delta:.1f}pp")
    if not all_ok:
        print(">>> Pre-flight FAILED for at least one condition. Aborting before spending eval time.")
    else:
        print(">>> Pre-flight passed for all conditions.")
    return all_ok


# --------------------------------------------------------------------------
# Running lm_eval_hf.py
# --------------------------------------------------------------------------

def run_lm_eval(model_path: str, tag: str, task: str, out_dir: Path,
                 batch_size: str, num_fewshot: int,
                 eval_dir: Path, limit: int | None = None):
    """Run a single (model, task) pair through lm_eval_hf.py.

    model_path is either the baseline HF model id, or a condition's
    already-quantized ModelOpt fake-quant checkpoint directory. There is no
    separate quant_cfg step anymore -- the checkpoint at model_path already
    IS the model to evaluate, quantized or not.

    out_path is a DIRECTORY (see module docstring) -- lm-eval-harness nests
    its own results_<timestamp>.json underneath it, which find_results_json
    below expects.
    """
    run_dir = out_dir / f"{tag}_{task}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Sampling settings matched to Nemotron 3 Nano's protocol in the QAD
    # report, used as the closest reference point in scale to a 1.5B model.
    gen_kwargs = "do_sample=False,max_gen_toks=512" #before was 4096

    cmd = [
        sys.executable, "lm_eval_hf.py",
        "--model", "hf",
        "--tasks", task,
        "--model_args", f"pretrained={model_path},trust_remote_code=True,dtype=bfloat16",
        "--batch_size", str(batch_size),
        "--num_fewshot", str(num_fewshot),
        "--output_path", str(run_dir),
        "--gen_kwargs", gen_kwargs,
        # Both models here are instruct-tuned (Qwen2.5-1.5B-Instruct and its QAD-trained
        # variants) -- lm-eval-harness's own GSM8K docs recommend these two flags together
        # to correctly wrap few-shot examples in the model's chat template, which matches
        # how instruct models are meant to be prompted and avoids the format-mismatch
        # warning seen without it.
        "--apply_chat_template",
        "--fewshot_as_multiturn",
    ]

    if limit is not None:
        # Passed straight through to lm-eval-harness. Use this for quick pipeline
        # smoke checks (e.g. limit=20) -- NOT for real accuracy numbers, since
        # GSM8K accuracy on 20 examples is extremely noisy.
        cmd += ["--limit", str(limit)]

    print(f"=== [{tag}] task={task} num_fewshot={num_fewshot} model_path={model_path} ===")
    subprocess.run(cmd, cwd=eval_dir, check=True)
    return run_dir


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run_comparison(baseline_model: str, conditions: dict[str, str],
                    batch_size: str = "8", num_fewshot: int = 5, out_dir: str = "./eval_results",
                    eval_dir: str = ".", skip_baseline: bool = False, limit: int | None = None,
                    skip_preflight: bool = False, expected_pct: dict[str, float] | None = None):
    out_dir = Path(out_dir)
    eval_dir = Path(eval_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not skip_preflight:
        if not run_preflight(conditions, expected_pct=expected_pct):
            print(">>> Refusing to run eval. Pass --skip_preflight to override (not recommended).")
            sys.exit(1)
    else:
        print(">>> Skipping preflight checkpoint verification (--skip_preflight set). "
              "Results may silently reflect an unquantized model if a checkpoint is broken.")

    tasks = TASK

    if not skip_baseline:
        print(f">>> BASELINE: {baseline_model}")
        for task in tasks:
            run_lm_eval(baseline_model, "baseline", task, out_dir,
                        batch_size, num_fewshot, eval_dir, limit)
    else:
        print(">>> Skipping baseline run, reusing existing results in out_dir.")

    for tag, model_path in conditions.items():
        print(f">>> CONDITION [{tag}]: {model_path}")
        for task in tasks:
            run_lm_eval(model_path, tag, task, out_dir,
                        batch_size, num_fewshot, eval_dir, limit)


def _parse_condition_arg(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            f"--condition must be in the form name=path, got: {raw!r}"
        )
    name, path = raw.split("=", 1)
    return name, path


def _parse_expected_pct_arg(raw: str) -> tuple[str, float]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            f"--expected_pct must be in the form name=pct, got: {raw!r}"
        )
    name, pct = raw.split("=", 1)
    return name, float(pct)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument(
        "--condition", action="append", type=_parse_condition_arg, default=[],
        dest="conditions", metavar="name=path",
        help="Repeatable. Each path must be a ModelOpt fake-quant checkpoint directory "
             "(has modelopt_state.pth), e.g. "
             "--condition condition2_zero=/kaggle/working/configs/condition2_zero_exemption_quantized",
    )
    ap.add_argument(
        "--expected_pct", action="append", type=_parse_expected_pct_arg, default=[],
        dest="expected_pcts", metavar="name=pct",
        help="Optional, repeatable. Expected pct_linear_quantized for a condition from "
             "config-build time, e.g. --expected_pct condition2_zero=99.5 "
             "--expected_pct condition3_heuristic=39.9. Preflight will warn if the loaded "
             "checkpoint's actual pct differs by more than 1 percentage point.",
    )
    ap.add_argument("--batch_size", type=str, default="8")
    ap.add_argument("--limit", type=int, default=None,
                     help="Cap the number of examples per task (passed to lm-eval-harness's "
                          "--limit). Use a small value (e.g. 20) for a fast pipeline smoke "
                          "check -- leave unset for real accuracy numbers.")
    ap.add_argument(
        "--num_fewshot", type=int, default=5,
        help="Locked across baseline and all conditions for comparability.",
    )
    ap.add_argument("--out_dir", default="./eval_results")
    ap.add_argument("--eval_dir", default=".", help="path to Model-Optimizer/examples/llm_eval")
    ap.add_argument("--skip_baseline", action="store_true",
                     help="Reuse existing baseline_<task> results in out_dir instead of re-running.")
    ap.add_argument("--skip_preflight", action="store_true",
                     help="Skip the pre-flight quantizer-state verification. Not recommended -- "
                          "this pipeline has twice produced checkpoints that load without error "
                          "but silently run as unquantized BF16.")
    args = ap.parse_args()

    run_comparison(
        baseline_model=args.baseline_model,
        conditions=dict(args.conditions),
        batch_size=args.batch_size,
        num_fewshot=args.num_fewshot,
        out_dir=args.out_dir,
        eval_dir=args.eval_dir,
        skip_baseline=args.skip_baseline,
        limit=args.limit,
        skip_preflight=args.skip_preflight,
        expected_pct=dict(args.expected_pcts) if args.expected_pcts else None,
    )