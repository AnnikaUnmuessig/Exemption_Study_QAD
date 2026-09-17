%%writefile build_quant_configs.py
#STEP 0.2 Build quantization configs
"""
build_quant_configs.py

Builds modelopt-compatible quantization configs:

  - condition2_zero_exemption.json  : quantize every Linear GEMM to NVFP4 (weight + activation)
                                       (matches the paper's actual recipe for dense transformers —
                                        Llama Nemotron Super V1 / AceReason Nemotron get zero
                                        exemptions )
  - condition3_fixed_heuristic.json : exempt attention + first/last two decoder layers
                                       (heuristic transferred from Nemotron Nano 9B V2's recipe,
                                        applied literally to Qwen2.5 )
  -condition4_autoquantize.json : search for a quantization config that matches the same effective-bits
  -condition5_attn_only.json : quantize all attention projections, exempt all MLP projections
  -condition6_mlp_only.json : quantize all MLP projections, exempt all attention projections


Uses weight and activation quantization (W4A4) with NVFP4 (E2M1, block size 16, two-level scaling).

Each config is validated with a calibration forward pass, just to confirm the config loads and
every module-name pattern actually matches real modules. 


Example usages:

python build_quant_configs.py --calib_dataset wikitext --run_autoquantize --effective_bits <printed_target>

Calibration dataset: wikitext
"""

import argparse
import copy
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import modelopt.torch.quantization as mtq
import modelopt.torch.opt as mto

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LAYER_MAP_PATH = "layer_map.json"
OUTPUT_DIR = Path("/kaggle/working/configs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# matching the Nemotron Nano 9B V2 recipe literally, per the finalized decision for condition 3
NUM_FIRST_LAST_LAYERS = 2


QUANTIZED_BITS = 4 #4 bit floating point
UNQUANTIZED_BITS = 16 #16 bit floating point


BASE_QUANT_CFG = {
    "quant_cfg": [
        {
            "quantizer_name": "*weight_quantizer",
            "cfg": {
                "num_bits": (2, 1),
                "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
                "axis": None,
            },
        },
        {
            "quantizer_name": "*input_quantizer",
            "cfg": {
                "num_bits": (2, 1),
                "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
                "axis": None,
            },
        },
        {
            "quantizer_name": "*lm_head*",
            "enable": False,
        },
        {
            "quantizer_name": "*output_layer*",
            "enable": False,
        },
        {
            "quantizer_name": "*q_bmm_quantizer",
            "enable": False,
        },
        {
            "quantizer_name": "*k_bmm_quantizer",
            "enable": False,
        },
        {
            "quantizer_name": "*v_bmm_quantizer",
            "enable": False,
        },
        {
            "quantizer_name": "*p_bmm_quantizer",
            "enable": False,
        },
    ],
    "algorithm": {"method": "max"},
}

#Calibration parameters for the forward pass
CALIB_NUM_EXAMPLES = 128
CALIB_SEQ_LEN = 512
CALIB_BATCH_SIZE = 4


# ---------------------------------------------------------------------------
# Layer map: opens layer map
# ---------------------------------------------------------------------------
def load_layer_map():
    with open(LAYER_MAP_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------
def build_condition2_config():
    """Zero exemptions: every Linear GEMM quantized to NVFP4 (weight + activation).
    Just the base config, unmodified — this is the paper's actual treatment for
    dense transformers (Sec 3.4), so no changes needed here beyond BASE_QUANT_CFG."""
    return copy.deepcopy(BASE_QUANT_CFG)


def build_condition3_config(layer_map):
    cfg = copy.deepcopy(BASE_QUANT_CFG)
    quant_cfg_list = cfg["quant_cfg"]

    num_layers = layer_map["num_layers"]
    first_indices = list(range(0, NUM_FIRST_LAST_LAYERS))
    last_indices = list(range(num_layers - NUM_FIRST_LAST_LAYERS, num_layers))
    exempt_layer_indices = sorted(set(first_indices + last_indices))

    attn_suffixes = layer_map["decoder_layer_linear_suffixes"]["attention"]
    mlp_suffixes = layer_map["decoder_layer_linear_suffixes"]["mlp"]

    exempt_patterns = set()

    # Rule 1: attention exempted at EVERY layer (literal paper heuristic)
    for suffix in attn_suffixes:
        exempt_patterns.add(f"*model.layers.*.{suffix}*")

    # Rule 2: first/last 2 layers exempted entirely 
    for idx in exempt_layer_indices:
        for suffix in attn_suffixes + mlp_suffixes:
            exempt_patterns.add(f"*model.layers.{idx}.{suffix}*")

    if layer_map.get("exempt_embed_lm_head", False):
        if layer_map.get("embed_tokens_module"):
            exempt_patterns.add(f"*{layer_map['embed_tokens_module']}*")
        if layer_map.get("lm_head_module"):
            exempt_patterns.add(f"*{layer_map['lm_head_module']}*")

    for pattern in sorted(exempt_patterns):
        quant_cfg_list.append({"quantizer_name": pattern, "enable": False})

    cfg["_exempt_patterns"] = sorted(exempt_patterns)
    cfg["_num_first_last_layers"] = NUM_FIRST_LAST_LAYERS
    cfg["_exempt_layer_indices"] = exempt_layer_indices
    return cfg


def build_condition5_attn_only_config(layer_map):
    """Attention-only quantization: exempt ALL MLP projections at every layer,
    quantize ALL attention projections at every layer."""
    cfg = copy.deepcopy(BASE_QUANT_CFG)
    quant_cfg_list = cfg["quant_cfg"]

    mlp_suffixes = layer_map["decoder_layer_linear_suffixes"]["mlp"]

    exempt_patterns = set()
    for suffix in mlp_suffixes:
        exempt_patterns.add(f"*model.layers.*.{suffix}*")

    if layer_map.get("exempt_embed_lm_head", False):
        if layer_map.get("embed_tokens_module"):
            exempt_patterns.add(f"*{layer_map['embed_tokens_module']}*")
        if layer_map.get("lm_head_module"):
            exempt_patterns.add(f"*{layer_map['lm_head_module']}*")

    for pattern in sorted(exempt_patterns):
        quant_cfg_list.append({"quantizer_name": pattern, "enable": False})

    cfg["_exempt_patterns"] = sorted(exempt_patterns)
    cfg["_structural_axis"] = "attention_only"
    return cfg


def build_condition6_mlp_only_config(layer_map):
    """MLP-only quantization: exempt ALL attention projections at every layer,
    quantize ALL MLP projections at every layer. """
    cfg = copy.deepcopy(BASE_QUANT_CFG)
    quant_cfg_list = cfg["quant_cfg"]

    attn_suffixes = layer_map["decoder_layer_linear_suffixes"]["attention"]

    exempt_patterns = set()
    for suffix in attn_suffixes:
        exempt_patterns.add(f"*model.layers.*.{suffix}*")

    if layer_map.get("exempt_embed_lm_head", False):
        if layer_map.get("embed_tokens_module"):
            exempt_patterns.add(f"*{layer_map['embed_tokens_module']}*")
        if layer_map.get("lm_head_module"):
            exempt_patterns.add(f"*{layer_map['lm_head_module']}*")

    for pattern in sorted(exempt_patterns):
        quant_cfg_list.append({"quantizer_name": pattern, "enable": False})

    cfg["_exempt_patterns"] = sorted(exempt_patterns)
    cfg["_structural_axis"] = "mlp_only"
    return cfg


def _strip_debug_keys(cfg):
    cfg = copy.deepcopy(cfg)
    for k in ("_exempt_patterns", "_num_first_last_layers", "_exempt_layer_indices", "_structural_axis"):
        cfg.pop(k, None)
    return cfg


# ---------------------------------------------------------------------------
# Validation (calibration pass)
# ---------------------------------------------------------------------------
def load_calib_texts(calib_dataset=None, max_examples: int = CALIB_NUM_EXAMPLES):
    """Load a shared text corpus for all PTQ calibration passes.

    Default to Wikitext so the three conditions use the same real calibration data.
    If a local file path or other Hugging Face dataset is provided, that is used instead.
    """
    p = Path(calib_dataset)
    if p.exists() and p.is_file():
        texts = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
        if len(texts) < max_examples:
            texts = texts * (max_examples // max(1, len(texts))) + texts[: max_examples % max(1, len(texts))]
        return texts[:max_examples]

    dataset_name = calib_dataset
    dataset_config = None
    if calib_dataset == "wikitext":
        dataset_name = "wikitext"
        dataset_config = "wikitext-2-raw-v1"

    try:
        if dataset_config is not None:
            ds = load_dataset(dataset_name, dataset_config, split="train")
        else:
            ds = load_dataset(dataset_name, split="train")
    except Exception:
        try:
            ds = load_dataset(dataset_name)
        except Exception:
            raise ValueError(f"Could not load calibration dataset '{calib_dataset}'. Use a local file path or a valid HF dataset id.")

    text_field = None
    for candidate in ("text", "sentence", "content", "review", "article", "body"):
        if candidate in ds.column_names:
            text_field = candidate
            break
    if text_field is None:
        text_field = ds.column_names[0]

    texts = []
    for ex in ds.select(range(min(len(ds), max_examples))):
        val = ex.get(text_field)
        if isinstance(val, list):
            val = " ".join(val)
        if val:
            texts.append(val)

    if len(texts) < max_examples:
        texts = texts * (max_examples // max(1, len(texts))) + texts[: max_examples % max(1, len(texts))]
    return texts[:max_examples]


def get_calib_inputs(tokenizer, device, calib_dataset=None, seed=42):
    """Construct a calibration batch from a shared text corpus. With activation
    quantization now enabled, this is no longer just a formality — it's exercising
    real input-activation statistics that the 'max' calibrator will use to set
    NVFP4 scale factors."""
    texts = load_calib_texts(calib_dataset=calib_dataset, max_examples=CALIB_NUM_EXAMPLES)
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=CALIB_SEQ_LEN,
    )
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def validate_config(model, tokenizer, quant_cfg, label, calib_dataset=None):
    """Runs modelopt calibration. """
    print(f"\n=== Validating {label} ===")

    device = next(model.parameters()).device
    input_ids, attention_mask = get_calib_inputs(tokenizer, device, calib_dataset=calib_dataset)

    def forward_loop(m):
        m.eval()
        with torch.no_grad():
            for i in range(0, CALIB_NUM_EXAMPLES, CALIB_BATCH_SIZE):
                m(
                    input_ids=input_ids[i : i + CALIB_BATCH_SIZE],
                    attention_mask=attention_mask[i : i + CALIB_BATCH_SIZE],
                )

    quantized_model = mtq.quantize(model, _strip_debug_keys(quant_cfg), forward_loop)

    total_linear = 0
    quantized_linear = 0
    total_params = 0
    quantized_params = 0
    for name, module in quantized_model.named_modules():
        if hasattr(module, "weight_quantizer"):
            total_linear += 1
            n = module.weight.numel()
            total_params += n
            if getattr(module.weight_quantizer, "is_enabled", True):
                quantized_linear += 1
                quantized_params += n

    pct = 100.0 * quantized_linear / max(total_linear, 1)
    pct_params = 100.0 * quantized_params / max(total_params, 1)
    print(f"{label}: {quantized_linear}/{total_linear} Linear modules quantized ({pct:.1f}% by module-count)") #summary of quantization
    print(f"{label}: {quantized_params}/{total_params} params quantized ({pct_params:.1f}% by param-weight)")
    return quantized_model, pct, pct_params


def build_calib_dataloader(tokenizer, batch_size: int, max_samples: int, seq_len: int = CALIB_SEQ_LEN, calib_dataset=None):
    """Return deterministic calibration batches sharing the same data source as condition 2/3."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = load_calib_texts(calib_dataset=calib_dataset, max_examples=max_samples)
    batches = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=seq_len)
        enc["labels"] = enc["input_ids"].clone()
        batches.append(enc)
    return batches


def make_forward_step_and_loss(method: str):
    if method == "gradient":
        def forward_step(model, batch):
            return model(**batch)

        def loss_func(output, batch):
            return output.loss
    elif method == "kl_div":
        def forward_step(model, batch):
            return model(**batch).logits

        def loss_func(output, batch):
            return None
    else:
        raise ValueError(f"method must be 'gradient' or 'kl_div', got {method!r}")
    return forward_step, loss_func


def timed(fn, timings: list):
    def wrapped(*args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
        return result
    return wrapped


def export_eval_checkpoint(model, tokenizer, export_dir: Path):
    """Save a ModelOpt-quantized model in the Hugging Face checkpoint format that
    plain HF generation can reload with `mto.enable_huggingface_checkpointing()`.
    This is the eval-ready format used for condition 4 and should be used for
    conditions 2/3 as well so all conditions share the same runtime contract."""
    mto.enable_huggingface_checkpointing()
    export_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(export_dir))
    tokenizer.save_pretrained(str(export_dir))
    return export_dir


def run_autoquantize_search(model_path: str, effective_bits: float, output_dir: str, method: str = "gradient", calib_dataset: str = "wikitext", dry_run: bool = False, num_calib_steps: int = None, num_score_steps: int = None, batch_size: int = 1, seq_len: int = CALIB_SEQ_LEN):
    """Run autoquantize search for a quantization config that matches the same effective-bits as condition 3. The search is done with a small number of calibration and scoring steps, and the resulting config is saved to output_dir."""
    if num_calib_steps is None:
        num_calib_steps = 32 if dry_run else 512
    if num_score_steps is None:
        num_score_steps = 16 if dry_run else 128
    if num_score_steps >= num_calib_steps:
        raise ValueError(f"num_score_steps ({num_score_steps}) should be < num_calib_steps ({num_calib_steps})")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run_autoquantize] dry_run={dry_run} method={method} num_calib_steps={num_calib_steps} num_score_steps={num_score_steps} effective_bits={effective_bits}")
    print("[run_autoquantize] loading tokenizer + model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()
    mto.enable_huggingface_checkpointing()

    calib_batches = build_calib_dataloader(tokenizer, batch_size, max_samples=num_calib_steps * batch_size, seq_len=seq_len, calib_dataset=calib_dataset)
    calib_batches = [{k: v.cuda() for k, v in b.items()} for b in calib_batches]

    def calib_dataloader():
        for b in calib_batches:
            yield b

    forward_step, loss_func = make_forward_step_and_loss(method)
    timings = []
    forward_step = timed(forward_step, timings)

    print(f"[run_autoquantize] starting auto_quantize search ({'DRY RUN — throwaway timing check' if dry_run else 'FULL RUN'})...")
    t0 = time.perf_counter()
    quantized_model, search_history = mtq.auto_quantize(
        model,
        constraints={"effective_bits": effective_bits},
        quantization_formats=[BASE_QUANT_CFG, None],
        data_loader=calib_batches,
        forward_step=forward_step,
        loss_func=loss_func,
        num_calib_steps=num_calib_steps,
        num_score_steps=num_score_steps,
        method=method,
        checkpoint=str(out_dir / "autoquantize_search_state.pt"),
        verbose=True,
    )

    total_linear = quantized_linear = 0
    total_params = quantized_params = 0
    for name, module in quantized_model.named_modules():
        if hasattr(module, "weight_quantizer"):
            total_linear += 1
            n = module.weight.numel()
            total_params += n
            if getattr(module.weight_quantizer, "is_enabled", True):
                quantized_linear += 1
                quantized_params += n
    pct_linear = 100.0 * quantized_linear / max(total_linear, 1)
    pct_params = 100.0 * quantized_params / max(total_params, 1)
    print(f"Condition 4: {quantized_linear}/{total_linear} Linear modules quantized ({pct_linear:.1f}% by module-count)")
    print(f"Condition 4: {quantized_params}/{total_params} params quantized ({pct_params:.1f}% by param-weight)")

    total_time = time.perf_counter() - t0
    if timings:
        avg_step = sum(timings) / len(timings)
        print(f"\n[timing] {len(timings)} forward_step calls, avg {avg_step:.3f}s/step, total wall time {total_time:.1f}s")
        if dry_run:
            full_calib, full_score = 512, 128
            projected = avg_step * (full_calib + full_score)
            print(f"[timing] naive projection for full run (num_calib_steps={full_calib}, num_score_steps={full_score}): ~{projected/60:.1f} min")
    else:
        print("[timing] no forward_step calls were captured — check that auto_quantize actually invoked the wrapped forward_step.")

    mtq.print_quant_summary(quantized_model)

    from collections import Counter
    fmt_counts = Counter()
    for name, module in quantized_model.named_modules():
        if hasattr(module, "weight_quantizer"):
            wq = module.weight_quantizer
            if not getattr(wq, "is_enabled", True):
                fmt_counts["exempt"] += 1
            else:
                num_bits = getattr(wq, "num_bits", None)
                fmt_counts["nvfp4" if num_bits == (2, 1) else "fp8" if num_bits == (4, 3) else f"other:{num_bits}"] += 1
    print(f"[run_autoquantize] format distribution: {dict(fmt_counts)}")
    if fmt_counts.get("exempt", 0) == 0:
        print("[run_autoquantize] WARNING: zero layers exempted — budget may be too tight to promote anything, or something's wrong with the constraint.")

    history_path = out_dir / ("dry_run_history.json" if dry_run else "search_history.json")
    with open(history_path, "w") as f:
        json.dump({
            "method": method,
            "effective_bits": effective_bits,
            "num_calib_steps": num_calib_steps,
            "num_score_steps": num_score_steps,
            "total_time_sec": total_time,
            "avg_step_time_sec": (sum(timings) / len(timings)) if timings else None,
            "format_distribution": dict(fmt_counts),
            "pct_linear_quantized": pct_linear,
            "pct_params_quantized": pct_params,
            "search_history": str(search_history),
        }, f, indent=2)
    print(f"[run_autoquantize] wrote {history_path}")

    if not dry_run:
        export_dir = out_dir / "quantized_checkpoint"
        export_eval_checkpoint(quantized_model, tokenizer, export_dir)
        print(f"[run_autoquantize] saved simulated-quant checkpoint to {export_dir}")

    cond4_out = OUTPUT_DIR / "condition4_autoquantize_quantized"
    export_eval_checkpoint(quantized_model, tokenizer, cond4_out)
    print(f"Condition 4 export complete: {cond4_out}")
    return quantized_model, history_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_autoquantize", action="store_true", help="Run auto_quantize for condition 4")
    p.add_argument("--effective_bits", type=float, default=None, help="Effective bits constraint for auto_quantize (required if --run_autoquantize). Use the param-weighted-derived value printed in the condition 2/3 summary, not a module-count percentage.")
    p.add_argument("--auto_output_dir", default=str(OUTPUT_DIR / "autoquantize_out"), help="Where to write autoquantize outputs")
    p.add_argument("--calib_dataset", default="wikitext", help="Calibration data: 'wikitext' (default), 'synthetic', HF dataset id, or local text file path")
    p.add_argument("--method", choices=["gradient", "kl_div"], default="gradient")
    p.add_argument("--dry_run", action="store_true", help="Run a short timing-only auto_quantize pass")
    p.add_argument("--num_calib_steps", type=int, default=None)
    p.add_argument("--num_score_steps", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--seq_len", type=int, default=CALIB_SEQ_LEN)
    args = p.parse_args()

    layer_map = load_layer_map()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    condition2_cfg = build_condition2_config()
    condition3_cfg = build_condition3_config(layer_map)
    condition5_cfg = build_condition5_attn_only_config(layer_map)
    condition6_cfg = build_condition6_mlp_only_config(layer_map)

    with open(OUTPUT_DIR / "condition2_zero_exemption.json", "w") as f:
        json.dump(_strip_debug_keys(condition2_cfg), f, indent=2, default=str)
    with open(OUTPUT_DIR / "condition3_fixed_heuristic.json", "w") as f:
        json.dump(_strip_debug_keys(condition3_cfg), f, indent=2, default=str)
    with open(OUTPUT_DIR / "condition5_attn_only.json", "w") as f:
        json.dump(_strip_debug_keys(condition5_cfg), f, indent=2, default=str)
    with open(OUTPUT_DIR / "condition6_mlp_only.json", "w") as f:
        json.dump(_strip_debug_keys(condition6_cfg), f, indent=2, default=str)
    print(f"Saved raw configs to {OUTPUT_DIR}/ before validation.")

    print("\nLoading model for validation (only GPU-touching step)...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    quant2_model, pct2, pct2p = validate_config(model, tokenizer, condition2_cfg, "condition2_zero_exemption", calib_dataset=args.calib_dataset)

    print("\nReloading fresh model for condition 3 (quantize() mutates in place)...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    quant3_model, pct3, pct3p = validate_config(model, tokenizer, condition3_cfg, "condition3_fixed_heuristic", calib_dataset=args.calib_dataset)

    print("\nReloading fresh model for condition 5 (attention-only)...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    quant5_model, pct5, pct5p = validate_config(model, tokenizer, condition5_cfg, "condition5_attn_only", calib_dataset=args.calib_dataset)

    print("\nReloading fresh model for condition 6 (mlp-only)...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    quant6_model, pct6, pct6p = validate_config(model, tokenizer, condition6_cfg, "condition6_mlp_only", calib_dataset=args.calib_dataset)

    cond2_out = OUTPUT_DIR / "condition2_zero_exemption_quantized"
    print(f"Exporting condition 2 quantized checkpoint to {cond2_out} ...")
    export_eval_checkpoint(quant2_model, tokenizer, cond2_out)
    print("Condition 2 export complete.")

    cond3_out = OUTPUT_DIR / "condition3_fixed_heuristic_quantized"
    print(f"Exporting condition 3 quantized checkpoint to {cond3_out} ...")
    export_eval_checkpoint(quant3_model, tokenizer, cond3_out)
    print("Condition 3 export complete.")

    cond5_out = OUTPUT_DIR / "condition5_attn_only_quantized"
    print(f"Exporting condition 5 quantized checkpoint to {cond5_out} ...")
    export_eval_checkpoint(quant5_model, tokenizer, cond5_out)
    print("Condition 5 export complete.")

    cond6_out = OUTPUT_DIR / "condition6_mlp_only_quantized"
    print(f"Exporting condition 6 quantized checkpoint to {cond6_out} ...")
    export_eval_checkpoint(quant6_model, tokenizer, cond6_out)
    print("Condition 6 export complete.")

    print("\n=== Summary ===")
    print(f"Condition 2 (zero exemption):  {pct2:.1f}% of Linear modules quantized ({pct2p:.1f}% of params)")
    print(f"Condition 3 (fixed heuristic): {pct3:.1f}% of Linear modules quantized ({pct3p:.1f}% of params)")
    print(f"Condition 5 (attention-only):  {pct5:.1f}% of Linear modules quantized ({pct5p:.1f}% of params)")
    print(f"Condition 6 (MLP-only):        {pct6:.1f}% of Linear modules quantized ({pct6p:.1f}% of params)")
    print(f"NUM_FIRST_LAST_LAYERS used:    {NUM_FIRST_LAST_LAYERS}")

    # Derive the effective_bits target for condition 4 from condition 3's PARAM-WEIGHTED
    # quantized fraction, not its module-count percentage. mtq.auto_quantize's
    # {"effective_bits": ...} constraint is itself a bits-per-parameter budget, so matching
    # it against a module-count % (as previously done) mismatches units and produces a
    # condition 4 checkpoint that quantizes a very different fraction of the model than
    # condition 3 actually does.
    target_effective_bits = (pct3p / 100.0) * QUANTIZED_BITS + (1 - pct3p / 100.0) * UNQUANTIZED_BITS
    print(f"\nCondition 3's param-weighted quantized fraction is {pct3p:.1f}% of parameters.")
    print(f"Matching effective_bits target for condition 4: {target_effective_bits:.2f}")
    print("Pass this value as --effective_bits when running --run_autoquantize (NOT the module-count percentage above).")
    print(f"\nConfigs saved to {OUTPUT_DIR}/")

    if args.run_autoquantize:
        if args.effective_bits is None:
            print(f"--run_autoquantize requires --effective_bits to be set. "
                  f"Skipping autoquantize. (Derived target from this run: {target_effective_bits:.2f})")
            return
        print(f"Running auto_quantize to match Condition 3 budget (effective_bits={args.effective_bits})...")
        run_autoquantize_search(
            model_path=MODEL_NAME,
            effective_bits=args.effective_bits,
            output_dir=args.auto_output_dir,
            method=args.method,
            calib_dataset=args.calib_dataset,
            dry_run=args.dry_run,
            num_calib_steps=args.num_calib_steps,
            num_score_steps=args.num_score_steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
        )


if __name__ == "__main__":
    main()