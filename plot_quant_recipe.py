%%writefile plot_quant_recipe.py
#STEP 1.2: Plot all six quantization recipes
"""
plot_quant_recipe.py

Generates per-layer quantization heatmaps for the six experiment conditions
from exported checkpoints:
    - Condition 1: BF16 baseline
    - Conditions 2 through 6: exported quantized checkpoints

Usage:
    python3 plot_quant_recipe.py \
        --condition condition2_zero=/kaggle/working/configs/condition2_zero_exemption_quantized \
        --condition condition3_heuristic=/kaggle/working/configs/condition3_fixed_heuristic_quantized \
        --condition condition4_autoquantize=/kaggle/working/configs/condition4_autoquantize_quantized \
        --condition condition5_attn_only=/kaggle/working/configs/condition5_attn_only_quantized \
        --condition condition6_mlp_only=/kaggle/working/configs/condition6_mlp_only_quantized \
        --output graphics/quant_recipe_map_all.png

Each condition path must contain config.json and model.safetensors. The
baseline is represented by an all-BF16 panel and needs no path.
"""

import argparse
import fnmatch
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

NUM_LAYERS = 28
COLS = ["Q", "K", "V", "O", "Gate", "Up", "Down"]

# Condition 4 columns map to the modules auto_quantize actually searched.
# K/V follow Q's format; Up follows Gate's format (grouped/tied, not searched independently).
COL_TO_MODULE = {
    "Q": "q_proj", "K": "q_proj", "V": "q_proj", "O": "o_proj",
    "Gate": "gate_proj", "Up": "gate_proj", "Down": "down_proj",
}
STATIC_COL_TO_MODULE = {
    "Q": "q_proj", "K": "k_proj", "V": "v_proj", "O": "o_proj",
    "Gate": "gate_proj", "Up": "up_proj", "Down": "down_proj",
}


def parse_checkpoint(checkpoint_dir):
    import modelopt.torch.opt as mto
    from transformers import AutoModelForCausalLM
    mto.enable_huggingface_checkpointing()
    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, torch_dtype="bfloat16").cuda()

    data = {}
    for name, module in model.named_modules():
        if not hasattr(module, "weight_quantizer"):
            continue
        m = re.match(
            r"model\.layers\.(\d+)\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
            name,
        )
        if not m:
            continue
        layer, block, mod = int(m.group(1)), m.group(2), m.group(3)
        wq = module.weight_quantizer
        is_quantized = bool(getattr(wq, "is_enabled", True))
        data.setdefault(layer, {})[(block, mod)] = "INT4" if is_quantized else "NONE"
    return data


def parse_static_config(path):
    """Parse a ModelOpt config using the most-specific matching rule."""
    with open(path) as f:
        cfg = json.load(f)
    grid = np.zeros((NUM_LAYERS, len(COLS)))
    for layer in range(NUM_LAYERS):
        for j, col in enumerate(COLS):
            module = STATIC_COL_TO_MODULE[col]
            block = "self_attn" if col in ("Q", "K", "V", "O") else "mlp"
            quantizer_name = f"model.layers.{layer}.{block}.{module}.weight_quantizer"
            matched_rule = None
            matched_score = (-1, -1)
            for index, rule in enumerate(cfg.get("quant_cfg", [])):
                pattern = rule.get("quantizer_name", "")
                if fnmatch.fnmatchcase(quantizer_name, pattern):
                    # Broad defaults such as "*" are overridden by more
                    # specific projection and layer rules.
                    score = (len(pattern.replace("*", "").replace("?", "")), index)
                    if score >= matched_score:
                        matched_rule = rule
                        matched_score = score
            enabled = matched_rule.get("enable", True) if matched_rule else False
            grid[layer, j] = 1 if enabled else 0
    return grid


def build_checkpoint_grid(recipe):
    grid = np.zeros((NUM_LAYERS, len(COLS)))  # 0 = BF16/exempt, 1 = INT4
    for layer in range(NUM_LAYERS):
        for j, col in enumerate(COLS):
            block = "self_attn" if col in ("Q", "K", "V", "O") else "mlp"
            module = STATIC_COL_TO_MODULE[col]
            fmt = recipe.get(layer, {}).get((block, module), "NONE")
            grid[layer, j] = 1 if fmt == "INT4" else 0
    return grid


def plot(recipes, output_path):
    fig, axes = plt.subplots(2, 3, figsize=(13, 10), sharex=True, sharey=True)
    cmap = plt.matplotlib.colors.ListedColormap(["#EAEAE4", "#639922"])

    for ax, (title, grid) in zip(axes.flat, recipes.items()):
        ax.imshow(grid, cmap=cmap, aspect="auto", vmin=0, vmax=1)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(COLS)))
        ax.set_xticklabels(COLS, fontsize=9)
        ax.set_yticks(range(NUM_LAYERS))
        ax.set_yticklabels(range(NUM_LAYERS), fontsize=7)
        ax.set_ylabel("Layer")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(COLS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, NUM_LAYERS, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", bottom=False, left=False)

    legend_elements = [
        plt.matplotlib.patches.Patch(facecolor="#639922", label="NVFP4"),
        plt.matplotlib.patches.Patch(facecolor="#EAEAE4", label="BF16 / exempt"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Per-layer quantization recipes", fontsize=13)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", action="append", required=True,
                         metavar="NAME=PATH",
                         help="Exported checkpoint; repeat once for each of conditions 2-6")
    parser.add_argument("--output", default="quant_recipe_map_all.png")
    args = parser.parse_args()
    if len(args.condition) != 5:
        parser.error("Provide exactly five --condition arguments for conditions 2 through 6")

    recipes = {
        "Condition 1: BF16 baseline": np.zeros((NUM_LAYERS, len(COLS))),
    }
    for condition in args.condition:
        if "=" not in condition:
            parser.error(f"Condition must use NAME=PATH format: {condition}")
        name, path = condition.split("=", 1)
        checkpoint_dir = Path(path)
        if not checkpoint_dir.is_dir():
            parser.error(f"Checkpoint directory not found: {checkpoint_dir}")
        recipes[name] = build_checkpoint_grid(parse_checkpoint(checkpoint_dir))
    plot(recipes, args.output)


if __name__ == "__main__":
    main()