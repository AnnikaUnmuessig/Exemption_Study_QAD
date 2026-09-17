#PREP
#STEP 0.1 Create layer analysis for model
import json
from transformers import AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

num_layers = model.config.num_hidden_layers
last_idx = num_layers - 1
mid_idx = num_layers // 2  # spot-check layer

print(f"Num layers: {num_layers}")

all_names = [name for name, _ in model.named_modules()]

def layer_modules(idx):
    prefix = f"layers.{idx}."
    return sorted(n for n in all_names if prefix in n)

layer0 = layer_modules(0)
layer_mid = layer_modules(mid_idx)
layer_last = layer_modules(last_idx)

# Verify layer 0 and mid-layer share identical *relative* suffixes
suffix0 = sorted(n.split(f"layers.0.")[1] for n in layer0)
suffix_mid = sorted(n.split(f"layers.{mid_idx}.")[1] for n in layer_mid)
pattern_consistent = suffix0 == suffix_mid
print(f"\nLayer 0 vs layer {mid_idx} structural match: {pattern_consistent}")
if not pattern_consistent:
    print("WARNING: layer pattern differs mid-stack — do not assume uniformity in configs.")

# Identify Linear modules specifically (what modelopt actually targets)
import torch.nn as nn
linear_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
layer0_linears = [n for n in linear_names if "layers.0." in n]
print(f"\nLinear modules in layer 0: {layer0_linears}")

# embed_tokens / lm_head
embed_name = None
lm_head_name = None
tied = False
for n, m in model.named_modules():
    if n == "model.embed_tokens":
        embed_name = n
    if n == "lm_head":
        lm_head_name = n

if hasattr(model, "lm_head") and hasattr(model.model, "embed_tokens"):
    tied = model.lm_head.weight is model.model.embed_tokens.weight

print(f"\nembed_tokens module: {embed_name}")
print(f"lm_head module: {lm_head_name}")
print(f"embed/lm_head tied: {tied}")

# GQA dims — note asymmetry between q/o and k/v proj
attn0 = model.model.layers[0].self_attn
print(f"\nq_proj: {attn0.q_proj.weight.shape}")
print(f"k_proj: {attn0.k_proj.weight.shape}")
print(f"v_proj: {attn0.v_proj.weight.shape}")
print(f"o_proj: {attn0.o_proj.weight.shape}")

# Build structured output
def relative_names(idx, names):
    return sorted(n for n in names if f"layers.{idx}." in n and "." in n.split(f"layers.{idx}.")[1])

attn_pattern = [n for n in suffix0 if "self_attn" in n and n.count(".") == 1]
mlp_pattern = [n for n in suffix0 if "mlp" in n and n.count(".") == 1]

layer_map = {
    "model_name": MODEL_NAME,
    "num_layers": num_layers,
    "pattern_consistent_layer0_vs_mid": pattern_consistent,
    "mid_layer_checked": mid_idx,
    "decoder_layer_linear_suffixes": {
        "attention": [s for s in attn_pattern if s.split(".")[-1] in ("self_attn",)] or 
                     [f"self_attn.{p}" for p in ("q_proj","k_proj","v_proj","o_proj")],
        "mlp": [f"mlp.{p}" for p in ("gate_proj","up_proj","down_proj")]
    },
    "first_last_layer_indices": [0, last_idx],
    "embed_tokens_module": embed_name,
    "lm_head_module": lm_head_name,
    "embed_lm_head_tied": tied,
    "gqa_dims": {
        "q_proj": list(attn0.q_proj.weight.shape),
        "k_proj": list(attn0.k_proj.weight.shape),
        "v_proj": list(attn0.v_proj.weight.shape),
        "o_proj": list(attn0.o_proj.weight.shape),
    }
}

with open("layer_map.json", "w") as f:
    json.dump(layer_map, f, indent=2)

print("\nSaved layer_map.json")