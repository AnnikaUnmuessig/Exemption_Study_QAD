# Quantization-Aware Distillation: An Exemption Study on Qwen2.5-1.5B-Instruct

This project investigates quantization-aware distillation (QAD) for the Qwen2.5-1.5B-Instruct model, focusing on which parts of the model should be quantized aggressively and which parts should be exempted and left at full precision.

## Overview
The repository implements a QAD evaluation and training pipeline for comparing six different quantization recipes on the same base model.

The main goal of this project is to fill NVIDIAs QAD paper's gap of not documenting the quantization configuration choices and offer an understanding of the importance of quantization settings in terms of performance for both PTQ and QAD.

## Project structure
- `inspecting_model.py` — generates a layer map of the model, which is later used to build the quantization configs.
- `build_quant_configs.py` — creates the quantization recipes and exported checkpoints for each condition.
- `qad_train.py` — trains the QAD model under each condition using the configured quantization policy.
- `qad_eval.py` — evaluates the trained or exported checkpoints on GSM8K.
- `plot_quant_recipe.py` — plots the layer-wise quantization recipe structure across conditions.
- `configs/` — stores the generated quantization configuration files.
- `eval_results/` — stores evaluation outputs.

## Experimental setup
The workflow follows a recipe-based comparison pipeline:

1. Build a quantization configuration for a given condition based on the model's layer map.
2. There are 6 quantization configurations:
   - Baseline: the unquantized, full-precision model (BF16).
   - Zero exemption: every eligible module is quantized, with no exceptions (NVFP4).
   - Fixed heuristic: the original paper's attention + first/last-layer exemption rule, applied to Qwen2.5 despite being designed for a different architecture.
   - AutoQuantize: an automated, sensitivity-based layer selection method that decides which modules to quantize.
   - Attention-only quantization: all MLP projections are exempted; only attention projections are quantized.
   - MLP-only quantization: all attention projections are exempted; only MLP projections are quantized.
3. Evaluate the quantization configurations in post-training quantization (PTQ) settings on GSM8K.
4. Train the quantized student model with QAD.
5. Evaluate on GSM8K using the standard language-model evaluation pipeline.
6. Compare the recipes across conditions to assess which exemption pattern performs best.

See `requirements.txt` for the dependency list.


