#%%writefile qad_train.py
#NVIDIA QAD (Quantization Aware Distillation) baseline implementation
#Datasets: GSM8K (very cheap dataset) after AIME24/25
#Model

"""
Dual-GPU (2x T4/P100) Quantization-Aware Distillation (QAD) for Kaggle.

The quantized "student" lives on GPU 0 and does all the training 
(optimizer states, gradients, activations).
The full-precision "teacher" lives entirely on GPU 1 and only ever runs
inference (no_grad, no optimizer, no gradient checkpointing needed) -- this
avoids the OOM you get from cramming both models onto a single 14-16GB GPU.

Usage (inside a Kaggle notebook cell or as a script, with 2x T4 enabled):

    !python qad_train.py \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --quant_config_path /kaggle/input/datasets/kerstgsu/ptq-configs/configs/condition6_mlp_only_quantized \
    --data_source nemotron_sft \ #matches papers SFT data however abiliation study shows data quality is not a major factor in QAD's success !
    --nemotron_domain math \
    --output_dir /kaggle/working/qad-output \
    --run_label mlp_only_fixed200 \
    --max_steps 200 \
    --eval_steps 50 #will save checkpoints at 50,100,150,200 (matches paper's approach of evaluating multiple checkpoints and picking the best)

"""

import argparse
import copy
import gc
import json
import os
from pathlib import Path


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.plugins.transformers_trainer import QADTrainer


def _parse_condition_arg(raw):
    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            f"--condition must be in the form name=path, got: {raw!r}"
        )
    name, path = raw.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError(
            f"--condition must contain both a name and path, got: {raw!r}"
        )
    return name, path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model_name",
        default="Qwen/Qwen2.5-1.5B-Instruct"
    )
    p.add_argument("--data_source", choices=["wikitext", "nemotron_sft"], default="wikitext",
                    help="'nemotron_sft' streams real QAD training data from the paper's own "
                         "public release (nvidia/Llama-Nemotron-Post-Training-Dataset), matching "
                         "the setup used for Llama Nemotron Super V1 in the QAD report.")
    p.add_argument("--nemotron_domain", choices=["math", "code", "science", "chat"], default="math",
                    help="Domain split to pull, mirroring the paper's math-only/code-only ablation "
                         "(Table 4/5) that tests cross-domain transfer from partial coverage.")
    p.add_argument("--dataset_name", default="wikitext")
    p.add_argument("--dataset_config", default="wikitext-2-raw-v1") #QAD is remarkably robust to the training data's content (Paper citation)
    p.add_argument("--teacher_model", default=None,
                    help="Teacher model name for distillation. Defaults to the same model.")
    p.add_argument("--output_dir", default="/kaggle/working/qad-output")
    p.add_argument(
        "--quantized_output_dir",
        default=None,
        help="Where to save the intermediate (post-quantization, pre-training) checkpoint. Defaults to <output_dir>/quantized.",
    )
    p.add_argument(
        "--quant_config_path",
        default=None,
        help="Single quantization config path (legacy; use --condition for comparisons).",
    )
    p.add_argument(
        "--condition", action="append", type=_parse_condition_arg, default=[],
        dest="conditions", metavar="name=path",
        help="Repeatable quantization config, e.g. --condition condition3_heuristic=/kaggle/working/configs/condition3_fixed_heuristic.json",
    )
    p.add_argument(
        "--run_label",
        default=None,
        help="Short identifier for this run, e.g. 'baseline', 'zero_exemption', "
             "'fixed_heuristic'. When set, it's appended as a subfolder under --output_dir "
             "(and --quantized_output_dir, if that wasn't explicitly overridden) so sibling "
             "comparison runs don't clobber each other's checkpoints, it's used as the HF "
             "Trainer run_name, and it's recorded (along with --quant_config_path and "
             "--recipe) in a run_metadata.json written to the run's output_dir for "
             "run_comparison.py to pick up.",
    )
    p.add_argument("--max_steps", type=int, default=300,
                    help="Upper bound on optimizer steps, not a target -- load_best_model_at_end "
                         "picks the lowest-eval-loss checkpoint along the way, so it's fine to set "
                         "this generously (budget permitting) and let early checkpoints win.")
    p.add_argument("--num_train_examples", type=int, default=20000,
                    help="Raised well past the old 2000 cap so a multi-thousand-step run isn't "
                         "just re-looping over the same tiny slice of data.")
    p.add_argument("--eval_fraction", type=float, default=0.02,
                    help="Fraction of the training set held out for periodic KL-divergence eval.")
    p.add_argument("--eval_steps", type=int, default=50,
                    help="How often to run eval / save a checkpoint candidate.")
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=1e-5,
                    help="Paper (Xin et al. 2026) recommends 1e-5 to 1e-6 for QAD")
    p.add_argument("--max_seq_length", type=int, default=512,
                    help="Kept short to control activation memory on 16GB.")
    return p.parse_args()


class DualGPUQADTrainer(QADTrainer):
    #QADTrainer variant that keeps the teacher on its own GPU.

    def _compute_teacher_outputs(self, inputs):
        teacher_device = next(self._teacher_model.parameters()).device
        student_device = next(self.model.parameters()).device

        moved_inputs = {
            k: (v.to(teacher_device) if torch.is_tensor(v) else v) for k, v in inputs.items()
        }

        with torch.no_grad(), self._ds_gather(self._teacher_model.parameters()):
            self._teacher_model.eval()
            outputs = self._teacher_model(**moved_inputs)

        if outputs.logits.device != student_device:
            outputs.logits = outputs.logits.to(student_device)
        return outputs


def load_quant_config(path):
    with open(path) as f:
        cfg = json.load(f)

    for rule in cfg.get("quant_cfg", []):
        c = rule.get("cfg")
        if not isinstance(c, dict):
            continue

        # num_bits round-trips through JSON as a list, but modelopt distinguishes
        # float-format quant (tuple of (mantissa, exponent) bits) from int quant
        # via isinstance(num_bits, tuple) -- MUST be restored or FP4 detection breaks.
        if isinstance(c.get("num_bits"), list):
            c["num_bits"] = tuple(c["num_bits"])

        bs = c.get("block_sizes")
        if isinstance(bs, dict):
            fixed_bs = {}
            for k, v in bs.items():
                if k not in ("type", "scale_bits", "scale_block_sizes", "four_over_six"):
                    k = int(k)
                if k == "scale_bits" and isinstance(v, list):
                    v = tuple(v)
                fixed_bs[k] = v
            c["block_sizes"] = fixed_bs

    n_rules = len(cfg.get("quant_cfg", []))
    print(
        f"Loaded quant config from {path}: {n_rules} quant_cfg rule(s), "
        f"algorithm={cfg.get('algorithm')}."
    )
    return cfg


def load_student(model_name, quant_config_path, device, torch_dtype, forward_loop, quantized_output_dir):
    path = Path(quant_config_path)
    if path.is_dir():
        # Already quantized + calibrated by build_quant_configs.py — just load it.
        print(f"Loading pre-quantized checkpoint from {path} (skipping re-quantization).")
        student = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch_dtype).to(device)
    else:
        # Legacy path: raw JSON recipe, no calibration baked in yet — quantize here.
        quant_cfg = load_quant_config(str(path))
        student = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype).to(device)
        student = mtq.quantize(student, quant_cfg, forward_loop)
        os.makedirs(quantized_output_dir, exist_ok=True)
        student.save_pretrained(quantized_output_dir)
    student.gradient_checkpointing_enable()
    return student


def run_training(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    num_gpus = torch.cuda.device_count()
    teacher_device = "cuda:1" if num_gpus > 1 else device  # run teacher on second GPU
    torch_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32  # bfloat (used in paper)
    teacher_model_name = args.teacher_model or args.model_name

    if args.run_label:
        # Keep sibling comparison runs (e.g. baseline / zero_exemption / fixed_heuristic)
        # from writing checkpoints on top of each other when they share --output_dir.
        args.output_dir = os.path.join(args.output_dir, args.run_label)
    quantized_output_dir = args.quantized_output_dir or os.path.join(args.output_dir, "quantized")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "run_metadata.json"), "w") as f:
        json.dump(
            {
                "run_label": args.run_label,
                "quant_config_path": args.quant_config_path,
                "model_name": args.model_name,
                "teacher_model": teacher_model_name,
                "data_source": args.data_source,
            },
            f,
            indent=2,
        )

    print(f"Detected {num_gpus} GPU(s). Student device: {device}. Teacher device: {teacher_device}.")

    mto.enable_huggingface_checkpointing()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Training set  ---
    if args.data_source == "nemotron_sft":
        streamed = load_dataset(
            "nvidia/Llama-Nemotron-Post-Training-Dataset",
            "SFT",
            split=args.nemotron_domain,
            streaming=True,
        )

        def to_text(example):
            prompt = example.get("input", "")
            response = example.get("output", "")
            return {"text": f"{prompt}{response}"}

        examples = []
        for i, ex in enumerate(streamed.map(to_text)):
            if i >= args.num_train_examples:
                break
            if ex["text"].strip():
                examples.append(ex["text"])

        train_dataset = Dataset.from_dict({"text": examples})
    else:
        raw_dataset = load_dataset(args.dataset_name, args.dataset_config)
        train_dataset = raw_dataset["train"].filter(lambda x: len(x["text"].strip()) > 0)
        train_dataset = train_dataset.select(range(min(args.num_train_examples, len(train_dataset))))

    IGNORE_INDEX = -100  # matches modelopt's KDTrainer padding mask

    def tokenize(example):
        tokenized = tokenizer(
            example["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding="max_length",
        )
        tokenized["labels"] = [
            [tok_id if mask == 1 else IGNORE_INDEX for tok_id, mask in zip(ids, attn)]
            for ids, attn in zip(tokenized["input_ids"], tokenized["attention_mask"])
        ]
        return tokenized

    train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["text"])

    # Hold out a small eval split so we can track validation KL-divergence (the paper's own
    # diagnostic: cross-entropy alone can look fine while the model still diverges from the
    # teacher -- see Table 1 of the QAD report) and pick the best checkpoint instead of trusting
    # a fixed step count.
    split = train_dataset.train_test_split(test_size=args.eval_fraction, seed=42)
    train_dataset, eval_dataset = split["train"], split["test"]
    print(f"Train examples: {len(train_dataset)}. Eval examples: {len(eval_dataset)}.")

    def make_forward_loop(target_device):
        def forward_loop(model):
            for i in range(min(128, len(train_dataset))):
                batch = train_dataset[i]
                input_ids = torch.tensor([batch["input_ids"]]).to(target_device)
                attention_mask = torch.tensor([batch["attention_mask"]]).to(target_device)
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
        return forward_loop

    # --- Load (and, if needed, quantize + calibrate) the student now that
    # train_dataset and make_forward_loop both exist. load_student already
    # handles quantizing, saving, and returning a ready-to-train model, so
    # there is no separate quantization step later. ---
    student = load_student(
        args.model_name, args.quant_config_path, device, torch_dtype,
        make_forward_loop(device), quantized_output_dir,
    )

    # Reload from the saved quantized checkpoint so ModelOpt state is restored
    # through the standard HF save/restore path (keeps behavior identical to
    # the original script's reload-after-quantize step).
    if Path(quantized_output_dir).is_dir() and any(Path(quantized_output_dir).iterdir()):
        del student
        gc.collect()
        torch.cuda.empty_cache()
        student = AutoModelForCausalLM.from_pretrained(
            quantized_output_dir, torch_dtype=torch_dtype
        ).to(device)
        student.gradient_checkpointing_enable()

    tokenizer.save_pretrained(quantized_output_dir)

    # Teacher model
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_model_name, torch_dtype=torch_dtype
    ).to(teacher_device)
    teacher.eval()
    teacher.requires_grad_(False)

    gc.collect()
    torch.cuda.empty_cache()

    distill_config = {
        "teacher_model": teacher,
        "temperature": 1.0,
        "criterion": "logits_loss",
        "liger_jsd_beta": None,  # check modelopt source/docs for the correct default if this errors
    }

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=args.run_label,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        fp16=False,
        bf16=(device.startswith("cuda")),  # NVIDIA's own default bf16
        gradient_checkpointing=True,
        optim="adafactor",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=4, #follows papers approach of valuating various checkpoints (paper:10, me:4)
        disable_tqdm=True,  # for clear training
        load_best_model_at_end=False, #like QAD paper did
    )
    
    if num_gpus > 1:
        training_args._n_gpu = 1

    trainer = DualGPUQADTrainer(
        model=student,
        processing_class=tokenizer,
        args=training_args,
        distill_args=distill_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    trainer.save_model()
    label_suffix = f" (run_label={args.run_label})" if args.run_label else ""
    print(f"Done{label_suffix}. Best QAD checkpoint (by eval_loss) saved to {args.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    if args.conditions and args.quant_config_path:
        raise SystemExit("Use either --condition or --quant_config_path, not both.")
    if not args.conditions and not args.quant_config_path:
        raise SystemExit("Provide --condition name=path or --quant_config_path path.")

    if args.conditions:
        for condition_name, condition_path in args.conditions:
            condition_args = copy.copy(args)
            condition_args.quant_config_path = condition_path
            condition_args.run_label = (
                condition_name if args.run_label is None
                else f"{args.run_label}_{condition_name}"
            )
            run_training(condition_args)
    else:
        run_training(args)