"""Shared patches we apply to the base notebook's arc_solver.py source when
building any of our experimental notebooks. Imported by the build scripts.
"""


# 1. Defensive model resolver — finds the qwen3 model wherever Kaggle mounts it
#    (sorokin or our mirror, several known paths, plus a glob fallback).
RESOLVER_FN = """

def _resolve_model_path():
    \"\"\"Find qwen3_4b wherever Kaggle mounted it; print diagnostics on miss.\"\"\"
    override = os.environ.get(\"AGI2_MODEL_PATH\")
    if override and os.path.exists(os.path.join(override, \"config.json\")):
        print(f\"[model] using AGI2_MODEL_PATH override: {override}\", flush=True)
        return override

    direct = [
        # Our mirror, if attached.
        \"/kaggle/input/qwen3-4b-grids15-sft139-mirror/transformers/bfloat16/1\",
        \"/kaggle/input/models/lucyellu/qwen3-4b-grids15-sft139-mirror/transformers/bfloat16/1\",
        \"/kaggle/input/datasets/lucyellu/qwen3-4b-grids15-sft139-mirror\",
        \"/kaggle/input/lucyellu/qwen3-4b-grids15-sft139-mirror\",
        # Sorokin's original.
        \"/kaggle/input/qwen3_4b_grids15_sft139/transformers/bfloat16/1\",
        \"/kaggle/input/datasets/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1\",
        \"/kaggle/input/datasets/sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1\",
        \"/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1\",
        \"/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1\",
        \"/kaggle/input/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1\",
    ]
    for path in direct:
        if os.path.exists(os.path.join(path, \"config.json\")):
            print(f\"[model] direct hit: {path}\", flush=True)
            return path

    if os.path.isdir(\"/kaggle/input\"):
        from pathlib import Path as _P
        for cfg in _P(\"/kaggle/input\").rglob(\"config.json\"):
            cfg_str = str(cfg)
            if \"qwen3_4b_grids15_sft139\" in cfg_str or \"qwen3-4b-grids15-sft139\" in cfg_str:
                model_dir = str(cfg.parent)
                print(f\"[model] glob hit: {model_dir}\", flush=True)
                return model_dir
        try:
            entries = sorted(os.listdir(\"/kaggle/input\"))
        except OSError:
            entries = []
        print(f\"[model] /kaggle/input/ entries: {entries}\", flush=True)

    raise RuntimeError(
        \"qwen3_4b_grids15_sft139 model not found under /kaggle/input/. \"
        \"Attach via web UI -> Add Input -> Models -> sorokin/qwen3_4b_grids15_sft139 \"
        \"or lucyellu/qwen3-4b-grids15-sft139-mirror.\"
    )

"""


# 2. EarlyStopOnLowLoss callback (ported verbatim from
#    notebooks/reference/nvarc-2025-bf16-faster-ttt-via-early-stopping.ipynb).
#    Stops TTT when training loss drops below `threshold` for `patience`
#    consecutive log steps. Saves wall clock on easy puzzles without
#    affecting harder ones.
EARLY_STOP_CALLBACK = """

class EarlyStopOnLowLoss(TrainerCallback):
    def __init__(self, threshold=0.05, patience=2):
        self.threshold = threshold
        self.patience = patience
        self.low_count = 0
        print(f\"[EarlyStopOnLowLoss] init threshold={threshold} patience={patience}\", flush=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Diagnostic: print every on_log to confirm wiring + show the loss key.
        keys = list(logs.keys()) if logs else None
        loss = logs.get(\"loss\") if logs else None
        print(f\"[EarlyStopOnLowLoss] on_log step={state.global_step} loss={loss} keys={keys}\", flush=True)
        if logs and \"loss\" in logs:
            if logs[\"loss\"] < self.threshold:
                self.low_count += 1
                if self.low_count >= self.patience:
                    print(f\"[Early stop] step {state.global_step} loss={logs['loss']:.4f}\", flush=True)
                    control.should_training_stop = True
            else:
                self.low_count = 0

"""


def apply_path_override(solver_src):
    """Make the test_path resolution check AGI2_CHALLENGES_PATH first."""
    OLD = (
        "    if rerun_mode:\n"
        "        test_path = \"/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json\"\n"
        "    else:\n"
        "        test_path = \"/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json\"\n"
    )
    NEW = (
        "    override = os.environ.get(\"AGI2_CHALLENGES_PATH\")\n"
        "    if override:\n"
        "        test_path = override\n"
        "    elif rerun_mode:\n"
        "        test_path = \"/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json\"\n"
        "    else:\n"
        "        test_path = \"/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json\"\n"
    )
    assert OLD in solver_src, "test_path block not found"
    return solver_src.replace(OLD, NEW)


def apply_model_resolver(solver_src):
    """Wire arc_solver.py to call _resolve_model_path() instead of a hardcoded path."""
    OLD = (
        "    model, tokenizer = FastLanguageModel.from_pretrained(\n"
        "        model_name=\"/kaggle/input/qwen3_4b_grids15_sft139/transformers/bfloat16/1\",\n"
    )
    NEW = (
        "    model_name = _resolve_model_path()\n"
        "    print(f\"[Rank {rank}] resolved model_name -> {model_name}\", flush=True)\n"
        "    model, tokenizer = FastLanguageModel.from_pretrained(\n"
        "        model_name=model_name,\n"
    )
    assert OLD in solver_src, "model_load block not found"
    src = solver_src.replace(OLD, NEW)
    # Inject _resolve_model_path() right after EOS_ID = 15 in the constants block.
    MARKER = "EOS_ID = 15\n"
    assert MARKER in src
    return src.replace(MARKER, MARKER + RESOLVER_FN, 1)


def apply_early_stop(solver_src):
    """Add EarlyStopOnLowLoss callback to TTT.

    Three changes:
    1. Add `TrainerCallback` to the transformers import.
    2. Inject the callback class definition after the import line.
    3. Switch logging_strategy to "steps" with logging_steps=16 (so on_log fires).
    4. Wire callbacks=[...] into the UnslothFixedTrainer instantiation.
    """
    # Step 1: import TrainerCallback
    OLD_IMPORT = "from transformers import DataCollatorForLanguageModeling"
    NEW_IMPORT = "from transformers import DataCollatorForLanguageModeling, TrainerCallback"
    assert OLD_IMPORT in solver_src, "transformers import line not found"
    src = solver_src.replace(OLD_IMPORT, NEW_IMPORT)

    # Step 2: inject the callback class right after the import line.
    src = src.replace(NEW_IMPORT, NEW_IMPORT + EARLY_STOP_CALLBACK, 1)

    # Step 3: switch logging_strategy from "no" to "steps", add logging_steps.
    OLD_LOG = (
        "        save_strategy=\"no\",\n"
        "        eval_strategy=\"no\",\n"
        "        logging_strategy=\"no\",\n"
    )
    NEW_LOG = (
        "        save_strategy=\"no\",\n"
        "        eval_strategy=\"no\",\n"
        "        logging_strategy=\"steps\",\n"
        "        logging_steps=16,\n"
    )
    assert OLD_LOG in src, "logging_strategy block not found"
    src = src.replace(OLD_LOG, NEW_LOG)

    # Step 4: add callbacks list to the trainer call.
    OLD_TRAINER = (
        "            trainer = UnslothFixedTrainer(\n"
        "                model=model,\n"
        "                tokenizer=tokenizer,\n"
        "                data_collator=collator,\n"
        "                train_dataset=Dataset.from_list(train_ds.as_list(formatter)),\n"
        "                dataset_text_field=\"text\",\n"
        "                max_seq_length=max_seq_length,\n"
        "                args=UnslothTrainingArguments(**train_args),\n"
        "            )\n"
    )
    NEW_TRAINER = (
        "            trainer = UnslothFixedTrainer(\n"
        "                model=model,\n"
        "                tokenizer=tokenizer,\n"
        "                data_collator=collator,\n"
        "                train_dataset=Dataset.from_list(train_ds.as_list(formatter)),\n"
        "                dataset_text_field=\"text\",\n"
        "                max_seq_length=max_seq_length,\n"
        "                args=UnslothTrainingArguments(**train_args),\n"
        "                callbacks=[EarlyStopOnLowLoss(threshold=0.05, patience=2)],\n"
        "            )\n"
    )
    assert OLD_TRAINER in src, "UnslothFixedTrainer instantiation block not found"
    return src.replace(OLD_TRAINER, NEW_TRAINER)
