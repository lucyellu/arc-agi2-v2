from unsloth import FastLanguageModel, UnslothTrainingArguments, UnslothTrainer
from arc_loader import ArcDataset, QwenFormatter

import gc
import os
import io
import json
import time
import torch
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from collections import defaultdict

from typing import Any, Union
from transformers import DataCollatorForLanguageModeling

import logging
from contextlib import redirect_stdout, redirect_stderr

from peft import get_peft_model_state_dict, set_peft_model_state_dict

import bz2
import pickle

logging.disable(logging.WARNING)

ARC_VOCAB = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "Ċ": 10,
    "<|im_end|>": 15,
}

ARC_TOKENS = list(ARC_VOCAB.values())
USER_TOKEN_ID = 11
ASSISTANT_TOKEN_ID = 12
PAD_ID = 13
EOS_ID = 15


class UnslothFixedTrainer(UnslothTrainer):

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if hasattr(unwrapped_model, "_get_name") and "unsloth" in unwrapped_model._get_name().lower():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        if hasattr(loss, "clone"):
            loss = loss.clone()
        if self.accelerator.num_processes > 1:
            loss = loss * self.accelerator.num_processes
        return (loss, outputs) if return_outputs else loss


class QwenDataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Detect actual role-header token IDs at runtime
        user_ids = self.tokenizer.encode("<|im_start|>user\n",      add_special_tokens=False)
        asst_ids = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        print(f"[Collator] token IDs — user: {user_ids}, asst: {asst_ids}, eos: [{EOS_ID}]", flush=True)
        self._user_start = user_ids[0]
        self._asst_hdr   = len(asst_ids)

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        batch = super().torch_call(examples)
        for i in range(len(examples)):
            input_ids = batch["input_ids"][i]
            labels = batch["labels"][i]
            last_user_pos = None
            for pos in range(len(input_ids) - 1, -1, -1):
                if input_ids[pos] == self._user_start:
                    last_user_pos = pos
                    break
            if last_user_pos is not None:
                labels[:last_user_pos + self._asst_hdr] = -100
            batch["labels"][i] = labels
        return batch


# =============================================================================
# INFERENCE — turbo DFS beam search (unchanged from rev2)
# =============================================================================

@torch.no_grad()
def turbo_dfs(model, logits, max_new_tokens, max_score, scores, pos, cache,
              start_time, end_time, timeout=45):
    n = logits.size(0)
    top_ids   = torch.argsort(logits, dim=-1, descending=False)[:, :30]
    top_probs = torch.gather(logits.log_softmax(-1), 1, top_ids)

    candidates = [[] for _ in range(n)]
    for i in range(n):
        for j in range(top_ids.size(1)):
            token_id = top_ids[i, j].item()
            if token_id not in ARC_TOKENS:
                continue
            token_score = -top_probs[i, j].item()
            if scores[i] + token_score > max_score:
                break
            candidates[i].append((scores[i] + token_score, [token_id]))

    if max_new_tokens <= 1:
        return {i: cands for i, cands in enumerate(candidates) if cands}

    suffixes = {i: [] for i in range(n)}

    for pos in range(pos, pos + max_new_tokens):

        if time.time() > end_time or time.time() > start_time + timeout:
            break

        batch_tokens = []
        batch_scores = []
        num_alive_beams = 0

        for i in range(n):
            if not candidates[i]:
                batch_tokens.append(EOS_ID)
                batch_scores.append(1000)
            else:
                score, t = candidates[i].pop(0)
                batch_tokens.append(t[-1])
                batch_scores.append(score)
                num_alive_beams += 1

        if num_alive_beams == 0:
            break

        outputs = model(
            input_ids=torch.tensor(batch_tokens, device=model.device, dtype=torch.long).view(-1, 1),
            position_ids=torch.full((n, 1), pos, device=model.device),
            past_key_values=cache,
            return_dict=True,
            use_cache=True,
        )

        next_suffixes = turbo_dfs(
            model,
            logits=outputs.logits[:, -1],
            max_new_tokens=max_new_tokens - 1,
            max_score=max_score,
            scores=batch_scores,
            pos=pos + 1,
            cache=outputs.past_key_values,
            start_time=start_time,
            end_time=end_time,
        )

        for batch_id, beams in next_suffixes.items():
            for score, suffix_tokens in beams:
                suffix_tokens.insert(0, batch_tokens[batch_id])
                suffixes[batch_id].append((score, suffix_tokens))

    return suffixes


@torch.no_grad()
def inference_turbo_dfs(model, prefix_tokens, max_new_tokens, max_score, end_time):
    input_ids = torch.tensor(prefix_tokens, device=model.device, dtype=torch.long)
    outputs = model(input_ids=input_ids, return_dict=True, use_cache=True)
    suffixes = turbo_dfs(
        model,
        logits=outputs.logits[:, -1],
        max_new_tokens=max_new_tokens,
        max_score=max_score,
        scores=[0.0] * input_ids.size(0),
        pos=input_ids.size(1),
        cache=outputs.past_key_values,
        start_time=time.time(),
        end_time=end_time,
    )
    result = []
    for batch_id, beams in suffixes.items():
        sorted_beams = sorted(beams, key=lambda x: x[0])
        result.append((batch_id, sorted_beams))
    return result


@torch.no_grad()
def calc_scores(queries, answers, tokenizer, model):
    batch_query_tokens = []
    batch_answer_tokens = []
    batch_tokens = []
    batch_lengths = []
    for query, answer in zip(queries, answers):
        query_tokens = tokenizer.encode(query)
        answer_tokens = tokenizer.encode(answer)
        tokens = query_tokens + answer_tokens
        batch_query_tokens.append(query_tokens)
        batch_answer_tokens.append(answer_tokens)
        batch_tokens.append(tokens)
        batch_lengths.append(len(tokens))
    max_len = max(batch_lengths)
    padded_tokens = []
    for tokens in batch_tokens:
        padded = tokens + [PAD_ID] * (max_len - len(tokens))
        padded_tokens.append(padded)
    input_ids = torch.tensor(padded_tokens, device=model.device, dtype=torch.long)
    outputs = model(input_ids=input_ids, return_dict=True, use_cache=True)
    batch_logits = outputs.logits.float().cpu().log_softmax(-1)
    result = []
    for logits, query_tokens, answer_tokens in zip(batch_logits, batch_query_tokens, batch_answer_tokens):
        query_length = len(query_tokens)
        answer_logits = logits[query_length - 1:query_length - 1 + len(answer_tokens)]
        answer_score = answer_logits[torch.arange(len(answer_tokens)), answer_tokens].sum()
        result.append(-answer_score.item())
    return result


# =============================================================================
# GRADING / OPERATION UTILITIES (unchanged from rev2)
# =============================================================================

def grid_diff_score(attempt, expected):
    """Cell-by-cell match fraction. Returns dict with overall_match (0–1)."""
    if attempt is None:
        return {"shape_match": False, "overall_match": 0.0,
                "color_diff_total": 2.0, "per_color": {}}
    exp = np.array(expected, dtype=int)
    att = np.array(attempt, dtype=int)
    if att.shape != exp.shape:
        return {"shape_match": False, "overall_match": 0.0,
                "color_diff_total": 2.0, "per_color": {}}
    overall = float(np.mean(att == exp))
    n = att.size
    per_color = {}
    for c in range(10):
        a_c = int(np.sum(att == c))
        e_c = int(np.sum(exp == c))
        if a_c > 0 or e_c > 0:
            per_color[c] = round(abs(a_c - e_c) / n, 4)
    return {
        "shape_match": True,
        "overall_match": round(overall, 4),
        "color_diff_total": round(sum(per_color.values()), 4),
        "per_color": per_color,
    }


def count_colors_grid(grid):
    arr = np.array(grid, dtype=int)
    return {str(c): int(np.sum(arr == c)) for c in range(10) if np.any(arr == c)}


def detect_operations(inp, out):
    """Detect what operations map inp -> out. Returns list of op dicts."""
    ops = []
    if inp is None or out is None:
        return ops
    inp = np.array(inp, dtype=int)
    out = np.array(out, dtype=int)
    if inp.shape == out.shape:
        ops.append({"type": "same_size", "shape": list(inp.shape)})
    else:
        ops.append({"type": "shape_change",
                    "from": list(inp.shape), "to": list(out.shape)})
        return ops

    n_changed = int(np.sum(inp != out))
    ops.append({"type": "pixel_diff", "changed": n_changed,
                "total": inp.size,
                "pct_changed": round(n_changed / max(inp.size, 1), 4)})

    mapping, consistent = {}, True
    for ic, oc in zip(inp.flat, out.flat):
        ic, oc = int(ic), int(oc)
        if ic in mapping:
            if mapping[ic] != oc:
                consistent = False
                break
        else:
            mapping[ic] = oc
    if consistent and mapping:
        ops.append({"type": "color_remap",
                    "mapping": {str(k): v for k, v in mapping.items()}})

    for k in [1, 2, 3]:
        if np.array_equal(np.rot90(inp, k), out):
            ops.append({"type": "rotation", "degrees": k * 90})
    if np.array_equal(np.flip(inp, 0), out):
        ops.append({"type": "reflect_vertical"})
    if np.array_equal(np.flip(inp, 1), out):
        ops.append({"type": "reflect_horizontal"})

    return ops


def apply_operations(ops, grid):
    g = np.array(grid, dtype=int).copy()
    for op in ops:
        t = op.get("type")
        if t == "color_remap":
            result = np.zeros_like(g)
            for ic, oc in op["mapping"].items():
                result[g == int(ic)] = int(oc)
            g = result
        elif t == "rotation":
            g = np.rot90(g, k=op["degrees"] // 90)
        elif t == "reflect_vertical":
            g = np.flip(g, 0).copy()
        elif t == "reflect_horizontal":
            g = np.flip(g, 1).copy()
    return g


def cross_validate_ops(ops, train_pairs, exclude_idx=0):
    results = []
    for i, pair in enumerate(train_pairs):
        if i == exclude_idx:
            continue
        predicted = apply_operations(ops, pair["input"])
        diff = grid_diff_score(predicted, pair["output"])
        results.append({
            "train_idx": i,
            "diff": diff,
            "cross_correct": diff["overall_match"] == 1.0,
        })
    return results


def infer_single_grid(model, formatter, tokenizer, prompt_tokens, max_new_tokens, end_time):
    """
    Greedy LOO inference. Returns (grid_or_None, raw_output_text).
    Time-limited to 45s.
    """
    input_ids = torch.tensor([prompt_tokens], device=model.device, dtype=torch.long)
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, return_dict=True, use_cache=True)
        suffixes = turbo_dfs(
            model,
            logits=outputs.logits[:, -1],
            max_new_tokens=max_new_tokens,
            max_score=-np.log(0.05),
            scores=[0.0],
            pos=input_ids.size(1),
            cache=outputs.past_key_values,
            start_time=time.time(),
            end_time=min(end_time, time.time() + 45),
        )
    for _batch_id, beams in suffixes.items():
        for _score, tokens in sorted(beams, key=lambda x: x[0]):
            raw_text = tokenizer.decode(tokens, skip_special_tokens=False)
            arr = formatter.convert_tokens_to_array(tokens)
            if arr is not None:
                return arr, raw_text
    return None, ""


def _json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def write_detailed_json(path, data):
    with open(path, "w") as _f:
        json.dump(data, _f, default=_json_safe, indent=2)


def append_run_log(log_path, line):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# =============================================================================
# MAIN WORKER
# =============================================================================

def worker(rank, queue, end_time):

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    import unsloth.models.qwen3 as qwen3_module
    import torch.nn.functional as F

    def patched_attention(Q, K, V, *args, **kwargs):
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        num_q_heads = Q.shape[1]
        num_kv_heads = K.shape[1]
        if num_q_heads != num_kv_heads:
            factor = num_q_heads // num_kv_heads
            K = K.repeat_interleave(factor, dim=1)
            V = V.repeat_interleave(factor, dim=1)
        scale = Q.shape[-1] ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        weights = torch.softmax(scores.float(), dim=-1).to(Q.dtype)
        out = torch.matmul(weights, V)
        return out.transpose(1, 2)

    qwen3_module.flash_attn_func = patched_attention
    import torch.cuda.amp as amp
    amp.GradScaler = lambda **kwargs: torch.cuda.amp.GradScaler(enabled=False, **kwargs)
    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")

    # ── Config (from env vars) ────────────────────────────────────────────────
    # copy_seed=True: LOO inference starts from a copy of the input grid
    # This makes operations trackable as input->output diffs
    copy_seed = os.environ.get("ARC_COPY_SEED", "1") == "1"   # default ON for rev3
    puzzle_timeout = int(os.environ.get("ARC_PUZZLE_TIMEOUT", "600"))  # 10min default

    print(f"[Rank {rank}] rev3  copy_seed={copy_seed}  puzzle_timeout={puzzle_timeout}s")

    peft_params = dict(
        r=256,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                        "embed_tokens", "lm_head"],
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing=False,
        random_state=42,
        use_rslora=True,
        loftq_config=None,
    )

    train_args = dict(
        per_device_eval_batch_size=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        warmup_steps=0,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        learning_rate=5e-5,
        optim="adamw_8bit",
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",
        save_strategy="no",
        eval_strategy="no",
        logging_strategy="no",
        fp16=False,
        bf16=False,
        fsdp="",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,
        gradient_checkpointing=False,
        half_precision_backend="cpu_amp",
        no_cuda=False,
    )

    max_seq_length = 8192

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1",
        full_finetuning=False,
        load_in_4bit=True,
        local_files_only=True,
        use_gradient_checkpointing=False,
        max_seq_length=max_seq_length,
        dtype=torch.float16,
        attn_implementation="eager",
    )

    model = FastLanguageModel.get_peft_model(model, **peft_params)
    model.config._attn_implementation = "eager"

    default_weights = get_peft_model_state_dict(model, adapter_name="default")
    default_weights = {k: v.clone().detach() for k, v in default_weights.items()}

    collator = QwenDataCollatorForCompletionOnlyLM(tokenizer=tokenizer, mlm=False)
    formatter = QwenFormatter(tokenizer=tokenizer)
    max_new_tokens = formatter.max_new_tokens()
    max_score = -np.log(0.2)

    if rerun_mode:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"
    else:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"

    arc_test_set = ArcDataset.from_file(test_path)

    dir_outputs = "/kaggle/inference_outputs"
    os.makedirs(dir_outputs, exist_ok=True)
    detail_dir = "/kaggle/working"
    os.makedirs(detail_dir, exist_ok=True)
    log_path = os.path.join(detail_dir, "run_log.txt")

    # ── Main puzzle loop ──────────────────────────────────────────────────────
    puzzle_idx = 0

    while not queue.empty():

        if time.time() > end_time:
            print(f"[Rank {rank}] stop!")
            break

        item = queue.get()
        if item is None:
            break

        if isinstance(item, dict):
            key             = item["key"]
            puzzle_level    = item.get("level", 0)
            puzzle_complexity = item.get("complexity", 0)
        else:
            key             = item
            puzzle_level    = 0
            puzzle_complexity = 0

        puzzle_idx += 1
        start_time = time.time()
        ts_start = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))

        torch.cuda.reset_peak_memory_stats()

        train_pairs = arc_test_set.queries[key]["train"]
        n_train     = len(train_pairs)
        n_test      = len(arc_test_set.queries[key]["test"])

        # Rev3: LOO possible with any n_train >= 2 (val = train[0], always fixed)
        # Progressive: start TTT with train[1] only, add train[2], train[3], ...
        has_loo = n_train >= 2

        # Baseline: input-copy diff (how far is input from expected output)
        baseline_diff = grid_diff_score(
            np.array(train_pairs[0]["input"], dtype=int),
            train_pairs[0]["output"]
        ) if has_loo else None

        puzzle_detail = {
            "puzzle_id"               : key,
            "level"                   : puzzle_level,
            "complexity"              : puzzle_complexity,
            "n_train"                 : n_train,
            "n_test"                  : n_test,
            "has_loo"                 : has_loo,
            "copy_seed"               : copy_seed,
            "start_ts"                : round(start_time, 2),
            "val_pair_idx"            : 0,           # always train[0]
            "val_input_color_counts"  : count_colors_grid(train_pairs[0]["input"])  if has_loo else None,
            "val_expected_color_counts": count_colors_grid(train_pairs[0]["output"]) if has_loo else None,
            "attempt0_baseline_diff"  : baseline_diff,
            "revision_rounds"         : [],
            "best_round"              : None,
            "best_val_match"          : None,
            "val_perfect"             : False,
            "token_counts"            : {},
            "timing"                  : {
                "ttt_seconds"             : [],
                "loo_inference_seconds"   : [],
                "test_inference_seconds"  : None,
                "total_seconds"           : None,
            },
            "test_outputs"            : {},
        }

        # ── Reset LoRA to defaults at start of each puzzle ────────────────────
        set_peft_model_state_dict(model, default_weights.copy(), adapter_name="default")

        # ── PROGRESSIVE REVISION LOOP ─────────────────────────────────────────
        # val_pair is always train[0] (fixed throughout)
        # n_ttt starts at 1 (TTT on train[1] only), grows each round
        # No LoRA reset between rounds — cumulative fine-tuning
        #
        # Stop when: LOO = 100%  OR  time budget hit  OR  all pairs exhausted
        # ─────────────────────────────────────────────────────────────────────

        best_val_match    = -1.0
        best_weights_ckpt = {k: v.clone().detach()
                             for k, v in model.named_parameters() if v.requires_grad}
        val_pair          = train_pairs[0]
        n_ttt             = 1          # number of pairs currently used for TTT

        if has_loo:
            round_idx = 0

            while True:

                # Time check (leave 30% for test inference)
                elapsed = time.time() - start_time
                if elapsed > puzzle_timeout * 0.70:
                    print(f"[Rank {rank}] time budget at {elapsed:.0f}s, moving to test")
                    break

                # Build TTT dataset from train[1 : 1+n_ttt]
                ttt_pairs = train_pairs[1 : 1 + n_ttt]
                loo_ttt_data = {key: {
                    "train": ttt_pairs,
                    "test" : arc_test_set.queries[key]["test"],
                }}
                puzzle_ds_ttt = ArcDataset(queries=loo_ttt_data)
                train_ds = puzzle_ds_ttt.augment(n=16, shfl_keys=True, seed=1 + round_idx)
                train_ds = train_ds.cut_to_len(formatter=formatter, name="text",
                                               max_len=max_seq_length)
                ttt_samples = train_ds.as_list(formatter)

                if round_idx == 0:
                    sample_lens = [len(tokenizer.encode(s["text"])) for s in ttt_samples[:4]]
                    puzzle_detail["token_counts"]["ttt_avg_seq_len"] = int(np.mean(sample_lens))
                    puzzle_detail["token_counts"]["ttt_n_samples"]   = len(ttt_samples)

                # TTT (no LoRA reset — cumulative from previous round)
                model = FastLanguageModel.for_training(model)
                ttt_start = time.time()

                with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
                    trainer = UnslothFixedTrainer(
                        model=model,
                        tokenizer=tokenizer,
                        data_collator=collator,
                        train_dataset=Dataset.from_list(ttt_samples),
                        dataset_text_field="text",
                        max_seq_length=max_seq_length,
                        args=UnslothTrainingArguments(**train_args),
                    )
                    stats = trainer.train()
                    model = trainer.accelerator.unwrap_model(model, keep_fp32_wrapper=False)
                    del trainer

                model = FastLanguageModel.for_inference(model)
                model.config._attn_implementation = "eager"
                gc.collect()
                torch.cuda.empty_cache()

                ttt_elapsed = round(time.time() - ttt_start, 1)
                puzzle_detail["timing"]["ttt_seconds"].append(ttt_elapsed)
                ttt_loss = round(float(stats.training_loss), 5)
                print(f"[Rank {rank}] {key} L{puzzle_level} round={round_idx} "
                      f"n_ttt={n_ttt} TTT={ttt_elapsed:.0f}s loss={ttt_loss}")

                # ── LOO inference on val_pair (train[0]) ──────────────────────
                loo_start = time.time()

                loo_context = formatter.fmt_train(ttt_pairs)   # same as TTT context
                loo_query   = formatter.fmt_query([{"input": val_pair["input"]}])

                if copy_seed:
                    from arc_loader import convert_grid_to_string as _cgts
                    seed_text  = _cgts(val_pair["input"])
                    loo_prompt = loo_context + loo_query + seed_text
                else:
                    loo_prompt = loo_context + loo_query

                loo_tokens = tokenizer.encode(loo_prompt)
                puzzle_detail["token_counts"]["loo_prompt_tokens"] = len(loo_tokens)

                val_attempt, raw_output = infer_single_grid(
                    model, formatter, tokenizer, loo_tokens, max_new_tokens, end_time)

                loo_elapsed = round(time.time() - loo_start, 1)
                puzzle_detail["timing"]["loo_inference_seconds"].append(loo_elapsed)

                diff = grid_diff_score(val_attempt, val_pair["output"])

                # Operations: input -> model_output (copy_seed makes this meaningful)
                ops = detect_operations(
                    val_pair["input"],
                    val_attempt.tolist() if val_attempt is not None else None,
                )
                cross_val = cross_validate_ops(ops, train_pairs, exclude_idx=0)

                round_detail = {
                    "round"              : round_idx,
                    "n_ttt_pairs"        : n_ttt,
                    "copy_seed"          : copy_seed,
                    "ttt_loss"           : ttt_loss,
                    "attempt"            : val_attempt.tolist() if val_attempt is not None else None,
                    "attempt_color_counts": count_colors_grid(val_attempt) if val_attempt is not None else {},
                    "diff"               : diff,
                    "operations"         : ops,
                    "cross_validation"   : cross_val,
                    # What the model saw and generated (truncated to 2000 chars each for size)
                    "loo_prompt_text"    : loo_prompt[:2000],
                    "loo_raw_output"     : raw_output[:500],
                    "elapsed_s"          : round(time.time() - start_time, 1),
                }
                puzzle_detail["revision_rounds"].append(round_detail)

                cross_ok = sum(1 for r in cross_val if r["cross_correct"])
                match    = diff["overall_match"]
                print(f"[Rank {rank}] LOO round={round_idx} n_ttt={n_ttt} "
                      f"match={match:.3f} cross={cross_ok}/{len(cross_val)} "
                      f"ops={[o['type'] for o in ops[:3]]} loo={loo_elapsed:.0f}s")

                # Track best
                if match > best_val_match:
                    best_val_match = match
                    best_weights_ckpt = {
                        k: v.clone().detach()
                        for k, v in model.named_parameters()
                        if v.requires_grad
                    }
                    puzzle_detail["best_round"]     = round_idx
                    puzzle_detail["best_val_match"] = match

                if match == 1.0:
                    puzzle_detail["val_perfect"] = True
                    print(f"[Rank {rank}] PERFECT LOO at round {round_idx}! Moving to test.")
                    break

                # Can we add another pair to TTT next round?
                if 1 + n_ttt < n_train:
                    n_ttt += 1   # add next pair (no LoRA reset)
                    print(f"[Rank {rank}] adding pair → n_ttt={n_ttt}")

                round_idx += 1

        # ── Restore best weights before test inference ─────────────────────────
        with torch.no_grad():
            for k, v in model.named_parameters():
                if k in best_weights_ckpt:
                    v.copy_(best_weights_ckpt[k])
        model = FastLanguageModel.for_inference(model)
        model.config._attn_implementation = "eager"
        gc.collect()
        torch.cuda.empty_cache()

        memory_after_ttt = torch.cuda.max_memory_allocated() // 1024**2
        print(f"[Rank {rank}] {memory_after_ttt}MB after TTT for {key}")
        torch.cuda.reset_peak_memory_stats()

        # ── Test inference (standard turbo_dfs — unchanged) ───────────────────
        inf_start   = time.time()
        puzzle_ds   = arc_test_set.change_keys([key])
        puzzle_ds_multi = puzzle_ds.split_multi_replies()
        eval_ds     = puzzle_ds_multi.augment(n=2, seed=2)
        eval_ds     = eval_ds.cut_to_len(formatter=formatter, name="input",
                                          max_len=max_seq_length - max_new_tokens)

        test_id_to_subkeys = defaultdict(list)
        for subkey in sorted(eval_ds.keys):
            test_id = subkey.split(".")[0].split("_")[1]
            test_id_to_subkeys[test_id].append(subkey)

        batches = []
        for test_id, subkeys in test_id_to_subkeys.items():
            batch = []
            for offset in [0, 4]:
                batch.extend(subkeys[offset:offset + 2])
            batches.append(batch)
            batch = []
            for offset in [2, 6]:
                batch.extend(subkeys[offset:offset + 2])
            batches.append(batch)
        for test_id, subkeys in test_id_to_subkeys.items():
            batch = []
            for offset in [8, 12]:
                batch.extend(subkeys[offset:offset + 2])
            batches.append(batch)
            batch = []
            for offset in [10, 14]:
                batch.extend(subkeys[offset:offset + 2])
            batches.append(batch)

        if eval_ds.keys:
            first_data = eval_ds.get(eval_ds.keys[0], formatter)
            puzzle_detail["token_counts"]["test_prompt_tokens"] = len(
                tokenizer.encode(first_data["input"]))

        with torch.inference_mode():
            known_scores = {}

            for subkeys in batches:
                spend_time = time.time() - start_time
                if spend_time > puzzle_timeout or time.time() > end_time:
                    print(f"[Rank {rank}] timeout after {spend_time:.1f}s for {key}")
                    break

                print(f"[Rank {rank}] decoding {subkeys}")

                tokens = []
                for subkey in subkeys:
                    data = eval_ds.get(subkey, formatter)
                    tokens.append(tokenizer.encode(data["input"]))

                dfs_result = inference_turbo_dfs(model, tokens, max_new_tokens,
                                                  max_score, end_time)

                for subkey_id, scored_beams in dfs_result:
                    subkey      = subkeys[subkey_id]
                    bk          = subkey.split(".")[0]
                    test_id_str = bk.split("_")[1]
                    decoded_result = []

                    for beam_score, beam_tokens in scored_beams:
                        array = formatter.convert_tokens_to_array(beam_tokens)
                        if array is None:
                            continue
                        solution = puzzle_ds_multi.invert_mod(array, subkey, inv_perm=True)
                        grid_id  = (bk, tuple(map(tuple, solution)))

                        if grid_id in known_scores:
                            augmented_scores = known_scores[grid_id]
                        else:
                            print(f"[Rank {rank}] scoring {subkey} #{len(decoded_result)}")
                            aug_dataset = ArcDataset(
                                keys=[bk],
                                queries={bk: puzzle_ds_multi.queries.get(bk)},
                                replies={bk: [solution.tolist()]},
                            )
                            aug_dataset = aug_dataset.augment(seed=hash(bk) % 1024**2)
                            aug_dataset = aug_dataset.cut_to_len(
                                formatter=formatter, name="input",
                                max_len=max_seq_length - max_new_tokens)
                            aug_queries = []
                            aug_answers = []
                            for s in aug_dataset.as_list(formatter):
                                aug_queries.append(s["input"])
                                aug_answers.append(s["reply"])
                            augmented_scores = (
                                calc_scores(aug_queries[:4], aug_answers[:4], tokenizer, model) +
                                calc_scores(aug_queries[4:], aug_answers[4:], tokenizer, model)
                            )
                            known_scores[grid_id] = augmented_scores

                        decoded_result.append({
                            "beam_score" : beam_score,
                            "score_aug"  : augmented_scores,
                            "solution"   : solution,
                        })

                    if test_id_str not in puzzle_detail["test_outputs"]:
                        puzzle_detail["test_outputs"][test_id_str] = {
                            "n_beams": 0, "top_candidates": []}
                    out_slot = puzzle_detail["test_outputs"][test_id_str]
                    out_slot["n_beams"] += len(decoded_result)
                    for entry in decoded_result[:3]:
                        tok_count = len(tokenizer.encode(
                            formatter.fmt_reply([entry["solution"].tolist()])))
                        out_slot["top_candidates"].append({
                            "beam_nll"           : round(float(entry["beam_score"]), 4),
                            "aug_score_mean"     : round(float(np.mean(entry["score_aug"])), 4),
                            "token_count"        : tok_count,
                            "attempt_color_counts": count_colors_grid(entry["solution"]),
                        })

                    if decoded_result:
                        with bz2.BZ2File(os.path.join(dir_outputs, subkey), "w") as f:
                            pickle.dump(decoded_result, f)

        memory_after_inf = torch.cuda.max_memory_allocated() // 1024**2
        spend_time = round(time.time() - start_time, 1)
        puzzle_detail["timing"]["test_inference_seconds"] = round(time.time() - inf_start, 1)
        puzzle_detail["timing"]["total_seconds"]          = spend_time

        print(f"[Rank {rank}] DONE {key} in {spend_time:.0f}s  "
              f"val_perfect={puzzle_detail['val_perfect']}  "
              f"best_val={puzzle_detail['best_val_match']}  "
              f"rounds={len(puzzle_detail['revision_rounds'])}")

        # ── Running log txt ───────────────────────────────────────────────────
        ts_end   = time.strftime("%H:%M:%S")
        n_rounds = len(puzzle_detail["revision_rounds"])
        best_val = puzzle_detail.get("best_val_match") or 0.0
        perfect  = "PERFECT" if puzzle_detail["val_perfect"] else f"val={best_val:.2f}"
        ops_summary = ""
        if puzzle_detail["revision_rounds"]:
            last_ops = puzzle_detail["revision_rounds"][-1].get("operations", [])
            ops_summary = ",".join(o["type"] for o in last_ops[:3])

        log_line = (
            f"[{ts_end}] puzzle={puzzle_idx:02d}  {key}  L{puzzle_level}  "
            f"n_train={n_train}  rounds={n_rounds}  {perfect}  "
            f"ops={ops_summary}  total={spend_time:.0f}s  copy_seed={copy_seed}"
        )
        append_run_log(log_path, log_line)
        print(f"[Rank {rank}] LOG: {log_line}")

        # ── Write per-puzzle JSON ─────────────────────────────────────────────
        existing_details = [
            f for f in os.listdir(detail_dir)
            if f.startswith("submission_detailed_") and f.endswith(".json")
        ]
        detail_idx  = len(existing_details) + 1
        detail_name = f"submission_detailed_{detail_idx:03d}.json"
        write_detailed_json(os.path.join(detail_dir, detail_name), puzzle_detail)
        print(f"[Rank {rank}] wrote {detail_name}")

        if puzzle_idx % 12 == 0:
            import zipfile
            batch_num = puzzle_idx // 12
            zip_path  = os.path.join(detail_dir, f"submission_batch_{batch_num:03d}.zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for fname in sorted(os.listdir(detail_dir)):
                    if fname.startswith("submission_detailed_") and fname.endswith(".json"):
                        zf.write(os.path.join(detail_dir, fname), fname)
                agg_zip = os.path.join(detail_dir, "submission_detailed.json")
                if os.path.exists(agg_zip):
                    zf.write(agg_zip, "submission_detailed.json")
            print(f"[Rank {rank}] wrote batch zip {zip_path}")

        # ── Update aggregate JSON ─────────────────────────────────────────────
        agg_path = os.path.join(detail_dir, "submission_detailed.json")
        try:
            with open(agg_path) as f:
                agg = json.load(f)
        except Exception:
            agg = {}
        agg[key] = puzzle_detail
        write_detailed_json(agg_path, agg)
