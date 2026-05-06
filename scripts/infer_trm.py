#!/usr/bin/env python3
"""
TRM inference runner for the ARC-AGI dashboard.

Runs test-time fine-tuning + prediction on a set of ARC puzzles using the
TinyRecursiveModel (arc_v2_public checkpoint from HuggingFace).

Usage (standalone):
  python infer_trm.py --challenges ../data/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json
                      --ids 00576224 007bbfb7
                      --output results/submission.json
                      --num-aug 8 --epochs 200 --gpus 1

The --num-aug and --epochs flags control speed vs quality:
  - Quick test:    --num-aug 4  --epochs 100
  - Normal:        --num-aug 16 --epochs 500
  - Competition:   --num-aug 128 --epochs 4000
"""
import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
TRM = Path(os.environ.get("AGI2_TRM_DIR", ROOT / "external" / "TinyRecursiveModels"))

import constraint_filter as CF
CKPT_V2 = Path(os.environ.get(
    "AGI2_TRM_CKPT_V2",
    ROOT / "external" / "models" / "trm_arc_prize_verification" / "arc_v2_public" / "step_723914",
))
CKPT_V1 = Path(os.environ.get(
    "AGI2_TRM_CKPT_V1",
    ROOT / "external" / "models" / "trm_arc_prize_verification" / "arc_v1_public" / "step_518071",
))

# Eval script: adapted from arc2-trm-v31.ipynb with H_cycles/halt params from HF checkpoint config
EVAL_SCRIPT = r"""
import os, sys, math, yaml, shutil, copy
from typing import Optional, Any, Sequence, List
from dataclasses import dataclass
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
import tqdm, coolname, hydra, pydantic
from omegaconf import DictConfig
from adam_atan2_pytorch import AdamAtan2
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from models.ema import EMAHelper


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig


class EvaluatorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class PretrainConfig(pydantic.BaseModel):
    arch: ArchConfig
    data_paths: List[str]
    data_paths_test: List[str] = []
    evaluators: List[EvaluatorConfig] = []
    global_batch_size: int
    epochs: int
    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int
    weight_decay: float
    beta1: float
    beta2: float
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    load_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    min_eval_interval: Optional[int] = 0
    eval_save_outputs: List[str] = []
    ema: bool = False
    ema_rate: float = 0.999
    freeze_weights: bool = False


@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any
    step: int
    total_steps: int


def create_dataloader(config, split, rank, world_size, **kwargs):
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=config.data_paths_test if len(config.data_paths_test) > 0 and split == "test" else config.data_paths,
        rank=rank,
        num_replicas=world_size,
        **kwargs
    ), split=split)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=1,
                            prefetch_factor=8, pin_memory=True, persistent_workers=True)
    return dataloader, dataset.metadata


def load_checkpoint(model, config):
    if config.load_checkpoint is not None:
        print(f"[TRM] Loading checkpoint: {config.load_checkpoint}", flush=True)
        state_dict = torch.load(config.load_checkpoint, map_location="cuda")
        # Checkpoint was saved from a torch.compile()-d model; strip the _orig_mod. prefix
        # so it loads cleanly into a non-compiled model (DISABLE_COMPILE=1).
        if any(k.startswith("_orig_mod.") for k in state_dict):
            state_dict = {k[len("_orig_mod."):]: v for k, v in state_dict.items()}
        puzzle_emb_name = "model.inner.puzzle_emb.weights"
        expected_shape = model.model.puzzle_emb.weights.shape
        if puzzle_emb_name in state_dict:
            puzzle_emb = state_dict[puzzle_emb_name]
            if puzzle_emb.shape != expected_shape:
                print(f"[TRM] Resizing puzzle embedding {puzzle_emb.shape} -> {expected_shape}", flush=True)
                state_dict[puzzle_emb_name] = (
                    torch.mean(puzzle_emb, dim=0, keepdim=True).expand(expected_shape).contiguous()
                )
        model.load_state_dict(state_dict, assign=True)
        print("[TRM] Checkpoint loaded.", flush=True)


def create_model(config, train_metadata, rank, world_size):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,
        batch_size=config.global_batch_size // world_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False
    )
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device("cuda"):
        model = model_cls(model_cfg)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model)
        if rank == 0:
            load_checkpoint(model, config)
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    if config.arch.puzzle_emb_ndim == 0:
        optimizers = [AdamAtan2(model.parameters(), lr=1e-4, weight_decay=config.weight_decay,
                                betas=(config.beta1, config.beta2))]
        optimizer_lrs = [config.lr]
    elif config.freeze_weights:
        optimizers = [CastedSparseEmbeddingSignSGD_Distributed(
            model.model.puzzle_emb.buffers(), lr=0, weight_decay=config.puzzle_emb_weight_decay,
            world_size=world_size)]
        optimizer_lrs = [config.puzzle_emb_lr]
    else:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(), lr=0, weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size),
            AdamAtan2(model.parameters(), lr=1e-4, weight_decay=config.weight_decay,
                      betas=(config.beta1, config.beta2))
        ]
        optimizer_lrs = [config.puzzle_emb_lr, config.lr]

    return model, optimizers, optimizer_lrs


def cosine_lr(step, base_lr, warmup, total, min_ratio=0.0):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))))


def init_train_state(config, train_metadata, rank, world_size):
    total_steps = int(config.epochs * train_metadata.total_groups *
                      train_metadata.mean_puzzle_examples / config.global_batch_size)
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, rank, world_size)
    return TrainState(step=0, total_steps=total_steps, model=model,
                      optimizers=optimizers, optimizer_lrs=optimizer_lrs, carry=None)


def train_batch(config, train_state, batch, global_batch_size, rank, world_size):
    train_state.step += 1
    if train_state.step > train_state.total_steps:
        return
    batch = {k: v.cuda() for k, v in batch.items()}
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)
    train_state.carry, loss, metrics, _, _ = train_state.model(
        carry=train_state.carry, batch=batch, return_keys=[])
    ((1 / global_batch_size) * loss).backward()
    if world_size > 1:
        for p in train_state.model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad)
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr = cosine_lr(train_state.step, base_lr, config.lr_warmup_steps, train_state.total_steps, config.lr_min_ratio)
        for pg in optim.param_groups:
            pg['lr'] = lr
        optim.step()
        optim.zero_grad()


def evaluate(config, train_state, eval_loader, eval_metadata, evaluators, rank, world_size, cpu_group):
    with torch.inference_mode():
        return_keys = set(config.eval_save_outputs)
        for ev in evaluators:
            ev.begin_eval()
            return_keys.update(ev.required_outputs)

        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}
        metric_keys, metric_values = [], None

        for set_name, batch, _ in eval_loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch)
            while True:
                carry, loss, metrics, preds, all_finish = train_state.model(
                    carry=carry, batch=batch, return_keys=return_keys)
                if all_finish:
                    break
            for ev in evaluators:
                ev.update_batch(batch, preds)
            del carry, loss, preds, batch, all_finish

            set_id = set_ids[set_name]
            if metric_values is None:
                metric_keys = sorted(metrics.keys())
                metric_values = torch.zeros((len(set_ids), len(metrics)), dtype=torch.float32, device="cuda")
            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])
            del metrics

        reduced = None
        if metric_values is not None and world_size > 1:
            dist.reduce(metric_values, dst=0)

    # ev.result() calls dist.gather_object which does inplace ops on CPU tensors;
    # must run outside torch.inference_mode() to avoid "inplace update to inference tensor" error.
    if rank == 0:
        for i, ev in enumerate(evaluators):
            save_path = None
            if config.checkpoint_path:
                save_path = os.path.join(config.checkpoint_path,
                                         f"evaluator_{ev.__class__.__name__}_step_{train_state.step}")
                os.makedirs(save_path, exist_ok=True)
            metrics = ev.result(save_path, rank=rank, world_size=world_size, group=cpu_group)
            if metrics:
                reduced = reduced or {}
                reduced.update(metrics)
    return reduced


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    RANK, WORLD_SIZE, CPU_GROUP = 0, 1, None
    if "LOCAL_RANK" in os.environ:
        _backend = "nccl" if dist.is_nccl_available() else "gloo"
        _world_size = int(os.environ.get("WORLD_SIZE", 1))
        _rank = int(os.environ.get("RANK", 0))
        if _world_size == 1:
            # FileStore avoids TCPStore/libuv (broken on some Windows PyTorch builds)
            import tempfile as _tf
            _store_path = _tf.mktemp(suffix=".dist_store")
            _store = dist.FileStore(_store_path, _world_size)
            dist.init_process_group(backend=_backend, store=_store,
                                    rank=_rank, world_size=_world_size)
        else:
            dist.init_process_group(backend=_backend)
        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        CPU_GROUP = None if _backend == "gloo" else dist.new_group(backend="gloo")

    objects = [None]
    if RANK == 0:
        config = PretrainConfig(**hydra_config)
        if config.project_name is None:
            config.project_name = "arc-dashboard"
        if config.run_name is None:
            config.run_name = "eval"
        if config.checkpoint_path is None:
            config.checkpoint_path = "eval_checkpoint"
        objects = [config]
    if WORLD_SIZE > 1:
        dist.broadcast_object_list(objects, src=0)
    config = objects[0]

    torch.random.manual_seed(config.seed + RANK)

    train_epochs = config.eval_interval or config.epochs
    train_loader, train_meta = create_dataloader(
        config, "train", test_set_mode=False,
        epochs_per_iter=train_epochs, global_batch_size=config.global_batch_size,
        rank=RANK, world_size=WORLD_SIZE)

    eval_loader, eval_meta = create_dataloader(
        config, "test", test_set_mode=True,
        epochs_per_iter=1, global_batch_size=config.global_batch_size,
        rank=RANK, world_size=WORLD_SIZE)

    klass = load_model_class("arc@ARC", "evaluators.")
    evaluators = [klass(data_path=config.data_paths[0], eval_metadata=eval_meta)]

    train_state = init_train_state(config, train_meta, RANK, WORLD_SIZE)
    ema_helper = None
    if config.ema:
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)

    if RANK == 0:
        total = train_state.total_steps
        print(f"[TRM] Fine-tuning {total} steps | checkpoint -> {config.checkpoint_path}", flush=True)

    train_state.model.train()
    for set_name, batch, gbs in train_loader:
        train_batch(config, train_state, batch, gbs, RANK, WORLD_SIZE)
        if RANK == 0 and train_state.step % 50 == 0:
            print(f"[TRM] step {train_state.step}/{train_state.total_steps}", flush=True)
        if config.ema:
            ema_helper.update(train_state.model)

    if RANK == 0:
        print("[TRM] Fine-tuning done. Running eval...", flush=True)

    if config.ema:
        eval_state = copy.deepcopy(train_state)
        eval_state.model = ema_helper.ema_copy(eval_state.model)
    else:
        eval_state = train_state
    eval_state.model.eval()

    evaluate(config, eval_state, eval_loader, eval_meta, evaluators,
             RANK, WORLD_SIZE, CPU_GROUP)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    launch()
"""


def build_dataset(tmp: Path, aug_dir: Path, num_aug: int) -> bool:
    """Run dataset.build_arc_dataset in the TRM directory."""
    result = subprocess.run(
        [sys.executable, "-m", "dataset.build_arc_dataset",
         "--input-file-prefix", str(tmp / "arc-agi"),
         "--output-dir", str(aug_dir),
         "--subsets", "test",
         "--test-set-name", "test",
         "--num-aug", str(num_aug)],
        cwd=str(TRM),
        capture_output=False,
        text=True,
    )
    return result.returncode == 0


def run_eval(eval_script: Path, aug_dir: Path, checkpoint: Path,
             ckpt_out: Path, epochs: int, batch_size: int, gpus: int) -> bool:
    """Run TRM test-time fine-tuning + eval via torchrun."""
    env = os.environ.copy()
    env["DISABLE_COMPILE"] = "1"          # skip torch.compile (safer on older GPUs)
    env["HYDRA_FULL_ERROR"] = "1"

    # Use forward slashes — backslashes are escape chars in Hydra/OmegaConf CLI overrides
    aug_dir_s    = aug_dir.as_posix()
    checkpoint_s = checkpoint.as_posix()
    ckpt_out_s   = ckpt_out.as_posix()

    # bfloat16 requires Ampere (compute 8.0+); fall back to float32 on older GPUs
    try:
        import torch
        bf16_ok = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    except Exception:
        bf16_ok = False
    dtype_override = [] if bf16_ok else ["arch.forward_dtype=float32"]

    hydra_args = [
        "arch=trm",
        f"data_paths=[{aug_dir_s}]",
        "arch.L_layers=2",
        "arch.H_cycles=3",   # matches arc_v2_public config
        "arch.L_cycles=4",
        "arch.halt_max_steps=16",
        *dtype_override,
        "freeze_weights=False",
        f"+load_checkpoint={checkpoint_s}",
        f"+checkpoint_path={ckpt_out_s}",
        f"eval_interval={epochs}",
        f"epochs={epochs}",
        f"global_batch_size={batch_size}",
        "ema=True",
        "lr_warmup_steps=50",
        "lr=0.0001",
        "lr_min_ratio=1.0",
    ]

    if gpus == 1:
        # Single-GPU: run python directly to avoid torchrun's libuv/TCPStore rendezvous
        # (broken on Windows when PyTorch is built without libuv support).
        # Set the standard distributed env vars so LOCAL_RANK is picked up by the script.
        with socket.socket() as _s:
            _s.bind(("", 0))
            free_port = _s.getsockname()[1]
        env.update({"LOCAL_RANK": "0", "RANK": "0", "WORLD_SIZE": "1",
                    "MASTER_ADDR": "localhost", "MASTER_PORT": str(free_port)})
        cmd = [sys.executable, "-u", str(eval_script)] + hydra_args
    else:
        cmd = (["torchrun", "--standalone", f"--nproc-per-node={gpus}", str(eval_script)]
               + hydra_args)

    result = subprocess.run(cmd, cwd=str(TRM), env=env)
    return result.returncode == 0


def find_submission(ckpt_out: Path) -> Path | None:
    hits = list(ckpt_out.glob("**/submission.json"))
    return hits[0] if hits else None


def _add_constraint_scores(submission_path: str, challenges_path: str):
    """
    Load a submission JSON, annotate each attempt dict with _constraint_scores,
    and re-save in place. Safe to call on any submission format.
    """
    try:
        with open(submission_path) as f:
            submission = json.load(f)
        with open(challenges_path) as f:
            all_challenges = json.load(f)

        for pid, attempts_list in submission.items():
            puzzle = all_challenges.get(pid)
            if not puzzle:
                continue
            test_input = puzzle["test"][0]["input"]
            examples   = puzzle["train"]
            for attempt_dict in attempts_list:
                if isinstance(attempt_dict, dict):
                    scores = CF.score_all_attempts(attempt_dict, test_input, examples)
                    attempt_dict["_constraint_scores"] = scores

        with open(submission_path, "w") as f:
            json.dump(submission, f)
    except Exception as e:
        print(f"[TRM] WARNING: constraint scoring failed: {e}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="TRM inference for ARC-AGI dashboard")
    parser.add_argument("--challenges", required=True, help="Path to ARC challenges JSON")
    parser.add_argument("--ids", nargs="*", help="Puzzle IDs to run (default: all)")
    parser.add_argument("--checkpoint", default=str(CKPT_V2), help="Checkpoint path")
    parser.add_argument("--output", required=True, help="Output submission JSON path")
    parser.add_argument("--num-aug", type=int, default=8,
                        help="Augmentations per puzzle (8=fast, 128=competition)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Fine-tuning epochs (200=fast, 4000=competition)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Global batch size (reduce if OOM)")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--checkpoint-name", default=None,
                        help="Human-readable label for this checkpoint (shown in PNG/dashboard)")
    args = parser.parse_args()

    # Derive checkpoint name from path if not specified
    if not args.checkpoint_name:
        args.checkpoint_name = Path(args.checkpoint).parent.name + "/" + Path(args.checkpoint).name

    # Validate
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"ERROR: Checkpoint not found: {ckpt}", flush=True)
        print(f"  Expected: {CKPT_V2}", flush=True)
        sys.exit(1)

    if not TRM.exists():
        print(f"ERROR: TRM code not found at {TRM}", flush=True)
        sys.exit(1)

    # Load and filter challenges
    with open(args.challenges) as f:
        all_challenges = json.load(f)

    if args.ids:
        challenges = {k: v for k, v in all_challenges.items() if k in args.ids}
        missing = set(args.ids) - set(challenges)
        if missing:
            print(f"WARNING: IDs not found: {missing}", flush=True)
    else:
        challenges = all_challenges

    print(f"[TRM] Puzzles: {len(challenges)} | aug: {args.num_aug} | epochs: {args.epochs}", flush=True)

    tmp = Path(tempfile.mkdtemp(prefix="arc_infer_"))
    try:
        # Write subset challenges as "test" split
        (tmp / "arc-agi_test_challenges.json").write_text(json.dumps(challenges))

        # Build augmented dataset
        aug_dir = tmp / "arc-aug"
        print("[TRM] Building augmented dataset...", flush=True)
        if not build_dataset(tmp, aug_dir, args.num_aug):
            print("ERROR: Dataset build failed", flush=True)
            sys.exit(1)
        print(f"[TRM] Dataset built at {aug_dir}", flush=True)

        # Write eval script into TRM dir so Hydra resolves config_path="config" correctly
        eval_script = TRM / "_arc_eval_tmp.py"
        eval_script.write_text(EVAL_SCRIPT)

        # Run inference
        ckpt_out = tmp / "checkpoint"
        print("[TRM] Starting inference (test-time fine-tuning)...", flush=True)
        try:
            ok = run_eval(eval_script, aug_dir, ckpt, ckpt_out,
                          args.epochs, args.batch_size, args.gpus)
        finally:
            eval_script.unlink(missing_ok=True)
        if not ok:
            print("ERROR: Inference failed", flush=True)
            sys.exit(1)

        # Collect results
        sub_file = find_submission(ckpt_out)
        if not sub_file:
            print("ERROR: submission.json not found in output", flush=True)
            sys.exit(1)

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(sub_file, args.output)
        print(f"[TRM] Done. Results saved to: {args.output}", flush=True)

        # Post-process: add constraint scores to every attempt
        _add_constraint_scores(args.output, args.challenges)
        print(f"[TRM] Constraint scores annotated.", flush=True)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
