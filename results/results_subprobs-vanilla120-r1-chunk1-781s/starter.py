import os
import time
import json
import torch
import argparse
import torch.multiprocessing as mp


def local_worker(rank, queue, end_time):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    torch.set_default_device("cpu")

    if rank > 0:
        while not os.path.exists(f"/kaggle/worker{rank-1}"):
            time.sleep(5)

    from arc_solver import worker

    with open(f"/kaggle/worker{rank}", "w") as f:
        f.write("Ok")

    print(f"[Rank {rank}] start!")

    worker(rank, queue, end_time)

    print(f"[Rank {rank}] done!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--end-time", type=float, default=0.0)
    args = parser.parse_args()

    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")

    override = os.environ.get("AGI2_CHALLENGES_PATH")
    if override:
        test_path = override
    elif rerun_mode:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"
    else:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"

    with open(test_path, "r") as f:
        data = json.load(f)

    queue = mp.Manager().Queue()

    env_keys = os.environ.get("AGI2_DEV_KEYS", "").split()
    if env_keys:
        eval_keys = [k for k in env_keys if k in data]
        missing = [k for k in env_keys if k not in data]
        if missing:
            print(f"WARNING: AGI2_DEV_KEYS missing from challenges: {missing}")
    else:
        eval_keys = sorted(data.keys())

    print(f"Queueing {len(eval_keys)} puzzles from {test_path}")
    for key in eval_keys:
        queue.put(key)

    nprocs = int(os.environ.get("AGI2_NPROCS", "0")) or min(4, max(1, torch.cuda.device_count()))
    print(f"Using nprocs={nprocs}")
    for _ in range(nprocs):
        queue.put(None)

    mp.spawn(local_worker, args=(queue, args.end_time), nprocs=nprocs)
