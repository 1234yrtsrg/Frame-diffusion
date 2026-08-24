import argparse
import datetime
import json
import os
import re
from pathlib import Path
import random
import sys

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CSDI_DIR = REPO_ROOT / "CSDI"
sys.path.insert(0, str(CSDI_DIR))

from dataset_keyframe_dataset_60fps import get_dataloader  # noqa: E402
from main_model import CSDI_Express4D  # noqa: E402
from utils import train  # noqa: E402


DEFAULT_SAVE_DIR = REPO_ROOT / "save"


def resolve_config_path(path):
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    for base in (Path(__file__).resolve().parent, REPO_ROOT, CSDI_DIR):
        candidate = base / path
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Config file not found: {path}")


def resolve_checkpoint_path(path):
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    if candidate.suffix:
        resolved = DEFAULT_SAVE_DIR / candidate
        if resolved.is_file():
            return resolved
    else:
        resolved = DEFAULT_SAVE_DIR / candidate / "model.pth"
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(f"Checkpoint not found: {path}")


def resolve_save_folder(path):
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if candidate.parts and candidate.parts[0].lower() == "save":
        return REPO_ROOT / candidate
    return DEFAULT_SAVE_DIR / candidate


def _checkpoint_step_key(path):
    match = re.search(r"checkpoint_step_(\d+)\.pth$", path.name)
    return int(match.group(1)) if match else -1


def find_resume_checkpoint(foldername):
    folder = Path(foldername)
    training_state = folder / "training_state.pth"
    if training_state.is_file():
        return training_state
    checkpoints = sorted(folder.glob("checkpoint_step_*.pth"), key=_checkpoint_step_key)
    if checkpoints:
        return checkpoints[-1]
    model_path = folder / "model.pth"
    if model_path.is_file():
        return model_path
    return None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_training_checkpoint(model, checkpoint_path, fallback_global_step=0):
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        model_state = state["model_state_dict"]
        resume_state = {
            "optimizer_state_dict": state.get("optimizer_state_dict"),
            "scheduler_state_dict": state.get("scheduler_state_dict"),
            "global_step": int(state.get("global_step", fallback_global_step)),
            "epoch_no": int(state.get("epoch_no", 0)),
        }
    elif isinstance(state, dict) and "state_dict" in state:
        model_state = state["state_dict"]
        resume_state = {
            "optimizer_state_dict": None,
            "scheduler_state_dict": None,
            "global_step": fallback_global_step,
            "epoch_no": 0,
        }
    else:
        model_state = state
        resume_state = {
            "optimizer_state_dict": None,
            "scheduler_state_dict": None,
            "global_step": fallback_global_step,
            "epoch_no": 0,
        }

    if all(key.startswith("module.") for key in model_state.keys()):
        model_state = {key.removeprefix("module."): value for key, value in model_state.items()}
    model.load_state_dict(model_state)
    return resume_state


def collect_forward_loss_stats(model, loader, max_batches, seed):
    """Collect unweighted loss terms without optimizer steps or parameter updates."""
    torch.manual_seed(int(seed))
    model.eval()
    totals = {}
    total_items = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= int(max_batches):
                break
            _, components = model(batch, is_train=1, return_loss_components=True)
            batch_items = int(batch["observed_data"].shape[0])
            total_items += batch_items
            for name, value in components.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().mean().cpu()) * batch_items
    if total_items == 0:
        raise ValueError("No batches were available for loss-scale statistics")
    return {
        "seed": int(seed),
        "num_batches": min(int(max_batches), batch_index + 1),
        "num_samples": total_items,
        "unweighted_losses": {name: value / total_items for name, value in sorted(totals.items())},
    }


def main():
    parser = argparse.ArgumentParser(description="Train CSDI on keyframe_dataset_60fps")
    parser.add_argument("--config", type=str, default="CSDI/config/keyframe_dataset_60fps.yaml")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--modelfolder",
        type=str,
        default="keyframe_dataset_60fps",
        help="Save/resume folder under save/. Empty creates save/keyframe_dataset_60fps_TIMESTAMP.",
    )
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--save_interval_steps", type=int, default=None)
    parser.add_argument("--dataset_root", type=str, default=None, help="Override dataset root path.")
    parser.add_argument(
        "--data_dirs",
        type=str,
        default=None,
        help="Comma-separated subdirectories under dataset_root, default: dfew,express4d",
    )
    parser.add_argument("--data_parallel", action="store_true")
    parser.add_argument(
        "--loss_stats_batches",
        type=int,
        default=0,
        help="Run this many deterministic forward-only batches and save raw loss scales before training.",
    )
    parser.add_argument(
        "--loss_stats_only",
        action="store_true",
        help="Exit after forward-only loss statistics; defaults to 4 batches if no count is given.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    config_path = resolve_config_path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["seed"] = args.seed

    if args.dataset_root is not None:
        config["dataset"]["root"] = args.dataset_root
    if args.data_dirs is not None:
        config["dataset"]["data_dirs"] = [item.strip() for item in args.data_dirs.split(",") if item.strip()]

    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError("--batch_size must be positive")
        config["train"]["batch_size"] = args.batch_size
    if args.max_train_steps is not None:
        if args.max_train_steps <= 0:
            raise ValueError("--max_train_steps must be positive")
        config["train"]["max_train_steps"] = args.max_train_steps
    if args.save_interval_steps is not None:
        if args.save_interval_steps <= 0:
            raise ValueError("--save_interval_steps must be positive")
        config["train"]["save_interval_steps"] = args.save_interval_steps

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    foldername = (
        resolve_save_folder(args.modelfolder)
        if args.modelfolder
        else DEFAULT_SAVE_DIR / f"keyframe_dataset_60fps_{current_time}"
    )
    os.makedirs(foldername, exist_ok=True)

    print(json.dumps(config, indent=4))
    print("model folder:", foldername)
    with open(foldername / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    train_loader, test_loader = get_dataloader(
        config,
        seed=args.seed,
        batch_size=config["train"]["batch_size"],
        num_workers=config["train"].get("num_workers", 0),
    )
    train_dataset = train_loader.dataset
    print("train dataset counts:", dict(sorted(train_dataset.dataset_counts.items())))
    print("train data source sequence counts:", dict(sorted(train_dataset.data_source_counts.items())))
    print("train condition counts:", dict(sorted(train_dataset.condition_counts.items())))
    print("target condition ratios:", dict(sorted(train_dataset.condition_ratios.items())))
    print("epoch target counts:", dict(sorted(train_loader.sampler.target_counts.items())))
    print("samples per epoch:", len(train_loader.sampler))
    sampling_stats = train_loader.sampler.epoch_statistics(epoch=0)
    print("epoch 0 condition/source/mask sampling stats:")
    print(json.dumps(sampling_stats, indent=2))
    with open(foldername / "sampling_stats_epoch_0.json", "w", encoding="utf-8") as f:
        json.dump(sampling_stats, f, indent=2)

    model = CSDI_Express4D(
        config,
        args.device,
        target_dim=config["dataset"].get("num_features", 52),
    ).to(args.device)

    checkpoint_path = None
    resume_state = None
    if args.checkpoint:
        checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    elif args.modelfolder:
        checkpoint_path = find_resume_checkpoint(foldername)

    if checkpoint_path is not None:
        fallback_global_step = max(0, _checkpoint_step_key(Path(checkpoint_path)))
        resume_state = load_training_checkpoint(
            model,
            checkpoint_path,
            fallback_global_step=fallback_global_step,
        )
        print(f"Resuming from {checkpoint_path} at global_step {resume_state['global_step']}")
    elif args.modelfolder:
        print(f"No checkpoint found in {foldername}; starting from scratch")

    stats_batches = int(args.loss_stats_batches)
    if args.loss_stats_only and stats_batches <= 0:
        stats_batches = 4
    if stats_batches < 0:
        raise ValueError("--loss_stats_batches must be non-negative")
    if stats_batches > 0:
        loss_stats = collect_forward_loss_stats(model, train_loader, stats_batches, args.seed)
        loss_stats["configured_weights"] = {
            "lambda_vel": float(config["loss"]["lambda_vel"]),
            "lambda_acc": float(config["loss"]["lambda_acc"]),
        }
        print("forward-only loss scale stats:")
        print(json.dumps(loss_stats, indent=2))
        with open(foldername / "forward_loss_stats.json", "w", encoding="utf-8") as f:
            json.dump(loss_stats, f, indent=2)
    if args.loss_stats_only:
        print("Forward-only loss-scale check complete; formal training was not started.")
        return

    if args.data_parallel:
        if not torch.cuda.is_available():
            raise RuntimeError("--data_parallel requires CUDA")
        gpu_count = torch.cuda.device_count()
        if gpu_count < 2:
            raise RuntimeError("--data_parallel requires at least 2 visible CUDA devices")
        print(f"Using DataParallel on {gpu_count} GPUs")
        model = torch.nn.DataParallel(model)

    train(
        model,
        config["train"],
        train_loader,
        valid_loader=test_loader,
        valid_epoch_interval=config["train"].get("valid_epoch_interval", 20),
        foldername=str(foldername),
        resume_state=resume_state,
    )


if __name__ == "__main__":
    main()
