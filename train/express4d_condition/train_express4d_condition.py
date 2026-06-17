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

from dataset_express4d_condition import get_dataloader
from main_model import CSDI_Express4D
from utils import train


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


def main():
    parser = argparse.ArgumentParser(description="Train CSDI Express4D with balanced condition windows")
    parser.add_argument("--config", type=str, default="CSDI/config/express4d_condition.yaml")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--modelfolder",
        type=str,
        default="express4d_condition",
        help="Save/resume folder under save/. Empty creates save/express4d_condition_TIMESTAMP.",
    )
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--save_interval_steps", type=int, default=None)
    parser.add_argument("--data_parallel", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    config_path = resolve_config_path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["seed"] = args.seed

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
    foldername = resolve_save_folder(args.modelfolder) if args.modelfolder else DEFAULT_SAVE_DIR / f"express4d_condition_{current_time}"
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
    print("train condition counts:", dict(sorted(train_dataset.condition_counts.items())))
    print("target condition ratios:", dict(sorted(train_dataset.condition_ratios.items())))
    print("epoch target counts:", dict(sorted(train_loader.sampler.target_counts.items())))
    print("samples per epoch:", len(train_loader.sampler))

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
