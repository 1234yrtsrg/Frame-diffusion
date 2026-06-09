import argparse
import datetime
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CSDI_DIR = REPO_ROOT / "CSDI"
sys.path.insert(0, str(CSDI_DIR))

from dataset_keyframe_segments_t30 import get_dataloader
from main_model import CSDI_KeyframeSegmentsT30
from utils import train


DEFAULT_SAVE_DIR = REPO_ROOT / "save"


def resolve_config_path(path):
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    for base in (Path(__file__).resolve().parent, REPO_ROOT, REPO_ROOT / "CSDI"):
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_state_dict(model, checkpoint_path, device):
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if all(key.startswith("module.") for key in state.keys()):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state)


def main():
    parser = argparse.ArgumentParser(description="Train CSDI for keyframe segment interpolation")
    parser.add_argument("--config", type=str, default="CSDI/config/keyframe_segments_T30.yaml")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--modelfolder", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Stop training after this many optimizer steps and save model.pth.",
    )
    parser.add_argument(
        "--save_interval_steps",
        type=int,
        default=None,
        help="Save checkpoint_step_*.pth every N optimizer steps.",
    )
    parser.add_argument(
        "--data_parallel",
        action="store_true",
        help="Use torch.nn.DataParallel across all visible CUDA devices.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    config_path = resolve_config_path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["seed"] = args.seed

    if args.max_train_steps is not None:
        if args.max_train_steps <= 0:
            raise ValueError("--max_train_steps must be positive")
        config["train"]["max_train_steps"] = args.max_train_steps
    if args.save_interval_steps is not None:
        if args.save_interval_steps <= 0:
            raise ValueError("--save_interval_steps must be positive")
        config["train"]["save_interval_steps"] = args.save_interval_steps

    print(json.dumps(config, indent=4))
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    foldername = DEFAULT_SAVE_DIR / f"keyframe_segments_T30_{current_time}"
    print("model folder:", foldername)
    os.makedirs(foldername, exist_ok=True)
    with open(foldername / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    train_loader, valid_loader, test_loader = get_dataloader(
        config,
        seed=args.seed,
        batch_size=config["train"]["batch_size"],
        num_workers=config["train"].get("num_workers", 0),
    )

    model = CSDI_KeyframeSegmentsT30(
        config,
        args.device,
        target_dim=config["dataset"].get("num_features", 52),
    ).to(args.device)

    checkpoint_path = None
    if args.checkpoint:
        checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    elif args.modelfolder:
        checkpoint_path = resolve_checkpoint_path(Path(args.modelfolder) / "model.pth")

    if checkpoint_path is not None:
        load_state_dict(model, checkpoint_path, args.device)

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
        valid_loader=valid_loader,
        valid_epoch_interval=config["train"].get("valid_epoch_interval", 20),
        foldername=str(foldername),
    )


if __name__ == "__main__":
    main()
