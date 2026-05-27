import argparse
import datetime
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
import yaml

from dataset_express4d import get_dataloader
from main_model import CSDI_Express4D
from utils import train


def resolve_config_path(path):
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    script_relative = Path(__file__).resolve().parent / path
    if script_relative.is_file():
        return script_relative
    raise FileNotFoundError(f"Config file not found: {path}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train CSDI for Express4D blendshape interpolation")
    parser.add_argument("--config", type=str, default="config/express4d.yaml")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--modelfolder", type=str, default="")
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
    args = parser.parse_args()

    set_seed(args.seed)
    config_path = resolve_config_path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
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
    foldername = "./save/express4d_" + current_time + "/"
    print("model folder:", foldername)
    os.makedirs(foldername, exist_ok=True)
    with open(os.path.join(foldername, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    train_loader, test_loader = get_dataloader(
        config,
        seed=args.seed,
        batch_size=config["train"]["batch_size"],
        num_workers=config["train"].get("num_workers", 0),
    )

    model = CSDI_Express4D(
        config,
        args.device,
        target_dim=config["dataset"].get("num_features", 52),
    ).to(args.device)

    if args.modelfolder:
        checkpoint = Path("./save") / args.modelfolder / "model.pth"
        model.load_state_dict(torch.load(checkpoint, map_location=args.device))
    else:
        train(
            model,
            config["train"],
            train_loader,
            valid_loader=test_loader,
            valid_epoch_interval=config["train"].get("valid_epoch_interval", 20),
            foldername=foldername,
        )


if __name__ == "__main__":
    main()
