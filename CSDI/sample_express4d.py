import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from dataset_express4d import load_vector_52
from main_model import CSDI_Express4D


def resolve_config_path(path):
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    script_relative = Path(__file__).resolve().parent / path
    if script_relative.is_file():
        return script_relative
    raise FileNotFoundError(f"Config file not found: {path}")


def main():
    parser = argparse.ArgumentParser(description="Sample Express4D middle blendshape frames")
    parser.add_argument("--config", type=str, default="config/express4d.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_start", type=str, default="")
    parser.add_argument("--input_end", type=str, default="")
    parser.add_argument("--duration", "--duraction", dest="duration", type=float, default=None)
    parser.add_argument("--output", type=str, default="output_middle.npy")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model = CSDI_Express4D(
        config,
        args.device,
        target_dim=config["dataset"].get("num_features", 52),
    ).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))

    if not args.input_start or not args.input_end or args.duration is None:
        raise ValueError("--input_start, --input_end, and --duration are required")

    dataset_config = config["dataset"]
    start = load_vector_52(
        args.input_start,
        clamp=dataset_config.get("clamp", True),
        clamp_min=dataset_config.get("clamp_min", 0.0),
        clamp_max=dataset_config.get("clamp_max", 1.0),
    )
    end = load_vector_52(
        args.input_end,
        clamp=dataset_config.get("clamp", True),
        clamp_min=dataset_config.get("clamp_min", 0.0),
        clamp_max=dataset_config.get("clamp_max", 1.0),
    )

    model.eval()
    middle = model.generate_middle(
        torch.from_numpy(start),
        torch.from_numpy(end),
        args.duration,
        num_samples=args.num_samples,
    )
    middle = middle.cpu().numpy()
    if args.num_samples == 1:
        middle = middle[0]
    np.save(args.output, middle.astype(np.float32))
    print(f"saved {middle.shape} to {args.output}")


if __name__ == "__main__":
    main()
