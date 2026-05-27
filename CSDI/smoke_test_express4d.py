import tempfile
from pathlib import Path

import numpy as np
import torch

from dataset_express4d import Express4D_Dataset, load_blendshape_file
from main_model import CSDI_Express4D


def build_config(root):
    return {
        "dataset": {
            "name": "express4d",
            "root": str(root),
            "data_dir": "data",
            "train_list": "train.txt",
            "test_list": "test.txt",
            "fps": 60,
            "num_features": 52,
            "seq_len": 12,
            "num_middle": 10,
            "gaps": [12],
            "use_npy_first": True,
            "clamp": True,
            "clamp_min": 0.0,
            "clamp_max": 1.0,
        },
        "train": {"batch_size": 2, "lr": 1.0e-3},
        "diffusion": {
            "layers": 1,
            "channels": 8,
            "nheads": 1,
            "diffusion_embedding_dim": 16,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "num_steps": 2,
            "schedule": "linear",
            "is_linear": False,
        },
        "model": {
            "is_unconditional": 0,
            "timeemb": 16,
            "featureemb": 4,
            "target_strategy": "express4d",
            "num_features": 52,
            "seq_len": 12,
            "use_duration": True,
            "duration_embed_dim": 8,
        },
        "loss": {
            "lambda_recon": 1.0,
            "lambda_vel": 0.5,
            "lambda_acc": 0.2,
            "lambda_range": 0.1,
        },
    }


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "dataset" / "Express4D"
        data_dir = root / "data"
        data_dir.mkdir(parents=True)
        values = np.random.default_rng(1).random((80, 61), dtype=np.float32)
        np.save(data_dir / "sample.npy", values)
        np.savetxt(data_dir / "sample_csv.csv", values, delimiter=",")
        (root / "train.txt").write_text("sample\n", encoding="utf-8")
        (root / "test.txt").write_text("data/sample_csv.csv\n", encoding="utf-8")

        config = build_config(root)
        loaded = load_blendshape_file(data_dir / "sample.npy")
        assert loaded.shape == (80, 52)

        dataset = Express4D_Dataset(config, split="train")
        item = dataset[0]
        assert item["observed_data"].shape == (12, 52)
        assert item["middle"].shape == (10, 52)
        assert item["observed_mask"][0].sum() == 52
        assert item["observed_mask"][1:-1].sum() == 0
        assert item["gt_mask"][1:-1].sum() == 10 * 52

        loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        assert batch["observed_data"].shape == (2, 12, 52)

        device = "cpu"
        model = CSDI_Express4D(config, device, target_dim=52).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])
        loss = model(batch)
        assert torch.isfinite(loss), loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        middle = model.generate_middle(
            torch.from_numpy(item["start"]),
            torch.from_numpy(item["end"]),
            float(item["duration"]),
            num_samples=1,
        )
        assert middle.shape == (1, 10, 52)
        assert torch.isfinite(middle).all()
        print("Express4D smoke test passed")


if __name__ == "__main__":
    main()
