from collections import Counter
from pathlib import Path
import sys
import unittest

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
CSDI_DIR = REPO_ROOT / "CSDI"
if str(CSDI_DIR) not in sys.path:
    sys.path.insert(0, str(CSDI_DIR))

from dataset_keyframe_dataset_60fps import (  # noqa: E402
    BalancedDeterministicConditionSampler,
    KeyframeDataset60fps,
    sample_nearest_sequence_with_keyframes,
    sample_timepoints,
)
from main_model import CSDI_Express4D  # noqa: E402


def frame_data(length=100, features=3):
    return np.arange(length * features, dtype=np.float32).reshape(length, features)


def test_real_frame_sampling_preserves_all_legal_keyframes():
    cases = [
        ([], []),
        ([7], [7]),
        (list(range(1, 11)), list(range(1, 11))),
        ([10, 3, 3, 7, 1, 7], [1, 3, 7, 10]),
    ]
    for keyframes, expected_internal in cases:
        data = frame_data()
        sampled, positions, slots = sample_nearest_sequence_with_keyframes(
            data, 0, 20, keyframes, seq_len=12
        )
        assert len(positions) == 12
        assert positions[0] == 0 and positions[-1] == 20
        assert np.all(np.diff(positions) > 0)
        assert len(np.unique(positions)) == 12
        assert np.array_equal(sampled, data[positions])
        assert set(expected_internal).issubset(set(positions.tolist()))
        assert set(positions[slots.astype(bool)].tolist()) == set(expected_internal)


def test_exactly_ten_and_more_than_ten_keyframes_are_deterministic():
    data = frame_data()
    exact = list(range(1, 11))
    _, exact_positions, exact_slots = sample_nearest_sequence_with_keyframes(
        data, 0, 20, exact
    )
    assert set(exact).issubset(set(exact_positions.tolist()))
    assert int(exact_slots.sum()) == 10

    dense = list(range(1, 20))
    first = sample_nearest_sequence_with_keyframes(data, 0, 20, dense)
    second = sample_nearest_sequence_with_keyframes(data, 0, 20, reversed(dense))
    assert np.array_equal(first[1], second[1])
    assert np.array_equal(first[2], second[2])
    assert int(first[2].sum()) == 10


def test_dense_keyframes_at_shortest_gap_stay_unique_and_real():
    data = frame_data(30)
    keyframes = [*range(6, 17), 8, 9]
    sampled, positions, slots = sample_nearest_sequence_with_keyframes(
        data, 5, 17, keyframes
    )
    assert positions.shape == (12,)
    assert positions[0] == 5 and positions[-1] == 17
    assert np.all(np.diff(positions) > 0)
    assert int(slots.sum()) == 10
    assert np.array_equal(sampled, data[positions])


def test_real_timepoints_use_relative_frame_locations():
    positions = np.asarray([10, 11, 13, 16, 20, 24, 29, 35, 42, 50, 59, 70])
    timepoints = sample_timepoints(positions, 10, 70)
    assert timepoints.dtype == np.float32
    assert timepoints[0] == np.float32(0.0)
    assert timepoints[-1] == np.float32(11.0)
    assert np.all(np.diff(timepoints) > 0)
    assert np.isclose(timepoints[1], 11.0 / 60.0)


def synthetic_dataset(mask_modes=None):
    dataset = KeyframeDataset60fps.__new__(KeyframeDataset60fps)
    dataset.seq_len = 12
    dataset.fps = 60.0
    dataset.partial_visible_ratio = 0.5
    dataset.mask_seed = 1234
    dataset.keyframe_overflow_strategy = "farthest_temporal_coverage"
    dataset.samples = [(0, 0, 20, 20, 1)]
    dataset.sequences = [
        {
            "data": frame_data(30, 52) / 2000.0,
            "keyframes": np.asarray([4, 8, 12, 16]),
            "data_source": "dfew",
            "sequence_name": "display-name/sequence-a",
            "dataset_name": "display-name",
            "path": Path("sequence-a.json"),
        }
    ]
    dataset.mask_modes = list(mask_modes or ["none"])
    return dataset


def test_mask_mode_semantics():
    for mode in ("none", "partial", "all"):
        dataset = synthetic_dataset([mode])
        item = dataset[0]
        keyframe_frames = int(item["keyframe_mask"][:, 0].sum())
        visible_frames = int(item["visible_keyframe_mask"][:, 0].sum())
        assert keyframe_frames == 4
        assert item["observed_mask"][0].all() and item["observed_mask"][-1].all()
        assert np.array_equal(item["target_mask"], 1.0 - item["observed_mask"])
        assert np.array_equal(item["observed_data"], item["data"])
        if mode == "none":
            assert visible_frames == 0
        elif mode == "partial":
            assert visible_frames == 2
            assert 1 <= visible_frames <= keyframe_frames - 1
        else:
            assert visible_frames == keyframe_frames


def mask_assignment_dataset():
    dataset = synthetic_dataset()
    dataset.mask_ratios = {
        1: {"none": 0.4, "partial": 0.3, "all": 0.3},
    }
    dataset.samples = [(0, start, start + 20, 20, 1) for start in range(8)]
    dataset.sequences[0]["data"] = frame_data(40, 52) / 3000.0
    dataset.sequences[0]["keyframes"] = np.arange(1, 39, dtype=np.int64)
    dataset.mask_modes = dataset._assign_mask_modes()
    return dataset


def test_zero_and_one_keyframe_modes_are_legal():
    dataset = synthetic_dataset()
    dataset.mask_ratios = {
        1: {"none": 0.4, "partial": 0.3, "all": 0.3},
        3: {"none": 0.0, "partial": 0.5, "all": 0.5},
    }
    dataset.samples = [(0, 0, 20, 20, 1), (1, 0, 20, 20, 1), (1, 1, 21, 20, 3)]
    no_keyframes = dict(dataset.sequences[0])
    no_keyframes["keyframes"] = np.asarray([], dtype=np.int64)
    one_keyframe = dict(dataset.sequences[0])
    one_keyframe["keyframes"] = np.asarray([7], dtype=np.int64)
    one_keyframe["sequence_name"] = "display-name/sequence-b"
    dataset.sequences = [no_keyframes, one_keyframe]
    modes = dataset._assign_mask_modes()
    assert modes[0] == "none"
    assert modes[1] in ("none", "all")
    assert modes[2] == "all"


def test_mask_is_stable_across_instances_workers_and_epochs():
    first = mask_assignment_dataset()
    second = mask_assignment_dataset()
    assert first.mask_modes == second.mask_modes
    before = [first[index]["visible_keyframe_mask"] for index in range(len(first))]
    after = [first[index]["visible_keyframe_mask"] for index in range(len(first))]
    assert all(np.array_equal(left, right) for left, right in zip(before, after))

    direct_modes = [batch["mask_mode"][0] for batch in torch.utils.data.DataLoader(first, batch_size=1)]
    worker_modes = [
        batch["mask_mode"][0]
        for batch in torch.utils.data.DataLoader(first, batch_size=1, num_workers=1)
    ]
    assert direct_modes == worker_modes == first.mask_modes


class SamplerFixture:
    def __init__(self):
        self.condition_ratios = {1: 0.4, 2: 0.1, 3: 0.4, 4: 0.1}
        self.data_source_ratios = {"dfew": 0.8739, "express4d": 0.1261}
        self.samples = []
        self.metadata = []
        available = {
            1: {"dfew": 30, "express4d": 10},
            2: {"dfew": 4, "express4d": 2},
            3: {"dfew": 6, "express4d": 3},
            4: {"dfew": 2, "express4d": 2},
        }
        source_id = {"dfew": 0, "express4d": 1}
        self.sequences = [{"data_source": "dfew"}, {"data_source": "express4d"}]
        self.condition_indices = {}
        self.condition_source_indices = {}
        for condition, source_counts in available.items():
            for source, count in source_counts.items():
                for local in range(count):
                    index = len(self.samples)
                    self.samples.append((source_id[source], local, local + 12, 12, condition))
                    mode = ("none", "partial", "all")[index % 3]
                    self.metadata.append((condition, source, mode))
                    self.condition_indices.setdefault(condition, []).append(index)
                    self.condition_source_indices.setdefault((condition, source), []).append(index)
        self.condition_counts = Counter(
            {condition: len(indices) for condition, indices in self.condition_indices.items()}
        )

    def target_epoch_counts(self, base_condition=1):
        base_count = self.condition_counts[base_condition]
        total = round(base_count / self.condition_ratios[base_condition])
        return {
            condition: base_count if condition == base_condition else round(total * ratio)
            for condition, ratio in self.condition_ratios.items()
        }

    def sample_sampling_metadata(self, index):
        return self.metadata[index]


def test_condition_and_source_sampler_targets_repeats_and_reproducibility():
    dataset = SamplerFixture()
    left = BalancedDeterministicConditionSampler(dataset, seed=77)
    right = BalancedDeterministicConditionSampler(dataset, seed=77)
    left_indices = list(iter(left))
    right_indices = list(iter(right))
    assert left_indices == right_indices
    stats = left.last_epoch_stats
    assert stats["actual_condition_counts"] == {1: 40, 2: 10, 3: 40, 4: 10}
    assert stats["actual_condition_source_counts"] == {
        1: {"dfew": 35, "express4d": 5},
        2: {"dfew": 9, "express4d": 1},
        3: {"dfew": 35, "express4d": 5},
        4: {"dfew": 9, "express4d": 1},
    }
    assert stats["total_repeats"] > 0
    assert len(left_indices) == 100
    assert list(iter(left)) != left_indices
    left.set_epoch(0)
    assert list(iter(left)) == left_indices


def test_irregular_time_uniform_motion_has_zero_acceleration():
    timepoints = torch.tensor([[0.0, 0.4, 1.7, 3.0, 5.5]])
    gt = timepoints[:, None, :].repeat(1, 2, 1)
    pred = 2.5 * gt + 0.7
    target_mask = torch.ones_like(gt)
    _, acceleration = CSDI_Express4D.motion_losses(pred, gt, target_mask, timepoints)
    assert abs(acceleration.item()) <= 1e-5


def test_known_known_intervals_do_not_contribute_to_motion_loss():
    timepoints = torch.tensor([[0.0, 0.5, 2.0, 5.0]])
    gt = torch.zeros(1, 1, 4)
    pred = gt.clone()
    pred[:, :, 0] = 10.0
    target_mask = torch.zeros_like(gt)
    target_mask[:, :, -1] = 1.0
    velocity, acceleration = CSDI_Express4D.motion_losses(pred, gt, target_mask, timepoints)
    assert velocity.item() == 0.0
    assert acceleration.item() == 0.0
    empty_velocity, empty_acceleration = CSDI_Express4D.motion_losses(
        pred, gt, torch.zeros_like(target_mask), timepoints
    )
    assert empty_velocity.item() == 0.0
    assert empty_acceleration.item() == 0.0


def tiny_model_config():
    return {
        "dataset": {"seq_len": 12, "num_middle": 10, "clamp_min": 0.0, "clamp_max": 1.0},
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
            "target_strategy": "express4d_condition",
            "seq_len": 12,
            "use_duration": False,
            "use_condition": True,
            "condition_embed_dim": 8,
        },
        "loss": {
            "lambda_recon": 1.0,
            "lambda_vel": 0.5,
            "lambda_acc": 0.2,
            "lambda_range": 0.1,
        },
    }


def test_all_losses_are_finite_and_backward_works():
    torch.manual_seed(9)
    batch_size, length, features = 2, 12, 4
    data = torch.rand(batch_size, length, features)
    observed_mask = torch.zeros_like(data)
    observed_mask[:, 0] = 1.0
    observed_mask[:, -1] = 1.0
    observed_mask[0, 4] = 1.0
    target_mask = 1.0 - observed_mask
    base_tp = torch.tensor([0.0, 0.2, 0.7, 1.4, 2.0, 3.1, 4.0, 5.7, 7.1, 8.0, 9.8, 11.0])
    batch = {
        "observed_data": data,
        "observed_mask": observed_mask,
        "target_mask": target_mask,
        "timepoints": base_tp.repeat(batch_size, 1),
        "condition": torch.tensor([1.0, 3.0]),
    }
    model = CSDI_Express4D(tiny_model_config(), "cpu", target_dim=features)
    loss, components = model(batch, return_loss_components=True)
    assert torch.isfinite(loss)
    assert {"diffusion", "recon", "velocity", "acceleration", "range"}.issubset(components)
    assert all(torch.isfinite(value) for value in components.values())
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


class KeyframeDataset60fpsTests(unittest.TestCase):
    def test_real_frame_sampling(self):
        test_real_frame_sampling_preserves_all_legal_keyframes()

    def test_keyframe_overflow(self):
        test_exactly_ten_and_more_than_ten_keyframes_are_deterministic()

    def test_shortest_gap(self):
        test_dense_keyframes_at_shortest_gap_stay_unique_and_real()

    def test_timepoints(self):
        test_real_timepoints_use_relative_frame_locations()

    def test_masks(self):
        test_mask_mode_semantics()

    def test_mask_stability(self):
        test_mask_is_stable_across_instances_workers_and_epochs()

    def test_legal_special_case_masks(self):
        test_zero_and_one_keyframe_modes_are_legal()

    def test_sampler(self):
        test_condition_and_source_sampler_targets_repeats_and_reproducibility()

    def test_uniform_acceleration(self):
        test_irregular_time_uniform_motion_has_zero_acceleration()

    def test_known_known_motion(self):
        test_known_known_intervals_do_not_contribute_to_motion_loss()

    def test_backward(self):
        test_all_losses_are_finite_and_backward_works()


if __name__ == "__main__":
    unittest.main()
