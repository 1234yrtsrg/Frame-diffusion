#!/usr/bin/env python3
"""Analyze keyframe-window and condition-sampler distributions without loading a model."""

from __future__ import annotations

import argparse
import bisect
import copy
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CSDI_DIR = REPO_ROOT / "CSDI"
if str(CSDI_DIR) not in sys.path:
    sys.path.insert(0, str(CSDI_DIR))

from dataset_keyframe_dataset_60fps import (  # noqa: E402
    BalancedDeterministicConditionSampler,
)


CATEGORIES = (
    "only_endpoints",
    "endpoints_with_all_internal",
    "endpoints_with_partial_internal",
)
HISTOGRAM_KEYS = tuple(str(value) for value in range(11)) + ("10+",)
NUM_FRAMES_RE = re.compile(r'"num_frames"\s*:\s*(\d+)')
KEYFRAMES_RE = re.compile(r'"(?:keyframe_indices|keyframes)"\s*:\s*\[([^\]]*)\]')
INTEGER_RE = re.compile(r"-?\d+")


def resolve_from_repo(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (REPO_ROOT / candidate).resolve()


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def percentage(count: int, total: int) -> float:
    return round(100.0 * count / total, 6) if total else 0.0


def histogram_key(value: int) -> str:
    return "10+" if value > 10 else str(value)


def canonical_source(value: str) -> str:
    normalized = value.strip().lower().replace("_", "").replace("-", "")
    if normalized == "dfew":
        return "DFEW"
    if normalized == "express4d":
        return "Express4D"
    return value.strip()


def progress(iterable: Iterable, total: int, description: str) -> Iterable:
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc=description, unit="item")
    except ImportError:
        def generator():
            every = max(1, total // 20)
            for index, item in enumerate(iterable, 1):
                if index == 1 or index % every == 0 or index == total:
                    print(f"{description}: {index}/{total}", flush=True)
                yield item

        return generator()


@dataclass(frozen=True)
class SequenceMetadata:
    source: str
    path: Path
    num_frames: int
    keyframes: Tuple[int, ...]


@dataclass
class Aggregate:
    total_samples: int = 0
    categories: Counter = field(default_factory=Counter)
    original_histogram: Counter = field(default_factory=Counter)
    sampled_histogram: Counter = field(default_factory=Counter)
    original_sum: int = 0
    sampled_sum: int = 0
    original_min: int | None = None
    original_max: int | None = None
    sampled_min: int | None = None
    sampled_max: int | None = None
    non_strictly_increasing: int = 0
    duplicate_positions: int = 0
    internal_over_10: int = 0
    incomplete_internal: int = 0
    groups: Dict[Tuple[int, int], Counter] = field(default_factory=lambda: defaultdict(Counter))

    def add(
        self,
        condition: int,
        gap: int,
        category: str,
        original_internal: int,
        sampled_internal: int,
        non_strict: bool,
        duplicate: bool,
    ) -> None:
        self.total_samples += 1
        self.categories[category] += 1
        self.original_histogram[histogram_key(original_internal)] += 1
        self.sampled_histogram[histogram_key(sampled_internal)] += 1
        self.original_sum += original_internal
        self.sampled_sum += sampled_internal
        self.original_min = original_internal if self.original_min is None else min(self.original_min, original_internal)
        self.original_max = original_internal if self.original_max is None else max(self.original_max, original_internal)
        self.sampled_min = sampled_internal if self.sampled_min is None else min(self.sampled_min, sampled_internal)
        self.sampled_max = sampled_internal if self.sampled_max is None else max(self.sampled_max, sampled_internal)
        self.non_strictly_increasing += int(non_strict)
        self.duplicate_positions += int(duplicate)
        self.internal_over_10 += int(original_internal > 10)
        self.incomplete_internal += int(sampled_internal < original_internal)
        self.groups[(condition, gap)][category] += 1

    def as_dict(self) -> dict:
        total = self.total_samples
        category_distribution = {
            category: {
                "count": int(self.categories[category]),
                "percentage": percentage(self.categories[category], total),
            }
            for category in CATEGORIES
        }
        by_group = []
        for (condition, gap), counts in sorted(self.groups.items()):
            group_total = int(sum(counts.values()))
            by_group.append(
                {
                    "condition": condition,
                    "gap": gap,
                    "total_samples": group_total,
                    "categories": {
                        category: {
                            "count": int(counts[category]),
                            "percentage": percentage(counts[category], group_total),
                        }
                        for category in CATEGORIES
                    },
                }
            )
        return {
            "total_samples": total,
            "category_distribution": category_distribution,
            "original_internal_keyframes": {
                "histogram": {key: int(self.original_histogram[key]) for key in HISTOGRAM_KEYS},
                "mean": round(self.original_sum / total, 6) if total else 0.0,
                "min": int(self.original_min or 0),
                "max": int(self.original_max or 0),
            },
            "sampled_internal_keyframes": {
                "histogram": {key: int(self.sampled_histogram[key]) for key in HISTOGRAM_KEYS},
                "mean": round(self.sampled_sum / total, 6) if total else 0.0,
                "min": int(self.sampled_min or 0),
                "max": int(self.sampled_max or 0),
            },
            "quality_checks": {
                "sample_positions_not_strictly_increasing": self.non_strictly_increasing,
                "sample_positions_with_duplicates": self.duplicate_positions,
                "internal_keyframes_over_10": self.internal_over_10,
                "internal_keyframes_not_fully_preserved": self.incomplete_internal,
            },
            "by_condition_gap": by_group,
        }


class SamplerDatasetView:
    """Minimal dataset interface required by the production sampler."""

    def __init__(self, condition_indices: Mapping[int, List[int]], ratios: Mapping[int, float]):
        self.condition_indices = dict(condition_indices)
        self.condition_counts = Counter({key: len(value) for key, value in condition_indices.items()})
        ratio_total = float(sum(ratios.values()))
        self.condition_ratios = {int(key): float(value) / ratio_total for key, value in ratios.items()}

    def target_epoch_counts(self, base_condition: int = 1) -> dict:
        base_count = int(self.condition_counts[base_condition])
        base_ratio = float(self.condition_ratios[base_condition])
        total_target = int(round(base_count / base_ratio))
        targets = {
            condition: base_count if condition == base_condition else int(round(total_target * ratio))
            for condition, ratio in sorted(self.condition_ratios.items())
        }
        targets[base_condition] = base_count
        return targets


def parse_mapping(items: Sequence[str], option_name: str) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"{option_name} must use SOURCE=PATH, got {item!r}")
        source, raw_path = item.split("=", 1)
        result[source.strip().lower()] = resolve_from_repo(raw_path.strip())
    return result


def read_split_entries(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}")
    entries = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not entries:
        raise ValueError(f"Split file is empty: {path}")
    return entries


def all_json_paths(dataset_dir: Path) -> List[Path]:
    return [
        path
        for path in sorted(dataset_dir.rglob("*.json"))
        if not path.name.startswith("keyframe_summary")
    ]


def entry_to_json_path(entry: str, dataset_dir: Path) -> Path:
    raw = entry.strip().replace("\\", "/")
    relative = Path(raw)
    candidates = [relative] if relative.is_absolute() else [dataset_dir / relative]
    if relative.suffix.lower() != ".json":
        candidates.append(relative.with_suffix(".json") if relative.is_absolute() else dataset_dir / relative.with_suffix(".json"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve split entry {entry!r} under {dataset_dir}")


def select_split_paths(
    dataset_config: Mapping,
    dataset_root: Path,
    split: str,
    overrides: Mapping[str, Path],
) -> Tuple[List[Tuple[str, List[Path]]], dict]:
    selected = []
    provenance = {}
    data_dirs = dataset_config.get("data_dirs") or [dataset_config.get("data_dir", "dfew")]
    if isinstance(data_dirs, str):
        data_dirs = [item.strip() for item in data_dirs.split(",") if item.strip()]
    random_dirs = set(dataset_config.get("random_split_data_dirs", ["dfew"]))
    split_files = dataset_config.get("split_files", {})
    ratios = np.asarray(dataset_config.get("split_ratios", [0.8, 0.2]), dtype=np.float64)
    ratios /= ratios.sum()
    split_seed = int(dataset_config.get("split_seed", 1))

    for data_dir in data_dirs:
        source = canonical_source(data_dir)
        source_key = str(data_dir).lower()
        dataset_dir = dataset_root / str(data_dir)
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"Dataset source directory not found: {dataset_dir}")
        configured = split_files.get(data_dir, {})
        configured_name = configured.get(split, f"{data_dir}_{split}.txt")
        configured_path = dataset_root / configured_name
        override = overrides.get(source_key)

        if override is not None:
            entries = read_split_entries(override)
            paths = [entry_to_json_path(entry, dataset_dir) for entry in entries]
            provenance[source] = {
                "method": "command_line_split_file_override",
                "path": display_path(override),
                "configured_path": display_path(configured_path),
            }
        elif configured_path.is_file():
            entries = read_split_entries(configured_path)
            paths = [entry_to_json_path(entry, dataset_dir) for entry in entries]
            provenance[source] = {"method": "existing_split_file", "path": display_path(configured_path)}
        elif data_dir in random_dirs:
            paths = all_json_paths(dataset_dir)
            indices = np.arange(len(paths), dtype=np.int64)
            rng = np.random.default_rng(split_seed)
            rng.shuffle(indices)
            train_count = int(len(indices) * ratios[0])
            chosen = indices[:train_count] if split == "train" else indices[train_count:]
            paths = [paths[int(index)] for index in chosen]
            provenance[source] = {
                "method": "read_only_deterministic_in_memory_split",
                "configured_path_missing": display_path(configured_path),
                "split_seed": split_seed,
                "split_ratios": ratios.tolist(),
                "files_written": False,
            }
        else:
            raise FileNotFoundError(
                f"Required {source} split file is missing: {configured_path}. "
                f"The analyzer never creates split files. Supply an existing file with "
                f"--train-split-file {data_dir}=PATH if the dataset was relocated."
            )
        selected.append((source, paths))
    return selected, provenance


def load_sequence_metadata(source: str, path: Path) -> SequenceMetadata:
    text = path.read_text(encoding="utf-8")
    frame_match = NUM_FRAMES_RE.search(text)
    keyframe_match = KEYFRAMES_RE.search(text)
    if frame_match is None or keyframe_match is None:
        payload = json.loads(text)
        frames = payload.get("frames")
        num_frames = int(payload.get("num_frames", len(frames) if frames is not None else 0))
        raw_keyframes = payload.get("keyframe_indices", payload.get("keyframes"))
        if raw_keyframes is None:
            raise ValueError(f"Missing keyframes: {path}")
        keyframes = []
        for item in raw_keyframes:
            if isinstance(item, dict):
                item = item.get("frame_index", item.get("index"))
            if item is not None:
                keyframes.append(int(item))
    else:
        num_frames = int(frame_match.group(1))
        keyframes = [int(value) for value in INTEGER_RE.findall(keyframe_match.group(1))]
    if num_frames <= 0:
        raise ValueError(f"Invalid num_frames={num_frames}: {path}")
    valid_keyframes = tuple(sorted(set(value for value in keyframes if 0 <= value < num_frames)))
    return SequenceMetadata(source=source, path=path, num_frames=num_frames, keyframes=valid_keyframes)


def exact_sample_positions(
    start_idx: int,
    end_idx: int,
    keyframes: Sequence[int],
    seq_len: int,
    num_frames: int,
) -> Tuple[np.ndarray, int, int]:
    interval = float(end_idx - start_idx) / float(seq_len - 1)
    base = np.rint(start_idx + np.arange(seq_len, dtype=np.float32) * interval).astype(np.int64)
    base[0] = start_idx
    base[-1] = end_idx
    positions = base.copy()
    left = bisect.bisect_right(keyframes, start_idx)
    right = bisect.bisect_left(keyframes, end_idx)
    internal = keyframes[left:right]
    available = set(range(1, seq_len - 1))
    sampled_internal = 0
    for keyframe in internal:
        if not available:
            break
        slot = min(available, key=lambda candidate: (abs(int(base[candidate]) - keyframe), candidate))
        positions[slot] = keyframe
        available.remove(slot)
        sampled_internal += 1
    return np.clip(positions, 0, num_frames - 1), len(internal), sampled_internal


def classify(original_internal: int, sampled_internal: int) -> str:
    if original_internal == 0:
        return "only_endpoints"
    if sampled_internal == original_internal:
        return "endpoints_with_all_internal"
    return "endpoints_with_partial_internal"


def distribution(counter: Mapping, total: int, ordered_keys: Iterable) -> dict:
    return {
        str(key): {"count": int(counter.get(key, 0)), "percentage": percentage(counter.get(key, 0), total)}
        for key in ordered_keys
    }


def analyze(config: dict, selected_sources: List[Tuple[str, List[Path]]], seed: int) -> Tuple[dict, list]:
    dataset_config = config["dataset"]
    seq_len = int(dataset_config.get("seq_len", 12))
    stride = int(dataset_config.get("window_stride", 5))
    condition_gaps = {int(key): int(value) for key, value in dataset_config["condition_gaps"].items()}
    condition_ratios = {int(key): float(value) for key, value in dataset_config["condition_ratios"].items()}
    if seq_len != 12:
        print(f"Warning: configured seq_len is {seq_len}, not 12", file=sys.stderr)

    sequences_by_source: Dict[str, List[SequenceMetadata]] = {}
    for source, paths in selected_sources:
        sequences_by_source[source] = [
            load_sequence_metadata(source, path)
            for path in progress(paths, len(paths), f"Loading {source} metadata")
        ]

    scopes = {"all": Aggregate()}
    scopes.update({source: Aggregate() for source in sequences_by_source})
    source_code = {source: code for code, source in enumerate(sequences_by_source)}
    code_source = {code: source for source, code in source_code.items()}
    source_by_sample = bytearray()
    condition_by_sample = bytearray()
    condition_indices: Dict[int, List[int]] = defaultdict(list)
    sample_index = 0

    total_sequences = sum(len(values) for values in sequences_by_source.values())
    sequence_items = (
        sequence
        for source in sequences_by_source
        for sequence in sequences_by_source[source]
    )
    for sequence in progress(sequence_items, total_sequences, "Analyzing training windows"):
        for condition, gap in sorted(condition_gaps.items()):
            if sequence.num_frames <= gap:
                continue
            for start_idx in range(0, sequence.num_frames - gap, stride):
                end_idx = start_idx + gap
                positions, original_internal, sampled_internal = exact_sample_positions(
                    start_idx, end_idx, sequence.keyframes, seq_len, sequence.num_frames
                )
                deltas = np.diff(positions)
                non_strict = bool(np.any(deltas <= 0))
                duplicate = len(np.unique(positions)) != len(positions)
                category = classify(original_internal, sampled_internal)
                for scope_name in ("all", sequence.source):
                    scopes[scope_name].add(
                        condition, gap, category, original_internal, sampled_internal, non_strict, duplicate
                    )
                source_by_sample.append(source_code[sequence.source])
                condition_by_sample.append(condition)
                condition_indices[condition].append(sample_index)
                sample_index += 1

    sampler_view = SamplerDatasetView(condition_indices, condition_ratios)
    sampler = BalancedDeterministicConditionSampler(
        sampler_view,
        base_condition=int(dataset_config.get("balance_base_condition", 1)),
        seed=seed,
    )
    epoch_indices = list(iter(sampler))
    epoch_condition = Counter()
    epoch_source = Counter()
    within_source_condition: Dict[str, Counter] = defaultdict(Counter)
    for index in progress(epoch_indices, len(epoch_indices), "Counting sampler epoch"):
        condition = condition_by_sample[index]
        source = code_source[source_by_sample[index]]
        epoch_condition[condition] += 1
        epoch_source[source] += 1
        within_source_condition[source][condition] += 1

    scope_dict = {name: aggregate.as_dict() for name, aggregate in scopes.items()}
    original_source_windows = {source: scopes[source].total_samples for source in sequences_by_source}
    all_windows = scopes["all"].total_samples
    original_source_sequences = {source: len(values) for source, values in sequences_by_source.items()}
    all_sequences = sum(original_source_sequences.values())
    ordered_conditions = sorted(condition_gaps)
    epoch_total = len(epoch_indices)
    source_names = list(sequences_by_source)

    sampler_source_distribution = distribution(epoch_source, epoch_total, source_names)
    original_source_distribution = distribution(original_source_windows, all_windows, source_names)
    original_condition_distribution = distribution(
        {condition: len(indices) for condition, indices in condition_indices.items()},
        all_windows,
        ordered_conditions,
    )
    source_percentage_change = {
        source: round(
            sampler_source_distribution[source]["percentage"]
            - original_source_distribution[source]["percentage"],
            6,
        )
        for source in source_names
    }
    result = {
        "original_training_set": {
            "scopes": scope_dict,
            "condition_distribution": original_condition_distribution,
            "data_source_distribution": {
                "windows": original_source_distribution,
                "sequences": distribution(original_source_sequences, all_sequences, source_names),
            },
        },
        "sampler_epoch": {
            "seed": seed,
            "epoch": 0,
            "total_samples": epoch_total,
            "target_counts": {str(key): int(value) for key, value in sorted(sampler.target_counts.items())},
            "condition_distribution": distribution(epoch_condition, epoch_total, ordered_conditions),
            "data_source_distribution": sampler_source_distribution,
            "within_data_source_condition_distribution": {
                source: distribution(within_source_condition[source], epoch_source[source], ordered_conditions)
                for source in source_names
            },
        },
        "comparison": {
            "original_data_source_window_distribution": original_source_distribution,
            "sampler_epoch_data_source_distribution": sampler_source_distribution,
            "sampler_minus_original_percentage_points": source_percentage_change,
            "condition_balanced": True,
            "data_source_explicitly_balanced": False,
            "interpretation": (
                "The sampler enforces configured condition ratios, but samples each condition bucket "
                "without a source quota; source proportions are therefore inherited from available windows."
            ),
            "recommended_data_source_sampling_ratio": {
                source: round(1.0 / len(source_names), 6) for source in source_names
            },
            "recommendation": (
                "Use hierarchical sampling: first apply condition_ratios, then sample data sources "
                "with an explicit 1:1 quota inside every condition bucket."
            ),
        },
    }
    return result, source_names


def write_csv(path: Path, scopes: Mapping[str, dict], source_names: Sequence[str]) -> None:
    fieldnames = [
        "data_source", "condition", "gap", "category", "count", "percentage", "group_total"
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for source in ["all", *source_names]:
            for group in scopes[source]["by_condition_gap"]:
                for category in CATEGORIES:
                    values = group["categories"][category]
                    writer.writerow(
                        {
                            "data_source": source,
                            "condition": group["condition"],
                            "gap": group["gap"],
                            "category": category,
                            "count": values["count"],
                            "percentage": f'{values["percentage"]:.6f}',
                            "group_total": group["total_samples"],
                        }
                    )


def markdown_distribution_table(values: Mapping[str, dict], label: str) -> List[str]:
    lines = [f"| {label} | 数量 | 比例 |", "|---|---:|---:|"]
    for key, item in values.items():
        lines.append(f'| {key} | {item["count"]:,} | {item["percentage"]:.2f}% |')
    return lines


def write_scope_markdown(lines: List[str], title: str, stats: dict) -> None:
    lines.extend([f"## {title}", "", f'样本总数：**{stats["total_samples"]:,}**。', ""])
    lines.extend(markdown_distribution_table(stats["category_distribution"], "类别"))
    lines.extend(["", "| 指标 | 原始内部关键帧 | 进入 12 帧的内部关键帧 |", "|---|---:|---:|"])
    original = stats["original_internal_keyframes"]
    sampled = stats["sampled_internal_keyframes"]
    lines.extend([
        f'| 平均值 | {original["mean"]:.3f} | {sampled["mean"]:.3f} |',
        f'| 最小值 | {original["min"]} | {sampled["min"]} |',
        f'| 最大值 | {original["max"]} | {sampled["max"]} |',
        "",
        "| 内部关键帧数 | " + " | ".join(HISTOGRAM_KEYS) + " |",
        "|---|" + "---:|" * len(HISTOGRAM_KEYS),
        "| 原始窗口 | " + " | ".join(f'{original["histogram"][key]:,}' for key in HISTOGRAM_KEYS) + " |",
        "| 进入 12 帧 | " + " | ".join(f'{sampled["histogram"][key]:,}' for key in HISTOGRAM_KEYS) + " |",
        "",
        "| condition | gap | 样本数 | only_endpoints | all_internal | partial_internal |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for group in stats["by_condition_gap"]:
        cats = group["categories"]
        lines.append(
            f'| {group["condition"]} | {group["gap"]} | {group["total_samples"]:,} | '
            f'{cats["only_endpoints"]["count"]:,} ({cats["only_endpoints"]["percentage"]:.2f}%) | '
            f'{cats["endpoints_with_all_internal"]["count"]:,} ({cats["endpoints_with_all_internal"]["percentage"]:.2f}%) | '
            f'{cats["endpoints_with_partial_internal"]["count"]:,} ({cats["endpoints_with_partial_internal"]["percentage"]:.2f}%) |'
        )
    quality = stats["quality_checks"]
    lines.extend([
        "",
        "| 检查项 | 样本数 |",
        "|---|---:|",
        f'| sample_positions 不严格递增 | {quality["sample_positions_not_strictly_increasing"]:,} |',
        f'| sample_positions 存在重复 | {quality["sample_positions_with_duplicates"]:,} |',
        f'| 原始内部关键帧超过 10 个 | {quality["internal_keyframes_over_10"]:,} |',
        f'| 内部关键帧未完整保留 | {quality["internal_keyframes_not_fully_preserved"]:,} |',
        "",
    ])


def write_summary_markdown(path: Path, payload: dict, source_names: Sequence[str]) -> None:
    original = payload["original_training_set"]
    sampler = payload["sampler_epoch"]
    comparison = payload["comparison"]
    all_stats = original["scopes"]["all"]
    partial = all_stats["category_distribution"]["endpoints_with_partial_internal"]
    duplicate = all_stats["quality_checks"]["sample_positions_with_duplicates"]
    lines = [
        "# 训练集关键帧分布统计",
        "",
        "## 总体结论",
        "",
        f'- train split 共 **{all_stats["total_samples"]:,}** 个窗口；因 12 帧容量而只保留部分内部关键帧的窗口为 '
        f'**{partial["count"]:,} ({partial["percentage"]:.2f}%)**。',
        f'- `sample_positions` 不严格递增的窗口为 **{all_stats["quality_checks"]["sample_positions_not_strictly_increasing"]:,}**，'
        f'其中存在重复位置的窗口为 **{duplicate:,}**。',
        "- 当前 sampler 明确平衡 condition（时间尺度），但 condition bucket 内没有数据源配额，因此没有主动平衡 DFEW 与 Express4D。",
        "- 若两类数据同等重要，建议采用“condition 配额 × 数据源配额”的分层采样，并在每个 condition 内按 **DFEW:Express4D = 1:1** 抽样。",
        "",
        "### 原始数据源窗口分布",
        "",
    ]
    lines.extend(markdown_distribution_table(original["data_source_distribution"]["windows"], "数据源"))
    lines.extend(["", "### 原始数据源序列分布", ""])
    lines.extend(markdown_distribution_table(original["data_source_distribution"]["sequences"], "数据源"))
    lines.append("")
    write_scope_markdown(lines, "整个训练集", all_stats)
    for source in source_names:
        write_scope_markdown(lines, source, original["scopes"][source])

    lines.extend([
        "## 当前 condition sampler：一个 epoch",
        "",
        "| condition | 原始窗口数 | 原始比例 | sampler 数量 | sampler 比例 |",
        "|---:|---:|---:|---:|---:|",
    ])
    for condition, sampler_values in sampler["condition_distribution"].items():
        original_values = original["condition_distribution"][condition]
        lines.append(
            f'| {condition} | {original_values["count"]:,} | {original_values["percentage"]:.2f}% | '
            f'{sampler_values["count"]:,} | {sampler_values["percentage"]:.2f}% |'
        )
    lines.extend(["", "### sampler epoch 数据源分布", ""])
    lines.extend(markdown_distribution_table(sampler["data_source_distribution"], "数据源"))
    lines.extend([
        "",
        "### 各数据源内部的 condition 比例",
        "",
        "| 数据源 | condition | 数量 | 比例 |",
        "|---|---:|---:|---:|",
    ])
    for source in source_names:
        for condition, values in sampler["within_data_source_condition_distribution"][source].items():
            lines.append(f'| {source} | {condition} | {values["count"]:,} | {values["percentage"]:.2f}% |')

    lines.extend([
        "",
        "### 原始训练集与 sampler epoch 对比",
        "",
        "| 数据源 | 原始窗口比例 | sampler epoch 比例 | 变化（百分点） |",
        "|---|---:|---:|---:|",
    ])
    for source in source_names:
        original_pct = comparison["original_data_source_window_distribution"][source]["percentage"]
        sampler_pct = comparison["sampler_epoch_data_source_distribution"][source]["percentage"]
        change = comparison["sampler_minus_original_percentage_points"][source]
        lines.append(f"| {source} | {original_pct:.2f}% | {sampler_pct:.2f}% | {change:+.2f} |")
    lines.extend([
        "",
        "结论：判断成立。sampler 只设置 condition 目标数量；每个 condition 内仍从混合数据源窗口池直接抽样，"
        "所以数据源比例由各 condition 中可用的 DFEW/Express4D 窗口数量决定。推荐的 1:1 是“来源同等重要”"
        "前提下的默认值；若验证目标有明确部署先验，应把该先验替换为显式、可配置的数据源比例。",
        "",
        "## 运行信息",
        "",
        f'- 配置：`{payload["metadata"]["config"]}`',
        f'- 数据根目录：`{payload["metadata"]["dataset_root"]}`',
        f'- split：`{payload["metadata"]["split"]}`',
        f'- 固定随机种子：`{payload["metadata"]["seed"]}`',
        "",
        "| 数据源 | split 读取方式 | 路径/说明 |",
        "|---|---|---|",
    ])
    for source, provenance in payload["metadata"]["split_provenance"].items():
        detail = provenance.get("path", provenance.get("configured_path_missing", ""))
        lines.append(f'| {source} | `{provenance["method"]}` | `{detail}` |')
    lines.extend(["", "完整 split 来源信息见 `summary.json -> metadata.split_provenance`。", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="CSDI/config/keyframe_dataset_60fps.yaml")
    parser.add_argument("--split", choices=("train",), default="train", help="Only train is supported by design.")
    parser.add_argument("--output-dir", default="analysis/keyframe_distribution/results")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dataset-root", default=None, help="Optional read-only override for a relocated dataset root.")
    parser.add_argument(
        "--train-split-file",
        action="append",
        default=[],
        metavar="SOURCE=PATH",
        help="Use an existing relocated train list (repeatable); no split file is created or changed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    config_path = resolve_from_repo(args.config)
    output_dir = resolve_from_repo(args.output_dir)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    config["seed"] = args.seed
    config["dataset"]["write_split_files"] = False
    dataset_root = resolve_from_repo(args.dataset_root or config["dataset"]["root"])
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Configured dataset root not found: {dataset_root}. "
            "If the data was relocated, pass --dataset-root PATH; the config is not modified."
        )
    config["dataset"]["root"] = str(dataset_root)
    overrides = parse_mapping(args.train_split_file, "--train-split-file")
    selected_sources, split_provenance = select_split_paths(
        config["dataset"], dataset_root, args.split, overrides
    )
    result, source_names = analyze(config, selected_sources, args.seed)
    payload = {
        "metadata": {
            "config": display_path(config_path),
            "dataset_root": display_path(dataset_root),
            "split": args.split,
            "seed": args.seed,
            "seq_len": int(config["dataset"].get("seq_len", 12)),
            "window_stride": int(config["dataset"].get("window_stride", 5)),
            "condition_gaps": {
                str(key): int(value) for key, value in config["dataset"]["condition_gaps"].items()
            },
            "split_provenance": split_provenance,
            "models_loaded": False,
            "gpu_required": False,
            "split_files_modified": False,
        },
        **result,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    csv_path = output_dir / "distribution.csv"
    markdown_path = output_dir / "summary.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(csv_path, payload["original_training_set"]["scopes"], source_names)
    write_summary_markdown(markdown_path, payload, source_names)
    print("Analysis complete:")
    for path in (markdown_path, json_path, csv_path):
        print(f"  {display_path(path)}")


if __name__ == "__main__":
    main()
