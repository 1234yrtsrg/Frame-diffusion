#!/usr/bin/env bash
set -e

NUM_GPUS="${NUM_GPUS:-8}"
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-128}"
FINE_SEGMENT_BATCH_SIZE="${FINE_SEGMENT_BATCH_SIZE:-512}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
OUTPUT="${OUTPUT:-outputs/model_eval/metrics.json}"
SHARD_DIR="${SHARD_DIR:-${OUTPUT%.json}_shards}"

IFS=',' read -r -a GPU_IDS <<< "$CUDA_DEVICES"
if [ "${#GPU_IDS[@]}" -lt "$NUM_GPUS" ]; then
  echo "NUM_GPUS=$NUM_GPUS but CUDA_DEVICES only has ${#GPU_IDS[@]} entries: $CUDA_DEVICES" >&2
  exit 1
fi

mkdir -p "$SHARD_DIR"
rm -f "$SHARD_DIR"/shard_*.json

pids=()
for shard_index in $(seq 0 $((NUM_GPUS - 1))); do
  gpu_id="${GPU_IDS[$shard_index]}"
  shard_output="$SHARD_DIR/shard_${shard_index}.json"
  echo "Launching shard ${shard_index}/${NUM_GPUS} on GPU ${gpu_id} with batch_size=${PER_GPU_BATCH_SIZE}, fine_segment_batch_size=${FINE_SEGMENT_BATCH_SIZE}"
  CUDA_VISIBLE_DEVICES="$gpu_id" python inference/evaluate_models.py \
    --express4d_duration_checkpoint "${EXPRESS4D_DURATION_CHECKPOINT:-save/express4d_20260528_032120/checkpoint_step_50000.pth}" \
    --express4d_condition_checkpoint "${EXPRESS4D_CONDITION_CHECKPOINT:-save/express4d_condition/checkpoint_step_50000.pth}" \
    --keyframe_dataset_60fps_checkpoint "${KEYFRAME_DATASET_60FPS_CHECKPOINT:-save/keyframe_dataset_60fps/checkpoint_step_50000.pth}" \
    --data_dirs "${DATA_DIRS:-dfew,express4d}" \
    --batch_size "$PER_GPU_BATCH_SIZE" \
    --fine_segment_batch_size "$FINE_SEGMENT_BATCH_SIZE" \
    --device cuda:0 \
    --num_shards "$NUM_GPUS" \
    --shard_index "$shard_index" \
    --output "$shard_output" \
    "$@" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

python - "$OUTPUT" "$SHARD_DIR" <<'PY'
import copy
import glob
import json
import os
import sys

output_path, shard_dir = sys.argv[1], sys.argv[2]
shard_files = sorted(glob.glob(os.path.join(shard_dir, "shard_*.json")))
if not shard_files:
    raise SystemExit(f"No shard outputs found in {shard_dir}")

payloads = []
for path in shard_files:
    with open(path, "r", encoding="utf-8") as f:
        payloads.append(json.load(f))

def feature_count(metrics):
    values = metrics.get("per_feature_non_keyframe_mae")
    if values:
        return len(values)
    return 52

def metric_count(metrics, name):
    counts = metrics["counts"]
    features = feature_count(metrics)
    if name.startswith("timeline_resampled_mae") or name.startswith("timeline_resampled_mse"):
        return counts["num_frames"] * features
    if name.startswith("non_keyframe"):
        return counts["num_non_keyframes"] * features
    if name.startswith("timeline_resampled_velocity"):
        return max(0, counts["num_frames"] - counts["num_sequences"]) * features
    if name.startswith("timeline_resampled_acceleration"):
        return max(0, counts["num_frames"] - 2 * counts["num_sequences"]) * features
    if name.startswith("boundary_velocity"):
        return counts["num_boundary_velocity_intervals"] * features
    if name.startswith("known_frame"):
        return counts["num_known_frames"] * features
    if name.startswith("endpoint"):
        return counts["num_sequences"] * 2 * features
    return 0

def weighted_metric(metric_payloads, name):
    numerator = 0.0
    denominator = 0
    for metrics in metric_payloads:
        value = metrics.get(name)
        if value is None:
            continue
        count = metric_count(metrics, name)
        numerator += float(value) * count
        denominator += count
    return (numerator / denominator) if denominator else None

def merge_counts(metric_payloads):
    merged = {}
    for metrics in metric_payloads:
        for key, value in metrics["counts"].items():
            merged[key] = merged.get(key, 0) + int(value)
    return merged

def merge_per_feature(metric_payloads):
    numerator = None
    denominator = 0
    for metrics in metric_payloads:
        values = metrics.get("per_feature_non_keyframe_mae")
        if values is None:
            continue
        count = metrics["counts"]["num_non_keyframes"]
        if numerator is None:
            numerator = [0.0] * len(values)
        for index, value in enumerate(values):
            numerator[index] += float(value) * count
        denominator += count
    if numerator is None or denominator == 0:
        return None
    return [value / denominator for value in numerator]

def merge_method(entries):
    merged = copy.deepcopy(entries[0])
    metric_payloads = [entry["metrics"] for entry in entries]
    metric_names = sorted(
        name
        for metrics in metric_payloads
        for name in metrics
        if name not in ("counts", "per_feature_non_keyframe_mae")
    )
    merged_metrics = {name: weighted_metric(metric_payloads, name) for name in metric_names}
    merged_metrics["per_feature_non_keyframe_mae"] = merge_per_feature(metric_payloads)
    merged_metrics["counts"] = merge_counts(metric_payloads)
    merged["metrics"] = merged_metrics
    if "num_selected_windows" in merged:
        merged["num_selected_windows"] = sum(entry.get("num_selected_windows", 0) for entry in entries)
    return merged

merged_payload = copy.deepcopy(payloads[0])
merged_payload["num_shards"] = len(payloads)
merged_payload["shard_index"] = None
merged_payload["num_selected_windows"] = sum(payload["num_selected_windows"] for payload in payloads)
merged_payload["shard_outputs"] = shard_files
merged_payload["methods"] = {}

method_names = sorted({name for payload in payloads for name in payload["methods"]})
for method_name in method_names:
    entries = [payload["methods"][method_name] for payload in payloads if method_name in payload["methods"]]
    merged_payload["methods"][method_name] = merge_method(entries)

os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(merged_payload, f, indent=2)

print(json.dumps(merged_payload["methods"], indent=2))
print(f"saved merged metrics to {output_path}")
PY
