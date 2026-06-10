param(
  [Parameter(Mandatory = $true)]
  [string]$CudaVisibleDevices,
  [string]$Device = "cuda:0",
  [int]$BatchSize = 0,
  [int]$MaxTrainSteps = 0,
  [int]$SaveIntervalSteps = 0
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$env:CUDA_VISIBLE_DEVICES = $CudaVisibleDevices
$env:NCCL_DEBUG = "WARN"

$trainArgs = @(
  "train/train_keyframe_segments_T30.py",
  "--config", "CSDI/config/keyframe_segments_T30.yaml",
  "--device", $Device,
  "--data_parallel"
)

if ($BatchSize -gt 0) {
  $trainArgs += @("--batch_size", "$BatchSize")
}
if ($MaxTrainSteps -gt 0) {
  $trainArgs += @("--max_train_steps", "$MaxTrainSteps")
}
if ($SaveIntervalSteps -gt 0) {
  $trainArgs += @("--save_interval_steps", "$SaveIntervalSteps")
}

python @trainArgs
