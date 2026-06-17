param(
  [string]$CudaVisibleDevices = "",
  [string]$Device = "cuda:0",
  [int]$BatchSize = 0,
  [int]$SamplesPerEpoch = 0,
  [int]$MaxTrainSteps = 0,
  [int]$SaveIntervalSteps = 0,
  [switch]$DataParallel
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repoRoot

if ($CudaVisibleDevices -ne "") {
  $env:CUDA_VISIBLE_DEVICES = $CudaVisibleDevices
}
if ($DataParallel) {
  $env:NCCL_DEBUG = "WARN"
}

$trainArgs = @(
  "train/express4d_condition/train_express4d_condition.py",
  "--config", "CSDI/config/express4d_condition.yaml",
  "--device", $Device
)

if ($BatchSize -gt 0) {
  $trainArgs += @("--batch_size", "$BatchSize")
}
if ($SamplesPerEpoch -gt 0) {
  $trainArgs += @("--samples_per_epoch", "$SamplesPerEpoch")
}
if ($MaxTrainSteps -gt 0) {
  $trainArgs += @("--max_train_steps", "$MaxTrainSteps")
}
if ($SaveIntervalSteps -gt 0) {
  $trainArgs += @("--save_interval_steps", "$SaveIntervalSteps")
}
if ($DataParallel) {
  $trainArgs += "--data_parallel"
}

python @trainArgs
