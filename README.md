# Express4D Blendshape 插帧条件扩散模型

本仓库基于官方 CSDI time-series imputation 代码，改造成用于 Express4D
blendshape 插帧的条件扩散模型。

模型目标：

- 输入 `B_start`: `[52]` ARKit blendshape 起始帧
- 输入 `B_end`: `[52]` ARKit blendshape 结束帧
- 输入 `duration`: 标量，单位为秒，表示两个关键帧之间的真实时间跨度
- 输出 `B_middle`: `[10, 52]` 中间 10 帧 blendshape 序列

训练时完整序列长度为 12：

```text
[B_start, B_1, B_2, ..., B_10, B_end]
```

第 `0` 帧和第 `11` 帧是 observed condition，中间第 `1` 到第 `10` 帧是
diffusion imputation / generation 的目标。

## 项目结构

```text
CSDI/
  config/express4d.yaml        Express4D 配置文件
  dataset_express4d.py         Express4D 数据集和 61 维到 52 维映射
  main_model.py                CSDI 主模型，包含 Express4D 专用模型类
  train_express4d.py           训练入口
  sample_express4d.py          推理和测试集评估入口
  baseline_express4d.py        linear / cubic 插值 baseline
  smoke_test_express4d.py      CPU smoke test

train_express4d.py             根目录训练 wrapper
sample_express4d.py            根目录推理 wrapper
```

官方 CSDI 原始说明保留在 `CSDI/README.md`。

## 数据目录

正式训练和测试只读取服务器路径：

```text
dataset/Express4D/
  train.txt
  test.txt
  data/
    *.npy
    *.csv
```

代码不会 fallback 到本地 `dataset/` 根目录。所有列表项只会围绕
`dataset/Express4D` 和 `dataset/Express4D/data` 解析。

`train.txt` / `test.txt` 每行可以是：

- 文件名
- 相对路径
- 已包含 `data/` 的路径
- 带扩展名路径
- 不带扩展名路径

如果 `.npy` 和 `.csv` 都可能存在，默认优先读取 `.npy`。

## 数据格式

Express4D 原始数据为 61 维。模型只读取其中 52 维 ARKit blendshape，
并忽略下面 9 维头部和眼球旋转：

```text
HeadYaw, HeadPitch, HeadRoll,
LeftEyeYaw, LeftEyePitch, LeftEyeRoll,
RightEyeYaw, RightEyePitch, RightEyeRoll
```

名字匹配使用大小写不敏感规则，例如：

```text
EyeBlinkLeft -> eyeblinkleft
eyeBlinkLeft -> eyeblinkleft
```

支持输入格式：

- `.npy`: `[T, 61]` 或 `[61, T]`
- `.csv`: 可以有表头，也可以没有表头

最终读取后的 shape 固定为：

```text
[T, 52] float32
```

NaN 和 Inf 会替换为 `0`。默认会 clamp 到 `[0, 1]`，因为头部和眼球旋转维度已经不再读取。

## 样本构造

原始数据按 60 FPS 处理。每条序列会用多尺度 gap 构造训练样本：

```yaml
gaps: [12, 24, 36, 60, 90, 120, 180, 240]
```

对每个样本：

```text
start_idx = t
end_idx = t + gap
duration = gap / fps
```

然后在 `[start_idx, end_idx]` 之间均匀采样 12 个点：

```text
positions = np.linspace(start_idx, end_idx, 12)
```

并通过时间线性插值得到：

```text
seq: [12, 52]
```

Dataset 每个 item 包含：

- `observed_data`: `[12, 52]`
- `observed_mask`: `[12, 52]`，只有第 `0` 和第 `11` 帧为 `1`
- `gt_mask` / `target_mask`: `[12, 52]`，只有第 `1` 到第 `10` 帧为 `1`
- `timepoints`: `[12]`
- `duration`: scalar float32
- `start`: `[52]`
- `end`: `[52]`
- `middle`: `[10, 52]`
- `sequence_name`
- `start_idx`
- `end_idx`
- `gap`

CSDI 模型内部使用的 batch shape 为：

```text
[B, K, L] = [B, 52, 12]
```

## 模型说明

Express4D 使用的模型类是：

```text
CSDI_Express4D
```

它保留原 CSDI conditional diffusion imputation 逻辑，并额外加入
`duration` 条件：

```text
duration -> MLP -> duration embedding -> side information
```

训练和推理都会使用 `duration`。

## Loss

总 loss 为：

```text
loss = diffusion_loss
     + lambda_recon * L_recon
     + lambda_vel * L_vel
     + lambda_acc * L_acc
     + lambda_range * L_range
```

默认权重：

```yaml
lambda_recon: 1.0
lambda_vel: 0.5
lambda_acc: 0.2
lambda_range: 0.1
```

说明：

- `diffusion_loss` 只在中间 10 帧 target mask 上计算
- `L_recon` 从 predicted noise 反推 `x0_pred`，并在中间 10 帧上计算 L1
- `L_vel` 在完整 12 帧序列上计算速度 L1
- `L_acc` 在完整 12 帧序列上计算加速度 L1
- `L_range` 惩罚超出 `[0, 1]` 的预测值

## 训练

在仓库根目录运行：

```shell
python train_express4d.py --config config/express4d.yaml
```

默认配置会训练 `100` 个 epoch。也可以在命令行指定最大训练步数，达到这个
optimizer step 数后会提前停止并保存 `model.pth`：

```shell
python train_express4d.py --config config/express4d.yaml --max_train_steps 100000
```

默认每 `50000` 个 optimizer steps 会额外保存一个中间模型：

```text
CSDI/save/express4d_xxx/checkpoint_step_50000.pth
CSDI/save/express4d_xxx/checkpoint_step_100000.pth
```

保存间隔也可以在命令行修改：

```shell
python train_express4d.py --config config/express4d.yaml --save_interval_steps 25000
```

如果要用多张 GPU，可以使用 `--data_parallel`。例如使用 4 张可见 GPU：

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_express4d.py \
  --config config/express4d.yaml \
  --device cuda:0 \
  --max_train_steps 100000 \
  --data_parallel
```

也可以进入 `CSDI/` 后运行：

```shell
cd CSDI
python train_express4d.py --config config/express4d.yaml
```

checkpoint 会保存到：

```text
CSDI/save/express4d_*/
```

## 推理

给定起始帧和结束帧，生成中间 10 帧：

```shell
python sample_express4d.py \
  --config config/express4d.yaml \
  --checkpoint CSDI/save/express4d_xxx/model.pth \
  --input_start path/to/start.npy \
  --input_end path/to/end.npy \
  --duration 1.0 \
  --output output_middle.npy
```

当 `--num_samples 1` 时，输出 shape 为：

```text
[10, 52]
```

当 `--num_samples N` 且 `N > 1` 时，输出 shape 为：

```text
[N, 10, 52]
```

## 测试集评估

在 `dataset/Express4D/test.txt` 上评估 checkpoint：

```shell
python sample_express4d.py \
  --config config/express4d.yaml \
  --checkpoint CSDI/save/express4d_xxx/model.pth \
  --eval_test
```

评估指标包括：

- middle L1
- middle MSE
- velocity L1
- acceleration L1
- endpoint continuity start
- endpoint continuity end

脚本也会输出 baseline：

- linear interpolation
- cubic interpolation。如果安装了 SciPy，会使用 SciPy cubic spline；否则使用 smoothstep fallback

## Smoke Test

安装依赖后运行：

```shell
cd CSDI
python smoke_test_express4d.py
```

smoke test 会临时创建一个 Express4D 风格数据目录，并检查：

- 能读取 `train.txt` / `test.txt`
- 能读取 `.npy` / `.csv`
- 能把 61 维转换成 52 维
- 能构造 `[12, 52]` 样本
- batch 后 shape 正确
- 模型 forward 正确
- 单步训练 loss 不为 NaN
- `generate_middle` 输出 shape 为 `[1, 10, 52]`

## 依赖

安装本项目依赖：

```shell
pip install -r requirements.txt
```

Express4D 默认配置中：

```yaml
diffusion:
  is_linear: false
```

因此默认不需要使用 `linear_attention_transformer`。

## 30 帧关键帧区间数据集训练

`dataset/keyframe_segments_T30.npz` 用 `train/train_keyframe_segments_T30.py` 训练。数据集按 `source_ids` 划分 train/valid/test，不按片段随机划分。

PowerShell 单卡示例：

```powershell
$env:CUDA_VISIBLE_DEVICES = "0"
python train/train_keyframe_segments_T30.py `
  --config CSDI/config/keyframe_segments_T30.yaml `
  --device cuda:0 `
  --max_train_steps 50000 `
  --save_interval_steps 10000
```

PowerShell 多卡示例：

```powershell
$env:CUDA_VISIBLE_DEVICES = "0,1,2,3"
python train/train_keyframe_segments_T30.py `
  --config CSDI/config/keyframe_segments_T30.yaml `
  --device cuda:0 `
  --max_train_steps 50000 `
  --save_interval_steps 10000 `
  --data_parallel
```

Bash 多卡示例：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train/train_keyframe_segments_T30.py \
  --config CSDI/config/keyframe_segments_T30.yaml \
  --device cuda:0 \
  --max_train_steps 50000 \
  --save_interval_steps 10000 \
  --data_parallel
```

也可以用脚本，但卡号和训练步数仍然从命令行传入：

```powershell
.\train\train_keyframe_segments_T30_single.ps1 `
  -CudaVisibleDevices "0" `
  -MaxTrainSteps 50000 `
  -SaveIntervalSteps 10000

.\train\train_keyframe_segments_T30_multi.ps1 `
  -CudaVisibleDevices "0,1,2,3" `
  -MaxTrainSteps 50000 `
  -SaveIntervalSteps 10000
```
