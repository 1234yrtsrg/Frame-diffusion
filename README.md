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
  config/
    express4d.yaml
    express4d_condition.yaml
    keyframe_dataset_60fps.yaml
  dataset_express4d.py
  dataset_express4d_condition.py
  dataset_keyframe_dataset_60fps.py
  diff_models.py
  main_model.py                CSDI_base 公共扩散代码和 CSDI_Express4D
  utils.py                     三条路线共用训练与 checkpoint 保存代码

train/
  express4d_duration/
  express4d_condition/
  keyframe_dataset_60fps/

inference/                     三条路线的推理和统一评测
algorithm/                     关键帧标注代码
```

`baseline_express4d.py`、线性插值和两阶段推理仍作为推理/基线代码保留。

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

仓库只保留三条训练路线，均在仓库根目录运行。

### express4d_duration

```shell
python train/express4d_duration/train_express4d.py \
  --config CSDI/config/express4d.yaml
```

### express4d_condition

```shell
python train/express4d_condition/train_express4d_condition.py \
  --config CSDI/config/express4d_condition.yaml
```

### keyframe_dataset_60fps

```shell
python train/keyframe_dataset_60fps/train_keyframe_dataset_60fps.py \
  --config CSDI/config/keyframe_dataset_60fps.yaml
```

三条入口继续支持既有的 `--max_train_steps`、`--save_interval_steps`、
`--device` 和 `--data_parallel` 等参数。对应目录中的 `.sh` 文件提供了
多卡训练示例。训练仍保存最终 `model.pth`，并按配置保存
`checkpoint_step_<step>.pth` 和 `training_state.pth`。

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python train/express4d_duration/train_express4d.py \
  --config CSDI/config/express4d.yaml \
  --device cuda:0 \
  --max_train_steps 100000 \
  --data_parallel
```

## 推理

给定起始帧和结束帧，生成中间 10 帧：

```shell
python inference/sample_express4d.py \
  --config CSDI/config/express4d.yaml \
  --checkpoint save/express4d_xxx/model.pth \
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

统一在 `keyframe_dataset_60fps` 的 test split 上评估三种训练方式的 checkpoint。
评估按两阶段推理生成数据：先用 `condition=1` 补全两个关键帧之间的 10 帧，
再用 `condition=3` 补全每个相邻粗帧区间里的 10 帧。

```shell
python inference/evaluate_models.py \
  --express4d_duration_checkpoint save/express4d_duration/model.pth \
  --express4d_condition_checkpoint save/express4d_condition/model.pth \
  --keyframe_dataset_60fps_checkpoint save/keyframe_dataset_60fps/model.pth
```

默认评估配置是 `CSDI/config/keyframe_dataset_60fps.yaml`。如需只评估某个子集，
可以加 `--data_dirs express4d` 或 `--data_dirs dfew`。

评估指标只包括：

- PSNR
- SSIM
- MS-SSIM
- MAE / L1
- MSE / L2

## Smoke Test

安装依赖后运行：

```shell
python CSDI/smoke_test_express4d.py
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
