# Frame-diffusion CSDI 模块

本目录包含 Frame-diffusion 三条 Express4D 训练路线共享的条件扩散实现。模型基于
[CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation](https://arxiv.org/abs/2107.03502)
改造，用于 52 维 ARKit blendshape 关键帧插值。

## 保留的训练路线

| 路线 | 训练入口 | 配置 | 数据集模块 |
| --- | --- | --- | --- |
| `express4d_duration` | `train/express4d_duration/train_express4d.py` | `CSDI/config/express4d.yaml` | `CSDI/dataset_express4d.py` |
| `express4d_condition` | `train/express4d_condition/train_express4d_condition.py` | `CSDI/config/express4d_condition.yaml` | `CSDI/dataset_keyframe_dataset_60fps.py` |
| `keyframe_dataset_60fps` | `train/keyframe_dataset_60fps/train_keyframe_dataset_60fps.py` | `CSDI/config/keyframe_dataset_60fps.yaml` | `CSDI/dataset_keyframe_dataset_60fps.py` |

`dataset_express4d_condition.py` 仍由统一评测按路线名动态加载，因此作为评测数据适配器保留。

三条路线都使用 `CSDI_Express4D`。`CSDI_base` 只保留它继承调用的网络初始化、
设备解析、时间/条件 side information、扩散输入拼接和反向采样公共代码。

## 训练

以下命令均从仓库根目录运行：

```shell
python train/express4d_duration/train_express4d.py \
  --config CSDI/config/express4d.yaml

python train/express4d_condition/train_express4d_condition.py \
  --config CSDI/config/express4d_condition.yaml

python train/keyframe_dataset_60fps/train_keyframe_dataset_60fps.py \
  --config CSDI/config/keyframe_dataset_60fps.yaml
```

各入口的完整参数可通过 `--help` 查看。对应训练目录也提供 Bash 脚本。

## 推理与评测

时长条件模型推理：

```shell
python inference/infer_keyframes_express4d.py \
  --config CSDI/config/express4d.yaml \
  --checkpoint path/to/checkpoint.pth
```

条件模型推理使用 `inference/infer_blendshapes_condition.py`。三条路线统一评测：

```shell
python inference/evaluate_models.py \
  --express4d_duration_checkpoint path/to/duration_checkpoint.pth \
  --express4d_condition_checkpoint path/to/condition_checkpoint.pth \
  --keyframe_dataset_60fps_checkpoint path/to/keyframe_checkpoint.pth
```

## Acknowledgements

扩散网络实现基于原 CSDI 项目，并包含来自 DiffWave 的相关设计。使用本代码时请引用：

```bibtex
@inproceedings{tashiro2021csdi,
  title={CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation},
  author={Tashiro, Yusuke and Song, Jiaming and Song, Yang and Ermon, Stefano},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```
