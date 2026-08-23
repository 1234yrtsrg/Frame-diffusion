# 训练集关键帧分布统计

本目录用于只读分析 `keyframe_dataset_60fps` 的 train split。脚本不加载模型、不使用 GPU，也不会创建、修改或重新生成训练/测试 split 文件。它遍历 train split 中构建出的全部滑动窗口；sampler 统计则另外复用训练代码中的 `BalancedDeterministicConditionSampler`，生成固定随机种子的第 0 个 epoch。

## 统计口径

“内部关键帧”严格指 `start_idx < keyframe < end_idx`，不含窗口首尾。脚本按训练数据集相同的最近槽位分配逻辑计算最终 12 个 `sample_positions`。

- `only_endpoints`：原始窗口没有内部关键帧，训练条件只有首尾帧。
- `endpoints_with_all_internal`：原始窗口有内部关键帧，且全部进入 12 帧序列。
- `endpoints_with_partial_internal`：原始窗口有内部关键帧，但只有部分进入 12 帧序列。这里的 `partial` 只表示 10 个内部槽位的容量限制，不表示随机 mask。

`sample_positions` 的检查以最终位置数组为准：相邻差值存在 `<= 0` 计为“不严格递增”，位置唯一值少于 12 计为“存在重复”。

## 运行

从仓库根目录运行：

```bash
python analysis/keyframe_distribution/analyze_keyframe_distribution.py \
  --config CSDI/config/keyframe_dataset_60fps.yaml \
  --split train \
  --output-dir analysis/keyframe_distribution/results
```

相对路径均从仓库根目录解析，默认随机种子为 `1`。脚本优先读取配置指定且已经存在的 split 文件。配置允许随机划分的数据源若缺少 split 文件，脚本只在内存中按配置的 `split_seed` 和 `split_ratios` 复现选择，绝不写出文件；该来源会记录到 `summary.json` 的 `metadata.split_provenance`。

如果数据目录或已有 split 列表被移到了仓库外，可使用只读覆盖，不会改写 YAML：

```bash
python analysis/keyframe_distribution/analyze_keyframe_distribution.py \
  --config CSDI/config/keyframe_dataset_60fps.yaml \
  --split train \
  --output-dir analysis/keyframe_distribution/results \
  --dataset-root /path/to/keyframe_dataset_60fps \
  --train-split-file express4d=/path/to/express4d_train.txt
```

## 输出

- `results/summary.md`：结论、总体/分数据源统计、condition/gap 分组、质量检查及 sampler 对比。
- `results/summary.json`：完整聚合结果和运行元数据，适合后续程序读取。
- `results/distribution.csv`：每行是“数据源 × condition/gap × 类别”的聚合结果，不包含逐样本明细。

建议在两类数据同等重要时使用分层 sampler：先按配置平衡 condition，再在每个 condition 内显式按 DFEW:Express4D = 1:1 抽样。若部署数据有明确先验，应把 1:1 替换为可配置的目标比例。
