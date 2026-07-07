# CSDI
This is the github repository for the NeurIPS 2021 paper "[CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation](https://arxiv.org/abs/2107.03502)".

## Requirement

Please install the packages in requirements.txt

## Preparation
### Download the healthcare dataset 
```shell
python download.py physio
```
### Download the air quality dataset 
```shell
python download.py pm25
```

### Download the elecricity dataset 
Please put files in [GoogleDrive](https://drive.google.com/drive/folders/1krZQofLdeQrzunuKkLXy8L_kMzQrVFI_?usp=drive_link) to the "data" folder.

## Experiments 

### training and imputation for the healthcare dataset
```shell
python exe_physio.py --testmissingratio [missing ratio] --nsample [number of samples]
```

### imputation for the healthcare dataset with pretrained model
```shell
python exe_physio.py --modelfolder pretrained --testmissingratio [missing ratio] --nsample [number of samples]
```

### training and imputation for the healthcare dataset
```shell
python exe_pm25.py --nsample [number of samples]
```

### training and forecasting for the electricity dataset
```shell
python exe_forecasting.py --datatype electricity --nsample [number of samples]
```

### Express4D blendshape interpolation
Express4D uses CSDI as a conditional diffusion imputer. Each sample is a
12-frame ARKit blendshape sequence with shape `[L,K] = [12,52]`; frame 0 and
frame 11 are observed endpoint conditions, and frames 1-10 are generated.
The model receives `duration` as an extra condition embedding.

Expected server data layout:
```text
dataset/Express4D/train.txt
dataset/Express4D/test.txt
dataset/Express4D/data/
```

Train:
```shell
python train_express4d.py --config config/express4d.yaml
```

Sample from endpoint vectors:
```shell
python sample_express4d.py --config config/express4d.yaml --checkpoint path/to/checkpoint.pth --input_start path/to/start.npy --input_end path/to/end.npy --duration 1.0 --output output_middle.npy
```

Evaluate on the unified keyframe_dataset_60fps test split with the two-stage condition=1 -> condition=3 pipeline:
```shell
python inference/evaluate_models.py --express4d_duration_checkpoint path/to/checkpoint.pth
```

Smoke test:
```shell
python smoke_test_express4d.py
```

### Visualize results
'visualize_examples.ipynb' is a notebook for visualizing results.

## Acknowledgements

A part of the codes is based on [BRITS](https://github.com/caow13/BRITS) and [DiffWave](https://github.com/lmnt-com/diffwave)

## Citation
If you use this code for your research, please cite our paper:

```
@inproceedings{tashiro2021csdi,
  title={CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation},
  author={Tashiro, Yusuke and Song, Jiaming and Song, Yang and Ermon, Stefano},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```
