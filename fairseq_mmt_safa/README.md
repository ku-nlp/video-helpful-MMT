# SAFA Model

**Note:** We are still organizing the code. The current version is for reference purpose only and **cannot** be directly used for training or inference.
**Note:** For installation and usage of the project, it is recommended to refer to https://github.com/libeineu/fairseq_mmt which is our base

## 1. Python Environment
The conda environment file `./env/environment.yaml` contains a list of packages. However, not all of these packages are necessary, as some may have been used for other experiments and are included for reference purposes only.

## 2. Prepare Required Data
Create a folder under `./data` and place your training data there. Ensure that the data has been tokenized.

The content of the `.video` files and the loading method for video features depend on the dataset being used. The relevant code is located in `./fairseq/data/image_dataset.py`. Additionally, some parameters are specified in the `train_*.sh` files.

## 3. Preprocess Text Data
Use the `preprocess_video.sh` script to preprocess the text data.

```
bash preprocess_video.sh
```

Refer to `preprocess_video.sh` for specific parameters.

## 4. Training
### NMT model
```
bash train_nmt.sh
```

### MMT model
```
bash train_mmt_video.sh
```

### SAFA
Example:
```
bash train_mmt_video_gaussian.sh 0,1 0.5 0.5 0
```

## 5. Evaluation
### NMT model
```
bash translate_nmt.sh
```

### MMT model and SAFA model
```
bash translate_mmt.sh
```
