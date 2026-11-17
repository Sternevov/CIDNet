# CIDNet: Causal Intervention for Debiased Weakly Supervised Referring Expression Comprehension

* This repo is the official implementation of the paper "CIDNet: Causal Intervention for Debiased Weakly Supervised Referring Expression Comprehension"

## 📋 Prerequisites

- Linux (or WSL)
- [Anaconda](https://www.anaconda.com/) / Miniconda
- CUDA 11.3 compatible GPU
- Git

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Sternevov/CIDNet.git
cd CIDNet
```

### 2. Create and activate a conda environment

```bash
conda create -n CIDNet python=3.9 -y
conda activate CIDNet
```

### 3. Install PyTorch

Install PyTorch following the [official installation instructions](https://pytorch.org/get-started/locally/).

> We run all our experiments on **PyTorch 1.11.0 with CUDA 11.3**.

### 4. Install apex

Install apex following the [official installation guide](https://github.com/NVIDIA/apex#quick-start).  
Or use the commands below (copied from the official repo):

```bash
git clone https://github.com/NVIDIA/apex
cd apex

# if pip >= 23.1 (supports multiple --config-settings with the same key)
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
  --config-settings "--build-option=--cpp_ext" \
  --config-settings "--build-option=--cuda_ext" ./

# otherwise
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
  --global-option="--cpp_ext" \
  --global-option="--cuda_ext" ./
```

### 5. Compile the DCN layer

```bash
cd utils/DCN
./make.sh
```

### 6. Install remaining dependencies

```bash
pip install -r requirements.txt

wget https://github.com/explosion/spacy-models/releases/download/en_vectors_web_lg-2.1.0/en_vectors_web_lg-2.1.0.tar.gz \
  -O en_vectors_web_lg-2.1.0.tar.gz
pip install en_vectors_web_lg-2.1.0.tar.gz
```

## 📂 Data Preparation

1. Download images and generate annotations according to [SimREC](https://github.com/luogen1996/SimREC/blob/main/DATA_PRE_README.md).
   > We also prepared the annotations inside the `data/anns` folder for saving your time.
2. Download the pretrained weights of YoloV3 from [Google Drive](https://1drv.ms/u/s!Aka1OHHnFhDktBV8GWXVWiuVpqOY?e=fSmr4D).
   > We recommend putting it in the main path of `CIDNet`; otherwise, please modify the path in the config files.

The `data` directory should look like this:

```text
├── data
│   ├── anns
│   │   ├── refcoco.json
│   │   ├── refcoco+.json
│   │   ├── refcocog.json
│   │   ├── refclef.json
│   │   └── cat_name.json
│   ├── images
│   │   ├── train2014
│   │   │   ├── COCO_train2014_000000515716.jpg
│   │   │   └── ...
│   │   ├── refclef
│   │   │   ├── 99.jpg
│   │   │   └── ...
│   └── ... the remaining directories
```

> **NOTE:** Our YoloV3 is trained on COCO's training images, excluding those in RefCOCO, RefCOCO+, and RefCOCOg's validation+testing sets.

## 🏋️ Training

```bash
python train.py --config ./configs/[DATASET_NAME].yaml
```

## 📊 Evaluation

```bash
python test.py --config ./config/[DATASET_NAME].yaml --eval-weights [PATH_TO_CHECKPOINT_FILE]
```

## 📝 Notes

- Replace `[DATASET_NAME]` with one of `refcoco`, `refcoco+`, `refcocog`, or `refclef`.
- Replace `[PATH_TO_CHECKPOINT_FILE]` with the actual path to your trained checkpoint.
- Make sure the YoloV3 pretrained weights path in the config files matches where you placed the file.

## 📄 License

Please refer to the original repository for license information.



