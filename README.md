# AIR-DETR
An improved RT-DETR based UAV visual object detection project supporting multiple drone datasets.
# Project Introduction
This project is an improved RT-DETR object detection algorithm specifically designed for small object detection tasks in UAV aerial images. It supports three mainstream drone datasets: VisDrone2019, AI-TOD, and CODrone. Through optimization for aerial scene characteristics, it achieves better detection performance in complex aerial environments.
# Key Features

✨ Based on RT-DETR architecture, optimized for small object detection
🚁 Supports multiple drone datasets: VisDrone2019, AI-TOD, CODrone
🔧 Supports various backbone and detection head configurations
📊 Complete training, validation, and testing pipeline
🎯 Data augmentation strategies tailored for aerial scenes

# Requirements
bashPython >= 3.8
PyTorch >= 1.10
CUDA >= 11.0 (Recommended for GPU)
# Tested Environment
This project has been tested in the following environment:
bashPython: 3.10.14
PyTorch: 2.2.2+cu121
TorchVision: 0.17.2+cu121
TIMM: 1.0.7
MMCV: 2.2.0
MMEngine: 0.10.4
Triton: 3.2.0
CUDA: 12.1
# Installation
## 1. Clone Repository
bashgit clone https://github.com/FY2hang/AIR-DETR.git
cd AIR-DETR
## 2. Environment Setup (Important!)
⚠️ Note: Follow these steps strictly to avoid dependency conflicts
Step 1: Install basic dependencies
bashpip install -r requirements.txt
Step 2: Uninstall existing ultralytics library
bash# Uninstall ultralytics library from environment
pip uninstall ultralytics

Execute again to confirm clean uninstall (should show WARNING: Skipping ultralytics as it is not installed.)
pip uninstall ultralytics
## Step 3: Install additional packages
bashpip install timm==1.0.7 thop efficientnet_pytorch==0.7.1 einops grad-cam==1.5.4 dill==0.3.8 albumentations==1.4.11 pytorch_wavelets==1.3.0 tidecv PyWavelets opencv-python prettytable -i https://pypi.tuna.tsinghua.edu.cn/simple
## Step 4: Install DyHead dependencies (Required)
bash# These packages are essential for DyHead functionality!
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
## Step 5: Install ultralytics (Optional)
bash# If you need to use official CLI running method
python setup.py develop

Note: Skip this step if you don't need official CLI method
3. Verification
Install any missing packages as prompted during runtime.
#     Dataset Preparation
This project supports three drone datasets: VisDrone2019, AI-TOD, CODrone
Dataset Directory Structure
Each dataset should be organized as follows:
## 
    datasets/
    ├── VisDrone2019/
    │   ├── images/
    │   │   ├── train/          # Training images
    │   │   ├── val/            # Validation images
    │   │   └── test/           # Test images
    │   └── labels/
    │       ├── train/          # Training labels
    │       ├── val/            # Validation labels
    │       └── test/           # Test labels
    ├── AI-TOD/
    │   ├── images/
    │   │   ├── train/
    │   │   ├── val/
    │   │   └── test/
    │   └── labels/
    │       ├── train/
    │       ├── val/
    │       └── test/
    └── CODrone/
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── labels/
            ├── train/
            ├── val/
            └── test/
Dataset Download
⚠️ Due to large dataset sizes, please download from official sources
VisDrone2019 Dataset

Official Website: http://aiskyeye.com/
GitHub: https://github.com/VisDrone/VisDrone-Dataset

AI-TOD Dataset

Official Website: https://github.com/jwwangchn/AI-TOD

CODrone Dataset

Official Website: https://github.com/AHideoKuzeA/CODrone-A-Comprehensive-Oriented-Object-Detection-benchmark-for-UAV
