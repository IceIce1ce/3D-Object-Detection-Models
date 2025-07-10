# A Simple Toolbox for 3D Object Detection Models

## 1. Setup
### 1.1. Using environment.yml
```bash
conda env create -f environment.yml
conda activate anomaly
```

### 1.2. Using requirements.txt
```bash
conda create --name anomaly python=3.10.13
conda activate anomaly
pip install -r requirements.txt
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

## 2. Acknowledgement
Our framework is built using multiple open source, thanks for their great contributions.
