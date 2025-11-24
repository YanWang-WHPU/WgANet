WgANet: A Wavelet-guided Attention Network for Remote Sensing Image Semantic Segmentation 
===

----
# Framework
.<div align=center><img src="URL" width="300" height="450" /></div>
# environment
Our experiments were implemented with the PyTorch framework done on a single NVIDIA A40 GPU equipped with 48GB RAM.  

# Prepare
## dataset 
All datasets including ISPRS Potsdam and ISPRS Vaihingen can be downloaded [here](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx)  

## Pretrained Weights of Backbones 

## Folder Structure
```
WgANet
├── pretrain (pretrained weights of backbones)
├── model (models)
├── train_WgANet.py (Training code)
├── utils_WgANet.py (Configuration)
├── results (The folder to save the results)
├── data
│   ├── vaihingen
│   │   ├── dsm (original)
│   │   ├── top (original)
│   │   ├── gts_eroded_for_participants (original)
│   │   ├── gts_for_participants (original)
│   ├── potsdam (the same with vaihingen)
```
# Install
### Open the folder WgANet using Linux Terminal and create python environment:
```
conda create -n WgANet python=3.8 -y
conda activate WgANet
```
### Install cuda=11.8
```
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
```
### Install torch=2.0.0
```
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```
or(pip):
```
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118 
```
### Install Mamba
#### Install mamba_ssm
```
pip install mamba-ssm
```
Because installing with pip can be problematic, we recommend downloading and installing it [here](https://github.com/state-spaces/mamba/releases)
#### Install causal-conv1d
```
pip install causal-conv1d
```
Because installing with pip can be problematic, we recommend downloading and installing it [here](https://github.com/Dao-AILab/causal-conv1d/releases)
# Train 
### Modify the parameters and addresses
1. Modify the address of data in utils_WgANet.py, batch_size, training mode =train, training model =WgANet.  
 
2. Modify the address of results in train_WgANet.py
### train
```
python train_WgANet.py
``` 
# test
### Modify the parameters and addresses
1. Modify the address of results in train_WgANet.py to be the address of the best trained model  
2. Modify the training mode =test in utils_WgANet.py
### test
```
python train_WgANet.py
```

| Method           | Imp. | Bui. | Low. | Tre. | Car. | OA (%) | mF1 (%) | mIoU (%) |
|------------------|------|------|------|------|------|--------|---------|----------|
| ABCNet           | 85.32 | 92.24 | 66.49 | 82.45 | 74.84 | 90.79 | 88.78 | 80.27 |
| UNetMamba        | 84.88 | 92.49 | 66.47 | 83.09 | 75.13 | 90.87 | 88.87 | 80.42 |
| TransUNet        | 85.54 | 92.48 | 67.77 | 83.27 | 81.16 | 91.21 | 89.91 | 82.04 |
| UNetFormer       | 85.58 | 92.93 | 67.70 | 83.55 | 82.43 | 91.29 | 90.14 | 82.44 |
| MAResU-Net       | 86.33 | 93.87 | 67.47 | 83.22 | 81.71 | 91.43 | 90.17 | 82.51 |
| CMTFNet          | 86.37 | 93.63 | 67.33 | 83.11 | 82.18 | 91.42 | 90.17 | 82.52 |
| RS³Mamba         | 86.38 | 93.55 | 67.42 | 82.79 | 82.64 | 91.30 | 90.20 | 82.56 |
| SFFNet           | 80.36 | 88.35 | 60.74 | 79.86 | 59.90 | 88.11 | 84.45 | 73.85 |
| PHA-UNet         | 82.71 | 89.30 | 64.51 | 81.89 | 71.56 | 89.53 | 87.36 | 78.00 |
| MCAFT            | 83.54 | 91.50 | 62.05 | 80.98 | 76.20 | 89.71 | 87.84 | 78.86 |
| DECS-Net         | 86.14 | **93.92** | 66.46 | 83.23 | 79.15 | 91.34 | 89.70 | 81.79 |
| MIFNet           | 86.04 | 93.75 | 67.72 | 83.57 | 82.44 | 91.43 | 90.29 | 82.71 |
| WgANet (ours)    | **86.70** | 93.64 | **67.96** | **83.60** | **83.29** | **91.45** | **90.50** | **83.04** |

# Citations
If these codes are helpful for your study, please cite:
```
@Article{s24227266,
AUTHOR = {Wang, Yan and Cao, Li and Deng, He},
TITLE = {MFMamba: A Mamba-Based Multi-Modal Fusion Network for Semantic Segmentation of Remote Sensing Images},
JOURNAL = {Sensors},
VOLUME = {24},
YEAR = {2024},
NUMBER = {22},
ARTICLE-NUMBER = {7266},
URL = {https://www.mdpi.com/1424-8220/24/22/7266},
ISSN = {1424-8220},
ABSTRACT = {Semantic segmentation of remote sensing images is a fundamental task in computer vision, holding substantial relevance in applications such as land cover surveys, environmental protection, and urban building planning. In recent years, multi-modal fusion-based models have garnered considerable attention, exhibiting superior segmentation performance when compared with traditional single-modal techniques. Nonetheless, the majority of these multi-modal models, which rely on Convolutional Neural Networks (CNNs) or Vision Transformers (ViTs) for feature fusion, face limitations in terms of remote modeling capabilities or computational complexity. This paper presents a novel Mamba-based multi-modal fusion network called MFMamba for semantic segmentation of remote sensing images. Specifically, the network employs a dual-branch encoding structure, consisting of a CNN-based main encoder for extracting local features from high-resolution remote sensing images (HRRSIs) and of a Mamba-based auxiliary encoder for capturing global features on its corresponding digital surface model (DSM). To capitalize on the distinct attributes of the multi-modal remote sensing data from both branches, a feature fusion block (FFB) is designed to synergistically enhance and integrate the features extracted from the dual-branch structure at each stage. Extensive experiments on the Vaihingen and the Potsdam datasets have verified the effectiveness and superiority of MFMamba in semantic segmentation of remote sensing images. Compared with state-of-the-art methods, MFMamba achieves higher overall accuracy (OA) and a higher mean F1 score (mF1) and mean intersection over union (mIoU), while maintaining low computational complexity.},
DOI = {10.3390/s24227266}
}
``` 
# Acknowledgement
Many thanks the following projects's contributions to **MFMamba**.  
[RS3Mamba](https://github.com/sstary/SSRS)  
[UNetFormer](https://github.com/WangLibo1995/GeoSeg)  
[PKINet](https://github.com/NUST-Machine-Intelligence-Laboratory/PKINet)  
[SwiftFormer](https://github.com/Amshaker/SwiftFormer)  

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=YanWang-WHPU/WgANet&type=date&legend=top-left)](https://www.star-history.com/#YanWang-WHPU/WgANet&type=date&legend=top-left)
