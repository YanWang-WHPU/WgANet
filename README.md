WgANet: A Wavelet-guided Attention Network for Remote Sensing Image Semantic Segmentation  
===
# Framework
![WgANet_model](https://github.com/YanWang-WHPU/WgANet/blob/main/figure_model.png "WgANet")
> Overall architecture of WgANet, comprising three parallel encoding branches: a spatial Mamba branch, a spatial convolutional branch, and a frequency
 band branch, along with a decoder.
# Environment
Our experiments were implemented with the PyTorch framework done on a single NVIDIA A40 GPU equipped with 48GB RAM.  

# Datasets 
 Supported Remote Sensing Datasets
| Datasets | Link|
| ---------- | -----------|
| ISPRS Vaihingen   | [download](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/default.aspx)   |
| ISPRS Potsdam   | [download](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/default.aspx)   |
| LoveDA   | [download](https://codalab.lisn.upsaclay.fr/competitions/421)   |

## Pretrained Weights of Backbones 
[vmamba_tiny_e292.pth](https://github.com/YanWang-WHPU/WgANet/blob/main/pretrain/vmamba_tiny_e292.pth)
## Folder Structure
Prepare the following folders to organize this repo:
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
│   ├── LoveDA
│   │   ├── Train
│   │   │   ├── Urban
│   │   │   │   ├── images_png (original images)
│   │   │   │   ├── masks_png (original masks)
│   │   │   │   ├── masks_png_convert (converted masks used for training)
│   │   │   │   ├── masks_png_convert_rgb (original rgb format masks)
│   │   │   ├── Rural
│   │   │   │   ├── images_png 
│   │   │   │   ├── masks_png 
│   │   │   │   ├── masks_png_convert
│   │   │   │   ├── masks_png_convert_rgb
│   │   ├── Val (the same with Train)
│   │   ├── Test
│   │   ├── train_val (Merge Train and Val)
```
# Install
### 1. Open the folder WgANet using Linux Terminal and create python environment:
```
conda create -n WgANet python=3.8 -y
conda activate WgANet
```
### 2. Install cuda=11.8
```
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
pip install cuda_11.8.0_520.61.05_linux.run
```
### 3. Install torch=2.0.0
```
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```
or(pip):
```
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118 
```
### 4. Install Mamba
#### 4.1 Install mamba_ssm
```
pip install mamba-ssm
```
Because installing with pip can be problematic, we recommend downloading and installing it [here](https://github.com/state-spaces/mamba/releases)
#### 4.2 Install causal-conv1d
```
pip install causal-conv1d
```
Because installing with pip can be problematic, we recommend downloading and installing it [here](https://github.com/Dao-AILab/causal-conv1d/releases)
# Training 
1. Modify the address of data in utils_WgANet.py, batch_size, training mode =train, training model =WgANet.  

2. Modify the address for saving the results in train_WgANet.py
  
4. train
```
python train_WgANet.py
``` 
# test
### Modify the parameters and addresses
1. Modify the address of results in train_WgANet.py to be the address of the best trained model  
2. Modify the training mode =test in utils_WgANet.py
3. test
```
python train_WgANet.py
```
# Reproduction Results on the ISPRS Vaihingen Dataset
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
# Visual Results on the ISPRS Vaihingen Dataset
![figure_vaihingen](https://github.com/YanWang-WHPU/WgANet/blob/main/figure_vaihingen.png "results")
> Qualitative comparisons on the ISPRS Vaihingen dataset. For each method, the left panel shows the original prediction map, and the right panel
 shows a cropped region from the left side with a size of 256 × 256. Red boxes highlight challenging regions: the lower box shows shadowed vehicles and
 building edges, while the upper box shows tree and low-vegetation boundaries. (a) original NIRRG image, (b) ground truth, segmentation results produced
 by (c) ABCNet, (d) UNetMamba, (e) TransUNet, (f) UNetFormer, (g) MAResU-Net, (h) CMTFNet, (i) RS3Mamba, (j) SFFNet, (k) PHA-UNet, (l) MCAFT,
 (m) DECS-Net, (n) MIFNet, and (o) our proposed WgANet.
# Citations
If these codes are helpful for your study, please cite:
```
@ARTICLE{11267048,
  author={Wang, Yan and Cao, Li and Deng, He},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={WgANet: A Wavelet-Guided Attention Network for Remote Sensing Image Semantic Segmentation}, 
  year={2025},
  volume={},
  number={},
  pages={1-15},
  keywords={Remote sensing;Wavelet transforms;Frequency-domain analysis;Feature extraction;Transforms;Wavelet analysis;Transformers;Semantic segmentation;Visualization;Land surface;Feature fusion;frequency-domain information;remote sensing;semantic segmentation;visual state space;wavelet transform},
  doi={10.1109/JSTARS.2025.3636614}}

``` 
# Acknowledgement
Many thanks the following projects's contributions to **WgANet**.  
[MFMamba](https://github.com/YanWang-WHPU/MFMamba)  
[Wave-Mamba](https://github.com/AlexZou14/Wave-Mamba)  
[RS3Mamba](https://github.com/sstary/SSRS)    
[UNetFormer](https://github.com/WangLibo1995/GeoSeg)   
[ConTriNet_RGBT-SOD](https://github.com/CSer-Tang-hao/ConTriNet_RGBT-SOD)  
## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=YanWang-WHPU/WgANet&type=date&legend=top-left)](https://www.star-history.com/#YanWang-WHPU/WgANet&type=date&legend=top-left)
