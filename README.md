# AD-SRF：Attack-Defense Semantic Reinforcement for Multi-Modal Image Fusion


---

## 📋 Overview
AD-SRF is a multi-modal image fusion (MMIF) algorithm that decouples visual and semantic optimization. It reinforces semantic attributes through Attack-Defend Game mechanism. By jointly optimizing visual fidelity and semantic consistency, AD-SRF achieves a balanced and superior representation in both visual quality and semantic understanding.

## 🖼️ Framework
![](framework.png)

## 🧰 Environment Setup
```bash
conda env create -f environment.yml
conda activate ad-srf
```



## 📂 Dataset Preparation
- 📎 [MFNet](https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/) & [FMB](https://github.com/JinyuanLiu-CV/SegMiF)
- 📦 [Our processed dataset (with HQ images)](https://xxx.com)
- The dataset should be organized as follows:
```
MFNet/ (or FMB/)
├── train/
│   ├── infrared/        # Infrared images
│   ├── infrared_HQ/     # High-quality infrared images
│   ├── visible/         # Visible images
│   ├── visible_HQ/      # High-quality visible images
│   └── label/           # Semantic segmentation labels
│
├── val/
│   ├── infrared/
│   ├── infrared_HQ/
│   ├── visible/
│   ├── visible_HQ/
│   └── label/
```
---

## 🔥 To Train

### 1️⃣ Train Fusion Network 
#### MFNet Dataset
*⚠️ Ensure all dataset paths are correctly configured in .py config files*

```bash
CUDA_VISIBLE_DEVICES=0 python trainFusion.py config_MFNet/config_train_fusion.py
```
#### FMB Dataset
```bash
CUDA_VISIBLE_DEVICES=0 python trainFusion.py config_FMB/config_train_fusion.py
```



### 2️⃣ Pretrain Semantic Segmentation 

- Download pretrained  [Overlock_B](https://github.com/LMMMEng/OverLoCK?tab=readme-ov-file) weight 
 
 *⚠️ Ensure all dataset paths are correctly configured in .py config files*
#### MFNet Dataset
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=25888 trainseg.py config_MFNet/config_train_seg.py --launcher pytorch
```
#### FMB Dataset
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=25888 trainseg.py config_FMB/config_train_seg.py --launcher pytorch
```




### 3️⃣ Attack-Defense Semantic Reinforcement

*⚠️ Ensure all dataset paths are correctly configured in .py config files*

***⚠️⚠️ Ensure all pretrained weights paths are correctly configured in .py config files***

#### MFNet Dataset
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=25888 trainADseg.py config_MFNet/config_train_generator_ADseg.py --launcher pytorch
```
#### FMB Dataset
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=25888 trainADseg.py config_FMB/config_train_generator_ADseg.py --launcher pytorch
```
---

## 🧪 To Test


### ✨ Visual Test 
*⚠️ Change your image path for test in `testfusion.py`*
#### MFNet Dataset
```bash
python testfusion.py config_MFNet/config_train_Fusion.py
```
#### FMB Dataset
```bash
python testfusion.py config_FMB/config_train_Fusion.py
```


### 🧠 Semantic Test 
Test semantic segmentation accuracy of fused images with pre-trained checkpoints<br>
*⚠️ Ensure all dataset paths are correctly configured in .py config files*<br>
***⚠️⚠️ Ensure all pretrained weights paths are correctly configured in .py config files***
#### MFNet Dataset
```bash
CUDA_VISIBLE_DEVICES=0 python testADseg.py config_MFNet/config_test_generator_ADseg.py --checkpoint path/to/your/MFNetckpt --work-dir path/to/your/workdir
```

#### FMB Dataset
```bash
CUDA_VISIBLE_DEVICES=0 python testADseg.py config_FMB/config_test_generator_ADseg.py --checkpoint path/to/your/FMBckpt --work-dir path/to/your/workdir
```
#### 📌 Reproducibility for FMB Dataset
- The released code computes 14-class mIoU by default. (Testset lacks bike class.)
- To obtain 15-class mIoU, a simple normalization adjustment is applied during reporting.

---
## 📎 Pretrained Weights
#### The pretrained weights of our AD-SRF can be found here:
|   Visual            | Semantic     | mIoU
|  ---------------------  | -----------      | ---- |
|  [MFNet](https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/)  | [MFNet](https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/) | 61.64
|   [FMB](https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/)      | [FMB](https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/) | 63.87

---

## 🤝 Acknowledgements
Our code is built upon the following libraries. We sincerely thank the authors for their contributions. If you use any components or pretrained weights from these works, please make sure to cite the corresponding references.
#### [OverLoCK](https://github.com/LMMMEng/OverLoCK?tab=readme-ov-file) | [EGS-TSSA](https://github.com/MisterRpeng/EGS-TSSA) | [MMSeg](https://github.com/open-mmlab/mmsegmentation)
---

## 📝 Citation
If you use AD-SRF in your research, please cite our paper:
```
@article{adsrf2026,
  title={AD-SRF: Attack-Defense Semantic Reinforcement for Multi-Modal Image Fusion},
  author={Zhang, Hao and Zheng, Zhiqian and Tang, Linfeng and Xiang, Xinyu and Ma, Jiayi},
  journal={...},
  year={2026}
}
```

---

