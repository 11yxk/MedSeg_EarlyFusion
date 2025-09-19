# MedSeg_EarlyFusion


Pytorch implementation of paper '' A Text-Image Fusion Method with Data Augmentation Capabilities for Referring Medical Image Segmentation '',the code will be updataed progressively.


# Data Preparation
For Kvasir and ISIC dataset:
Please follow the instructions of [medvlsm](https://github.com/naamiinepal/medvlsm).

For QaTa-COV19 dataset:
Please follow the instructions of [LViT](https://github.com/HUANGLIZI/LViT).



### File Descriptions
For Example (UNET):

| Filename                                      | Data Augmentation | Text Guidance | Fusion Strategy     | Special Notes                          |
|----------------------------------------------|-------------------|----------------|---------------------|----------------------------------------|
| `github_UNET.py`                              | ❌                | ❌             | N/A                 | Baseline UNet                          |
| `github_UNET_aug.py`                          | ✅                | ❌             | N/A                 | Data augmentation only                 |
| `github_UNET_text_gene.py`                   | ❌                | ✅             | Early Fusion    | Text guidance only                     |
| `github_UNET_aug_text_gene.py`               | ✅                | ✅             | Early Fusion    | Augmentation + text guidance           |




### Training

1. Change the dataset path (Kvasir, ISIC and QaTa-COV19) in train.py

2. Run

```
# Example: training network on kvasir dataset using UNet
python train.py --model_name utils.githubUNET.github_UNET.UNet --dataset_name kvasir  --expriment_name your_exp_name
```
### Testing

```
# Example: testing network on kvasir dataset using UNet
python evaluate.py --model_name utils.githubUNET.github_UNET.UNet --dataset_name kvasir --ckpt path/to/your/ckpt
```

