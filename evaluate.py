import argparse
import os

from engine.wrapper import LanGuideMedSegWrapper

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl  

from utils.new_dataset import ImageTextMaskDataset
import utils.config as config


def get_parser():
    parser = argparse.ArgumentParser(
        description='Language-guide Medical Image Segmentation'
    )
    parser.add_argument('--config',
                        default='./config/training.yaml',
                        type=str,
                        help='config file')

    parser.add_argument('--model_name',
                        type=str,
                        default=None,
                        help='Full import path for the model class, e.g., utils.githubUNET.github_UNET')

    parser.add_argument('--ckpt',
                        type=str,
                        default=None,
                        help='Full import path for the model class, e.g., utils.githubUNET.github_UNET')
    parser.add_argument('--dataset_name',
                        type=str,
                        default=None,
                        help='Full import path for the model class, e.g., utils.githubUNET.github_UNET')

    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)

    if args.model_name is not None:
        cfg['model_name'] = args.model_name
        cfg['ckpt'] = args.ckpt
        cfg['dataset_name'] = args.dataset_name

    return cfg

if __name__ == '__main__':

    import random
    import numpy as np

    seed = 43
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = get_parser()

    # # load model
    # model = LanGuideMedSegWrapper(args)
    # checkpoint = torch.load('./save_model_github_unet/medseg_run_2.ckpt', map_location='cpu')["state_dict"]
    # model.load_state_dict(checkpoint, strict=True)
    # ds_test = ImageTextMaskDataset(tokenizer_type="microsoft/BiomedVLP-CXR-BERT-specialized",
    #                                prompt_type='p9',
    #                                images_dir='/media/iipl/35f051be-def5-48dd-b3a9-6db9e762c2d6/early_fusion/kvasir/kvasir-seg/Kvasir-SEG/images',
    #                                masks_dir='/media/iipl/35f051be-def5-48dd-b3a9-6db9e762c2d6/early_fusion/kvasir/kvasir-seg/Kvasir-SEG/masks',
    #                                caps_file='/media/iipl/35f051be-def5-48dd-b3a9-6db9e762c2d6/early_fusion/text_data/kvasir_polyp/anns/test.json')
    # dl_test = DataLoader(ds_test, batch_size=args.valid_batch_size, shuffle=False, num_workers=8, drop_last=False)
    # trainer = pl.Trainer(accelerator='gpu', devices=1)
    # model.eval()
    # result = trainer.test(model, dl_test)


    # path = '/github_unet/save_model_github_unet_aug'
    path = args.ckpt
    results = []
    for ckpt in os.listdir(path):
        if not ckpt.endswith('.ckpt'):
            continue
        # if ckpt[-6:-5] not in ['3']:
        #     continue
        print(ckpt)
        model = LanGuideMedSegWrapper(args)
        checkpoint = torch.load(os.path.join(path,ckpt), map_location='cpu')["state_dict"]
        model.load_state_dict(checkpoint,strict=True)

        if args.dataset_name == 'kvasir':
            print('using dataset: kvasir')
            print('using dataset: kvasir')
            print('using dataset: kvasir')
            print('using dataset: kvasir')

            ds_test = ImageTextMaskDataset(
                tokenizer_type="microsoft/BiomedVLP-CXR-BERT-specialized",
                prompt_type='p6',
                images_dir='/media/iipl/35f051be-def5-48dd-b3a9-6db9e762c2d6/early_fusion/kvasir/kvasir-seg/Kvasir-SEG/images',
                masks_dir='/media/iipl/35f051be-def5-48dd-b3a9-6db9e762c2d6/early_fusion/kvasir/kvasir-seg/Kvasir-SEG/masks',
                caps_file='/media/iipl/35f051be-def5-48dd-b3a9-6db9e762c2d6/early_fusion/text_data/kvasir_polyp/anns/test.json'

            )

        elif args.dataset_name == 'ClinicDB':
            print('using dataset: ClinicDB')
            print('using dataset: ClinicDB')
            print('using dataset: ClinicDB')
            print('using dataset: ClinicDB')

            ds_test =  ImageTextMaskDataset(
            tokenizer_type="microsoft/BiomedVLP-CXR-BERT-specialized",
            prompt_type='p9',
            images_dir='/media/iipl/35f051be-def5-48dd-b3a9-6db9e762c2d6/early_fusion/Medvlsm/CVC-ClinicDB/images',
            masks_dir='/media/iipl/35f051be-def5-48dd-b3a9-6db9e762c2d6/early_fusion/Medvlsm/CVC-ClinicDB/masks',
            caps_file='/media/iipl/35f051be-def5-48dd-b3a9-6db9e762c2d6/early_fusion/code/text_data/clinicdb_polyp/anns/test.json'
        )

        elif args.dataset_name == 'colondb':
            print('using dataset: colondb_polyp')
            print('using dataset: colondb_polyp')
            print('using dataset: colondb_polyp')
            print('using dataset: colondb_polyp')

            ds_test = ImageTextMaskDataset(
                tokenizer_type="microsoft/BiomedVLP-CXR-BERT-specialized",
                prompt_type='p6',
                images_dir='/media/iipl/35f051be-def5-48dd-b3a9-6db9e762c2d6/early_fusion/Medvlsm/CVC-ColonDB/images',
                masks_dir='/media/iipl/35f051be-def5-48dd-b3a9-6db9e762c2d6/early_fusion/Medvlsm/CVC-ColonDB/masks',
                caps_file='/media/iipl/35f051be-def5-48dd-b3a9-6db9e762c2d6/early_fusion/code/text_data/colondb_polyp/anns/test.json'
            )

        else:
            ValueError('No dataset')

        dl_test = DataLoader(ds_test, batch_size=args.valid_batch_size, shuffle=False, num_workers=8,drop_last=False)

        trainer = pl.Trainer(accelerator='gpu',devices=1)
        model.eval()
        result = trainer.test(model, dl_test)
        results.append(result[0])


    keys = results[0].keys()
    stats = {key: [] for key in keys}

    for d in results:
        for key in d:
            stats[key].append(d[key])

    # 计算均值和标准差
    print(stats.items())
    results = {key: {'mean': np.mean(values), 'std': np.std(values)} for key, values in stats.items()}

    # 打印结果
    for key, value in results.items():
        print(f"{key} - Mean: {value['mean']:.6f}, Std: {value['std']:.6f}")

    stats_file_path = os.path.join(args.ckpt, "stats.txt")
    with open(stats_file_path, "w") as f:
        for key, values in stats.items():
            f.write(f"{key}: {values}\n")

    # 保存均值和标准差到 txt 文件
    results_file_path = os.path.join(args.ckpt, "results.txt")
    with open(results_file_path, "w") as f:
        for key, value in results.items():
            f.write(f"{key} - Mean: {value['mean']:.6f}, Std: {value['std']:.6f}\n")

    print(f"Stats and results saved to {args.ckpt}")

# CUDA_VISIBLE_DEVICES=1 python evaluate.py --model_name utils.githubUNET.github_UNET.UNet --ckpt github_UNET --dataset_name kvasir --ckpt kvasir/github_UNET_norm
# CUDA_VISIBLE_DEVICES=1 python evaluate.py --model_name utils.githubUNET.github_UNET_aug.UNet --dataset_name kvasir --ckpt kvasir/github_UNET_aug_norm
# CUDA_VISIBLE_DEVICES=1 python evaluate.py --model_name utils.githubUNET.github_UNET_aug_text_gene.UNet --dataset_name kvasir --ckpt kvasir/github_UNET_aug_text_gene
# CUDA_VISIBLE_DEVICES=1 python evaluate.py --model_name utils.githubUNET.github_UNET_text_gene.UNet  --dataset_name kvasir --ckpt kvasir/github_UNET_text_norm_gene
# CUDA_VISIBLE_DEVICES=1 python evaluate.py --model_name utils.githubUNET.github_UNET_aug_text_img_gene.UNet --dataset_name kvasir --ckpt kvasir/github_UNET_aug_text_img_gene_norm
# CUDA_VISIBLE_DEVICES=1 python evaluate.py --model_name utils.githubUNET.github_UNET_text.UNet --dataset_name kvasir --ckpt kvasir/github_UNET_text_norm
# python evaluate.py --model_name utils.githubUNET.github_UNET_text_gene.UNet  --dataset_name kvasir --ckpt kvasir/github_UNET_text_gene_v1
# CUDA_VISIBLE_DEVICES=1 python evaluate.py --model_name utils.githubUNET.github_UNET_aug_text_gene.UNet --dataset_name kvasir --ckpt kvasir/github_UNET_aug_text_gene_norm_v1_compare

# CUDA_VISIBLE_DEVICES=1 tsp python evaluate.py --model_name utils.githubUNET.github_UNET.UNet --dataset_name ClinicDB --ckpt ClinicDB/github_UNET_norm
# CUDA_VISIBLE_DEVICES=1 tsp python evaluate.py --model_name utils.githubUNET.github_UNET_aug.UNet --dataset_name ClinicDB --ckpt ClinicDB/github_UNET_aug_norm
# CUDA_VISIBLE_DEVICES=1 tsp python evaluate.py --model_name utils.githubUNET.github_UNET_aug_text_gene.UNet --dataset_name ClinicDB --ckpt ClinicDB/github_UNET_aug_text_gene_norm
# CUDA_VISIBLE_DEVICES=1 tsp python evaluate.py --model_name utils.githubUNET.github_UNET_text.UNet --dataset_name ClinicDB --ckpt ClinicDB/github_UNET_text_norm
# CUDA_VISIBLE_DEVICES=1 tsp python evaluate.py --model_name utils.githubUNET.github_UNET_aug_text.UNet --dataset_name ClinicDB  --ckpt ClinicDB/github_UNET_aug_text_norm
# CUDA_VISIBLE_DEVICES=1 tsp python evaluate.py --model_name utils.githubUNET.github_UNET_text_gene.UNet --dataset_name ClinicDB  --ckpt ClinicDB/github_UNET_text_gene_norm


# CUDA_VISIBLE_DEVICES=1 tsp python evaluate.py --model_name utils.githubUNET.github_UNET.UNet --dataset_name colondb --ckpt colondb/github_UNET_norm
# CUDA_VISIBLE_DEVICES=1 tsp python evaluate.py --model_name utils.githubUNET.github_UNET_aug.UNet --dataset_name colondb --ckpt colondb/github_UNET_aug_norm
# CUDA_VISIBLE_DEVICES=1 tsp python evaluate.py --model_name utils.githubUNET.github_UNET_aug_text_gene.UNet --dataset_name colondb --ckpt colondb/github_UNET_aug_text_gene_p6
# CUDA_VISIBLE_DEVICES=1 tsp python evaluate.py --model_name utils.githubUNET.github_UNET_text.UNet --dataset_name colondb --ckpt colondb/github_UNET_text_p6
# CUDA_VISIBLE_DEVICES=1 tsp python evaluate.py --model_name utils.githubUNET.github_UNET_aug_text.UNet --dataset_name colondb  --ckpt colondb/github_UNET_aug_text_p6
# CUDA_VISIBLE_DEVICES=1 tsp python evaluate.py --model_name utils.githubUNET.github_UNET_text_gene.UNet --dataset_name colondb  --ckpt colondb/github_UNET_text_gene_p6


# CUDA_VISIBLE_DEVICES=1 tsp python evaluate.py --model_name utils.missformer.missformer.MISSFormer --dataset_name kvasir --ckpt kvasir/missformer_norm
# CUDA_VISIBLE_DEVICES=1 python evaluate.py --model_name utils.missformer.missformer_aug_text_gene.MISSFormer --dataset_name kvasir  --ckpt kvasir/missformer_aug_text_gene_norm
# CUDA_VISIBLE_DEVICES=1 tsp python evaluate.py --model_name utils.missformer.missformer_text.MISSFormer --dataset_name kvasir   --ckpt kvasir/missformer_text_norm
# CUDA_VISIBLE_DEVICES=1 tsp python evaluate.py --model_name utils.missformer.missformer_aug_text.MISSFormer --dataset_name kvasir --ckpt kvasir/missformer_aug_text_norm
# CUDA_VISIBLE_DEVICES=1 python evaluate.py --model_name utils.missformer.missformer_text_gene.MISSFormer --dataset_name kvasir  --ckpt kvasir/missformer_text_gene_norm
# CUDA_VISIBLE_DEVICES=1 tsp python evaluate.py --model_name utils.missformer.missformer_aug.MISSFormer --dataset_name kvasir   --ckpt kvasir/missformer_aug_norm

# CUDA_VISIBLE_DEVICES=1 python evaluate.py --model_name utils.transunet.transunet.VisionTransformer --dataset_name kvasir --ckpt kvasir/transunet_norm
# CUDA_VISIBLE_DEVICES=1 python evaluate.py --model_name utils.transunet.transunet_aug.VisionTransformer --dataset_name kvasir  --ckpt kvasir/transunet_aug_norm
# CUDA_VISIBLE_DEVICES=1 python evaluate.py --model_name utils.transunet.transunet_text_gene.VisionTransformer --dataset_name kvasir  --ckpt kvasir/transunet_text_gene_norm
# CUDA_VISIBLE_DEVICES=1 python evaluate.py --model_name utils.transunet.transunet_text.VisionTransformer --dataset_name kvasir  --ckpt kvasir/transunet_text_norm
# CUDA_VISIBLE_DEVICES=1 python evaluate.py --model_name utils.transunet.transunet_aug_text.VisionTransformer --dataset_name kvasir  --ckpt kvasir/transunet_aug_text_norm
# CUDA_VISIBLE_DEVICES=1 python evaluate.py --model_name utils.transunet.transunet_aug_text_gene.VisionTransformer --dataset_name kvasir  --ckpt kvasir/transunet_aug_text_gene_norm

# CUDA_VISIBLE_DEVICES=1 python evaluate.py --model_name utils.Unet3plus.unet3.UNet_3Plus --dataset_name kvasir  --ckpt kvasir/unet3_norm
# CUDA_VISIBLE_DEVICES=1 python evaluate.py --model_name utils.Unet3plus.unet3_text.UNet_3Plus --dataset_name kvasir  --ckpt kvasir/unet3_text_norm
# CUDA_VISIBLE_DEVICES=1 python evaluate.py --model_name utils.Unet3plus.unet3_aug.UNet_3Plus --dataset_name kvasir  --ckpt kvasir/unet3_aug_norm
# CUDA_VISIBLE_DEVICES=1 python evaluate.py --model_name utils.Unet3plus.unet3_text_gene.UNet_3Plus --dataset_name kvasir  --ckpt kvasir/unet3_text_gene_norm
# CUDA_VISIBLE_DEVICES=1 tsp python evaluate.py --model_name utils.Unet3plus.unet3_text_aug.UNet_3Plus --dataset_name kvasir  --ckpt kvasir/unet3_text_aug_norm
# CUDA_VISIBLE_DEVICES=1 tsp python evaluate.py --model_name utils.Unet3plus.unet3_text_aug_gene.UNet_3Plus --dataset_name kvasir  --ckpt kvasir/unet3_text_aug_gene_norm



# CUDA_VISIBLE_DEVICES=1 python evaluate.py --model_name utils.unet_plus.unet_plus.NestedUNet --dataset_name kvasir  --ckpt kvasir/unet_plus_norm
# CUDA_VISIBLE_DEVICES=1 python evaluate.py --model_name utils.unet_plus.unet_plus_aug.NestedUNet --dataset_name kvasir  --ckpt kvasir/unet_plus_aug_norm
# CUDA_VISIBLE_DEVICES=1 python evaluate.py --model_name utils.unet_plus.unet_plus_text.NestedUNet --dataset_name kvasir  --ckpt kvasir/unet_plus_text_norm
# CUDA_VISIBLE_DEVICES=1 python evaluate.py --model_name utils.unet_plus.unet_plus_text_gene.NestedUNet --dataset_name kvasir  --ckpt kvasir/unet_plus_text_gene_norm
# CUDA_VISIBLE_DEVICES=1 python evaluate.py --model_name utils.unet_plus.unet_plus_aug_text.NestedUNet --dataset_name kvasir  --ckpt kvasir/unet_plus_aug_text_norm
# CUDA_VISIBLE_DEVICES=1 python evaluate.py --model_name utils.unet_plus.unet_plus_aug_text_gene.NestedUNet --dataset_name kvasir  --ckpt kvasir/unet_plus_aug_text_gene_v1