import torch
from torch.utils.data import DataLoader
from utils.new_dataset import ImageTextMaskDataset
import utils.config as config
from torch.optim import lr_scheduler
from engine.wrapper import LanGuideMedSegWrapper

import pytorch_lightning as pl
from torchmetrics import Accuracy, Dice
from torchmetrics.classification import BinaryJaccardIndex
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import os
import torch.multiprocessing
import argparse

import warnings

# 忽略所有警告
warnings.filterwarnings("ignore")



torch.multiprocessing.set_sharing_strategy('file_system')

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

    parser.add_argument('--dataset_name',
                        type=str,
                        default=None,
                        help='Full import path for the model class, e.g., utils.githubUNET.github_UNET')
    parser.add_argument('--expriment_name',
                        type=str,
                        default=None,
                        help='Full import path for the model class, e.g., utils.githubUNET.github_UNET')

    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)

    if args.model_name is not None:
        cfg['model_name'] = args.model_name
        cfg['dataset_name'] = args.dataset_name
        cfg['expriment_name'] = args.expriment_name

    return cfg

if __name__ == '__main__':

    args = get_parser()
    print("cuda:", torch.cuda.is_available())

    if args.dataset_name == 'kvasir':
        print('using dataset: kvasir')
        print('using dataset: kvasir')
        print('using dataset: kvasir')
        print('using dataset: kvasir')

        ds_train = ImageTextMaskDataset(
            tokenizer_type="microsoft/BiomedVLP-CXR-BERT-specialized",
            prompt_type='p9',
            images_dir='/media/iipl/35f051be-def5-48dd-b3a9-6db9e762c2d6/early_fusion/kvasir/kvasir-seg/Kvasir-SEG/images',
            masks_dir='/media/iipl/35f051be-def5-48dd-b3a9-6db9e762c2d6/early_fusion/kvasir/kvasir-seg/Kvasir-SEG/masks',
            caps_file='/media/iipl/35f051be-def5-48dd-b3a9-6db9e762c2d6/early_fusion/text_data/kvasir_polyp/anns/train.json'

        )

        ds_valid = ImageTextMaskDataset(
            tokenizer_type="microsoft/BiomedVLP-CXR-BERT-specialized",
            prompt_type='p9',
            images_dir='/media/iipl/35f051be-def5-48dd-b3a9-6db9e762c2d6/early_fusion/kvasir/kvasir-seg/Kvasir-SEG/images',
            masks_dir='/media/iipl/35f051be-def5-48dd-b3a9-6db9e762c2d6/early_fusion/kvasir/kvasir-seg/Kvasir-SEG/masks',
            caps_file='/media/iipl/35f051be-def5-48dd-b3a9-6db9e762c2d6/early_fusion/text_data/kvasir_polyp/anns/val.json'
        )
    elif args.dataset_name == 'ClinicDB':
        print('using dataset: ClinicDB')
        print('using dataset: ClinicDB')
        print('using dataset: ClinicDB')
        print('using dataset: ClinicDB')

        ds_train = ImageTextMaskDataset(
            tokenizer_type="microsoft/BiomedVLP-CXR-BERT-specialized",
            prompt_type='p9',
            images_dir='/media/iipl/35f051be-def5-48dd-b3a9-6db9e762c2d6/early_fusion/Medvlsm/CVC-ClinicDB/images',
            masks_dir='/media/iipl/35f051be-def5-48dd-b3a9-6db9e762c2d6/early_fusion/Medvlsm/CVC-ClinicDB/masks',
            caps_file='/media/iipl/35f051be-def5-48dd-b3a9-6db9e762c2d6/early_fusion/code/text_data/clinicdb_polyp/anns/train.json'
        )

        ds_valid = ImageTextMaskDataset(
            tokenizer_type="microsoft/BiomedVLP-CXR-BERT-specialized",
            prompt_type='p9',
            images_dir='/media/iipl/35f051be-def5-48dd-b3a9-6db9e762c2d6/early_fusion/Medvlsm/CVC-ClinicDB/images',
            masks_dir='/media/iipl/35f051be-def5-48dd-b3a9-6db9e762c2d6/early_fusion/Medvlsm/CVC-ClinicDB/masks',
            caps_file='/media/iipl/35f051be-def5-48dd-b3a9-6db9e762c2d6/early_fusion/code/text_data/clinicdb_polyp/anns/val.json'
        )
    elif args.dataset_name == 'colondb':
        print('using dataset: colondb_polyp')
        print('using dataset: colondb_polyp')
        print('using dataset: colondb_polyp')
        print('using dataset: colondb_polyp')

        ds_train = ImageTextMaskDataset(
            tokenizer_type="microsoft/BiomedVLP-CXR-BERT-specialized",
            prompt_type='p6',
            images_dir='/media/iipl/35f051be-def5-48dd-b3a9-6db9e762c2d6/early_fusion/Medvlsm/CVC-ColonDB/images',
            masks_dir='/media/iipl/35f051be-def5-48dd-b3a9-6db9e762c2d6/early_fusion/Medvlsm/CVC-ColonDB/masks',
            caps_file='/media/iipl/35f051be-def5-48dd-b3a9-6db9e762c2d6/early_fusion/code/text_data/colondb_polyp/anns/train.json'
        )

        ds_valid = ImageTextMaskDataset(
            tokenizer_type="microsoft/BiomedVLP-CXR-BERT-specialized",
            prompt_type='p6',
            images_dir='/media/iipl/35f051be-def5-48dd-b3a9-6db9e762c2d6/early_fusion/Medvlsm/CVC-ColonDB/images',
            masks_dir='/media/iipl/35f051be-def5-48dd-b3a9-6db9e762c2d6/early_fusion/Medvlsm/CVC-ColonDB/masks',
            caps_file='/media/iipl/35f051be-def5-48dd-b3a9-6db9e762c2d6/early_fusion/code/text_data/colondb_polyp/anns/val.json'
        )

    else:
        ValueError('No dataset')

    dl_train = DataLoader(ds_train, batch_size=args.train_batch_size, shuffle=True, num_workers=8)
    dl_valid = DataLoader(ds_valid, batch_size=args.valid_batch_size, shuffle=False, num_workers=8)

    for i in range(5):  # Repeat the experiment 5 times
        print(f"Starting experiment {i + 1}")

        # Reinitialize model
        model = LanGuideMedSegWrapper(args)

        # Setting checkpoint and early stopping callbacks
        model_ckpt = ModelCheckpoint(
            # dirpath=args.model_save_path,
            dirpath=os.path.join(args.dataset_name, args.model_name.split('.')[-2] + '_' + args.expriment_name),
            filename=f"{args.model_save_filename}_run_{i + 1}",
            monitor='val_loss',
            save_top_k=1,
            mode='min',
            verbose=True,
        )

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=args.patience,
            mode='min'
        )

        # Initialize trainer
        trainer = pl.Trainer(
            logger=True,
            min_epochs=args.min_epochs,
            max_epochs=args.max_epochs,
            accelerator='gpu',
            devices=args.device,
            callbacks=[model_ckpt, early_stopping],
            enable_progress_bar=False,
        )

        # Set random seed for reproducibility
        pl.seed_everything(42 + i)

        # Start training
        print('Start training')
        trainer.fit(model, dl_train, dl_valid)
        print('Done training')

# CUDA_VISIBLE_DEVICES=1  python train.py