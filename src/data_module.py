
import os
from typing import Optional, Tuple
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import pytorch_lightning as pl
from albumentations import (
    HorizontalFlip, VerticalFlip, Rotate, ShiftScaleRotate,
    RandomBrightnessContrast, Normalize, Compose
)
from albumentations.pytorch import ToTensorV2
import numpy as np



class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = './data',
        batch_size: int = 64,
        num_workers: int = 4,
        val_split: float = 0.2,
        random_seed: int = 42,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.random_seed = random_seed

        # Define transforms
        self.train_transforms = Compose([
            HorizontalFlip(),
            RandomBrightnessContrast(),
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15),
            Normalize(mean=(0.5071, 0.4865, 0.4409),
                      std=(0.2673, 0.2564, 0.2762)),
            ToTensorV2(),
        ])

        self.val_transforms = Compose([
            Normalize(mean=(0.5071, 0.4865, 0.4409),
                      std=(0.2673, 0.2564, 0.2762)),
            ToTensorV2(),
        ])
    def get_class_names(self):
        return self.test_dataset.classes

    def prepare_data(self):
        # Download CIFAR-100 dataset
        datasets.CIFAR100(root=self.data_dir, train=True, download=True)
        datasets.CIFAR100(root=self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            full_train = datasets.CIFAR100(
                root=self.data_dir, train=True, transform=self.train_transforms, download=False
            )
            val_size = int(len(full_train) * self.val_split)
            train_size = len(full_train) - val_size
            self.train_dataset, self.val_dataset = random_split(
                full_train, [train_size, val_size],
                generator=torch.Generator().manual_seed(self.random_seed)
            )
            # Apply validation transforms
            self.val_dataset.dataset.transform = self.val_transforms

        if stage == 'test' or stage is None:
            self.test_dataset = datasets.CIFAR100(
                root=self.data_dir, train=False, transform=self.val_transforms, download=False
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
