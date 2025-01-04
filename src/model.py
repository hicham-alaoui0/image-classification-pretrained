# src/model.py

import torch
from torch import nn
import pytorch_lightning as pl
from transformers import ViTModel, ViTConfig



class VisionTransformerClassifier(pl.LightningModule):
    def __init__(
        self,
        model_name: str = 'google/vit-base-patch16-224-in21k',
        num_classes: int = 100,
        learning_rate: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Load pretrained ViT model
        self.vit = ViTModel.from_pretrained(model_name)

        # Classification head
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        pooled_output = outputs.pooler_output  # Shape: (batch_size, hidden_size)
        logits = self.classifier(pooled_output)  # Shape: (batch_size, num_classes)
        return logits

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]
