import torch
import torch.nn as nn
import pytorch_lightning as pl
from src.utils.utils import load_class

class ModelInterface(pl.LightningModule):
    def __init__(self, model_name, loss_name, lr, **kwargs):
        super().__init__()
        # Store hyperparameters (model_name, loss_name, lr, and any extra model args)
        self.save_hyperparameters()
        
        # 1. Dynamically load the model architecture
        model_class = load_class(self.hparams.model_name, base_path="src.models")
        self.model = model_class(**kwargs)
        
        # 2. Setup loss function
        self.loss_function = self._get_loss(self.hparams.loss_name)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Main training logic: forward pass + loss calculation
        """
        x, y = batch
        out = self(x)
        loss = self.loss_function(out, y)
        
        # Logging metrics
        acc = (out.argmax(dim=1) == y).float().mean()
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Main validation logic: evaluation and metrics
        """
        x, y = batch
        out = self(x)
        loss = self.loss_function(out, y)
        
        # Calculate accuracy
        acc = (out.argmax(dim=1) == y).float().mean()
        
        # Log metrics (Lightning automatically averages these over the epoch)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        
        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        """
        Define optimizer and learning rate scheduler
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        
        # Example: Optional scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def _get_loss(self, loss_name):
        # Mapping string names to actual loss functions
        losses = {
            "ce": nn.CrossEntropyLoss(),
            "mse": nn.MSELoss(),
            "l1": nn.L1Loss()
        }
        return losses.get(loss_name.lower(), nn.CrossEntropyLoss())