import argparse
import sys
import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import hatasını önlemek için try-except bloğu veya doğrudan import kullanımı
try:
    from .model import HSIModel
    from .dataset import HSIDataset, collate_fn
    from .config import get_default_config
except ImportError:
    # Eğer script doğrudan çalıştırılıyorsa (örn: python train.py) burası çalışır
    from model import HSIModel
    from dataset import HSIDataset, collate_fn
    from config import get_default_config

class HSITrainer(pl.LightningModule):
    """
    LightningModule for training the HSI model.
    """
    def __init__(self, config=None, model=None):
        super().__init__()
        
        if config is None:
            config = get_default_config()
        self.config = config
        
        # Initialize model
        if model is None:
            self.model = HSIModel(
                num_classes=self.config["num_classes"],
                nms_thresh=self.config["nms_thresh"],
                score_thresh=self.config["score_thresh"],
                in_channels=self.config.get("in_channels", 3)
            )
        else:
            self.model = model
            
        self.save_hyperparameters(ignore=['model'])

    def forward(self, x):
        return self.model.predict(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        
        # RetinaNet returns a dict of losses during training
        loss_dict = self.model(images, targets)
        
        # Sum up losses
        # typical keys: 'classification', 'bbox_regression'
        losses = sum(loss for loss in loss_dict.values())
        
        self.log("train_loss", losses, on_step=True, on_epoch=True, prog_bar=True)
        return losses

    def validation_step(self, batch, batch_idx):
        # Validation is tricky with RetinaNet because it returns boxes in eval mode, not losses.
        # To compute validation loss, we must enable training mode temporarily or implementation a custom evaluator.
        # However, usually we want to evaluate mAP.
        
        # Current torchvision RetinaNet behavior:
        # train() mode -> returns losses
        # eval() mode -> returns detections
        
        # For validation loss:
        self.model.train() # Switch to train mode to get losses
        with torch.no_grad():
            images, targets = batch
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            self.log("val_loss", losses, on_epoch=True, prog_bar=True)
            
        # Switch back to eval if needed for other metrics (omitted here for brevity)
        # In a real scenario, you'd calculate mAP here.
        self.model.eval() 

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.config["train"]["lr"],
            momentum=0.9,
            weight_decay=0.0005
        )
        
        # Optional: Add scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        if self.config["train"]["csv_file"] and self.config["train"]["root_dir"]:
            dataset = HSIDataset(
                csv_file=self.config["train"]["csv_file"],
                root_dir=self.config["train"]["root_dir"],
                label_dict=self.config["label_dict"],
                selected_bands=self.config.get("selected_bands")
            )
            return DataLoader(
                dataset,
                batch_size=self.config["batch_size"],
                shuffle=True,
                num_workers=self.config["workers"],
                collate_fn=collate_fn
            )
        return None

    def val_dataloader(self):
        if self.config["validation"]["csv_file"] and self.config["validation"]["root_dir"]:
            dataset = HSIDataset(
                csv_file=self.config["validation"]["csv_file"],
                root_dir=self.config["validation"]["root_dir"],
                label_dict=self.config["label_dict"],
                selected_bands=self.config.get("selected_bands")
            )
            return DataLoader(
                dataset,
                batch_size=self.config["batch_size"],
                shuffle=False,
                num_workers=self.config["workers"],
                collate_fn=collate_fn
            )
        return None

def train_model(config=None):
    """
    Entry point function to start training.
    """
    if config is None:
        config = get_default_config()
        
    model = HSITrainer(config)
    
    # Validation verisi var mı kontrol et
    has_validation = config["validation"]["csv_file"] and config["validation"]["root_dir"]
    
    trainer = pl.Trainer(
        max_epochs=config["train"]["epochs"],
        accelerator=config["accelerator"],
        devices=config["devices"],
        fast_dev_run=config["train"].get("fast_dev_run", False),
        # Eğer validation verisi yoksa, sanity check ve validation adımlarını kapat
        limit_val_batches=1.0 if has_validation else 0,
        num_sanity_val_steps=2 if has_validation else 0
    )
    
    trainer.fit(model)
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="Train HSI Model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--bands", type=int, default=3, help="Number of input bands/channels")
    parser.add_argument("--csv_train", type=str, required=False, help="Path to training CSV annotation file")
    parser.add_argument("--root_train", type=str, required=False, help="Path to training images directory")
    parser.add_argument("--csv_val", type=str, required=False, help="Path to validation CSV annotation file")
    parser.add_argument("--root_val", type=str, required=False, help="Path to validation images directory")
    parser.add_argument("--selected_bands", type=str, default=None, help="Comma separated list of 0-based band indices (e.g., '0,10,20')")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    config = get_default_config()
    
    # Update config with args
    config["train"]["epochs"] = args.epochs
    config["train"]["lr"] = args.lr
    config["batch_size"] = args.batch_size
    config["in_channels"] = args.bands
    
    if args.selected_bands:
        config["selected_bands"] = [int(x) for x in args.selected_bands.split(",")]

    if args.csv_train:
        config["train"]["csv_file"] = args.csv_train
    if args.root_train:
        config["train"]["root_dir"] = args.root_train
    if args.csv_val:
        config["validation"]["csv_file"] = args.csv_val
    if args.root_val:
        config["validation"]["root_dir"] = args.root_val
        
    # Start training
    train_model(config)
