import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MNIST with Lightning DDP.")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory for MNIST data.")
    parser.add_argument("--batch-size", type=int, default=128, help="Per-device batch size.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader worker count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--save-dir", type=str, default="./checkpoints_ddp", help="Checkpoint directory.")
    parser.add_argument("--devices", type=int, default=2, help="Number of GPUs for DDP.")
    return parser.parse_args()


class MNISTDataModule(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int, seed: int) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self) -> None:
        datasets.MNIST(root=self.data_dir, train=True, download=True)
        datasets.MNIST(root=self.data_dir, train=False, download=True)

    def setup(self, stage: str | None = None) -> None:
        if stage in ("fit", None):
            full_train_dataset = datasets.MNIST(
                root=self.data_dir,
                train=True,
                download=False,
                transform=self.transform,
            )
            train_size = int(0.9 * len(full_train_dataset))
            val_size = len(full_train_dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(
                full_train_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.seed),
            )

        if stage in ("test", None):
            self.test_dataset = datasets.MNIST(
                root=self.data_dir,
                train=False,
                download=False,
                transform=self.transform,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )


class LitMNISTClassifier(LightningModule):
    def __init__(self, lr: float = 1e-3) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        predictions = logits.argmax(dim=1)
        acc = (predictions == labels).float().mean()

        # sync_dist=True 让 DDP 下跨进程聚合日志指标
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="val")

    def test_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed, workers=True)

    if not torch.cuda.is_available():
        raise RuntimeError("DDP example requires GPUs, but CUDA is not available.")
    if torch.cuda.device_count() < args.devices:
        raise RuntimeError(
            f"Requested {args.devices} GPUs for DDP, but only found {torch.cuda.device_count()} visible GPU(s)."
        )

    datamodule = MNISTDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    model = LitMNISTClassifier(lr=args.lr)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename="best-ddp-{epoch:02d}-{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )

    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=args.devices,
        strategy="ddp",
        callbacks=[checkpoint_callback],
        default_root_dir=str(save_dir),
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")

    if checkpoint_callback.best_model_path:
        print(f"Best checkpoint: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
