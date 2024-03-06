import argparse

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping

from src.dataset import MyDataModule
from src.model import TransformerWrapper

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True

pl.seed_everything(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    my_dataset = MyDataModule(args.data_dir, args.config)
    model = TransformerWrapper(args.config)
    trainer = pl.Trainer(
        callbacks=[EarlyStopping(**model.config.early_stopping)], **model.config.trainer
    )
    trainer.fit(model, my_dataset)
