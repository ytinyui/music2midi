import argparse

import pytorch_lightning as pl
import torch

from src.dataset import Music2MidiDataModule
from src.model import Music2Midi

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True

pl.seed_everything(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    my_dataset = Music2MidiDataModule(args.data_dir, args.config)
    model = Music2Midi(args.config)
    model.train()
    trainer = pl.Trainer(**model.model.config.trainer)
    trainer.fit(model, my_dataset)
