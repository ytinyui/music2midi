import argparse
from datetime import datetime

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger

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

    dataset = Music2MidiDataModule(args.data_dir, args.config)
    model = Music2Midi(args.config)
    model.train()

    now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config_dict = OmegaConf.to_container(model.config)
    logger = WandbLogger(name=now_time, project="music2midi", config=config_dict)
    trainer = pl.Trainer(logger=logger, **model.config.trainer)
    logger.watch(model)

    trainer.fit(model, dataset)
