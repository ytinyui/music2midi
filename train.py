import argparse
from datetime import datetime

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger

from music2midi.dataset import Music2MIDIDataModule
from music2midi.model import Music2MIDI

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True

pl.seed_everything(0)

if __name__ == "__main__":
    now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("--name", type=str, default=now_time, help="name of the run")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--ckpt", type=str, default=None, help="checkpoint path to resume training"
    )
    parser.add_argument(
        "--run_id", type=str, default=None, help="wandb run id to resume training"
    )
    args = parser.parse_args()

    dataset = Music2MIDIDataModule(args.data_dir, args.config)
    model = Music2MIDI(args.config)
    model.train()

    config_dict = OmegaConf.to_container(model.config)
    logger = WandbLogger(
        name=args.name, project="music2midi", id=args.run_id, config=config_dict
    )
    logger.watch(model)
    trainer = pl.Trainer(logger=logger, **model.config.trainer)
    trainer.fit(model, dataset, ckpt_path=args.ckpt)
