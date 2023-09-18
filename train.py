import torch
import pytorch_lightning as pl
from dataset import PopDataModule
from transformer_wrapper import TransformerWrapper
from pytorch_lightning import seed_everything
import argparse

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True

seed_everything(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../pop2piano-dataset")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    pop_dataset = PopDataModule(args.data_dir, args.config)
    model = TransformerWrapper(args.config)
    trainer = pl.Trainer(**model.config.trainer)
    trainer.fit(model, pop_dataset)
