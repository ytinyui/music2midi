import pytorch_lightning as pl
import torch_optimizer
import soundfile as sf
from torch.nn.utils.rnn import pad_sequence
from transformers import T5Config, T5ForConditionalGeneration
from dataset import PopDataModule
from midi_tokenizer import MidiTokenizer
from omegaconf import OmegaConf


DEFAULT_COMPOSERS = {"various composer": 2052}


class TransformerWrapper(pl.LightningModule):
    def __init__(self, config_path:str):
        super().__init__()
        self.config = OmegaConf.load(config_path)

        self.tokenizer = MidiTokenizer(self.config.tokenizer)
        self.t5config = T5Config.from_pretrained("t5-small")

        for k, v in self.config.t5.items():
            self.t5config.__setattr__(k, v)

        self.transformer = T5ForConditionalGeneration(self.t5config)
        
    def configure_optimizers(self):
        return torch_optimizer.Adafactor(self.parameters(), lr=self.config.training.lr)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.transformer(inputs_embeds=x, labels=y)
        loss = y_pred.loss
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.transformer(inputs_embeds=x, labels=y)
        loss = y_pred.loss
        self.log("valid_loss", loss, prog_bar=True)
        return loss
    
if __name__ == "__main__":
    pop_dataset = PopDataModule("../pop2piano-dataset")
    model = TransformerWrapper("config.yaml")
    
    trainer = pl.Trainer(max_epochs=100)
    trainer.fit(model, pop_dataset)