from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    TrOCRConfig,
    TrOCRForCausalLM,
    VisionEncoderDecoderModel,
)
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.vit import ViTConfig, ViTModel

from .input import Conditioning, LogMelSpectrogram, ModelInputs
from .tokenizer import MidiTokenizer


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.temp = nn.Parameter(torch.ones(1))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        assert x1.shape == x2.shape
        sim = F.cosine_similarity(x1.unsqueeze(0), x2.unsqueeze(1), dim=-1).mean(dim=-1)
        return F.cross_entropy(sim / self.temp, torch.arange(x1.shape[0]).to(x1.device))


class VitModelConditioned(ViTModel):
    def __init__(
        self,
        config,
        cond_embeds,
        add_pooling_layer=True,
        use_mask_token=False,
    ):
        """
        Adds an embedding layer conditioning at output
        """
        super().__init__(config, add_pooling_layer, use_mask_token)

        self.conditioning = Conditioning(self.config.hidden_size, cond_embeds)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cond_index: Optional[list[torch.LongTensor]] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        embedding_output = self.embeddings(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        # add conditioning
        if cond_index is not None:
            sequence_output = self.conditioning(sequence_output, cond_index)
        sequence_output = self.layernorm(sequence_output)
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        if not return_dict:
            head_outputs = (
                (sequence_output, pooled_output)
                if pooled_output is not None
                else (sequence_output,)
            )
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class VisionTransformer(nn.Module):
    def __init__(self, config_path: str):
        super().__init__()
        self.config = OmegaConf.load(config_path)
        encoder_config = ViTConfig(**self.config.model.vit)
        decoder_config = TrOCRConfig(**self.config.model.trocr)

        encoder = VitModelConditioned(
            encoder_config, [len(v) for v in self.config.conditioning.values()]
        )
        decoder = TrOCRForCausalLM(decoder_config)
        self.transformer = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)
        self.transformer.config.pad_token_id = decoder.config.pad_token_id
        self.transformer.config.decoder_start_token_id = decoder.config.bos_token_id
        self.transformer.config.eos_token_id = decoder.config.eos_token_id

        self.tokenizer = MidiTokenizer(self.config)
        self.spectrogram = LogMelSpectrogram(
            sample_rate=self.config.dataset.sample_rate,
            n_mels=self.config.model.vit.image_size,
            **self.config.spectrogram,
        )
        self.contrastive_loss = ContrastiveLoss()

    def forward(self, inputs: ModelInputs, **kwargs) -> dict:
        labels = self.tokenizer(inputs.notes_batch)
        labels[labels == self.transformer.config.pad_token_id] = -100
        labels = labels.to(self.transformer.device)

        encoder_inputs = self.spectrogram(inputs.input_waveform).unsqueeze(1)

        outputs = self.transformer(
            encoder_inputs,
            labels=labels,
            genre_index=inputs.genre_id,
            difficulty_index=inputs.difficulty_id,
            **kwargs,
        )

        if inputs.notes_waveform is not None:
            notes_spectrogram = self.spectrogram(inputs.notes_waveform).unsqueeze(1)
            encoder_out2 = self.transformer.encoder(notes_spectrogram)
            ct_loss = self.contrastive_loss(
                outputs.encoder_last_hidden_state[:, 2:, :],
                encoder_out2.last_hidden_state,
            )
            outputs["ct_loss"] = ct_loss

        return outputs

    def generate(self, inputs: ModelInputs, **kwargs) -> torch.Tensor:
        encoder_inputs = self.spectrogram(inputs.input_waveform).unsqueeze(1)
        outputs = self.transformer.generate(
            inputs=encoder_inputs,
            genre_index=inputs.genre_id,
            difficulty_index=inputs.difficulty_id,
            **kwargs,
        )
        return outputs


class T5Transformer(nn.Module):
    def __init__(self, config_path):
        super().__init__()
        self.config = OmegaConf.load(config_path)

        self.t5config = T5Config(**self.config.model.t5)
        self.t5config.pad_token_id = 0
        self.t5config.decoder_start_token_id = 1
        self.t5config.eos_token_id = 2

        self.transformer = T5ForConditionalGeneration(self.t5config)
        self.tokenizer = MidiTokenizer(self.config)
        self.spectrogram = LogMelSpectrogram(
            sample_rate=self.config.dataset.sample_rate,
            n_mels=self.config.model.t5.d_model,
            **self.config.spectrogram,
        )
        self.conditioning = Conditioning(
            self.config.model.t5.d_model,
            [len(v) for v in self.config.conditioning.values()],
        )
        self.contrastive_loss = ContrastiveLoss()

    def forward(self, inputs: ModelInputs, **kwargs) -> dict:
        labels = self.tokenizer(inputs.notes_batch)
        labels[labels == self.transformer.config.pad_token_id] = -100
        labels = labels.to(self.transformer.device)

        encoder_inputs = self.spectrogram(inputs.input_waveform)
        encoder_inputs = self.conditioning(encoder_inputs, inputs.cond_index)
        outputs = self.transformer(
            inputs_embeds=encoder_inputs, labels=labels, **kwargs
        )

        if inputs.notes_waveform is not None:
            notes_spectrogram = self.spectrogram(inputs.notes_waveform)
            notes_spectrogram = self.conditioning(notes_spectrogram, inputs.cond_index)
            out2 = self.transformer.encoder(inputs_embeds=notes_spectrogram)
            ct_loss = self.contrastive_loss(
                outputs.encoder_last_hidden_state,
                out2.last_hidden_state,
            )
            outputs["ct_loss"] = ct_loss

        return outputs

    def generate(self, inputs: ModelInputs, **kwargs) -> torch.Tensor:
        encoder_inputs = self.spectrogram(inputs.input_waveform)
        encoder_inputs = self.conditioning(encoder_inputs, inputs.cond_index)
        outputs = self.transformer.generate(inputs_embeds=encoder_inputs, **kwargs)
        return outputs
