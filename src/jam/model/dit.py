# Copyright (c) 2025 ASLP-LAB
#               2025 Ziqian Ning   (ningziqian@mail.nwpu.edu.cn)
#               2025 Huakang Chen  (huakang@mail.nwpu.edu.cn)
#               2025 Yuepeng Jiang (Jiangyp@mail.nwpu.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" This implementation is adapted from github repo:
    https://github.com/SWivid/F5-TTS.
"""

from __future__ import annotations

import torch
from torch import nn
import torch
from torch.utils.checkpoint import checkpoint

from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding
from transformers.models.llama import LlamaConfig

from .modules import (
    TimestepEmbedding,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    AdaLayerNormZero_Final,
    precompute_freqs_cis,
    get_pos_embed_indices,
    _prepare_decoder_attention_mask,
)

# Text embedding
class TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds, text_dim, max_pos, conv_layers=0, conv_mult=2):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # use 0 as filler token

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = max_pos  # ~44s of 24khz audio
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, 4 * self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
            )
            self.downsample_blocks = nn.Sequential(
                *[torch.nn.Conv1d(text_dim, text_dim, kernel_size=2 * 2, stride=2, padding=1) for _ in range(2)]
            )
                
        else:
            self.extra_modeling = False

    def forward(self, text: int["b nt"], seq_len, drop_text=False):  # noqa: F722
        batch, text_len = text.shape[0], text.shape[1]

        if drop_text:  # cfg for text
            text = torch.zeros_like(text)

        text = self.text_embed(text)  # b n -> b n d

        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            batch_start = torch.zeros((batch,), dtype=torch.long)
            pos_idx = get_pos_embed_indices(batch_start, 4*seq_len, max_pos=4*self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx]
            text = text + text_pos_embed

            # convnextv2 blocks
            text = self.text_blocks(text)
            text = self.downsample_blocks(text.transpose(1, 2)).transpose(1, 2)

        return text


# noised input audio and context mixing embedding
class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim, cond_dim):
        super().__init__()
        # Updated to handle mel, cond, text, style, and time+duration embeddings
        self.proj = nn.Linear(mel_dim * 2 + text_dim + cond_dim * 2, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x: float["b n d"], cond: float["b n d"], text_embed: float["b n d"], style_emb, time_emb, drop_audio_cond=False):  # noqa: F722
        if drop_audio_cond:  # cfg for cond audio
            cond = torch.zeros_like(cond)
        style_emb = style_emb.unsqueeze(1).repeat(1, x.shape[1], 1)
        time_emb = time_emb.unsqueeze(1).repeat(1, x.shape[1], 1)
        x = self.proj(torch.cat((x, cond, text_embed, style_emb, time_emb), dim=-1))
        x = self.conv_pos_embed(x) + x
        return x


# Transformer backbone using Llama blocks
class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        text_num_embeds=256,
        text_dim=None,
        conv_layers=0,
        long_skip_connection=False,
        max_frames=2048,
        grad_ckpt=False,
        use_implicit_duration=False
    ):
        super().__init__()
        
        self.max_frames = max_frames

        cond_dim = 512
        self.time_embed = TimestepEmbedding(cond_dim)
        self.start_time_embed = TimestepEmbedding(cond_dim)
        self.duration_abs_embed = TimestepEmbedding(cond_dim)
        self.duration_rel_embed = TimestepEmbedding(cond_dim)
        if text_dim is None:
            text_dim = mel_dim
        self.text_embed = TextEmbedding(text_num_embeds, text_dim, conv_layers=conv_layers, max_pos=self.max_frames)
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim, cond_dim=cond_dim)
        self.grad_ckpt = grad_ckpt
        self.dim = dim
        self.depth = depth

        self.use_implicit_duration = use_implicit_duration
        if self.use_implicit_duration:
            self.duration_pad_bias = nn.Parameter(torch.zeros(text_dim))
            self.register_buffer("arange_T", torch.arange(self.max_frames), persistent=False)

        llama_config = LlamaConfig(hidden_size=dim, intermediate_size=dim * ff_mult, hidden_act='silu', max_position_embeddings=self.max_frames)
        llama_config._attn_implementation = 'sdpa'
        self.transformer_blocks = nn.ModuleList(
            [LlamaDecoderLayer(llama_config, layer_idx=i) for i in range(depth)]
        )
        self.rotary_emb = LlamaRotaryEmbedding(config=llama_config)
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.text_fusion_linears = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(cond_dim, dim),
                    nn.SiLU()
                ) for i in range(depth // 2)
            ]
        )
        for layer in self.text_fusion_linears:
            for p in layer.parameters():
                p.detach().zero_()

        self.norm_out = AdaLayerNormZero_Final(dim, cond_dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)

    def forward(
        self,
        x: float["b n d"],  # nosied input audio  # noqa: F722
        cond: float["b n d"],  # masked cond audio  # noqa: F722
        text: int["b nt"],  # text  # noqa: F722
        time: float["b"] | float[""],  # time step  # noqa: F821 F722
        drop_audio_cond,  # cfg for cond audio
        drop_text,  # cfg for text
        drop_prompt=False,
        style_prompt=None, # [b d t]
        start_time=None,
        duration_abs=None,
        duration_rel=None,
    ):

        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        # t: conditioning time, c: context (text + masked cond audio), x: noised input audio
        t = self.time_embed(time)
        time_embed = self.start_time_embed(start_time) + self.duration_abs_embed(duration_abs) + self.duration_rel_embed(duration_rel)

        c = t + time_embed
        text_embed = self.text_embed(text, seq_len, drop_text=drop_text)
        if self.use_implicit_duration:
            T_real = (self.max_frames * duration_rel).long() # (B,)
            token_mask = self.arange_T[None, :] >= T_real[:, None] # (B, T)
            # print(T_real, token_mask.shape, token_mask[:, -100:])
            text_embed = text_embed + self.duration_pad_bias[None, None, :] * token_mask.unsqueeze(-1)

        if drop_prompt:
            style_prompt = torch.zeros_like(style_prompt)
        
        style_embed = style_prompt # [b, 512]

        x = self.input_embed(x, cond, text_embed, style_embed, c, drop_audio_cond=drop_audio_cond)

        if self.long_skip_connection is not None:
            residual = x

        pos_ids = torch.arange(x.shape[1], device=x.device)
        pos_ids = pos_ids.unsqueeze(0).repeat(x.shape[0], 1)
        rotary_embed = self.rotary_emb(x, pos_ids)
        
        attention_mask = torch.ones(
            (batch, seq_len),
            dtype=torch.bool,
            device=x.device,
        )
        attention_mask = _prepare_decoder_attention_mask(
            attention_mask,
            (batch, seq_len),
            x,
        )

        for i, block in enumerate(self.transformer_blocks):
            if self.grad_ckpt and self.training:
                # print(f"Using gradient checkpointing for block {i}")
                x, *_ = checkpoint(block, x, attention_mask=attention_mask, position_embeddings=rotary_embed, use_reentrant=False)
            else:
                x, *_ = block(x, attention_mask=attention_mask, position_embeddings=rotary_embed)
            if i < self.depth // 2:
                x = x + self.text_fusion_linears[i](text_embed)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, c)
        output = self.proj_out(x)

        return output
