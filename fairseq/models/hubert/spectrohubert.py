# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
import math
import torch
import torch.nn as nn
import torchaudio.transforms as transforms
import torch.nn.functional as F
from fairseq.modules import GradMultiply, LayerNorm
from fairseq.models.wav2vec.wav2vec2 import (
    EXTRACTOR_MODE_CHOICES,
    MASKING_DISTRIBUTION_CHOICES,
    LAYER_TYPE_CHOICES,
    ConvFeatureExtractionModel,
    TransformerEncoder,
)
import torchaudio.functional as func
import numpy as np
from fairseq.data.dictionary import Dictionary

from fairseq.models import register_model
from fairseq.models.speech_to_text.s2t_transformer import (
    S2TTransformerEncoder,
    S2TTransformerModel,
    Conv1dSubsampler,
    base_architecture as transformer_base_architecture,
)
from fairseq.data.audio.feature_transforms.specaugment import SpecAugmentTransform
# from ..task.spectrohubert_pretraining import *
from fairseq.tasks.hubert_pretraining import HubertPretrainingConfig
from fairseq.models.hubert.hubert import HubertModel,HubertConfig
from fairseq.tasks.hubert_pretraining import (
    HubertPretrainingConfig,
    HubertPretrainingTask
)
from fairseq.modules import (
    Fp32GroupNorm,
    Fp32LayerNorm,
    TransposeLast
)

logger = logging.getLogger(__name__)

SPEC_EXTRACTOR_MODE_CHOICES = ChoiceEnum(["default", "complex", "linear", "logmel"])
PROJECT_FEATURES_CHOICES = ChoiceEnum(["none", "same", "new"])
ACTIVATION_CHOICES = ChoiceEnum(["relu", "gelu"])

@dataclass
class SpectroHubertConfig(HubertConfig):  
    extractor_type: SPEC_EXTRACTOR_MODE_CHOICES = field(    
        default="default",  
        metadata={  
            "help": "mode for feature extractor. default has a single group "   
            "norm with d groups in the first conv block. complex_linear uses complex linear spectrogram,"   
            "linear_power uses the "    
        }
    )

@register_model("spectrohubert", dataclass=SpectroHubertConfig)
class SpectroHubertModel(HubertModel):
    def __init__(
        self,
        cfg: SpectroHubertConfig,
        task_cfg: HubertPretrainingConfig,
        dictionaries: List[Dictionary],
    ) -> None:
        super().__init__(cfg, task_cfg, dictionaries)
        logger.info(f"SpectroHubertModelConfig: {cfg}") 
         
        # self.post_extract_proj = (  
        #     nn.Linear(self.embed, cfg.encoder_embed_dim)
        #     if self.embed != cfg.encoder_embed_dim
        #     else None
        # )
        # should be 512x768
        self.num_classes = len(dictionaries[0]) 
        final_dim = cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim
        if self.untie_final_proj:
            self.final_proj = nn.Linear(
                cfg.encoder_embed_dim, final_dim * len(dictionaries)
            )
        else:
            self.final_proj = nn.Linear(cfg.encoder_embed_dim, final_dim)
        self.encoder = TransformerEncoder(cfg)
        # for param in self.encoder.parameters():
        #     param.requires_grad = False
        self.constrain = nn.Linear(514, 512, device='cuda')
        self.extractor_type = cfg.extractor_type
        feature_enc_layers = eval(cfg.conv_feature_layers)  # noqa
        self.embed = feature_enc_layers[-1][0]
        self.feature_extractor = None
        # self.feature_extractor = ConvFeatureExtractionModel(
        #     conv_layers=feature_enc_layers,
        #     dropout=0.0,
        #     mode=cfg.extractor_mode,
        #     conv_bias=cfg.conv_bias,
        # )
        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )

    
          
    @classmethod
    def build_model(cls, cfg: SpectroHubertConfig, task: HubertPretrainingTask):
        """Build a new model instance."""

        model = SpectroHubertModel(cfg, task.cfg, task.dictionaries)
        return model

    def spectrogram_extractor(self, source: torch.Tensor) -> torch.Tensor:
        n_fft_size = 512
        win_len = n_fft_size
        hop_len = 320
        window_func = torch.hamming_window(dtype=torch.float32, window_length=n_fft_size, periodic=True, device='cuda')
        spec = func.spectrogram(
            waveform=source,
            pad=1,
            window=window_func,
            normalized=False,
            hop_length=hop_len,
            n_fft=n_fft_size,
            power=None, # returns complex spectrogram - raw spectrogram complex-valued
            win_length=win_len
            
        )
        # spectrogram = transforms.Spectrogram(power=None).to(device='cuda')
        # spec = spectrogram(source)
        magnitude = torch.abs(spec)**2
        angle = torch.angle(spec)
        features = torch.cat((magnitude, angle),dim=1).to(device='cuda')
        # B T 2 * F, put a linear layer to get it to 512.
        # then re.view the output (B, T, D)
        B, F, T = features.shape
        # print('B*T', B*T, 'F', F)
        features = features.transpose(1, 2)
        features = features.reshape(B*T, F)
        # print('before linear layer', features.dtype)
        features = self.constrain(features)
        features = features.reshape(B, T, 512)
        features = features.transpose(1, 2)
        # print('featureshape', features)
        return features


    def forward(
        self,
        source: torch.Tensor,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """output layer is 1-based"""
        # check that the feature extraction is the problem
        if self.feature_grad_mult > 0:
            features = self.spectrogram_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.spectrogram_extractor(source)
        
        if target_list is not None:
            features, target_list = self.forward_targets(features, target_list)
        features_pen = features.float().pow(2).mean()
        features = features.transpose(1, 2)
        # print('after transposing', features.shape)
        features = self.layer_norm(features)

        unmasked_features = features.clone()
        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        # print('post extraction proj', features.shape)
        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        if mask:
            x, mask_indices = self.apply_mask(features, padding_mask, target_list)
        else:
            x = features
            mask_indices = None

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        x, _ = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1,
        )
        # print('and howbout after masking', x.shape)

        if features_only:
            return {"x": x, "padding_mask": padding_mask, "features": features}

        def compute_pred(proj_x, target, label_embs):
            # compute logits for the i-th label set
            y = torch.index_select(label_embs, 0, target.long())
            negs = label_embs.unsqueeze(1).expand(-1, proj_x.size(0), -1)
            if self.target_glu:
                y = self.target_glu(y)
                negs = self.target_glu(negs)
            # proj_x: (S, D)
            # y: (S, D)
            # negs: (Neg, S, D)
            return self.compute_nce(proj_x, y, negs)

        label_embs_list = self.label_embs_concat.split(self.num_classes, 0)
        if not self.skip_masked:
            masked_indices = torch.logical_and(~padding_mask, mask_indices)
            proj_x_m = self.final_proj(x[masked_indices])
            if self.untie_final_proj:
                proj_x_m_list = proj_x_m.chunk(len(target_list), dim=-1)
            else:
                proj_x_m_list = [proj_x_m for _ in range(len(target_list))]
            
            # print('targets', target_list[0].shape) # 6, 703
            # print('projected?', proj_x_m_list[0].shape) # 2160, 504
            logit_m_list = [
                compute_pred(proj_x_m, t[masked_indices], label_embs_list[i])
                for i, (proj_x_m, t) in enumerate(zip(proj_x_m_list, target_list))
            ]
        else:
            logit_m_list = [None for _ in target_list]

        if not self.skip_nomask:
            nomask_indices = torch.logical_and(~padding_mask, ~mask_indices)
            proj_x_u = self.final_proj(x[nomask_indices])
            if self.untie_final_proj:
                proj_x_u_list = proj_x_u.chunk(len(target_list), dim=-1)
            else:
                proj_x_u_list = [proj_x_u for _ in range(len(target_list))]
            logit_u_list = [
                compute_pred(proj_x_u, t[nomask_indices], label_embs_list[i])
                for i, (proj_x_u, t) in enumerate(zip(proj_x_u_list, target_list))
            ]
        else:
            logit_u_list = [None for _ in target_list]
        result = {
            "logit_m_list": logit_m_list,
            "logit_u_list": logit_u_list,
            "padding_mask": padding_mask,
            "features_pen": features_pen,
        }
        return result

