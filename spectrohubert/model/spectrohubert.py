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
        self.final_proj = nn.Linear(cfg.encoder_embed_dim, self.num_classes)
        
        self.extractor_type = cfg.extractor_type
        self.feature_extractor = None
        spectrogram = transforms.Spectrogram(
            n_fft=512,
            power=None, # returns complex spectrogram - raw spectrogram complex-valued
        )
        self.spectrogram = spectrogram.to(device='cuda')

        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )

    
          
    @classmethod
    def build_model(cls, cfg: SpectroHubertConfig, task: HubertPretrainingTask):
        """Build a new model instance."""
        print('buildmodelinspectrohubert', cfg)
        print('buildmodelinspectrohubert', task.cfg)

        model = SpectroHubertModel(cfg, task.cfg, task.dictionaries)
        return model


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

        # basically forward_features function        
        if self.extractor_type == "complex":
            # print('slkdjfksldfjsd', source.shape) 5, 236160
            spec = self.spectrogram(source).to(device='cuda')
            # print('skdfhisdf', spec.shape) 5, 257, 923
            spec = torch.narrow(spec, 1, 1, 256)
            # print('slklskldsd', spec.shape) 5, 256, 923
            magnitude = torch.abs(spec)**2
            angle = torch.angle(spec)
            features = torch.cat((magnitude, angle),dim=1).to(device='cuda')
        # print('lkwjfeklwef', features.shape) 5 512 923
        features = features.transpose(1, 2)
        # print('wljwjfjwejfwje', features.shape) 5 923 512
        # print('didnt even get to lanroem')
        features = self.layer_norm(features)

        unmasked_features = features.clone()
        # print('padme masks')
        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)  
        # print('extract the projs')
        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)  

        features = self.dropout_input(features)   
        # print('droputhe feats')
        unmasked_features = self.dropout_features(unmasked_features)

        if mask:
            x, mask_indices = self.apply_mask(features, padding_mask, target_list) 
        else:
            x = features
            mask_indices = None 

        # print('maskup')

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        x, layer_results = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1,
        )  
        # print('sad no encoder')
        # if self.ils and self.fine_tuning==False:
        #     ils_results = []
        #     for layer in self.predict_layers:
        #         if layer < len(layer_results):
        #             ils_results.append(layer_results[layer][0].transpose(0, 1))
        #         else:
        #             ils_results.append(layer_results[-1][0].transpose(0, 1))

        if features_only:  
            return {"x": x, "padding_mask": padding_mask, "features": features}

        # if self.ils:
        #     logit_m_list = []
        #     logit_u_list = []
        #     targ_m_list_all = []
        #     targ_u_list_all = []

        #     masked_indices = torch.logical_and(~padding_mask, mask_indices)
        #     nomask_indices = torch.logical_and(~padding_mask, ~mask_indices)

        #     targ_m_list=target_list[0][masked_indices]
        #     targ_m_list=targ_m_list.long()
            
        #     targ_u_list=target_list[0][nomask_indices]
        #     targ_u_list = targ_u_list.long()
            
        #     for idx, layer_x in enumerate(ils_results):
        #         if not self.skip_masked:
        #             proj_x_m = self.final_proj[idx](layer_x[masked_indices])
        #             proj_x_m /= self.logit_temp
        #             logit_m_list.append(proj_x_m )
        #         else:
        #             logit_m_list += [None for _ in target_list]

        #         if not self.skip_nomask:
        #             proj_x_u = self.final_proj[idx](layer_x[nomask_indices])
        #             proj_x_u /= self.logit_temp
        #             logit_u_list.append(proj_x_u )
        #         else:
        #             logit_u_list += [None for _ in target_list]
                    
        #         targ_m_list_all.append(targ_m_list)
        #         targ_u_list_all.append(targ_u_list)
                

        # else: 
        if not self.skip_masked:
            masked_indices = torch.logical_and(~padding_mask, mask_indices)
            proj_x_m = self.final_proj(x[masked_indices])
            proj_x_m /= self.logit_temp
            logit_m_list = [proj_x_m for _ in range(len(target_list))]
        else:
            logit_m_list = [None for _ in target_list]

        if not self.skip_nomask:
            nomask_indices = torch.logical_and(~padding_mask, ~mask_indices)
            proj_x_u = self.final_proj(x[nomask_indices])
            proj_x_u /= self.logit_temp
            logit_u_list = [proj_x_u for _ in range(len(target_list))]
        else:
            logit_u_list = [None for _ in target_list]

        # targ_m_list=target_list[0][masked_indices]
        # targ_m_list=targ_m_list.long()
        # targ_m_list_all = [targ_m_list for _ in range(len(target_list))]

        # targ_u_list=target_list[0][nomask_indices]
        # targ_u_list = targ_u_list.long()
        # targ_u_list_all = [targ_u_list for _ in range(len(target_list))]
        
        print('whatlogitlooklike', logit_m_list)
        result = {
            "logit_m_list": logit_m_list,
            "logit_u_list": logit_u_list,
            "padding_mask": padding_mask, 
        }
        # print('shapepfpfppf', features.shape)

        return result
