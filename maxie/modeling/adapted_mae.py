#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn            as nn
import torch.nn.functional as F

from transformers import ViTMAEForPreTraining

from dataclasses import dataclass, field, asdict, is_dataclass
from typing import List, Dict, Optional

# ----------------------------------------------------------------------- #
#  Helper
# ----------------------------------------------------------------------- #
def update_num_channels(model, new_channels=1):
    for child_name, child in model.named_children():
        if hasattr(child, 'num_channels'):
            print(f"Updating {child_name} num_channels from {child.num_channels} to {new_channels}")
            child.num_channels = new_channels

        # Recursively update submodules
        update_num_channels(child, new_channels)


# ----------------------------------------------------------------------- #
#  Model
# ----------------------------------------------------------------------- #
@dataclass
class AdaptedViTMAEForPreTrainingConfig:
    model_name  : str   = "facebook/vit-mae-base"
    mask_ratio  : float = 0.75
    from_scratch: bool  = False

class AdaptedViTMAEForPreTraining(nn.Module):
    NUM_RGB_CHANNEL    = 3
    DECODE_IN_FEATURES = 512

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model  = AdaptedViTMAEForPreTraining.adapt_pretrained_model(
            self.config.model_name,
            self.config.mask_ratio,
        )

        # Conditionally init from scratch
        if self.config.from_scratch:
            self.model.apply(self._init_weights)

    @staticmethod
    def adapt_pretrained_model(model_name, mask_ratio):
        # -- Which pretrained model is in use
        vit_mae_model_config_dict = {
            "facebook/vit-mae-base"  : { "emb_size" : 768,  "win_size" : 16 },
            "facebook/vit-mae-large" : { "emb_size" : 1024, "win_size" : 16 },
            "facebook/vit-mae-huge"  : { "emb_size" : 1280, "win_size" : 14 },
        }
        vit_mae_model_config = vit_mae_model_config_dict[model_name]
        emb_size = vit_mae_model_config['emb_size']
        win_size = vit_mae_model_config['win_size']

        # -- Initialize the pretrained model
        model = ViTMAEForPreTraining.from_pretrained(model_name)

        # -- Adapt
        # --- Update channel number
        update_num_channels(model)
        model.config.num_channels = 1

        # --- Adapt to one channel input
        avg_weight_patch_embd = model.vit.embeddings.patch_embeddings.projection.weight.data.mean(dim = 1, keepdim = True)
        model.vit.embeddings.patch_embeddings.projection = nn.Conv2d(
            in_channels  = 1,
            out_channels = emb_size,
            kernel_size  =(win_size, win_size),
            stride       =(win_size, win_size),
        )
        model.vit.embeddings.patch_embeddings.projection.weight.data = avg_weight_patch_embd

        # --- Adapt to correct output
        avg_weight_decoder_pred = model.decoder.decoder_pred.weight.data.view(
            AdaptedViTMAEForPreTraining.NUM_RGB_CHANNEL, win_size, win_size, -1
        ).mean(dim = 0).view(win_size * win_size, -1)
        model.decoder.decoder_pred = nn.Linear(
            in_features  = AdaptedViTMAEForPreTraining.DECODE_IN_FEATURES,
            out_features = win_size*win_size,
            bias         = True)
        model.decoder.decoder_pred.weight.data = avg_weight_decoder_pred

        # --- Adapt the mask ratio
        model.config.mask_ratio = mask_ratio

        return model

    def forward(self, x):
        return self.model(x)

    def _init_weights(self, module):
        """ Refer to https://github.com/facebookresearch/mae/blob/main/models_mae.py
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
