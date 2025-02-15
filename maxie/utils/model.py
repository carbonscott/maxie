import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

# -- Monkey patch the _init_weights
# Account for the (pre)activation spread due to the accumulation of residual paths
# --- Encoder
def _init_weights_in_encoder(self, module):
    """Initialize the weights"""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        # Normalize the init std by the number of residual paths
        std  = self.config.initializer_range
        std *= (2 * self.config.num_hidden_layers)**-0.5  # 1/sqrt(num_residual_layers), cf: GPT-2 paper

        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=std)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
## ViTMAEModel._init_weights = _init_weights_in_encoder

# --- Decoder
# HF's MAE doesn't have a _init_weights for decoder, but it initializes the
# decoder in the end through the _init_weights from ViTMAEPreTrainedModel
def _init_weights_in_decoder(self, module):
    """Initialize the weights"""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        # Normalize the init std by the number of residual paths
        std  = self.config.initializer_range
        std *= (2 * self.config.decoder_num_hidden_layers)**-0.5  # 1/sqrt(num_residual_layers), cf: GPT-2 paper

        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=std)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
## ViTMAEPreTrainedModel._init_weights = _init_weights_in_decoder

def logging_model_init(dist_config, model):
    if dist_config.rank == 0:
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                mean = module.weight.data.mean()
                std  = module.weight.data.std()
                logger.info(f"logevent='INIT' | rank={dist_config.rank} | module={name} | mean={mean:.6f} | std={std:.6f}")

# !! Make all params trainable, a workaround for pytorch 2.0.1
def unfreeze_model(model):
    for name, param in model.named_parameters():
        if not param.requires_grad:
            param.requires_grad = True

