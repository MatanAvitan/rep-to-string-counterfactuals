import os

from transformers.modeling_outputs import SequenceClassifierOutput

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaForSequenceClassification, RobertaConfig
import wandb

KL_FACTOR = 0.75


class RobertaWithCounterfactualKL(RobertaForSequenceClassification):
    def __init__(self, config):
        wandb.log({'KL_FACTOR': KL_FACTOR})
        wandb.log({'NO counterfactual CEL': True})
        super().__init__(config)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            counterfactual_input_ids=None,
            counterfactual_attention_mask=None,
            labels=None,
            **kwargs
    ):
        hard_text_output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return hard_text_output