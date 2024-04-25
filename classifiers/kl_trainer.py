import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
from torch import nn
import torch.nn.functional as F
from transformers import Trainer

N_LABELS = 28
KL_FACTOR = 0.95


class KLTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        counterfactual_input_ids = inputs.get("counterfactual_input_ids")
        counterfactual_attention_mask = inputs.get("counterfactual_attention_mask")
        labels = inputs.get("labels")
        ce_loss_fct = nn.CrossEntropyLoss()

        hard_text_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        hard_text_logits = hard_text_outputs.get('logits')
        hard_text_cel = ce_loss_fct(hard_text_logits.view(-1, N_LABELS), labels.view(-1))
        if counterfactual_input_ids is None:
            return (hard_text_cel, hard_text_outputs) if return_outputs else hard_text_cel

        counterfactual_outputs = model(input_ids=counterfactual_input_ids, attention_mask=counterfactual_attention_mask)
        counterfactual_logits = counterfactual_outputs.get('logits')
        counterfactual_cel = ce_loss_fct(counterfactual_logits.view(-1, N_LABELS), labels.view(-1))

        kl_loss_fct = nn.KLDivLoss(reduction='batchmean')
        kl_loss_left = kl_loss_fct(F.log_softmax(hard_text_logits, dim=1), F.softmax(counterfactual_logits, dim=1))
        kl_loss_right = kl_loss_fct(F.log_softmax(counterfactual_logits, dim=1), F.softmax(hard_text_logits, dim=1))
        # loss = (1 - KL_FACTOR) * (hard_text_cel + counterfactual_cel) + KL_FACTOR * (kl_loss_left + kl_loss_right)
        loss = (1-KL_FACTOR) * hard_text_cel + KL_FACTOR * (kl_loss_left + kl_loss_right)
        return (loss, hard_text_outputs) if return_outputs else loss
