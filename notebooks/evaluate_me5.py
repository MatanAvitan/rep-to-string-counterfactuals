#!/usr/bin/env python
# coding: utf-8


import dill
import numpy as np
import os
import ot
import pandas as pd
import pickle
import torch
import torch.nn.functional as F
import tqdm
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
from typing import List

import vec2text
from vec2text.models.model_utils import device

BS = 4
N_CORRECTIONS = 50


def average_pool(last_hidden_states,
                 attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def get_e5_embeddings(text_list,
                      encoder: PreTrainedModel,
                      tokenizer: PreTrainedTokenizer) -> [torch.Tensor, List]:
    # text_list = ['query: ' + i for i in text_list]
    text_list = [i for i in text_list]
    samples_len = [len(s) for s in tokenizer(text_list)['input_ids']]

    batch_dict = tokenizer(text_list, max_length=MAX_SEQUENCE_LENGTH, truncation=True, padding='max_length',
                           return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = encoder(**batch_dict)
        embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.detach().cpu(), samples_len


encoder = AutoModel.from_pretrained("intfloat/multilingual-e5-base").to(device)
tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")
corrector = vec2text.load_corrector("yiyic/t5_me5_base_mtg_en_5m_64")
sents = [
    '']
embs, _ = get_e5_embeddings(sents, encoder, tokenizer)

transformed = vec2text.invert_embeddings(
    torch.tensor(embs).to(device).float(),
    corrector=corrector,
    num_steps=N_CORRECTIONS,
    sequence_beam_width=BS,
)
print(sents)
print(transformed)
