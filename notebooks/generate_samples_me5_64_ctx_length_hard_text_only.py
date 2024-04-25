#!/usr/bin/env python
# coding: utf-8


import os
import pickle
from typing import List

import dill
import numpy as np
import ot
import pandas as pd
import torch
import tqdm

import vec2text
import torch.nn.functional as F
from vec2text.models.model_utils import device
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel


def log_print(str_to_print):
    eval_str_to_print = eval(str_to_print)
    if type(eval_str_to_print) == list:
        eval_str_to_print = '\n'.join(eval_str_to_print)
    print(f"{str_to_print}: {eval_str_to_print}")


IS_FIRST = False
IS_DOWNLOAD = False
# BASE_MODEL = 'gtr-base'
MAX_SEQUENCE_LENGTH = 64  # Don't it should be 32 for the gtr model?
BATCH_SIZE = 16
N_SAMPLES_TO_INVERT = 10_000
TRANSFORMED_COLUMN = 'hard_text'
BS = 4
N_CORRECTIONS = 50

if IS_DOWNLOAD:
    os.system(
        'wget -r --no-clobber --no-parent -R "index.html*" -nH --cut-dirs=4 -P bios_data https://nlp.biu.ac.il/~ravfogs/rlace-cr/bios/bios_data/')

with open("/home/nlp/matan_avitan/git/vec2text_inter/bios_data/bios_train.pickle", "rb") as f:
    bios_train = pickle.load(f)

with open("/home/nlp/matan_avitan/git/vec2text_inter/bios_data/bios_dev.pickle", "rb") as f:
    bios_dev = pickle.load(f)

with open("/home/nlp/matan_avitan/git/vec2text_inter/bios_data/bios_test.pickle", "rb") as f:
    bios_test = pickle.load(f)

"""
A look into the dataset
"""
bios_train_df = pd.DataFrame(bios_train)
bios_dev_df = pd.DataFrame(bios_dev)
bios_test_df = pd.DataFrame(bios_test)
log_print('bios_train_df.shape')
log_print('bios_dev_df.shape')
log_print('bios_test_df.shape')

"""
Preprocess for gender counterfactuals creation
"""
z_train = bios_train_df['g'].replace('f', 0).replace('m', 1).astype(int).to_numpy()
z_dev = bios_dev_df['g'].replace('f', 0).replace('m', 1).astype(int).to_numpy()

y_train = bios_train_df['p'].to_numpy()
y_dev = bios_dev_df['p'].to_numpy()

sequences_length = np.zeros((bios_train_df.shape[0]))


def average_pool(last_hidden_states,
                 attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def get_e5_embeddings(text_list,
                      encoder: PreTrainedModel,
                      tokenizer: PreTrainedTokenizer) -> [torch.Tensor, List]:
    text_list = ['query: ' + i for i in text_list]
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
if IS_FIRST:
    encodings = []
    for i in tqdm.tqdm(range(0, len(bios_train_df), BATCH_SIZE)):
        sents_batch = bios_train_df.loc[i:i + BATCH_SIZE - 1, TRANSFORMED_COLUMN].tolist()
        embeddings, samples_len = get_e5_embeddings(sents_batch, encoder, tokenizer)
        encodings.append(embeddings)
        sequences_length[i: i + BATCH_SIZE] = samples_len
    with open('generate_samples_me5_64_ctx_length_hard_text_only_encodings', 'wb') as f:
        dill.dump(encodings, f)
    with open('sequences_length', 'wb') as f:
        dill.dump(sequences_length, f)
else:
    with open('generate_samples_me5_64_ctx_length_hard_text_only_encodings', 'rb') as f:
        encodings = dill.load(f)
    with open('sequences_length', 'rb') as f:
        sequences_length = dill.load(f)
print('after encoding')
x = np.concatenate(encodings, axis=0)
x = x[:len(bios_train_df)]  # Nan are added to the last batch, let's remove them

female_mask = (z_train == 0)
x_source = x[female_mask]  # Female
x_target = x[~female_mask]  # Male

"""
Linear transformation of the female repr dist to the male repr dist
"""
ot_linear = ot.da.LinearTransport(reg=1e-7)
ot_linear.fit(Xs=x_source, Xt=x_target)
train_x_transformed = x.copy()
train_x_transformed[female_mask] = ot_linear.transform(Xs=x_source)  # Optimal transport intervention

"""
Generate text from repr
"""
print('before run correction on females')
female_mask_indices = np.arange(len(bios_train_df))[(female_mask) & (sequences_length <= MAX_SEQUENCE_LENGTH)]
for lower_bound in tqdm.tqdm(range(0, len(female_mask_indices), BATCH_SIZE)):
    batch_indices = female_mask_indices[lower_bound: lower_bound + BATCH_SIZE]
    bios_train_df.loc[batch_indices, 'transformed_hard_text'] = \
        vec2text.invert_embeddings(
            torch.tensor(train_x_transformed[batch_indices]).to(device).float(),
            corrector=corrector,
            num_steps=N_CORRECTIONS,
            sequence_beam_width=BS,
        )
    print(bios_train_df.loc[batch_indices[0], 'hard_text'])
    print(bios_train_df.loc[batch_indices[0], 'transformed_hard_text'])

"""
Linear transformation of the male repr dist to the female repr dist
"""
print('before run correction on males')
male_mask = ~female_mask
ot_linear = ot.da.LinearTransport(reg=1e-7)
ot_linear.fit(Xs=x_target, Xt=x_source)
train_x_transformed = x.copy()
train_x_transformed[male_mask] = ot_linear.transform(Xs=x_target)  # Optimal transport intervention

"""
Generate text from repr
"""
male_mask_indices = np.arange(len(bios_train_df))[(male_mask) & ((sequences_length <= MAX_SEQUENCE_LENGTH))]
for lower_bound in tqdm.tqdm(range(0, len(male_mask_indices), BATCH_SIZE)):
    batch_indices = male_mask_indices[lower_bound: lower_bound + BATCH_SIZE]
    bios_train_df.loc[batch_indices, 'transformed_hard_text'] = \
        vec2text.invert_embeddings(
            torch.tensor(train_x_transformed[batch_indices]).to(device).float(),
            corrector=corrector,
            num_steps=N_CORRECTIONS,
            sequence_beam_width=BS,
        )

bios_train_df.to_csv(
    f'/home/nlp/matan_avitan/git/vec2text_inter/bios_data/me5_bs_{BS}_n_corrections_{N_CORRECTIONS}.csv',
    index=False, escapechar='\\')
# bios_train_df.to_csv('bios_data/bios_train_df.csv', index=False)
