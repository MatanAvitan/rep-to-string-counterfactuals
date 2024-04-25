#!/usr/bin/env python
# coding: utf-8


import os
import pickle
from typing import List

import numpy as np
import ot
import pandas as pd
import torch
import tqdm

import vec2text
from vec2text.models.model_utils import device
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel


def log_print(str_to_print):
    eval_str_to_print = eval(str_to_print)
    if type(eval_str_to_print) == list:
        eval_str_to_print = '\n'.join(eval_str_to_print)
    print(f"{str_to_print}: {eval_str_to_print}")


IS_FIRST = False
# BASE_MODEL = 'gtr-base'
MAX_SEQUENCE_LENGTH = 64  # Don't it should be 32 for the gtr model?
BATCH_SIZE = 16

# corrector = vec2text.load_corrector(BASE_MODEL)
# corrector.model.to(device)
# corrector.embedder.to(device)
if IS_FIRST:
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
log_print('bios_train_df.shape');
log_print('bios_dev_df.shape');
log_print('bios_test_df.shape')

"""
Preprocess for gender counterfactuals creation
"""
z_train = bios_train_df['g'].replace('f', 0).replace('m', 1).astype(int).to_numpy()
z_dev = bios_dev_df['g'].replace('f', 0).replace('m', 1).astype(int).to_numpy()

y_train = bios_train_df['p'].to_numpy()
y_dev = bios_dev_df['p'].to_numpy()

hard_text_sequences_length = np.zeros((bios_train_df.shape[0]))
text_sequences_length = np.zeros((bios_train_df.shape[0]))


def get_gtr_embeddings(text_list,
                       encoder: PreTrainedModel,
                       tokenizer: PreTrainedTokenizer) -> torch.Tensor:
    samples_len = [len(s) for s in tokenizer(text_list)['input_ids']]

    inputs = tokenizer(text_list,
                       return_tensors="pt",
                       max_length=MAX_SEQUENCE_LENGTH,
                       truncation=True,
                       padding="max_length", ).to("cuda")

    with torch.no_grad():
        model_output = encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        hidden_state = model_output.last_hidden_state
        embeddings = vec2text.models.model_utils.mean_pool(hidden_state,
                                                           inputs['attention_mask']).cpu().detach().numpy()

    return embeddings, samples_len


encoder = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").encoder.to("cuda")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
corrector = vec2text.load_corrector("gtr-base")

hard_text_encodings, text_encodings = [], []
for i in tqdm.tqdm(range(0, len(bios_train_df), BATCH_SIZE)):
    sents_batch = bios_train_df.loc[i:i + BATCH_SIZE - 1, 'hard_text'].tolist()
    hard_text_embeddings, hard_text_samples_len = get_gtr_embeddings(sents_batch, encoder, tokenizer)
    sents_batch = bios_train_df.loc[i:i + BATCH_SIZE - 1, 'text'].tolist()
    text_embeddings, text_samples_len = get_gtr_embeddings(sents_batch, encoder, tokenizer)
    hard_text_encodings.append(hard_text_embeddings)
    text_encodings.append(text_embeddings)
    hard_text_sequences_length[i: i + BATCH_SIZE] = hard_text_samples_len
    text_sequences_length[i: i + BATCH_SIZE] = text_samples_len

hard_text_x = np.concatenate(hard_text_encodings, axis=0)
text_x = np.concatenate(text_encodings, axis=0)
hard_text_x = hard_text_x[:len(bios_train_df)]  # Nan are added to the last batch, let's remove them
text_x = text_x[:len(bios_train_df)]  # Nan are added to the last batch, let's remove them

female_mask = (z_train == 0)
hard_text_x_source = hard_text_x[female_mask]  # Female
text_x_source = text_x[female_mask]  # Female

hard_text_x_target = hard_text_x[~female_mask]  # Male
text_x_target = text_x[~female_mask]  # Male

"""
Stack the text and the hard text
"""
x_source = np.stack([hard_text_x_source, text_x_source])
x_target = np.stack([hard_text_x_target, text_x_target])

"""
Linear transformation of the female repr dist to the male repr dist
"""
ot_linear = ot.da.LinearTransport(reg=1e-7)
ot_linear.fit(Xs=x_source, Xt=x_target)
train_x_source_transformed = x_source.copy()
train_x_source_transformed[female_mask] = ot_linear.transform(Xs=x_source)  # Optimal transport intervention

"""
Generate text from repr
"""
female_mask_indices = np.arange(len(bios_train_df))[(female_mask) & (hard_text_sequences_length <= MAX_SEQUENCE_LENGTH)]
for lower_bound in tqdm.tqdm(range(0, len(female_mask_indices), BATCH_SIZE)):
    batch_indices = female_mask_indices[lower_bound: lower_bound + BATCH_SIZE]
    bios_train_df.loc[batch_indices, 'transformed_hard_text'] = \
        vec2text.invert_embeddings(
            torch.tensor(train_x_transformed[batch_indices]).to(device).float(),
            corrector=corrector,
            num_steps=20,
            sequence_beam_width=4,
        )

"""
Linear transformation of the male repr dist to the female repr dist
"""
male_mask = ~female_mask
ot_linear = ot.da.LinearTransport(reg=1e-7)
ot_linear.fit(Xs=x_target, Xt=x_source)
train_x_transformed = x.copy()
train_x_transformed[male_mask] = ot_linear.transform(Xs=x_target)  # Optimal transport intervention

"""
Generate text from repr
"""
male_mask_indices = np.arange(len(bios_train_df))[(male_mask) & ((hard_text_sequences_length <= MAX_SEQUENCE_LENGTH))]
for lower_bound in tqdm.tqdm(range(0, len(male_mask_indices), BATCH_SIZE)):
    batch_indices = male_mask_indices[lower_bound: lower_bound + BATCH_SIZE]
    bios_train_df.loc[batch_indices, 'transformed_hard_text'] = \
        vec2text.invert_embeddings(
            torch.tensor(train_x_transformed[batch_indices]).to(device).float(),
            corrector=corrector
        )

bios_train_df.to_csv('/home/nlp/matan_avitan/git/vec2text_inter/bios_data/bios_train_bsw_4_n_steps_20_df.csv',
                     index=False, escapechar='\\')
# bios_train_df.to_csv('bios_data/bios_train_df.csv', index=False)
