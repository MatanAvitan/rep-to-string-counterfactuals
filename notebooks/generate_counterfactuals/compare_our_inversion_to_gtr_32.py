#!/usr/bin/env python
# coding: utf-8


import os
import pickle
import pathlib

import numpy as np
import ot
import pandas as pd
import torch
import tqdm
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel

import vec2text
import wandb
from vec2text.models.model_utils import device

IS_FIRST = False
THEIR_BASE_MODEL = 'gtr-base'
OUR_BASE_MODEL = 'checkpoint-166284'
MAX_SEQUENCE_LENGTH = 64
PROCESSING_BATCH_SIZE = 64
N_SAMPLES = 30
INVERSION_BATCH_SIZE = 16
NUM_CORRECTION_STEPS = 0
BEAM_SEARCH_SIZE = 0

file_name = pathlib.Path(__file__).name
wandb.init(project=file_name,
           # Track hyperparameters and run metadata
           config={
               "MAX_SEQUENCE_LENGTH": MAX_SEQUENCE_LENGTH,
               "PROCESSING_BATCH_SIZE": PROCESSING_BATCH_SIZE,
               "INVERSION_BATCH_SIZE": INVERSION_BATCH_SIZE,
               "THEIR_BASE_MODEL": THEIR_BASE_MODEL,
               "OUR_BASE_MODEL": OUR_BASE_MODEL,
               "NUM_CORRECTION_STEPS": NUM_CORRECTION_STEPS,
               "BEAM_SEARCH_SIZE": BEAM_SEARCH_SIZE,
           }
           )


def log_print(str_to_print):
    eval_str_to_print = eval(str_to_print)
    if type(eval_str_to_print) == list:
        eval_str_to_print = '\n'.join(eval_str_to_print)
    print(f"{str_to_print}: {eval_str_to_print}")


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
log_print('bios_train_df.shape')

sequences_length = np.zeros((bios_train_df.shape[0]))


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
their_corrector = vec2text.load_corrector(THEIR_BASE_MODEL)
our_corrector = vec2text.load_corrector(OUR_BASE_MODEL)

encodings = []
for i in tqdm.tqdm(range(0, len(bios_train_df), PROCESSING_BATCH_SIZE)):
    sents_batch = bios_train_df.loc[i:i + PROCESSING_BATCH_SIZE - 1, 'hard_text'].tolist()
    embeddings, samples_len = get_gtr_embeddings(sents_batch, encoder, tokenizer)
    encodings.append(embeddings)
    sequences_length[i: i + PROCESSING_BATCH_SIZE] = samples_len

x = np.concatenate(encodings, axis=0)
x = x[:len(bios_train_df)]  # Nan are added to the last batch, let's remove them

decoded_tb = wandb.Table(columns=['hard_text', 'decoded_hard_text_option_1', 'decoded_hard_text_option_2'])

"""
Generate text from repr
"""
mask_indices = np.arange(len(bios_train_df))[(sequences_length <= MAX_SEQUENCE_LENGTH)][:N_SAMPLES]
for lower_bound in tqdm.tqdm(range(0, len(mask_indices), INVERSION_BATCH_SIZE)):
    batch_indices = mask_indices[lower_bound: lower_bound + INVERSION_BATCH_SIZE]
    bios_train_df.loc[batch_indices, 'decoded_hard_text_option_1'] = \
        vec2text.invert_embeddings(
            torch.tensor(x[batch_indices]).to(device).float(),
            corrector=their_corrector
        )
    bios_train_df.loc[batch_indices, 'decoded_hard_text_option_2'] = \
        vec2text.invert_embeddings(
            torch.tensor(x[batch_indices]).to(device).float(),
            corrector=our_corrector
        )
    [decoded_tb.add_data(*i) for i
     in bios_train_df.loc[batch_indices,
    ['hard_text', 'decoded_hard_text_option_1', 'decoded_hard_text_option_2']].to_records(index=False).tolist()]
    wandb.log({'decoded_tb': decoded_tb})

bios_train_df.to_csv(
    f'/home/nlp/matan_avitan/git/vec2text_inter/bios_data/{file_name}_{MAX_SEQUENCE_LENGTH}_{NUM_CORRECTION_STEPS}_{BEAM_SEARCH_SIZE}.csv',
    index=False, escapechar='\\')
wandb.finish()
# bios_train_df.to_csv('bios_data/bios_train_df.csv', index=False)
