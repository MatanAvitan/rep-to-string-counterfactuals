#!/usr/bin/env python
# coding: utf-8


import os
import pickle
import pathlib

import datasets
import numpy as np
import ot
import pandas as pd
import torch
import tqdm
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel

import wandb

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'
# os.environ["WORLD_SIZE"] = "2"

MODEL_ID = 'mistralai/Mistral-7B-v0.1'
# MODEL_ID = 'openai-community/gpt2'
SEED=0
torch.manual_seed(SEED)
np.random.seed(SEED)

IS_FIRST = False
SAMPLE = 1_000

file_name = pathlib.Path(__file__).name

from evaluate import load

perplexity = load("perplexity", module_type="metric")

base = '/home/nlp/matan_avitan/tmp/vec2text_inter/notebooks/train_class/bios_extracted_data'
leace_females = f'{base}/leace_females.csv'
leace_females_df = pd.read_csv(leace_females)
leace_males = f'{base}/leace_males.csv'
leace_males_df = pd.read_csv(leace_males)
ot_males_to_females = f'{base}/ot_males_to_females.csv'
ot_males_to_females_df = pd.read_csv(ot_males_to_females)
ot_females_to_males = f'{base}/ot_females_to_males.csv'
ot_females_to_males_df = pd.read_csv(ot_females_to_males)
new_ot_males_to_females = f'{base}/new_ot_males_to_females.csv'
new_ot_males_to_females_df = pd.read_csv(new_ot_males_to_females)
new_ot_females_to_males = f'{base}/new_ot_females_to_males.csv'
new_ot_females_to_males_df = pd.read_csv(new_ot_females_to_males)

leace_df = pd.concat((leace_females_df, leace_males_df))
idxs = np.random.choice(np.arange(len(leace_df)), size=SAMPLE)
leace_df = leace_df.iloc[idxs]
ot_df = pd.concat((ot_females_to_males_df, ot_males_to_females_df))
ot_df = ot_df.iloc[idxs]
new_ot_df = pd.concat((new_ot_females_to_males_df, new_ot_males_to_females_df))
new_ot_df = new_ot_df.iloc[idxs]

wandb.init(project=file_name,
           # Track hyperparameters and run metadata
           config={
               "model": MODEL_ID,
           })

res = perplexity.compute(predictions=leace_df['intervention_hard_text'].tolist(), model_id=MODEL_ID)
wandb.log({'leace_intervention_hard_text': res})
res = perplexity.compute(predictions=ot_df['intervention_hard_text'].tolist(), model_id=MODEL_ID)
wandb.log({'ot_intervention_hard_text': res})
res = perplexity.compute(predictions=new_ot_df['intervention_hard_text'].tolist(), model_id=MODEL_ID)
wandb.log({'new_ot_intervention_hard_text': res})


res = perplexity.compute(predictions=leace_df['hard_text'].tolist(), model_id=MODEL_ID)
wandb.log({'hard_text': res})
res = perplexity.compute(predictions=leace_df['transformed_hard_text'].tolist(), model_id=MODEL_ID)
wandb.log({'transformed_hard_text': res})

res = perplexity.compute(predictions=ot_df['hard_text'].tolist(), model_id=MODEL_ID)
wandb.log({'hard_text_double_check': res})
res = perplexity.compute(predictions=ot_df['transformed_hard_text'].tolist(), model_id=MODEL_ID)
wandb.log({'transformed_hard_text_double_check': res})

wandb.finish()

