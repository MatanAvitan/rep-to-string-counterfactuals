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

SEED=0
torch.manual_seed(SEED)
np.random.seed(SEED)

IS_FIRST = False
SAMPLE = 1_000

file_name = pathlib.Path(__file__).name

from evaluate import load

metric = load("bertscore")

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
               "model": 'bertscore',
           })

res = metric.compute(predictions=leace_df['intervention_hard_text'].tolist(), references=leace_df['hard_text'].tolist(), lang="en")
wandb.log({'leace_intervention_hard_text': np.mean(res['f1'])})
res = metric.compute(predictions=ot_df['intervention_hard_text'].tolist(), references=leace_df['hard_text'].tolist(), lang="en")
wandb.log({'ot_intervention_hard_text': np.mean(res['f1'])})
res = metric.compute(predictions=new_ot_df['intervention_hard_text'].tolist(), references=leace_df['hard_text'].tolist(), lang="en")
wandb.log({'new_ot_intervention_hard_text': np.mean(res['f1'])})


res = metric.compute(predictions=leace_df['transformed_hard_text'].tolist(), references=leace_df['hard_text'].tolist(), lang="en")
wandb.log({'transformed_hard_text': np.mean(res['f1'])})


wandb.finish()

