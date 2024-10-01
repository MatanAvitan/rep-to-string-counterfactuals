#!/usr/bin/env python
# coding: utf-8
import os
import pickle
import pathlib

# import datasets
import numpy as np
# import ot
import pandas as pd
import torch
import tqdm
import random
import wandb

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '3,4'
# os.environ["WORLD_SIZE"] = "2"

MODEL_ID = 'mistralai/Mistral-7B-v0.1'
# MODEL_ID = 'openai-community/gpt2'
SEED=0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
IS_FIRST = False

file_name = pathlib.Path(__file__).name

from evaluate import load

perplexity = load("perplexity", module_type="metric")

base = '/home/nlp/matan_avitan/git/rep-to-string-counterfactuals/data/cebab_gpt_cfs'

# LEACE
food_leace = f'{base}/cebab_gpt_leace_food.csv'
service_leace = f'{base}/cebab_gpt_leace_service.csv'
food_leace = pd.read_csv(food_leace)
service_leace = pd.read_csv(service_leace)
# MiMiC
food_mimic = f'{base}/cebab_gpt_mimic_food.csv'
service_mimic = f'{base}/cebab_gpt_mimic_service.csv'
food_mimic = pd.read_csv(food_mimic)
service_mimic = pd.read_csv(service_mimic)
# MiMiC plus
food_mimic_plus = f'{base}/cebab_gpt_mimic_plus_food.csv'
service_mimic_plus = f'{base}/cebab_gpt_mimic_plus_service.csv'
food_mimic_plus = pd.read_csv(food_mimic_plus)
service_mimic_plus = pd.read_csv(service_mimic_plus)
# No intervetion applied
no_int = f'{base}/cebab_gpt_no_int.csv'
no_int = pd.read_csv(no_int)


wandb.init(project=file_name,
           # Track hyperparameters and run metadata
           config={
               "model": MODEL_ID,
               "dataset": 'cebab-gpt'
           })

res = perplexity.compute(predictions=food_leace['transformed_hard_text'].tolist(), model_id=MODEL_ID)
wandb.log({'food_leace_int': res})
res = perplexity.compute(predictions=service_leace['transformed_hard_text'].tolist(), model_id=MODEL_ID)
wandb.log({'service_leace_int': res})

res = perplexity.compute(predictions=food_mimic['transformed_hard_text'].tolist(), model_id=MODEL_ID)
wandb.log({'food_mimic_int': res})
res = perplexity.compute(predictions=service_mimic['transformed_hard_text'].tolist(), model_id=MODEL_ID)
wandb.log({'service_mimic_int': res})

res = perplexity.compute(predictions=food_mimic_plus['transformed_hard_text'].tolist(), model_id=MODEL_ID)
wandb.log({'food_mimic_plus_int': res})
res = perplexity.compute(predictions=service_mimic_plus['transformed_hard_text'].tolist(), model_id=MODEL_ID)
wandb.log({'service_mimic_plus_int': res})

res = perplexity.compute(predictions=no_int['text'].tolist(), model_id=MODEL_ID)
wandb.log({'text_wo_modification': res})
res = perplexity.compute(predictions=no_int['transformed_hard_text'].tolist(), model_id=MODEL_ID)
wandb.log({'transformed_hard_text': res})


wandb.finish()

