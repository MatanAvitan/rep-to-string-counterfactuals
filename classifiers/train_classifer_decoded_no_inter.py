import os
import pickle

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from datasets import load_metric
from transformers import AutoTokenizer, Trainer, TrainingArguments, EvalPrediction, RobertaForSequenceClassification, \
    RobertaConfig
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
# import spacy
import datasets
from datetime import datetime
import wandb
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

EXP_NAME = 'decoded_no_inter'
MAX_LENGTH = 64
SEED = 0
IS_FIRST = True

file_name = Path(__file__).name
wandb.init(project=file_name)


def log_print(str_to_print):
    eval_str_to_print = eval(str_to_print)
    if type(eval_str_to_print) == list:
        eval_str_to_print = '\n'.join(eval_str_to_print)
    print(f"{str_to_print}: {eval_str_to_print}")


# # Load data
with open("/home/nlp/matan_avitan/git/vec2text_inter/bios_data/bios_dev.pickle", "rb") as f:
    validation_df = pd.DataFrame(pickle.load(f))

ot_females_to_males_path = f'bios_extracted_data/ot_females_to_males.csv'
ot_males_to_females_path = f'bios_extracted_data/ot_males_to_females.csv'

ot_females_to_males_df = pd.read_csv(ot_females_to_males_path)
ot_males_to_females_df = pd.read_csv(ot_males_to_females_path)

ot_stacked = pd.concat([ot_females_to_males_df, ot_males_to_females_df]).reset_index(drop=True)

labels = list(ot_stacked['p'].unique())
id2label = {i: l for i, l in enumerate(labels)}
label2id = {l: i for i, l in enumerate(labels)}

formatted_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# Define the directory where the output/results will be saved
output_dir = f'/home/nlp/matan_avitan/tmp/vec2text_inter/trained_models/{EXP_NAME}_classifier_{formatted_datetime}'
# Define the specific pre-trained model to be used
model_name = "FacebookAI/roberta-base"
# Init config with labels
config = RobertaConfig.from_pretrained(model_name, label2id=label2id, id2label=id2label, num_labels=len(labels))
# Retrieve the tokenizer component from the pipeline object
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Clear the CUDA cache to free up GPU memory
torch.cuda.empty_cache()
# Retrieve the model component from the pipeline object
model = RobertaForSequenceClassification.from_pretrained(model_name,
                                                         config=config)  # Adding the labels dicts is equivalent to num_labels=3, and therfore init a classifier head.

num_labels = len(label2id)  # Number of classes in your classification task


# # Create/Load data


def create_input_sequence(sample):
    features = sample['transformed_hard_text']
    labels = sample['labels']
    # Encoding the sequence using the tokenizer
    encoded_sequence = tokenizer(features, truncation=True, padding='max_length', max_length=MAX_LENGTH,
                                 return_tensors='pt')
    # Decode the input_ids
    encoded_sequence['input_sentence'] = tokenizer.batch_decode(encoded_sequence.input_ids)
    # Assign label to the encoded sequence
    encoded_sequence['labels'] = labels
    return encoded_sequence


label_encoder = LabelEncoder()
ot_stacked['labels'] = label_encoder.fit_transform(ot_stacked['p'])
validation_df['labels'] = label_encoder.transform(validation_df['p'])
validation_df = validation_df.rename({'hard_text': 'transformed_hard_text'}, axis=1)

# Define id2label and label2id mappings
id2label = {i: label for i, label in enumerate(label_encoder.classes_)}
label2id = {label: i for i, label in enumerate(label_encoder.classes_)}
wandb.log({'labels': list(id2label.values())})

# ot_stacked['labels'] = ot_stacked['p']
train_df = ot_stacked
train_ds = datasets.Dataset.from_pandas(train_df)
validation_ds = datasets.Dataset.from_pandas(validation_df)
len(train_ds), len(validation_ds), train_ds

if IS_FIRST:
    features = ['transformed_hard_text']
    labels = ['labels']
    columns_to_preserve = features + labels + ['g']
    columns_to_remove = list(set(train_df.columns) - set(columns_to_preserve))

    train_ds = train_ds.map(create_input_sequence, batched=True, batch_size=1, remove_columns=columns_to_remove,
                            num_proc=16)
    train_ds.save_to_disk(f'{EXP_NAME}_train_dataset')
    columns_to_remove = list(set(validation_df.columns) - set(columns_to_preserve))
    validation_ds = validation_ds.map(create_input_sequence, batched=True, batch_size=1,
                                      remove_columns=columns_to_remove,
                                      num_proc=16)
    validation_ds.save_to_disk(f'{EXP_NAME}_validation_dataset')
else:
    train_ds = datasets.load_from_disk(f'{EXP_NAME}_train_dataset')
    validation_ds = datasets.load_from_disk(f'{EXP_NAME}_validation_dataset')


def calculate_tpr(y_pred, y_true, z_true):
    rms_tpr_gap = 0.0
    for y in set(y_pred):
        tpr_1 = (((y_pred[y_true == y])[z_true[y_true == y] == 1]) == y).mean()
        tpr_0 = (((y_pred[y_true == y])[z_true[y_true == y] == 0]) == y).mean()
        rms_tpr_gap += (tpr_1 - tpr_0) ** 2

    rms_tpr_gap = np.sqrt(rms_tpr_gap / len(set(y_pred)))
    return rms_tpr_gap


def compute_metrics(p: EvalPrediction):
    # Extracting predictions from EvalPrediction object
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions

    # Obtaining the predicted classes
    preds = np.argmax(preds, axis=1)

    # Calculating the ratio of predictions equal to 0 (arbitrary label)
    ratio = np.mean(preds == 0)

    # Dictionary to store computed metrics
    result = {}

    # Loading evaluation metrics
    metric_f1 = load_metric("f1")
    metric_precision = load_metric("precision")
    metric_recall = load_metric("recall")
    metric_acc = load_metric("accuracy")

    # Computing various metrics
    result["accuracy"] = metric_acc.compute(predictions=preds, references=p.label_ids)["accuracy"]
    result["precision"] = metric_precision.compute(predictions=preds, references=p.label_ids, average='macro')[
        'precision']
    result["recall"] = metric_recall.compute(predictions=preds, references=p.label_ids, average='macro')["recall"]
    result["f1"] = metric_f1.compute(predictions=preds, references=p.label_ids, average='macro')["f1"]
    result["ratio"] = ratio

    return result


training_args = TrainingArguments(
    output_dir=output_dir,  # Output directory
    logging_dir=output_dir + "/logs",  # Output directory for logging
    num_train_epochs=30,  # Total number of training epochs
    per_device_train_batch_size=512,  # Batch size per device during training
    per_device_eval_batch_size=2048,  # Batch size for evaluation
    weight_decay=0.01,  # Strength of weight decay
    gradient_accumulation_steps=1,  # The number of steps whose gradients are accumulated
    learning_rate=2e-5,  # Controls the magnitude of updates to the model weights
    warmup_ratio=0.06,  # Represents the proportion of training steps
    label_smoothing_factor=0.05,  # Regularization technique to prevent the model from becoming overconfident
    evaluation_strategy='steps',  # Frequency or timing of evaluating
    logging_strategy='steps',  # Frequency or timing of logging
    logging_steps=50,  # Frequency or timing of logging
    eval_steps=50,  # Frequency or timing of evaluating
    logging_first_step=True,
    fp16=True,
    do_eval=True,
    report_to="wandb",
    save_total_limit=5,
    load_best_model_at_end=True,
    save_steps=50
    # eval_accumulation_steps=16
)

trainer = Trainer(
    model=model,  # The instantiated model to be trained
    args=training_args,  # Training arguments, defined above
    compute_metrics=compute_metrics,  # A function to compute the metrics
    train_dataset=train_ds,  # Training dataset
    eval_dataset=validation_ds  # Evaluation dataset
)

print(trainer.args.device)

trainer.evaluate()

torch.cuda.empty_cache()

trainer.train()

trainer.evaluate()

model.eval()

Path(output_dir + "/modeldir_new").mkdir()
# Save the state_dict of your custom model
torch.save(model.state_dict(), output_dir + "/modeldir_new/model.pth")

# Save the tokenizer associated with the model
tokenizer.save_pretrained(output_dir + "/modeldir_new/tokenizer")

trainer.save_model(output_dir + "/modeldir_new/trainer_model")
