import pickle
import pandas as pd
import datasets

MAX_SEQUENCE_LENGTH = 64
SAVING_PATH = '/home/nlp/matan_avitan/git/vec2text/train_data/bias_bios'

with open("/home/nlp/matan_avitan/git/vec2text_inter/bios_data/bios_train.pickle", "rb") as f:
    train_df = pd.DataFrame(pickle.load(f))
with open("/home/nlp/matan_avitan/git/vec2text_inter/bios_data/bios_dev.pickle", "rb") as f:
    dev_df = pd.DataFrame(pickle.load(f))
with open("/home/nlp/matan_avitan/git/vec2text_inter/bios_data/bios_test.pickle", "rb") as f:
    test_df = pd.DataFrame(pickle.load(f))


def preprocess(df):
    text_series = pd.concat([df['hard_text'], df['text']], ignore_index=True)
    text_df = pd.DataFrame(text_series, columns=['text'])
    text_df = text_df[text_df['text_len'] <= MAX_SEQUENCE_LENGTH]
    text_df['text_len'] = text_df['text'].str.split().apply(len)
    text_df = text_df.drop(['text_len'], axis=1)
    text_df['text'] = text_df['text'].sample(frac=1)
    return text_df


new_train_df = preprocess(train_df)
new_dev_df = preprocess(dev_df)
new_test_df = preprocess(test_df)

# Convert DataFrames to DatasetDict and assign features
dataset_dict = datasets.DatasetDict({
    "train": datasets.Dataset.from_dict({'text': new_train_df['text'].tolist()}),
    "validation": datasets.Dataset.from_dict({'text': new_dev_df['text'].tolist()}),
    "test": datasets.Dataset.from_dict({'text': new_test_df['text'].tolist()})
})

dataset_dict.save_to_disk(SAVING_PATH)
