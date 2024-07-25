#load neccesary packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import DatasetDict, Dataset
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()


def tokenize_words(text):
    """Tokenize words in text and remove punctuation"""
    text = word_tokenize(str(text).lower())
    text = [token for token in text if token.isalnum()]
    return text


def remove_stopwords(text):
    """Remove stopwords from the text"""
    text = [token for token in text if token not in stopwords.words("english")]
    return text


def stem(text):
    """Stem the text (originate => origin)"""
    text = [ps.stem(token) for token in text]
    return text


def transform(text):
    """Tokenize, remove stopwords, stem the text"""
    text = tokenize_words(text)
    text = remove_stopwords(text)
    text = stem(text)
    text = " ".join(text)
    return text

def transform_df(df):
    """Apply the transform function to the dataframe"""
    df["transformed_text"] = df["text"].apply(transform)
    return df

#convert processed data into hugging face datasets format

def train_val_test_split(df, train_size=0.8, has_val=True):
    """Return a tuple (Dataframe, DatasetDict) with a custom train/val/split"""
    # Convert int train_size into float
    if isinstance(train_size, int):
        train_size = train_size / len(df)

    # Shuffled train/val/test split
    df = df.sample(frac=1, random_state=0)
    df_train, df_test = train_test_split(
        df, test_size=1 - train_size, stratify=df["label"]
    )

    if has_val:
        df_test, df_val = train_test_split(
            df_test, test_size=0.5, stratify=df_test["label"]
        )
        return (
            (df_train, df_val, df_test),
            DatasetDict(
                {
                    "train": Dataset.from_pandas(df_train),
                    "val": Dataset.from_pandas(df_val),
                    "test": Dataset.from_pandas(df_test),
                }
            ),
        )

    else:
        return (
            (df_train, df_test),
            DatasetDict(
                {
                    "train": Dataset.from_pandas(df_train),
                    "test": Dataset.from_pandas(df_test),
                }
            ),
        )

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"


def tokenize_function(example):

    start_prompt = 'Classify the following text as spam or ham' + '\n\n '
    end_prompt = '\n\n ' +'label: '
    prompt = [start_prompt + dialogue + end_prompt for dialogue in example["transformed_text"]]
    example['input_ids'] = tokenizer(prompt,max_length=128, padding="max_length", truncation=True, return_tensors="pt").input_ids
    example['labels'] = tokenizer(example["label"], padding="max_length", truncation=True, return_tensors="pt").input_ids
    labels = example['labels']
    labels[labels == tokenizer.pad_token_id] = -100
    example['labels'] = labels


    return example



