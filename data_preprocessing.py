#load neccesary packages
import time
import evaluate
import pandas as pd
import numpy as np

from urllib.request import urlopen
from zipfile import ZipFile
from pathlib import Path
from io import BytesIO

from utils import *



def preprocess(url_path):
    #create directory for storing raw and processed data
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    # Download and unzip the data to csv
    url = url_path
    with urlopen(url) as zurl:
        with ZipFile(BytesIO(zurl.read())) as zfile:
            zfile.extractall("data/raw")
    # Load the spam data 
    df = pd.read_csv("data/raw/enron_spam_data.csv", encoding="ISO-8859-1")
    
    # Preprocess
    df = df.fillna("")
    df["text"] = df["Subject"] + df["Message"]
    df["label"] = df["Spam/Ham"].map({"ham": 0, "spam": 1})
    df = df[["text", "label"]]
    df = df.dropna()
    df = df.drop_duplicates()
    df = transform_df(df)
    
    
    #further cleaning
    df['label'] = df['label'].astype('str')
    df['transformed_text'] = df['transformed_text'].astype('str')
    df.drop(columns = ['text'], inplace=True)
    df['label'] = df['label'].replace({'0': 'ham', '1': 'spam'})
    
    # Save the processed data 
    df.to_csv("data/processed/data.csv", index=False)
    
    _, dataset = train_val_test_split(
                    df, train_size=0.8, has_val=True
                )

    return dataset



