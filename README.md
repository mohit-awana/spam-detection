# spam-detection

## General info

The project concerns spam detection in email messages to classify the text is spam or not(ham). It includes data analysis, data preparation, text mining and create model by using llm and **FLANT T5 model**. 

## The data
The dataset comes from Emai Spam Collection and can be find [here]([https://www.kaggle.com/uciml/sms-spam-collection-dataset](https://github.com/MWiechmann/enron_spam_data/)). This email Spam Collection is a set of email tagged messages that have been collected for research. The dataset contains a total of 17.171 spam and 16.545 non-spam ("ham") e-mail messages (33.716 e-mails total). The original dataset and documentation can be found [here](https://www2.aueb.gr/users/ion/data/enron-spam/readme.txt).

## Motivation
The aim of the project was build a spam detector using exisiting LLM model with fine tuning using [Parameter-Efficient Fine-Tuning ](https://github.com/huggingface/peft). The spam filtering is one of the way to reduce the number of scams messages.

## Project contains:
- Python script for preprocessing - **data_preprocessing.py**
- Python script for helper function- **utils.py**
- Python script to train and use spam model with customised FLAN T5 - **training.py**
- checkpoints - saved fine tune model
- requirement - neccasary packages to run the project

## Summary
The goal of the project is to detect if an email is a spam or not. I began with data analysis and data pre-processing from the dataset. Following I have used NLP methods to prepare and clean our text data (tokenization, remove stop words, stemming). The preprocessed data then was converted into Hugging face DataDict format and tokenised using Flant T5 tokeniser. Third the fine tuning was performed on 0.71% of trainable parameters(1769472) resulting in 0.61 F1 score for class spam and overall improvement of .15 when comapared to Flant T5 base model.
