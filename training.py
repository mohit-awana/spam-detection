from data_preprocessing import *
from utils import *
from datasets import load_dataset,DatasetDict, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
import time
import evaluate

#Setup the PEFT/LoRA model for Fine-Tuning
from peft import LoraConfig, get_peft_model, TaskType


dataset = preprocess(url_path="https://github.com/MWiechmann/enron_spam_data/raw/master/enron_spam_data.zip")


#Load the pre-trained FLAN-T5 model and its tokenizer directly from HuggingFace. 
#Notice that we will be using the small version of FLAN-T5. 
#Setting torch_dtype=torch.bfloat16 specifies the memory type to be used by this model.
model_name='google/flan-t5-base'

original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)


print(print_number_of_trainable_model_parameters(original_model))


lora_config = LoraConfig(
    r=32, # Rank
    lora_alpha=64,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM # FLAN-T5
)

#Add LoRA adapter layers/parameters to the original LLM to be trained.
peft_model = get_peft_model(original_model,
                            lora_config)
print(print_number_of_trainable_model_parameters(peft_model))


# The dataset actually contains 3 diff splits: train, validation, test.
# The tokenize_function code is handling all data across all splits in batches.
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['transformed_text', 'label','__index_level_0__'])

print(f"Shapes of the datasets:")
print(f"Training: {tokenized_datasets['train'].shape}")
print(f"Validation: {tokenized_datasets['val'].shape}")
print(f"Test: {tokenized_datasets['test'].shape}")

print(tokenized_datasets)


#Setup peft trainer 

output_dir = f'./peft-spam-text-training-{str(int(time.time()))}'
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    learning_rate=1e-3,
    gradient_accumulation_steps=1,
    auto_find_batch_size=True,
    num_train_epochs=1,
    save_steps=100,
    save_total_limit=8,
)
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

trainer.train()

peft_model_path="./checkpoint"

trainer.model.save_pretrained(peft_model_path)
tokenizer.save_pretrained(peft_model_path)
