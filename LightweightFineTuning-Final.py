#!/usr/bin/env python
# coding: utf-8

# # Lightweight Fine-Tuning Project

# TODO: In this cell, describe your choices for each of the following
# 
# * PEFT technique: 
# * Model: 
# * Evaluation approach: 
# * Fine-tuning dataset: 

# ## Loading and Evaluating a Foundation Model
# 
# TODO: In the cells below, load your chosen pre-trained Hugging Face model and evaluate its performance prior to fine-tuning. This step includes loading an appropriate tokenizer and dataset.

# In[15]:


import torch
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer, AutoModelForSequenceClassification
from datasets import load_dataset
import pandas as pd
import numpy as np


# In[16]:


splits = ["train", "test"]
ds = {split: ds for split, ds in zip(splits, load_dataset("rotten_tomatoes", split=splits))}

for split in splits:
    ds[split] = ds[split].shuffle(seed=42).select(range(500))

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_ds = {}
for split in splits:
    tokenized_ds[split] = ds[split].map(preprocess_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
    id2label={0: "NEGATIVE", 1: "POSITIVE"},
    label2id={"NEGATIVE": 0, "POSITIVE": 1},
)

# Freeze all the parameters of the base model
for param in base_model.base_model.parameters():
    param.requires_grad = False


# In[17]:


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": (predictions == labels).mean()}


# In[18]:


trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./finalresults",
        learning_rate=2e-3,
        per_device_train_batch_size=6,
        per_device_eval_batch_size=6,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    ),
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


# In[19]:


trainer.train()

evaluation_result = trainer.evaluate()

print(evaluation_result)


# In[ ]:





# ## Performing Parameter-Efficient Fine-Tuning
# 
# TODO: In the cells below, create a PEFT model from your loaded model, run a training loop, and save the PEFT model weights.

# In[20]:


for param in model.parameters():
    param.requires_grad = True


# In[21]:


trainer_peft = Trainer(
    model=base_model,
    args=TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=12,
        per_device_eval_batch_size=12,
        num_train_epochs=4,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    ),
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

trainer_peft.train()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Performing Inference with a PEFT Model
# 
# TODO: In the cells below, load the saved PEFT model weights and evaluate the performance of the trained PEFT model. Be sure to compare the results to the results from prior to fine-tuning.

# In[22]:


peft_model_evaluation = trainer_peft.evaluate()


# In[23]:


print("Base Model Evaluation:")
print(evaluation_result)

print("\nPEFT Model Evaluation:")
print(peft_model_evaluation)


# In[ ]:





# In[ ]:





# In[ ]:




