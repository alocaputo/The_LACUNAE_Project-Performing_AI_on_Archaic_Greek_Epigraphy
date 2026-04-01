import os
import random
from collections import Counter
import json
from tqdm import tqdm
import editdistance
import numpy as np

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import pipeline
from transformers import set_seed
from datasets import load_dataset

import torch

test_set = json.load(open('data_bak/archaic/test_common.json'))
max_l = max([ len(x['masked_gt']) for x in test_set])

from transformers import EarlyStoppingCallback
import os

set_seed(0)
model_name = "bowphs/PhilTa"
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=config)

tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = 0
model.config.pad_token_id = tokenizer.pad_token_id

early_stopping = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0)

def load_text_dataset(file_path):
    dataset = load_dataset('text', data_files=file_path, split='train')
    
    def preprocess_function(examples):
        inputs = ["restore: " + text for text in examples['text']]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding='max_length')
        labels = tokenizer(examples['text'], max_length=128, truncation=True, padding='max_length')
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    return dataset.map(preprocess_function, batched=True, remove_columns=['text'])

train_dataset = load_text_dataset("data_bak/archaic/train.txt")

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, model=model
)

training_args = Seq2SeqTrainingArguments(
    output_dir="models/philta-archaic",
    overwrite_output_dir=True,
    num_train_epochs=10000,
    per_device_train_batch_size=32,
    save_steps=500,
    logging_steps=500,
    eval_steps=500,
    save_total_limit=2,
    eval_strategy='steps',
    metric_for_best_model='eval_loss',
    load_best_model_at_end=True,
    greater_is_better=False,
    logging_dir='./logs/philta',
    run_name="philta-iphi-archaic",
    predict_with_generate=False,
)

val_dataset = load_text_dataset("data_bak/archaic/validation.txt")

test_dataset = load_text_dataset("data_bak/archaic/test.txt")

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[early_stopping]
)

test_result = trainer.evaluate(test_dataset)

trainer.train()
trainer.save_model('models/philta-archaic')
tokenizer.save_pretrained('models/philta-archaic')

test_result = trainer.evaluate(test_dataset)
print("Evaluation result:", test_result)