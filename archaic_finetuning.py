import os
import random
from collections import Counter
import json
from tqdm import tqdm
import editdistance
import numpy as np

from transformers import AutoTokenizer, BertForMaskedLM, BertTokenizer, BertConfig
from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import pipeline
from transformers import set_seed

import torch

test_set = json.load(open('data/archaic/test_common.json'))
max_l = max([ len(x['masked_gt']) for x in test_set])

from transformers import EarlyStoppingCallback
import os

# Load pre-trained BERT model and tokenizer
#model_name = "bert-base-uncased"
set_seed(0)
model_name = "pranaydeeps/Ancient-Greek-BERT"
tokenizer = BertTokenizer.from_pretrained(model_name)
config = BertConfig.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name, config=config)

early_stopping = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0)

train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="data/archaic/train.txt",#"data/archaic/iphi_archaic_train.txt",
    block_size=128,
)

data_collator = DataCollatorForLanguageModeling(
    #tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="models/agBERT-iphi-archaic",
    overwrite_output_dir=True,
    num_train_epochs=10000,
    per_device_train_batch_size=32,
    save_steps=500,
    logging_steps=500,
    eval_steps=500,
    save_total_limit=2,
    evaluation_strategy = 'steps',
    metric_for_best_model= 'eval_loss',
    load_best_model_at_end = True,
    greater_is_better=False,
    logging_dir='./logs/archaic',
    run_name="agBERT-iphi-archaic",
)

val_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="data/archaic/validation.txt",#"data/archaic/iphi_archaic_validation.txt",
    block_size=128,
)

test_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="data/archaic/test.txt",#"data/archaic/iphi_archaic_test.txt",
    block_size=128,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[early_stopping]
)

test_result = trainer.evaluate(test_dataset)
print("Evaluation result:", test_result)
"""
Evaluation result: {'eval_loss': 11.396125793457031, 'eval_runtime': 22.5231, 'eval_samples_per_second': 346.8, 'eval_steps_per_second': 14.474}
"""

trainer.train()
trainer.save_model('models/agBERT-archaic')
tokenizer.save_pretrained('models/agBERT-archaic')

test_result = trainer.evaluate(test_dataset)
print("Evaluation result:", test_result)

# with open('log.txt', 'w') as f:
#     f.write(cap.stdout)