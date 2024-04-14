import torch
import numpy as np
import json
import re
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, TrainingArguments, Trainer
import os

os.environ["WANDB_PROJECT"] = "AbstractToTitle"  # name your W&B project
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

def show_samples(dataset, num_samples=2):
    sample = dataset['train'].shuffle().select(range(num_samples))
    for example in sample:
        print(f">> Abstract : {example['abstract']}")
        print(f">> Title : {example['title']}\n")
        
def main():
    # Load Dataset
    ds = load_dataset("json", data_files="./arxiv-metadata-oai-snapshot.json", split="train[:10%]")

    # Train test split
    dataset = ds.train_test_split(test_size=0.2) # Dict {train:Dataset, test:Dataset}

    print(dataset)
    
    model_name = 'facebook/bart-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    if torch.cuda.device_count() > 1:
        print(f"Using Multiple GPUs")
        model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])

    # abstracts_only = dataset['train']['abstract']

    # max_token_length = max(len(tokenizer.encode(abstract, truncation=True)) for abstract in abstracts_only)
    # print(f"The longest text is {max_token_length} tokens long.") # Get nax token length

    def get_feature(batch):
        encodings = tokenizer(batch['abstract'], text_target=batch['title'],
                            max_length=1024, truncation=True)

        encodings = {'input_ids': encodings['input_ids'],
                'attention_mask': encodings['attention_mask'],
                'labels': encodings['labels']}

        return encodings
    
    dataset_pt = dataset.map(get_feature, batched=True)
    columns = ['input_ids', 'labels', 'attention_mask']
    dataset_pt.set_format(type='torch', columns=columns)

    print(dataset_pt)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # we're using the Trainer API which abstracts away a lot of complexity
    training_args = TrainingArguments(
        output_dir = 'bart_abs_title', # rename to what you want it to be called
        learning_rate = 2e-3,
        num_train_epochs=25, # your choice
        warmup_steps = 500,
        per_device_train_batch_size=32, # keep a small batch size when working with a small GPU
        per_device_eval_batch_size=32,
        weight_decay = 0.01, # helps prevent overfitting
        logging_steps = 200,
        evaluation_strategy = 'epoch',
        save_steps=3e3,
        report_to="wandb",
        fp16=True,
        run_name="batch_32_4gpu",
        remove_unused_columns=False
    )

    trainer = Trainer(model=model, args=training_args, tokenizer=tokenizer, data_collator=data_collator,
                    train_dataset = dataset_pt['train'], eval_dataset = dataset_pt['test'])

    trainer.train()
    
    
if __name__ == "__main__":
    main()

