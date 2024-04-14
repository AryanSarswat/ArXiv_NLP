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

def main():
    # Load Dataset
    ds = load_dataset("arxiv-tokenized.hf")
    
    # Train test split
    dataset = ds.train_test_split(test_size=0.2) # Dict {train:Dataset, test:Dataset}

    print(dataset)
    
    
if __name__ == "__main__":
    main()

