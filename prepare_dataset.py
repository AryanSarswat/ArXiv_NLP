from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_CHECKPOINT = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

def preprocess_function(examples):
    model_inputs = tokenizer(examples["abstract"], max_length=1024, truncation=True)
    labels = tokenizer(examples["title"], max_length=1024, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    dataset = load_dataset("json", data_files="./arxiv-metadata-oai-snapshot.json", split="train")
    print(dataset)
    
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(dataset.column_names)
    print(tokenized_datasets)
    
    tokenized_datasets.save_to_disk("arxiv-tokenized.hf")

if __name__ == '__main__':
    main()
    