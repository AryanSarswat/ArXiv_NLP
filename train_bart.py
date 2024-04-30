import torch
from torch.optim import AdamW
from torch.optim import AdamW
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_from_disk
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq
import os
from tqdm.auto import tqdm
import wandb
import evaluate
import nltk

os.environ["WANDB_PROJECT"] = "AbstractToTitle"  # name your W&B project
os.environ["CUDA_VISIBLE_DEVICES"]="0"
MODEL_CHECKPOINT = "facebook/bart-base"
OUTPUT_DIR = "./bart_abs_title"
SUBSAMPLE = True
WANDB = False
nltk.download('punkt')
torch.backends.cudnn.benchmark = True

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # ROUGE expects a newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def main():
    accelerator = Accelerator(mixed_precision="fp16")
    
    # Load Dataset
    ds = load_from_disk("arxiv-tokenized.hf")
    num_samples = len(ds)
    ratio = 0.01
    
    # # Select Random Subset
    if SUBSAMPLE:
        ds = ds.select(np.random.randint(0, num_samples, size=int(ratio * num_samples)))
    #ds = ds.select(np.random.randint(0, num_samples, size=int(1000)))
    
    # Train test split
    dataset = ds.train_test_split(test_size=0.2) # Dict {train:Dataset, test:Dataset}

    print(dataset)
    NUM_GPU = torch.cuda.device_count()
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)
    
    # if NUM_GPU > 1:
    #     model = torch.nn.DataParallel(model)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    rouge_score = evaluate.load("rouge")  
    
    batch_size = 64
    train_dataloader = DataLoader(dataset["train"], shuffle=True, collate_fn=data_collator, batch_size=batch_size, num_workers=20, pin_memory=True)
    eval_dataloader = DataLoader(dataset["test"], collate_fn=data_collator, batch_size=batch_size, num_workers=20, pin_memory=True)
    
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader)
    
    num_train_epochs = 1
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=100, num_training_steps=num_training_steps)
    
    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)
    
    
    if WANDB:
        run = wandb.init(project="AbstractToTitle", name=f"BART_LARGE_A2T_{batch_size}_{NUM_GPU}gpu")
        columns = ["Prompt", "Predicted Title", "True Title"]
        table = wandb.Table(columns=columns)
    
    for epoch in range(num_train_epochs):
        model.train()
        
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            
            total_loss += loss.item()
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            progress_bar.update(1)
            
            if WANDB:
                run.log({"loss" : loss.item()})
            
        total_loss /= step

        model.eval()
        
        STEP_ = np.random.randint(low=0, high=len(eval_dataloader), size=50)
        
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(model).generate(batch["input_ids"], attention_mask=batch["attention_mask"])
                generated_tokens = accelerator.pad_across_processes(generated_tokens, dim=1, pad_index=tokenizer.pad_token_id)
                
                labels = batch["labels"]
                labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)
                
                generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                labels = accelerator.gather(labels).cpu().numpy()
                
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                
                inp_text = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
                
                rouge_score.add_batch(predictions=decoded_preds, references=decoded_labels)
                
                
                if step in STEP_:
                    IDX_ = np.random.randint(low=0, high=len(batch["input_ids"]), size=1)[0]
                    print(f"Sample prediction : {decoded_preds[IDX_]}")
                    print(f"Sample label : {decoded_labels[IDX_]}")
                    
                    if WANDB:
                        table.add_data(inp_text[IDX_], decoded_preds[IDX_], decoded_labels[IDX_])
        
        # Compute metrics
        result = rouge_score.compute()
        # Extract the median ROUGE scores
        result = {key: value * 100 for key, value in result.items()}
        result = {k: round(v, 4) for k, v in result.items()}
        result["Epoch Loss"] = total_loss
        
        print(f"Epoch {epoch}:", result)
        
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(OUTPUT_DIR, save_function=accelerator.save)
        
        if WANDB:
            run.log(result)
        
        if accelerator.is_main_process:
            tokenizer.save_pretrained(OUTPUT_DIR)
    
    if WANDB:
        run.log({"Examples" : table})
        
if __name__ == "__main__":
    main()

