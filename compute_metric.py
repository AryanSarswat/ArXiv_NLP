import torch
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
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
import argparse

os.environ["WANDB_PROJECT"] = "AbstractToTitle"  # name your W&B project
os.environ["CUDA_VISIBLE_DEVICES"]="0"
MODEL_CHECKPOINT = "google-t5/t5-base"
OUTPUT_DIR = "./t5_base_abs_title"
SUBSAMPLE = False
WANDB = True
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
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    args = parser.parse_args()

    accelerator = Accelerator(mixed_precision="fp16")
    
    # Load Dataset
    ds = load_from_disk("arxiv-tokenized.hf")
    num_samples = len(ds)
    
    print(ds)
    NUM_GPU = torch.cuda.device_count()

    # Load Model
    if args.model == "base":
        MODEL_CHECKPOINT = "./bart_abs_title"
    elif args.model == "large":
        MODEL_CHECKPOINT = "./bart_large_abs_title"

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    batch_size = 64
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    data_loader = DataLoader(ds, shuffle=True, collate_fn=data_collator, batch_size=batch_size, num_workers=20, pin_memory=True)

    # Load Metrics
    rouge = evaluate.load('rouge')
    bleu = evaluate.load("sacrebleu")

    model, dataloader = accelerator.prepare(model, data_loader)

    running_rouge1 = []
    running_rouge2 = []
    running_rougeL = []
    running_rougeLsum = []
    running_bleu = []
    running_bp = []

    num_steps = len(dataloader)
    p_bar = tqdm(range(num_steps), disable=not accelerator.is_local_main_process)

    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            generated_tokens = accelerator.unwrap_model(model).generate(batch["input_ids"].cuda(), attention_mask=batch["attention_mask"].cuda(), max_new_tokens=100, do_sample=True)
            generated_tokens = accelerator.pad_across_processes(generated_tokens, dim=1, pad_index=tokenizer.pad_token_id)
            
            labels = batch["labels"]
            labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)
            
            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
            labels = accelerator.gather(labels).cpu().numpy()
            
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            
            inp_text = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

            scores_r = rouge.compute(predictions=decoded_preds, references=decoded_labels)
            scores_b = bleu.compute(predictions=decoded_preds, references=decoded_labels)

            running_rouge1.append(scores_r["rouge1"])
            running_rouge2.append(scores_r["rouge2"])
            running_rougeL.append(scores_r["rougeL"])
            running_rougeLsum.append(scores_r["rougeLsum"])
            running_bleu.append(scores_b['score'])
            running_bp.append(scores_b['bp'])

            p_bar.update(1)
            p_bar.set_description(f"R1 : {np.array(running_rouge1).mean()} | R2 : {np.array(running_rouge2).mean()} | RL : {np.array(running_rougeL).mean()} | RLS : {np.array(running_rougeLsum).mean()} | BS : {np.array(running_bleu).mean()} | BP : {np.array(running_bp).mean()}")

    print(f"Average Rouge 1 : {np.array(running_rouge1).mean()}")
    print(f"Average Rouge 2 : {np.array(running_rouge2).mean()}")
    print(f"Average Rouge L : {np.array(running_rougeL).mean()}")
    print(f"Average Rouge Lsum : {np.array(running_rougeLsum).mean()}")
    print(f"Average BLEU Score : {np.array(running_bleu).mean()}")
    print(f"Average Brevity Penalty : {np.array(running_bp).mean()}")

if __name__ == "__main__":
    main()
