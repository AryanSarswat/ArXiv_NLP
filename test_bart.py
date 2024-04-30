import torch
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq
import os
import evaluate
import nltk

os.environ["WANDB_PROJECT"] = "AbstractToTitle"  # name your W&B project
os.environ["CUDA_VISIBLE_DEVICES"]="0"
torch.backends.cudnn.benchmark = True

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # ROUGE expects a newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def main():
    MODEL_CHECKPOINT = "./bart_large_abs_title"

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT).cuda()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    
    abstracts = [
        "This project explores the application of sequence-to-sequence (seq2seq) models for the task of generating concise titles from research paper abstracts. Leveraging a large dataset of papers from arXiv, we investigate the performance of several prominent architectures including recurrent neural networks (RNNs), gated recurrent units (GRUs), and the state-of-the-art transformer model BART. The models are trained to generate titles by taking abstracts as input sequences. Quantitative evaluation is performed using standard metrics like ROUGE and BLEU to measure the overlap between the generated titles and reference human-written titles from the dataset. Experimental results demonstrate the clear superiority of the BART models, especially the larger BART_LARGE variant, in accurately capturing the key points from abstracts and generating informative titles that overlap well with the reference titles. The findings highlight the capabilities of modern seq2seq models, particularly transformers, for this abstractive summarization task.",
    ]

    abstracts_t = [tokenizer(p, return_tensors="pt").to("cuda:0") for p in abstracts]

    titles = []

    for abstract in abstracts_t:
        output = model.generate(**abstract, max_new_tokens=50, do_sample=True, num_beams=20, temperature=2.5, num_return_sequences=15)
        titles.extend(tokenizer.decode(out, skip_special_tokens=True) for out in output)

    for (abstract, title) in zip(abstracts, titles):
        print(f"=" * 20)
        print(abstracts)
        print("\n")
        print("\n".join(titles))
        print(f"=" * 20)




        
if __name__ == "__main__":
    main()

