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
        "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.",

        "Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth. On the ImageNet dataset we evaluate residual nets with a depth of up to 152 layers---8x deeper than VGG nets but still having lower complexity. An ensemble of these residual nets achieves 3.57% error on the ImageNet test set. This result won the 1st place on the ILSVRC 2015 classification task. We also present analysis on CIFAR-10 with 100 and 1000 layers. The depth of representations is of central importance for many visual recognition tasks. Solely due to our extremely deep representations, we obtain a 28% relative improvement on the COCO object detection dataset. Deep residual nets are foundations of our submissions to ILSVRC & COCO 2015 competitions, where we also won the 1st places on the tasks of ImageNet detection, ImageNet localization, COCO detection, and COCO segmentation."
    ]

    abstracts_t = [tokenizer(p, return_tensors="pt").to("cuda:0") for p in abstracts]

    titles = []

    for abstract in abstracts_t:
        output = model.generate(**abstract, max_new_tokens=50, do_sample=True)
        titles.append(tokenizer.decode(output[0], skip_special_tokens=True))

    for (abstract, title) in zip(abstracts, titles):
        print(f"=" * 20)
        print(abstracts)
        print("\n")
        print(titles)
        print(f"=" * 20)




        
if __name__ == "__main__":
    main()

