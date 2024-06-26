# -*- coding: utf-8 -*-
"""Abstract_to_title_vanilla_rnn.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1gigMwf8IsTDp82MeNum4WRUhIahDLdZC

Abstract to title NLP model

## 1. Load and Preprocess Data
"""

!gdown 1qYdSlDJ89AvgozK3V5tik8Op93zPbG6e -O processed_CMDC.pkl
!pip install rouge_score

from google.colab import drive
import os
drive.mount('/content/drive')
os.chdir('/content/drive/MyDrive/datasetNLP') #change this to the directory where the data is
print(os.getcwd())

# ===========================================================================
# Run some setup code for this notebook. Don't modify anything in this cell.
# ===========================================================================

import csv, random, re, os, math, pickle, statistics, tqdm, numpy as np
from io import open
from google.colab import files
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.jit import trace
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# ===========================================================================
# A quick note on CUDA functionality (and `.to(model.device)`):
# CUDA is a parallel GPU platform produced by NVIDIA and is used by most GPU
# libraries in PyTorch. CUDA organizes GPUs into device IDs (i.e., "cuda:X" for GPU #X).
# "device" will tell PyTorch which GPU (or CPU) to place an object in. Since
# collab only uses one GPU, we will use 'cuda' as the device if a GPU is available
# and the CPU if not. You will run into problems if your tensors are on different devices.
# ===========================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Select the Runtime > "Change runtime type" menu to enable a GPU accelerator, ')
  print('and then re-execute this cell.')
else:
  print(gpu_info)

"""### 1.1 Preparing Data"""

def print_list(l, K=None):
	for i, e in enumerate(l):
		if i == K:
			break
		print(e)
	print()

def load_from_pickle(pickle_file):
	with open(pickle_file, "rb") as pickle_in:
		return pickle.load(pickle_in)

cols = ['id', 'title', 'abstract', 'categories']
data = []
file_name = './arxiv-metadata-oai-snapshot.json'

# Open the file and read data
with open(file_name, encoding='latin-1') as f:
    count = 0
    for line in f:
        doc = json.loads(line)
        lst = [doc['id'], doc['title'], doc['abstract'], doc['categories']]
        data.append(lst)
        count += 1
        # Read 11,000 entries
        if count >= 11000:
            break

# Create a DataFrame from the data list
df = pd.DataFrame(data=data, columns=cols)
# df = df.sample(frac=1, random_state=68).reset_index(drop=True)


train_df = df.iloc[:10000]  #
test_df = df.iloc[10000:]


print("Training DataFrame:")
print(train_df.head())
print("\nTesting DataFrame:")
print(test_df.head())

pad_word = "<pad>"
bos_word = "<s>"
eos_word = "</s>"
unk_word = "<unk>"
pad_id = 0
bos_id = 1
eos_id = 2
unk_id = 3

def normalize_sentence(s):
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

class Vocabulary:
    def __init__(self):
        self.word_to_id = {pad_word: pad_id, bos_word: bos_id, eos_word:eos_id, unk_word: unk_id}
        self.word_count = {}
        self.id_to_word = {pad_id: pad_word, bos_id: bos_word, eos_id: eos_word, unk_id: unk_word}
        self.num_words = 4

    def get_ids_from_sentence(self, sentence):
        sentence = normalize_sentence(sentence)
        sent_ids = [bos_id] + [self.word_to_id[word] if word in self.word_to_id \
                               else unk_id for word in sentence.split()] + \
                               [eos_id]
        return sent_ids

    def tokenized_sentence(self, sentence):
        sent_ids = self.get_ids_from_sentence(sentence)
        return [self.id_to_word[word_id] for word_id in sent_ids]

    def decode_sentence_from_ids(self, sent_ids):
        words = list()
        for i, word_id in enumerate(sent_ids):
            if word_id in [bos_id, eos_id, pad_id]:
                # Skip these words
                continue
            else:
                words.append(self.id_to_word[word_id])
        return ' '.join(words)

    def add_words_from_sentence(self, sentence):
        sentence = normalize_sentence(sentence)
        for word in sentence.split():
            if word not in self.word_to_id:
                # add this word to the vocabulary
                self.word_to_id[word] = self.num_words
                self.id_to_word[self.num_words] = word
                self.word_count[word] = 1
                self.num_words += 1
            else:
                # update the word count
                self.word_count[word] += 1
abstracts_only = []
titles_only = []
vocab = Vocabulary()
for ids, title, abstract, categories in data:
    abstracts_only.append(abstract)
    titles_only.append(title)
for index, row in train_df.iterrows():
    vocab.add_words_from_sentence(row['abstract'])
    vocab.add_words_from_sentence(row["title"])
print(f"Total words in the vocabulary = {vocab.num_words}")
print(len(titles_only))

for ids, title, abstract, categories in data[:2]:
    sentence = abstract
    word_tokens = vocab.tokenized_sentence(sentence)

    # Automatically adds bos_id and eos_id before and after sentence ids respectively
    word_ids = vocab.get_ids_from_sentence(sentence)
    print(sentence)
    print(word_tokens)
    print(word_ids)
    print(vocab.decode_sentence_from_ids(word_ids))
    print()

word = "the"
word_id = vocab.word_to_id[word]
print(f"Word = {word}")
print(f"Word ID = {word_id}")
print(f"Word decoded from ID = {vocab.decode_sentence_from_ids([word_id])}")

"""### 1.3 Dataset Preparation"""

class ABS_Title_Data(Dataset):
    """Single-Turn version of Cornell Movie Dialog Cropus dataset."""

    def __init__(self, abstracts,titles,  vocab, device):
        """
        Args:
            conversations: list of tuple (src_string, tgt_string)
                         - src_string: String of the source sentence
                         - tgt_string: String of the target sentence
            vocab: Vocabulary object that contains the mapping of
                    words to indices
            device: cpu or cuda
        """
        self.abstract_title = list(zip(abstracts, titles))
        self.vocab = vocab
        self.device = device

        def encode(src, tgt):
            src_ids = self.vocab.get_ids_from_sentence(src)
            tgt_ids = self.vocab.get_ids_from_sentence(tgt)
            return (src_ids, tgt_ids)

        # We will pre-tokenize the conversations and save in id lists for later use
        self.tokenized_conversations = [encode(src, tgt) for src, tgt in self.abstract_title]

    def __len__(self):
        return len(self.abstract_title)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return {"conv_ids":self.tokenized_conversations[idx], "conv":self.abstract_title[idx]}

def collate_fn(data):

    # Sort conv_ids based on decreasing order of the src_lengths.
    # This is required for efficient GPU computations.
    src_ids = [torch.LongTensor(e["conv_ids"][0]) for e in data]
    tgt_ids = [torch.LongTensor(e["conv_ids"][1]) for e in data]
    src_str = [e["conv"][0] for e in data]
    tgt_str = [e["conv"][1] for e in data]
    data = list(zip(src_ids, tgt_ids, src_str, tgt_str))
    data.sort(key=lambda x: len(x[0]), reverse=True)
    src_ids, tgt_ids, src_str, tgt_str = zip(*data)

    ### BEGIN YOUR CODE ###

    # Pad the src_ids and tgt_ids using token pad_id to create src_seqs and tgt_seqs
    src_seqs = nn.utils.rnn.pad_sequence(src_ids, padding_value=0)

    tgt_seqs = nn.utils.rnn.pad_sequence(tgt_ids, padding_value=0)

    ### END YOUR CODE ###

    return {"conv_ids":(src_ids, tgt_ids), "conv":(src_str, tgt_str), "conv_tensors":(src_seqs.to(device), tgt_seqs.to(device))}

# Create the DataLoader for all_conversations
dataset = ABS_Title_Data(abstracts_only, titles_only, vocab, device)

batch_size = 64

data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Test one batch of training data
first_batch = next(iter(data_loader))
print(f"Testing first training batch of size {len(first_batch['conv'][0])}")
print(f"List of source strings:")
print_list(first_batch["conv"][0])
print(f"Tokenized source ids:")
print_list(first_batch["conv_ids"][0])
print(f"Padded source ids as tensor (shape {first_batch['conv_tensors'][0].size()}):")
print(first_batch["conv_tensors"][0])

class Seq2seqBaseline(nn.Module):
    def __init__(self, vocab, emb_dim=300, hidden_dim=300, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_words = num_words = vocab.num_words
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_embeddings=num_words, embedding_dim=emb_dim)

        # Encoder with Bidirectional RNN
        self.erNN = nn.RNN(input_size=emb_dim, hidden_size=hidden_dim, num_layers=num_layers,
                           dropout=dropout, bidirectional=True)

        # Decoder RNN uses doubled hidden size from bidirectional encoder
        DH = hidden_dim * 2  # Adjusted hidden dimension for decoder
        self.drNN = nn.RNN(input_size=emb_dim, hidden_size=DH, num_layers=num_layers,
                           dropout=dropout, bidirectional=False)

        # Output layer remains the same
        self.linear = nn.Linear(DH, num_words)

    def encode(self, source, pad_id):
        source_lengths = torch.sum((source != pad_id).int(), dim=0).cpu()
        mask = (source != pad_id)
        w2i = self.embedding(source)
        packedbatch = nn.utils.rnn.pack_padded_sequence(w2i, source_lengths.cpu())
        packedO, hidden = self.erNN(packedbatch)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packedO, batch_first=True)
        outputs = outputs[:, :, :self.hidden_dim] + outputs[:, :, self.hidden_dim:]
        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_dim)
        hidden = torch.cat((hidden[:, 0, :, :], hidden[:, 1, :, :]), dim=2)
        return outputs, mask, hidden

    def decode(self, decoder_input, last_hidden):
        output, hidden = None, None
        embed = self.embedding(decoder_input)
        gOut, hidden = self.drNN(embed, last_hidden)
        output = self.linear(gOut.squeeze(0))
        return output, hidden

    def compute_loss(self, source, target):
        loss = 0
        nonPadTok = 0
        loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, reduction='none')
        eout, emask, ehidden = self.encode(source, pad_id)
        din = torch.full((1, source.size(1)), bos_id, device=source.device)
        dhidden = ehidden
        for i in range(target.size(0)):
            dout, dhidden = self.decode(din, dhidden)
            currLoss = loss_fn(dout, target[i])
            m = target[i] != pad_id
            loss += currLoss[m].sum()
            nonPadTok += m.sum()
            din = target[i].unsqueeze(0)
        loss = loss / nonPadTok
        return loss

def train(model, data_loader, num_epochs, model_file, learning_rate=0.0001):
    decoder_learning_ratio = 5.0
    encoder_parameter_names = ['embedding', 'egru']
    encoder_named_params = list(filter(lambda kv: any(key in kv[0] for key in encoder_parameter_names), model.named_parameters()))
    decoder_named_params = list(filter(lambda kv: not any(key in kv[0] for key in encoder_parameter_names), model.named_parameters()))
    encoder_params = [e[1] for e in encoder_named_params]
    decoder_params = [e[1] for e in decoder_named_params]
    optimizer = torch.optim.AdamW([
        {'params': encoder_params},
        {
            'params': decoder_params,
            'lr': learning_rate * decoder_learning_ratio
        }
    ], lr = learning_rate)

    clip = 50.0
    for epoch in tqdm.trange(num_epochs, desc="training", unit="epoch"):
        with tqdm.tqdm(data_loader, desc=f"epoch {epoch + 1}", unit="batch", total=len(data_loader), position=0, leave=True) as batch_iterator:
            model.train()
            total_loss = 0.0
            for i, batch_data in enumerate(batch_iterator, start=1):
                source, target = batch_data["conv_tensors"]
                optimizer.zero_grad()
                loss = model.compute_loss(source, target)
                total_loss += loss.item()
                loss.backward()
                _ = nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

                batch_iterator.set_postfix(mean_loss=total_loss / i, current_loss=loss.item())
    torch.save(model.state_dict(), model_file)

num_epochs = 10
baseline_model = Seq2seqBaseline(vocab).to(device)
train(baseline_model, data_loader, num_epochs, "baseline_model.pt")
files.download('baseline_model.pt')

baseline_model = Seq2seqBaseline(vocab).to(device)
baseline_model.load_state_dict(torch.load("baseline_model.pt", map_location=device))

print(baseline_model)

def predict_greedy(model, sentence, max_length=100):
    model.eval()
    generation = None
    generation = vocab.get_ids_from_sentence(sentence)
    sentence_ids = vocab.get_ids_from_sentence(sentence)
    sentence_tensor = torch.tensor(sentence_ids, dtype=torch.long).unsqueeze(1)
    sentence_tensor = sentence_tensor.to(device)
    with torch.no_grad():
      eout, emask, ehidden = model.encode(sentence_tensor, pad_id)
      tensorInput = [bos_id]
      din = torch.tensor(tensorInput,device= device)
      dhidden = ehidden
      dwords = []
      for i in range(max_length):
        din = din.unsqueeze(1)
        dout, dhidden = model.decode(din, dhidden)
        topValues, topi = dout.topk(1)
        if topi.item() == eos_id:
          break
        else:
          dwords.append(topi.item())
        din = topi.squeeze().detach()
        din = din.unsqueeze(0)

      generation = ' '.join([vocab.id_to_word[index] for index in dwords])
      generation = vocab.decode_sentence_from_ids(dwords)
    return generation

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
first_five_pairs = dataset.abstract_title[10000:10100]
rouge_scores = []
bleu_scores = []
total_rouge1 = total_rouge2 = total_rougeL = total_rougeLsum = 0
total_bleu = 0
for abstract, reference_title in first_five_pairs:
    generated_title = predict_greedy(baseline_model, abstract, max_length=100)
    rouge_score = scorer.score(reference_title, generated_title)
    rouge_scores.append(rouge_score)
    total_rouge1 += rouge_score['rouge1'].fmeasure
    total_rouge2 += rouge_score['rouge2'].fmeasure
    total_rougeL += rouge_score['rougeL'].fmeasure
    total_rougeLsum += rouge_score['rougeLsum'].fmeasure
    bleu_score = sentence_bleu([reference_title.split()], generated_title.split(), smoothing_function=SmoothingFunction().method1)
    bleu_scores.append(bleu_score)
    total_bleu += bleu_score
average_rouge1 = total_rouge1 / len(first_five_pairs)
average_rouge2 = total_rouge2 / len(first_five_pairs)
average_rougeL = total_rougeL / len(first_five_pairs)
average_rougeLsum = total_rougeLsum / len(first_five_pairs)
average_bleu = total_bleu / len(first_five_pairs)
print("Average ROUGE-1 Score:", average_rouge1)
print("Average ROUGE-2 Score:", average_rouge2)
print("Average ROUGE-L Score:", average_rougeL)
print("Average ROUGE-Lsum Score:", average_rougeLsum)
print("Average BLEU Score:", average_bleu)