# -*- coding: utf-8 -*-
"""Untitled3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1gbiCRqj1WN6XaSzQ3HtlLb4pG76jmd6B
"""

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %cd "/content/drive/MyDrive/CS7650_Final_Project"
# !ls
# %cd "../../../../"

import torch
!pip install datasets
!pip install transformers

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import csv, random, re, os, math, pickle, statistics, tqdm, numpy as np
from io import open
from google.colab import files

import torch
import torch.nn as nn
import torch.nn.functional as F
from queue import PriorityQueue
import operator
from collections import Counter
from torch import optim
from torch.jit import trace
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Select the Runtime > "Change runtime type" menu to enable a GPU accelerator, ')
  print('and then re-execute this cell.')
else:
  print(gpu_info)

print(f'GPU available: {torch.cuda.is_available()}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from datasets import load_dataset
from transformers import AutoTokenizer

!unzip "archive.zip"
# dataset_json = load_dataset('json',data_files='arxiv-metadata.json')

def print_list(l, K=None):
	for i, e in enumerate(l):
		if i == K:
			break
		print(e)
	print()

dataset_json = load_dataset('json',data_files='arxiv-metadata-oai-snapshot.json',split="train")
# dataset_json 2463961
abstract = dataset_json["abstract"]
titles = dataset_json["title"]
# print(dataset_json['train']['title'][0])

# torch.save(all_conversations,"all_conversations.hf")
# torch.save(abstract,"data_abstracts.hf")
# torch.save(titles,"data_titles.hf")
all_conversations = torch.load("all_conversations.hf")

all_conversations = []
for j in range(len(abstract)):
  all_conversations.append((abstract[j],titles[j]))



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

# vocab = Vocabulary()
# for src, tgt in all_conversations:
#     vocab.add_words_from_sentence(src)
#     vocab.add_words_from_sentence(tgt)
# print(f"Total words in the vocabulary = {vocab.num_words}")

# torch.save(vocab,"vocab.hf")
vocab = torch.load("vocab.hf")
print(vocab.num_words)

class SingleTurnMovieDialog_dataset(Dataset):
    """Single-Turn version of Cornell Movie Dialog Cropus dataset."""

    def __init__(self, conversations, vocab, device):
        """
        Args:
            conversations: list of tuple (src_string, tgt_string)
                         - src_string: String of the source sentence
                         - tgt_string: String of the target sentence
            vocab: Vocabulary object that contains the mapping of
                    words to indices
            device: cpu or cuda
        """
        self.conversations = conversations
        self.vocab = vocab
        self.device = device

        def encode(src, tgt):
            src_ids = self.vocab.get_ids_from_sentence(src)
            tgt_ids = self.vocab.get_ids_from_sentence(tgt)
            return (src_ids, tgt_ids)

        # We will pre-tokenize the conversations and save in id lists for later use
        self.tokenized_conversations = [encode(src, tgt) for src, tgt in self.conversations]

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return {"conv_ids":self.tokenized_conversations[idx], "conv":self.conversations[idx]}

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq).
    We should build a custom collate_fn rather than using default collate_fn,
    because merging sequences (including padding) is not supported in default.
    Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).
    Args:
        data: list of dicts {"conv_ids":(src_ids, tgt_ids), "conv":(src_str, trg_str)}.
            - src_ids: list of src piece ids; variable length.
            - tgt_ids: list of tgt piece ids; variable length.
            - src_str: String of src
            - tgt_str: String of tgt
    Returns: dict { "conv_ids":     (src_ids, tgt_ids),
                    "conv":         (src_str, tgt_str),
                    "conv_tensors": (src_seqs, tgt_seqs)}
            src_seqs: torch tensor of shape (src_padded_length, batch_size).
            trg_seqs: torch tensor of shape (tgt_padded_length, batch_size).
            src_padded_length = length of the longest src sequence from src_ids
            tgt_padded_length = length of the longest tgt sequence from tgt_ids

    Implementation tip: You can use the nn.utils.rnn.pad_sequence utility
    function to combine a list of variable-length sequences with padding.
    """
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

    src_seqs = pad_sequence(src_ids)
    tgt_seqs = pad_sequence(tgt_ids)

    ### END YOUR CODE ###

    return {"conv_ids":(src_ids, tgt_ids), "conv":(src_str, tgt_str), "conv_tensors":(src_seqs.to(device), tgt_seqs.to(device))}

# Create the DataLoader for all_conversations
ds = SingleTurnMovieDialog_dataset(all_conversations[:int(len(all_conversations)/480)], vocab, device)

# batch_size = 5

# data_loader = DataLoader(dataset=ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Test one batch of training data
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
    def __init__(self, vocab, emb_dim = 300, hidden_dim = 300, num_layers = 2, dropout=0.1):
        """
        Initialize your model's parameters here. To get started, we suggest
        setting all embedding and hidden dimensions to 300, using encoder and
        decoder GRUs with 2 layers, and using a dropout rate of 0.1.

        Implementation tip: To create a bidirectional GRU, you don't need to
        create two GRU networks. Instead use nn.GRU(..., bidirectional=True).
        """
        super().__init__()

        self.num_words = num_words = vocab.num_words
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        ### BEGIN YOUR CODE ###
        self.GRU = nn.GRU(self.emb_dim,self.hidden_dim,self.num_layers,batch_first=False,dropout=dropout,bidirectional=True).cuda()
        self.GRU_dec = nn.GRU(self.hidden_dim,self.hidden_dim,self.num_layers,dropout=dropout).cuda()
        self.embeddings = nn.Embedding(self.num_words,self.emb_dim).cuda()
        self.Linear = nn.Linear(self.hidden_dim,self.num_words).cuda()
        self.dropout_rate = dropout
        # self.logSM = nn.LogSoftmax(dim=-1)

        # self.criterion = nn.NLLLoss()
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        ### END YOUR CODE ###

    def encode(self, source):
        """Encode the source batch using a bidirectional GRU encoder.

        Args:
            source: An integer tensor with shape (max_src_sequence_length,
                batch_size) containing subword indices for the source sentences.

        Returns:
            A tuple with three elements:
                encoder_output: The output hidden representation of the encoder
                    with shape (max_src_sequence_length, batch_size, hidden_size).
                    Can be obtained by adding the hidden representations of both
                    directions of the encoder bidirectional GRU.
                encoder_mask: A boolean tensor with shape (max_src_sequence_length,
                    batch_size) indicating which encoder outputs correspond to padding
                    tokens. Its elements should be True at positions corresponding to
                    padding tokens and False elsewhere.
                encoder_hidden: The final hidden states of the bidirectional GRU
                    (after a suitable projection) that will be used to initialize
                    the decoder. This should be a tensor h_n with shape
                    (num_layers, batch_size, hidden_size). Note that the hidden
                    state returned by the bi-GRU cannot be used directly. Its
                    initial dimension is twice the required size because it
                    contains state from two directions.

        The first two return values are not required for the baseline model and will
        only be used later in the attention model. If desired, they can be replaced
        with None for the initial implementation.

        Implementation tip: consider using packed sequences to more easily work
        with the variable-length sequences represented by the source tensor.
        See https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.PackedSequence.

        https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch

        Implementation tip: there are many simple ways to combine the forward
        and backward portions of the final hidden state, e.g. addition, averaging,
        or a linear transformation of the appropriate size. Any of these
        should let you reach the required performance.
        """
        # Compute a tensor containing the length of each source sequence.
        source_lengths = torch.sum(source != pad_id, axis=0).cpu()

        ### BEGIN YOUR CODE ###

        # Compute the mask first
        mask = encoder_mask = (source != pad_id)

        # Convert word indexes to embeddings
        embeds = self.embeddings(source.cuda())
        # embeds = F.dropout(embeds,self.dropout_rate).cuda()

        # Pack padded batch of sequences for RNN module
        packed_seqs = pack_padded_sequence(embeds,source_lengths,enforce_sorted=False).cuda()

        # Forward pass through GRU
        packed_outputs, hn = self.GRU(packed_seqs)

        # Unpack padding
        unpacked_outputs, unpacked_lengths = pad_packed_sequence(packed_outputs)

        # Sum bidirectional GRU outputs
        # outputs = unpacked_outputs.view(source.shape[0],source.shape[1],self.num_layers,self.hidden_dim)
        outputs = torch.reshape(unpacked_outputs,(source.shape[0],source.shape[1],2,self.hidden_dim)).cuda()
        outputs = torch.sum(outputs,dim=-2).cuda()
        outputs.cpu()

        hn = torch.reshape(hn,(2,self.num_layers,source.shape[1],-1)).cuda()
        hidden = torch.sum(hn,dim=0).cuda()
        hidden.cpu()

        ### END YOUR CODE ###

        return outputs, mask, hidden

    def decode(self, decoder_input, last_hidden, encoder_output, encoder_mask):
        """Run the decoder GRU for one decoding step from the last hidden state.

        The third and fourth arguments are not used in the baseline model, but are
        included for compatibility with the attention model in the next section.

        Args:
            decoder_input: An integer tensor with shape (1, batch_size) containing
                the subword indices for the current decoder input.
            last_hidden: A pair of tensors h_{t-1} representing the last hidden
                state of the decoder, each with shape (num_layers, batch_size,
                hidden_size). For the first decoding step the last_hidden will be
                encoder's final hidden representation.
            encoder_output: The output of the encoder with shape
                (max_src_sequence_length, batch_size, hidden_size).
            encoder_mask: The output mask from the encoder with shape
                (max_src_sequence_length, batch_size). Encoder outputs at positions
                with a True value correspond to padding tokens and should be ignored.

        Returns:
            A tuple with three elements:
                logits: A tensor with shape (batch_size,
                    vocab_size) containing unnormalized scores for the next-word
                    predictions at each position.
                decoder_hidden: tensor h_n with the same shape as last_hidden
                    representing the updated decoder state after processing the
                    decoder input.
                attention_weights: This will be implemented later in the attention
                    model, but in order to maintain compatible type signatures, we also
                    include it here. This can be None or any other placeholder value.
        """
        # These arguments are not used in the baseline model.
        del encoder_output
        del encoder_mask

        output, hidden = None, None

        ### BEGIN YOUR CODE ###

        # First process the decoder_input via embedding layer
        dec_embeds = self.embeddings(decoder_input.type(torch.LongTensor).cuda())
        dec_embeds = F.dropout(dec_embeds,self.dropout_rate).cuda()

        # print(dec_embeds.shape, last_hidden.shape)

        # Forward through unidirectional GRU
        dec_outs, hidden = self.GRU_dec(dec_embeds,last_hidden)
        hidden.cpu()
        output = self.Linear(torch.squeeze(dec_outs,0).cuda())
        # print(output.shape)
        output.cpu()
        # output = torch.squeeze(output,0)
        ### END YOUR CODE ###

        return output, hidden, None

    def compute_loss(self, source, target):
        """Run the model on the source and compute the loss on the target.

        Args:
            source: An integer tensor with shape (max_source_sequence_length,
                batch_size) containing subword indices for the source sentences.
            target: An integer tensor with shape (max_target_sequence_length,
                batch_size) containing subword indices for the target sentences.

        Returns:
            A scalar float tensor representing cross-entropy loss on the current batch
            divided by the number of target tokens in the batch.
            Many of the target tokens will be pad tokens. You should mask the loss
            from these tokens using appropriate mask on the target tokens loss.

        Implementation tip: don't feed the target tensor directly to the decoder.
        To see why, note that for a target sequence like <s> A B C </s>, you would
        want to run the decoder on the prefix <s> A B C and have it predict the
        suffix A B C </s>.

        You may run self.encode() on the source only once and decode the target
        one step at a time.
        """

        loss = 0

        ### BEGIN YOUR CODE ###

        # Forward pass through encoder
        encoder_outs, encoder_mask, encoder_hiddens = self.encode(source)

        # Create initial decoder input (start with SOS tokens for each sentence)
        decoder_in = torch.empty(1,source.shape[-1]).fill_(bos_id)

        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hiddens
        target_mask = (target != pad_id).type(torch.LongTensor)
        real_tokens = 0

        # contr_loss = 0
        # Forward batch of sequences through decoder one time step at a time
        for i in range(1,target.shape[0]):
          decoder_outs, decoder_hidden, _ = self.decode(decoder_in,decoder_hidden,encoder_outs,encoder_mask)

            # Teacher forcing: next input is current target
          if target is not None:
            # print(target[i,:].shape,source.shape,target.shape)
            decoder_in = target[i,:].unsqueeze(0)

          else:
            _, top_id = decoder_outs.topk(1,dim=-1)
            decoder_in = top_id.unsqueeze(0).detach()

            # Calculate and accumulate loss
          loss_1 = self.criterion(decoder_outs.view(-1,decoder_outs.size(-1)),target[i,:])
          # loss_2 = loss_func(decoder_outs.view(-1,decoder_outs.size(-1)),target[i,:])
          # print(loss_1.shape, loss_2)
          # contr_loss += loss_2
          loss_w_mask = target_mask[i,:].cuda() * loss_1
          # loss.backward()
          loss += loss_w_mask.sum()
          real_tokens += target_mask[i,:].sum()
        loss /= real_tokens

        # decoder_out (batch_size,vocab_size)  target(max_seq_lenth,batch_size)
        #NLLoss: input(n,classes)  target(n)
        ### END YOUR CODE ###

        return loss

class Seq2seqAttention(Seq2seqBaseline):
    def __init__(self, vocab):
        """
        Initialize any additional parameters needed for this model that are not
        already included in the baseline model.
        """
        super().__init__(vocab)

        ### BEGIN YOUR CODE ###
        self.Linear2 = nn.Linear(2*self.hidden_dim,self.num_words).cuda()




        ### END YOUR CODE ###

    def decode(self, decoder_input, last_hidden, encoder_output, encoder_mask):
        """
        Run the decoder GRU for one decoding step from the last hidden state.

        The third and fourth arguments are not used in the baseline model, but are
        included for compatibility with the attention model in the next section.

        Args:
            decoder_input: An integer tensor with shape (1, batch_size) containing
                the subword indices for the current decoder input.
            last_hidden: A pair of tensors h_{t-1} representing the last hidden
                state of the decoder, each with shape (num_layers, batch_size,
                hidden_size). For the first decoding step the last_hidden will be
                encoder's final hidden representation.
            encoder_output: The output of the encoder with shape
                (max_src_sequence_length, batch_size, hidden_size).
            encoder_mask: The output mask from the encoder with shape
                (max_src_sequence_length, batch_size). Encoder outputs at positions
                with a True value correspond to padding tokens and should be ignored.

        Returns:
            A tuple with three elements:
                logits: A tensor with shape (batch_size,
                    vocab_size) containing unnormalized scores for the next-word
                    predictions at each position.
                decoder_hidden: tensor h_n with the same shape as last_hidden
                    representing the updated decoder state after processing the
                    decoder input.
                attention_weights: A tensor with shape (batch_size,
                    max_src_sequence_length) representing the normalized
                    attention weights. This should sum to 1 along the last dimension.
        """
        output, hidden, attn_weights = None, None, None

        ### BEGIN YOUR CODE ###

        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        dec_embeds = self.embeddings(decoder_input.type(torch.LongTensor).cuda())

        # Forward through unidirectional GRU
        dec_outs, hidden = self.GRU_dec(dec_embeds,last_hidden)
        # print(dec_outs.shape)
        hidden.cpu()
        # output = self.Linear(torch.squeeze(dec_outs,0).cuda())
        # output.cpu()
        # Calculate attention weights from the current GRU output
        # encoder_output = self.attn(encoder_output)
        att_to_dec = torch.sum(encoder_output * dec_outs,dim=-1).cuda()
        # print(encoder_output.shape,att_to_dec.shape)
        att_logits = F.softmax(att_to_dec,dim=0).cuda().transpose(0,1).cpu()
        attn_weights = torch.clone(att_logits.cuda()).cuda().cpu()
        att_logits = att_logits.unsqueeze(1)




        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context_wt = torch.bmm(att_logits.cuda(),encoder_output.cuda().transpose(0,1))
        # Concatenate weighted context vector and GRU output
        output = self.Linear2(torch.cat((dec_outs.cuda(),context_wt.transpose(0,1).cuda()),dim=-1).cuda()).squeeze()
        # print(output.shape)
        output.cpu()
        # print(output.shape)
        ### END YOUR CODE ###

        return output, hidden, attn_weights

def train(model, data_loader, num_epochs, model_file, learning_rate=0.0001):
    """
    Train the model for given number of epochs and save the trained model in
    the final model_file.
    """
    decoder_learning_ratio = 5.0
    # step_down = 0.6
    # dec_lr = 0.00025
    # enc_lr = 0.0025

    ### BEGIN YOUR CODE ###

    encoder_parameter_names = ['embeddings', 'GRU'] # <- Add a list of encoder parameter names here!

    ### END YOUR CODE ###

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

    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.04)
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

                # Gradient clipping before taking the step
                _ = nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

                batch_iterator.set_postfix(mean_loss=total_loss / i, current_loss=loss.item())

        scheduler.step()

    # Save the model after training
    torch.save(model.state_dict(), model_file)

# You are welcome to adjust these parameters based on your model implementation.
num_epochs = 6
batch_size = 64

# Reloading the data_loader to increase batch_size
data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
baseline_model = Seq2seqBaseline(vocab).to(device)
train(baseline_model, data_loader, num_epochs, "baseline_model.pt",0.00025)

# Download the trained model to local for future use
files.download('baseline_model.pt')

# You are welcome to adjust these parameters based on your model implementation.
num_epochs = 4
batch_size = 128
leng = len(ds)
# del data_loader
data_loader = DataLoader(dataset=ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
# attention_model = Seq2seqAttention(vocab).to(device)
train(attention_model, data_loader, num_epochs, "attention_model.hf",.0025)

# Download the trained model to local for future use
# files.download('attention_model.pt')



def predict_greedy(model, sentence, max_length=100):
    """
    Make predictions for the given input using greedy inference.

    Args:
        model: A sequence-to-sequence model.
        sentence: A input string.
        max_length: The maximum length at which to truncate outputs in order to
            avoid non-terminating inference.

    Returns:
        Model's predicted greedy response for the input, represented as string.
    """

    # You should make only one call to model.encode() at the start of the function,
    # and make only one call to model.decode() per inference step.
    model.eval()

    generation = None

    ### BEGIN YOUR CODE ###

    # Forward input through encoder model
    sent_ids = torch.tensor(vocab.get_ids_from_sentence(sentence)).unsqueeze(-1)
    encoder_outs, encoder_mask, encoder_hiddens = model.encode(sent_ids)

    # Prepare encoder's final hidden layer to be first hidden input to the decoder
    last_hidden = encoder_hiddens

    # Initialize decoder input with SOS_token
    decoder_in = torch.empty(1,sent_ids.shape[-1]).fill_(bos_id)

    # Initialize tensors to append decoded words to
    dec_words = []

    # Iteratively decode one word token at a time
    for i in range(len(sent_ids)):


        # Forward pass through decoder
        decoder_outs, decoder_hidden, _ = model.decode(decoder_in,last_hidden,encoder_outs,encoder_mask)

        # Obtain most likely word token and its softmax score
        sm_scores = F.log_softmax(decoder_outs,dim=-1)
        # print(sent_ids)
        cur_score, curr_id = torch.max(sm_scores,dim=-1)
        # print(curr_id.tolist())
        # print(curr_id)
        # print(torch.is_tensor(curr_id))
        # if(not (torch.is_tensor(curr_id))):
        #   print(torch.is_tensor(curr_id))
        #   curr_id = curr_id.unsqueeze(0)
        # print(curr_id.shape)
        curr_id = torch.tensor([curr_id.item()])
        # curr_id = curr_id.unsqueeze(0)
        # print(decoder_hidden.shape,last_hidden.shape)
        dec_words += (curr_id).tolist()
        # temp = torch.tensor([4,5,6,7,8,9,10])
        # print(temp[-3:])
        # print(curr_id)

        # Record token and score

        # Prepare current token to be next decoder input (add a dimension)
        if curr_id.item() == eos_id:
          break
        decoder_in = curr_id.unsqueeze(-1)
        # print(curr_id.shape)
        last_hidden = decoder_hidden
    # Return collections of word tokens and scores
    generation = vocab.decode_sentence_from_ids([bos_id]+dec_words)
    ### END YOUR CODE ###

    return generation

attention_model = Seq2seqAttention(vocab).to(device)
attention_model.load_state_dict(torch.load("attention_model.hf", map_location=device))
# PROMPT = all_conversations[-100][0]
# print(PROMPT)
# print(f'Greedy decoding:\t{predict_greedy(attention_model, PROMPT)}\n')

# !pip install pytorch_metric_learning
# !pip install evaluate
# !pip install bleu
# !pip install rouge
!pip install rouge_score

from pytorch_metric_learning import losses
import evaluate

# loss_func = losses.NTXentLoss(temperature=0.5)
loss_func = losses.ContrastiveLoss()

# from rouge_score import rouge_scorer, scoring
import evaluate

rouge = evaluate.load('rouge')
print(rouge)
PROMPTs = all_conversations[-1000:-600]
print(len(PROMPTs))
refs = [sent[1] for sent in PROMPTs]
cands = []
for sents in PROMPTs:
  cands.append(predict_greedy(attention_model, sents[0]))
# torch.save(cands,"candidates3.hf")
rouge_results = rouge.compute(predictions=cands,references=refs)
print(rouge_results)
bleu = evaluate.load('bleu')
bleu_results = bleu.compute(predictions=cands,references=refs)
print(bleu_results)
