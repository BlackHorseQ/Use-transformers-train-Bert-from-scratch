##查看训练集和测试集中字
import pandas as pd
from pathlib import Path
from torch import nn
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm
from transformers import BertConfig, BertForPreTraining, BertModel
from torch.utils.data import Dataset, DataLoader
import random
import copy
import numpy as np
import torch
import os
from sklearn.metrics import accuracy_score, roc_auc_score
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import json
from sklearn.model_selection import train_test_split
train = pd.read_csv('./data/gaiic_track3_round1_train_20210228.tsv',sep='\t', names=['text_a', 'text_b', 'label'])
test = pd.read_csv('data/gaiic_track3_round1_testA_20210228.tsv',sep='\t', names=['text_a', 'text_b', 'label'])
test['label'] = 2
from collections import defaultdict
def get_dict(data):
    words_dict = defaultdict(int)
    for i in tqdm(range(data.shape[0])):
        text = data.text_a.iloc[i].split() + data.text_b.iloc[i].split()
        for c in text:
            words_dict[c] += 1
    return words_dict
word_dict = get_dict(train.append(test))
min_count = 5
word_dict =  {i: j for i, j in word_dict.items() if j >= min_count}
word_dict = dict(sorted(word_dict.items(), key=lambda s: -s[1]))
word_dict = list(word_dict.keys())
special_tokens = ["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]", "sim", 'sim_no', 'unlabeled']
WORDS = special_tokens + word_dict
pd.Series(WORDS).to_csv('Bert-vocab.txt', header=False,index=0)
vocab = pd.read_csv('Bert-vocab.txt', names=['word'])
vocab_dict = {}
for key, value in vocab.word.to_dict().items():
    vocab_dict[value] = key
##为了使用初始化embedding
with open('bert-base-chinese-1/vocab.txt', 'r') as f:
    lines = f.read()
    tokens = lines.split('\n')
token_dict = dict(zip(tokens, range(len(tokens))))
counts = json.load(open('counts.json'))
del counts['[CLS]']
del counts['[SEP]']
freqs = [
    counts.get(i, 0) for i, j in sorted(token_dict.items(), key=lambda s: s[1])
]
keep_tokens = list(np.argsort(freqs)[::-1])
keep_tokens = [0, 100, 101, 102, 103, 6, 7, 8] + keep_tokens[:len(vocab_dict)]

class PretrainedBERT(nn.Module):
    def __init__(self, embeding_size=6933, embedding_dim=768, maxlen=64, checkpoint_path='embeddingBert.pth',keep_tokens=None):
        super(PretrainedBERT, self).__init__()
#         self.bert = BertModel(BertConfig(embeding_size))
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.lr = nn.Linear(in_features=embedding_dim, out_features=768)
        self.layer_norm = nn.LayerNorm((maxlen, embedding_dim))
        self.lr1 = nn.Linear(in_features=768, out_features=embeding_size)
        ##使用bert的预训练embedding作为初始化embedding
        if keep_tokens is not None:
            self.embedding = nn.Embedding(embeding_size, embedding_dim)
            weight = torch.load('embeddingBert.pth')
            weight = nn.Parameter(weight['weight'][keep_tokens])
            self.embedding.weight = weight
            self.bert.embeddings.word_embeddings = self.embedding
    def forward(self, x):
        x = self.bert(**x)
        x = self.lr(x['last_hidden_state'])
        x = self.layer_norm(x)
        x = self.lr1(x)
        return x
class BERTDataset(Dataset):
    
    def __init__(self, corpus, vocab:dict, seq_len:int=128): 
        self.vocab = vocab
        self.seq_len = seq_len
        self.lines = corpus
        self.corpus_lines = self.lines.shape[0]
    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, idx):
        t1, t2, is_next_label = self.get_sentence(idx)
        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)
        label = self.lines.label.iloc[idx]
        t1 = [self.vocab['[CLS]']] + t1_random + [self.vocab['[SEP]']]
        t2 = t2_random + [self.vocab['[SEP]']]
        ## 相似为6，不相似为5，未标注为7
        t1_label = [label+5] + t1_label + [self.vocab['[SEP]']]
        t2_label = t2_label + [self.vocab['[SEP]']]

        segment_label = ([0 for _ in range(len(t1))] + [1 for _ in range(len(t2))])[:self.seq_len]
        bert_input = (t1 + t2)[:self.seq_len]
        bert_label = (t1_label + t2_label)[:self.seq_len]

        padding = [self.vocab['[PAD]'] for _ in range(self.seq_len - len(bert_input))]
        padding_label = [-100 for _ in range(self.seq_len - len(bert_input))]
        attention_mask = len(bert_input) * [1] + len(padding) * [0]
        bert_input.extend(padding), bert_label.extend(padding_label), segment_label.extend(padding)
        attention_mask = np.array(attention_mask)
        bert_input = np.array(bert_input)
        segment_label = np.array(segment_label)
        bert_label = np.array(bert_label)
        is_next_label = np.array(is_next_label)
        output = {"input_ids": bert_input,
                  "token_type_ids": segment_label,
                  'attention_mask': attention_mask,
                  "bert_label": bert_label}, label
        return output
    def random_word(self, sentence):
        import random
        tokens = sentence.split()
        output_label = []
        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% 
                if prob < 0.8:
                    tokens[i] = self.vocab['[MASK]']

                # 10%
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))

                # 10% 
                else:
                    tokens[i] = self.vocab.get(token, self.vocab['[UNK]'])

                output_label.append(self.vocab.get(token, self.vocab['[UNK]']))
            else:
                tokens[i] = self.vocab.get(token, self.vocab['[UNK]'])
                output_label.append(-100)
        return tokens, output_label

    def get_sentence(self, idx):
        
        t1, t2, _ = self.lines.iloc[idx].values
        if random.random() > 0.5:
            return t1, t2, 0
        else:
            return t2, t1, 0
train_index, valid_index = train_test_split(range(train.shape[0]), test_size=0.1,random_state=427)

train_data = train.iloc[train_index]
valid_data = train.iloc[valid_index]
pretrain_dataset = BERTDataset(train_data, vocab_dict, 64)
prevalid_dataset = BERTDataset(valid_data, vocab_dict, 64)
pretest_dataset = BERTDataset(test, vocab_dict, 64)
train_loader = DataLoader(pretrain_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(prevalid_dataset, batch_size=32)
def evaluate(model, data_loader,  device='cuda'):
    model.eval()
    losses = []
    labels_list = []
    next_list = []
    pre_list = []
    pbar = tqdm(data_loader)
    data_dict = {}
    for data_label in pbar:
        data = data_label[0]
        data_dict['input_ids'] = data['input_ids'].to(device).long()
        data_dict['token_type_ids'] = data['token_type_ids'].to(device).long()
        data_dict['attention_mask'] =  data['attention_mask'].to(device).long()
        labels = data['bert_label'].to(device).long()
        optim.zero_grad()
        outputs = model(data_dict)
        ## 取出第6个和第7个tokens，因为这个是预留的两个token作为yes 和 no
        preds = outputs[:,0,5:7].cpu().detach().numpy()
        preds = preds[:, 1] / (preds.sum(axis=1) + 1e-8)
        labels = labels[:, 0] - 5
        labels_list.append(labels.cpu().numpy())
        pre_list.append(preds)
    pre_list = np.concatenate(pre_list)
    labels_list = np.concatenate(labels_list)
    auc_score = roc_auc_score(labels_list, pre_list)
    print(f'auc_pre:{auc_score}')
    return auc_score
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = PretrainedBERT(len(vocab_dict), keep_tokens=keep_tokens)
optim = torch.optim.Adam(model.parameters(), lr=2e-5)
criterion  = nn.CrossEntropyLoss()
model = model.to(device)

NUM_EPOCHS = 100
best_auc = 0.5
data_dict = {}
for epoch  in range(NUM_EPOCHS):
    pbar = tqdm(train_loader)
    losses = []
    
    for data_label in pbar:
        data = data_label[0]
        next_sentence_label = data_label[1].to(device).long()

#         data['next_sentence_label'] = next_sentence_label
        data_dict['input_ids'] = data['input_ids'].to(device).long()
        data_dict['token_type_ids'] = data['token_type_ids'].to(device).long()
        data_dict['attention_mask'] =  data['attention_mask'].to(device).long()
        labels = data['bert_label'].to(device).long()
        outputs = model(data_dict)
        mask = (labels!=-100)
        loss = criterion(outputs[mask].view(-1, len(vocab_dict)), labels[mask].view(-1))
        losses.append(loss.cpu().detach().numpy())
        optim.zero_grad()
        loss.backward()
        optim.step()
        pbar.set_description(f'epoch:{epoch} loss:{np.mean(losses):.4f}')
#         break
    valid_auc = evaluate(model,valid_loader)
    print('=*'*50)
    print('valid loss:', loss)
    if valid_auc > best_auc:
        best_auc = valid_auc
        torch.save(model.state_dict(), f'pretrainBERT/preTrainModel{best_auc:.3f}.pth', _use_new_zipfile_serialization=False)
    print('=*'*50)
torch.save(model.state_dict(), f'pretrainBERT/preTrainModel{best_auc:.3f}.pth', _use_new_zipfile_serialization=False)