import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import copy
import logging
import os
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader

def logger_init(log_file_name='monitor',
                log_level=logging.DEBUG,
                log_dir='data/log/',
                only_file=False):
    #
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path=os.path.join(log_dir,log_file_name+'_'+str(datetime.now())[:10]+'.txt')
    formatter='[%(asctime)s] - %(levelname)s: %(message)s'
    if only_file:
        logging.basicConfig(filename=log_path,
                            level=log_level,
                            format=formatter,
                            datefmt='%Y-%m-%d %H:%M:%S'
                            )
    else:
        logging.basicConfig(
            level=log_level,
            format=formatter,
            handlers=[logging.FileHandler(log_path),
                      logging.StreamHandler(sys.stdout)]
        )

class Vocab:
    UNK='[UNK]'
    def __init__(self,vocab_path):
        self.dict={}
        self.wordlist=[]
        with open(vocab_path,'r',encoding='utf-8') as f:
            for i, word in enumerate(f):
                w=word.strip('\n')
                self.dict[w]=i
                self.wordlist.append(w)

    def __getitem__(self, token):
        return self.dict.get(token,self.dict.get(Vocab.UNK))

    def __len__(self):
        return len(self.wordlist)

def build_vocab(vocab_path):
    return Vocab(vocab_path)

class LoadDataset:
    def __init__(self,
                 vocab_path='bert_pretrained/bert-case-chinese/vocab.txt',
                 tokenizer=None,
                 batch_size=16,
                 max_sen_len=None,
                 split_sep='\t',
                 max_position_embeddings=512,
                 pad_index=0,
                 is_sample_shuffle=True):
        '''
        :param vocab_path:
        :param tokenizer:
        :param batch_size:
        :param max_sen_len:
        :param split_sep:
        :param max_position_embeddings:
        :param pad_index:
        :param is_sample_shuffle:
        '''
        self.tokenizer=tokenizer
        self.vocab=build_vocab(vocab_path)
        self.PAD_IDX=pad_index
        self.SEP_IDX=self.vocab['[SEP]']
        self.CLS_IDX=self.vocab['[CLS]']
        self.batch_size=batch_size
        self.split_sep=split_sep
        self.max_position_embeddings=max_position_embeddings
        if isinstance(max_sen_len,int) and max_sen_len>=max_position_embeddings:
            max_sen_len=max_position_embeddings
        self.max_sen_len=max_sen_len
        self.is_sample_shuffle=is_sample_shuffle

    def sentence_process(self,content):
        data = [self.CLS_IDX] + [self.vocab[token] for token in self.tokenizer(content)]
        seq_len = len(data)
        if seq_len >= self.max_sen_len - 1:
            data = data[:self.max_sen_len - 1]
            data += [self.SEP_IDX]
            seq_len = self.max_sen_len
        else:
            data += [self.SEP_IDX]
            seq_len += 1
            data.extend([0] * (self.max_sen_len - seq_len))
        return seq_len,data

    def data_process(self,filepath):
        postfix = os.path.splitext(filepath)[1]
        if postfix == '.xlsx':
            data = pd.read_excel(filepath, header=1)
            data_column = data.columns.values.tolist()
            if data_column[-1] != 'text':
                data = data.rename(columns={data_column[-1]: 'label'})
                # label=list(data['label'])
                # data['label'].astype('int64')
                data['content'] = data['text'].map(
                    lambda x: x.replace(' ', '').replace('_x000D_', '，').replace('\n', ''))
                data['seq_len'] = data['content'].map(lambda x: self.sentence_process(x)[0])
                data['word_id'] = data['content'].map(lambda x: self.sentence_process(x)[1])
                data['result'] = data.apply(lambda x: (x['word_id'], x['label'], x['seq_len']), axis=1)
                contents = list(data['result'])
            else:
                data['content'] = data['text'].map(
                    lambda x: x.replace(' ', '').replace('_x000D_', '，').replace('\n', ''))
                data['seq_len'] = data['content'].map(lambda x: self.sentence_process(x)[0])
                data['word_id'] = data['content'].map(lambda x: self.sentence_process(x)[1])
                data['result'] = data.apply(lambda x: (x['word_id'], x['seq_len']), axis=1)
                contents = list(data['result'])
        elif postfix == '.csv':
            data = pd.read_csv(filepath)
            data_column = data.columns.values.tolist()
            if data_column[-1] != 'text':
                data.rename(column={data_column[-1]: 'label'})
                # label=list(data['label'])
                data['label'].astype('int64')
                data['content'] = data['text'].map(lambda x: x.replace(' ', '').replace('\r', '，').replace('\n', ''))
                data['seq_len'] = data['content'].map(lambda x: self.sentence_process(x)[0])
                data['word_id'] = data['content'].map(lambda x: self.sentence_process(x)[1])
                data['result'] = data.apply(lambda x: (x['word_id'], x['label'], x['seq_len']), axis=1)
                contents = list(data['result'])
            else:
                data['content'] = data['text'].map(
                    lambda x: x.replace(' ', '').replace('\r', '，').replace('\n', ''))
                data['seq_len'] = data['content'].map(lambda x: self.sentence_process(x)[0])
                data['word_id'] = data['content'].map(lambda x: self.sentence_process(x)[1])
                data['result'] = data.apply(lambda x: (x['word_id'], x['seq_len']), axis=1)
                contents = list(data['result'])
        else:#'txt'
            contents=[]
            with open(filepath,'r',encoding='utf-8') as f:
                raw_data=f.readlines()
            with open(filepath,'r',encoding='utf-8') as f:
                for line in tqdm(raw_data,ncols=80):
                    lin=line.strip().split(self.split_sep)
                    if len(lin)==2:
                        content,label=lin
                    else:
                        content=lin[0]
                    data=[self.CLS_IDX]+[self.vocab[token] for token in self.tokenizer(content)]
                    seq_len=len(data)
                    if seq_len>=self.max_sen_len-1:
                        data=data[:self.max_sen_len-1]
                        data += [self.SEP_IDX]
                        seq_len=self.max_sen_len
                    else:
                        data += [self.SEP_IDX]
                        seq_len+=1
                        data.extend([0]*(self.max_sen_len-seq_len))

                    try:
                        label
                    except NameError:
                        label_exists=False
                    else:
                        label_exists=True
                    if label_exists:
                        contents.append((data,int(label),seq_len))
                    else:
                        contents.append((data,seq_len))
        return contents #[([...],1,12),([...],0,32)]
    def load_data(self,file_path=None,only_test=False):
        data=self.data_process(file_path)
        data=[(torch.tensor(i[0],dtype=torch.long),torch.tensor(int(i[1]),dtype=torch.long)) for i in data]
        if only_test:
            data_iter=DataLoader(data,batch_size=self.batch_size)
        else:
            data_iter=DataLoader(data,batch_size=self.batch_size,shuffle=self.is_sample_shuffle)
        return data_iter

class DataIterater(object):
    def __init__(self,original_data,batch_size,device):
        self.batch_size=batch_size
        self.batches=original_data
        self.n_batches=len(original_data)//batch_size
        self.residue=False
        if len(original_data)%self.n_batches !=0:
            self.residue=True
        self.index=0
        self.device=device

    def _to_tensor(self,data):
        x=torch.LongTensor([item[0] for item in data]).to(self.device)
        if len(data[0])==3:
            y=torch.LongTensor([item[1] for item in data]).to(self.device)
            seq_len=torch.LongTensor([item[2] for item in data]).to(self.device)
            return (x,seq_len),y
        else:
            seq_len = torch.LongTensor([item[2] for item in data]).to(self.device)
            return (x,seq_len)

    def __next__(self):
        if self.residue and self.index==self.n_batches:
            batches=self.batches[self.index*self.batch_size:len(self.batches)]
            self.index+=1
            batches=self._to_tensor(batches)
            return batches
        elif self.index>=self.n_batches:
            self.index=0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size:(self.index+1)*self.batch_size]
            self.index+=1
            batches=self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches+1
        else:
            return self.n_batches

def build_iter(dataset,config):
    return DataIterater(dataset,config.batch_size,config.device)

'''
 class LoadSingleSentenceClassificationDataset:
     def __init__(self,
                  vocab_path='./vocab.txt',  #
                  tokenizer=None,
                  batch_size=32,
                  max_sen_len=None,
                  split_sep='\n',
                  max_position_embeddings=512,
                  pad_index=0,
                  is_sample_shuffle=True):
         self.tokenizer = tokenizer
         self.vocab = build_vocab(vocab_path)
         self.PAD_IDX = pad_index
         self.SEP_IDX = self.vocab['[SEP]']
         self.CLS_IDX = self.vocab['[CLS]']
         self.batch_size = batch_size
         self.split_sep = split_sep
         self.max_position_embeddings = max_position_embeddings
         if isinstance(max_sen_len, int) and max_sen_len > max_position_embeddings:
             max_sen_len = max_position_embeddings
         self.max_sen_len = max_sen_len
         self.is_sample_shuffle = is_sample_shuffle
     def data_process(self, filepath):
         raw_iter = open(filepath, encoding="utf8").readlines()
         data = []
         max_len = 0
         for raw in tqdm(raw_iter, ncols=80):
             line = raw.rstrip("\n").split(self.split_sep)
             s, l = line[0], line[1]
             tmp = [self.CLS_IDX] + [self.vocab[token] for token in self.tokenizer(s)]
             if len(tmp) > self.max_position_embeddings - 1:
                 tmp = tmp[:self.max_position_embeddings - 1]  # BERT预训练模型只取前512个字符
             tmp += [self.SEP_IDX]
             tensor_ = torch.tensor(tmp, dtype=torch.long)
             l = torch.tensor(int(l), dtype=torch.long)
             max_len = max(max_len, tensor_.size(0))
             data.append((tensor_, l))
         return data, max_len        
     def pad_sequence(sequences, batch_first=False, max_len=None, padding_value=0):
         if max_len is None:
             max_len = max([s.size(0) for s in sequences])
         out_tensors = []
         for tensor in sequences:
             if tensor.size(0) < max_len:
                 tensor = torch.cat([tensor, torch.tensor(
                   [padding_value] * (max_len - tensor.size(0)))], dim=0)
             else:
                 tensor = tensor[:max_len]
             out_tensors.append(tensor)
         out_tensors = torch.stack(out_tensors, dim=1)
         if batch_first:
             return out_tensors.transpose(0, 1)
         return out_tensors
     def generate_batch(self, data_batch):
         batch_sentence, batch_label = [], []
         for (sen, label) in data_batch:  # 开始对一个batch中的每一个样本进行处理。
             batch_sentence.append(sen)
             batch_label.append(label)
         batch_sentence = pad_sequence(batch_sentence,  # [batch_size,max_len]
                                       padding_value=self.PAD_IDX,
                                       batch_first=False,
                                       max_len=self.max_sen_len)
         batch_label = torch.tensor(batch_label, dtype=torch.long)
         return batch_sentence, batch_label
     def load_train_val_test_data(self, train_file_path=None,
                                  val_file_path=None,
                                  test_file_path=None,
                                  only_test=False):
         test_data, _ = self.data_process(test_file_path)
         test_iter = DataLoader(test_data, batch_size=self.batch_size,
                                shuffle=False, collate_fn=self.generate_batch)
         if only_test:
             return test_iter
         train_data, max_sen_len = self.data_process(train_file_path)  # 得到处理好的所有样本
         if self.max_sen_len == 'same':
             self.max_sen_len = max_sen_len
         val_data, _ = self.data_process(val_file_path)
         train_iter = DataLoader(train_data, batch_size=self.batch_size,  # 构造DataLoader
                                 shuffle=self.is_sample_shuffle, collate_fn=self.generate_batch)
         val_iter = DataLoader(val_data, batch_size=self.batch_size,
                               shuffle=False, collate_fn=self.generate_batch)
         return train_iter, test_iter, val_iter
        '''
