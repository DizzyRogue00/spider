import os
import sys
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta

MAX_VOCAB_SIZE=10000 # length of dictionary
UNK,PAD="<UNK>","<PAD>" # unkown, padding symbol

def build_vocab(file_path, tokenizer,max_size,min_freq): # construct the vocabuary dictionary
    """
    :param file_path:
    :param tokenizer:
    :param max_size:
    :param min_freq:
    :return:
    """
    vocab_dic={}
    with open(file_path,'r',encoding="UTF-8") as f:
        num_line=sum(1 for line in f)
    with open(file_path,'r',encoding="utf-8") as f:
        for line in tqdm(f,total=num_line):
            lin=line.strip()#remove leading and trailing characters (spaces)
            if not lin:
                continue
            content=lin.split('\t')[0]
            for word in tokenizer(content):
                vocab_dic[word]=vocab_dic.get(word,0)+1 #word frequency
        vocab_list=sorted([item for item in vocab_dic.items() if item[1]>=min_freq], key=lambda x:x[1],reverse=True)[:max_size]
        vocab_dic={word_count[0]:idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK:len(vocab_dic),PAD:len(vocab_dic)+1})
    return vocab_dic

def build_dataset(config,use_word):
    if use_word:
        tokenizer=lambda x:x.split(' ') # word-level
    else:
        tokenizer=lambda x:[y for y in x] # char-level
    if os.path.exists(config.vocab_path):
        vocab_dict=pkl.load(open(config.vocab_path))
    else:
        vocab_dict=build_vocab(config.train_path,tokenizer=tokenizer,max_size=MAX_VOCAB_SIZE,min_freq=1)
        pkl.dump(vocab_dict,open(config.vocab_path,'wb'))
    print(f"Vocabuary size: {len(vocab_dict)}")

    def load_dataset(path,pad_size=32):
        contents=[]
        with open(path,'r',encoding="UTF-8") as f:
            for line in tqdm(f):
                lin=line.strip()
                if not lin:
                    continue
                if len(lin.split('\t'))==2:
                    content,label=lin.split('\t')
                else:
                    content=lin.split('\t')
                token=tokenizer(content)
                seq_len=len(token)
                if seq_len<pad_size:
                    token.extend([PAD]*(pad_size-seq_len))
                else:
                    token=token[:pad_size]
                    seq_len=pad_size
                #word to id
                words=[vocab_dict.get(word,vocab_dict.get(UNK)) for word in token]
                try:
                    label
                except NameError:
                    label_exists=False
                else:
                    label_exists=True
                if label_exists:
                    contents.append((words,int(label),seq_len))
                else:
                    contents.append((words,seq_len))
        return contents #[([...],0,32),([...],1,32)]

    train=load_dataset(config.train_path,config.pad_size)
    valid=load_dataset(config.dev_path,config.pad_size)
    test=load_dataset(config.test_path,config.pad_size)
    return vocab_dict, train, valid, test

class DataIterater(object):
    def __init__(self,batches,batch_size,device):
        self.batch_size=batch_size
        self.batches=batches
        self.n_batches=len(batches)//batch_size
        self.residue=False
        if len(batches)%self.n_batches !=0:
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

def get_time_diff(start_time):
    end_time=time.time()
    time_diff=end_time-start_time
    return timedelta(seconds=round(time_diff))

if __name__ == "__main__":
    '''提取预训练词向量'''
    train_dir ="./data/Bulletin-screen train 200.txt"
    vocab_dir ="./data/vocab.pkl"
    pretrain_dir = "./data/sgns.sogou.char"
    emb_dim = 300
    filename_trimmed_dir = "./data/embedding_SougouNews"
    if os.path.exists(vocab_dir):
        with open(vocab_dir,'rb') as f:
            word_to_id=pkl.load(f)
    else:
        # tokenizer = lambda x: x.split(' ')  # 以词为单位构建词表(数据集中词之间以空格隔开)
        tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
        word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        with open(vocab_dir,'wb') as f:
            pkl.dump(word_to_id, f)

    embeddings = np.random.rand(len(word_to_id), emb_dim)
    f = open(pretrain_dir, "r", encoding='UTF-8')
    for i, line in enumerate(f.readlines()):
        # if i == 0:  # 若第一行是标题，则跳过
        #     continue
        lin = line.strip().split(" ")
        if lin[0] in word_to_id:
            idx = word_to_id[lin[0]]
            emb = [float(x) for x in lin[1:301]]
            embeddings[idx] = np.asarray(emb, dtype='float32')
    f.close()
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)