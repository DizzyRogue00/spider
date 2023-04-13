import os
import sys
import torch
import numpy as np
import pickle as pkl
import tqdm as tqdm
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
    with open(file_path,'r',encoding="utf-8") as f:
        for line in tqdm(f):
            lin=line.strip()#remove leading and trailing characters (spaces)
            if not lin:
                continue
            content=lin.split('\t')[0]
            for word in tokenizer(content):
                vocab_dic[word]=vocab_dic.get(word,0)+1 #word frequency
        vocab_list=sorted([item for item in vocab_dic.items() if item(1)>=min_freq], key=lambda x:x[1],reverse=True)[:max_size]
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

    train=load_dataset()