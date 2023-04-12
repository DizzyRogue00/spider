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

def build_vocab(file_path, tokenizer,max_size,min_freq):
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
    if os.path
