import time
import numpy as np
import torch
from train import init_network, train
from importlib import import_module
import argparse
from word2vec import build_vocab,build_dataset,build_iter,get_time_diff
import copy
import os
#python -m tensorboard.main --logdir=./data/log --port=6006
#model.load_state_dict(torch.load(config.save_path))
#loaded_paras = torch.load('pytorch_model.bin')

parser=argparse.ArgumentParser(description='Text Classification')
parser.add_argument('--model',type=str,required=True,help='Choose a model: TextCNN_trail,TextRNN,TextRNN_Attention,RCNN,DPCNN,Transformer,Bert') #python xx.py --model=
parser.add_argument('--embedding',default='pre_trained',type=str,help='random or pre_trained')
parser.add_argument('--word',default=False,type=bool,help='True for word, False for char')
parser.add_argument('--mode',default='Danmaku',type=str,help='Danmaku or Comment')
args=parser.parse_args()

def model_train(model_name,embedding,mode):
    if mode=='Danmaku':
        train_path='data/Bulletin-screen train 200.txt'
        dev_path='data/Bulletin-screen train 200.txt'
    else:
        train_path='data/Comment 200 label data.xlsx'
        dev_path='data/Comment 200 label data.xlsx'

    x=import_module('classify')
    config=x.config(model_name,mode,embedding,train_path,dev_path)
    if mode=='Danmaku':
        config.pad_size=32
    else:
        config.pad_size=160

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic=True

    if model_name !='Bert':
        start_time=time.time()
        print('Loading data...')
        vocab_dict,train_data=build_dataset(config, args.word, config.train_path)#train_data :[(sentence,label,seq_len)]
        train_iter=build_iter(train_data, config)
        dev_iter=copy.deepcopy(train_iter)
        time_diff=get_time_diff(start_time)
        print("Time Usage:",time_diff)

        config.n_vocab = len(vocab_dict)
        temp_model_class=getattr(x,model_name)
        model=temp_model_class(config).to(config.device)
        init_network(model)
        print(model.parameters)
        train(config,model,train_iter,dev_iter)#(config,model,train_iter,train_original)
    else:
        start_time = time.time()
        print('Basic bert model...')
        train(config)
        time_diff = get_time_diff(start_time)
        print("Time Usage:", time_diff)

if __name__=='__main__':
    embedding = 'embedding.npz'
    model_name = args.model
    if args.embedding == 'random':
        embedding = 'random'
    mode=args.mode
    model_train(model_name, embedding, mode)
    '''
    embedding='embedding.npz'
    model_name=args.model
    if args.embedding=='random':
        embedding='random'
    train_path='data/Bulletin-screen train 200.txt'
    dev_path='data/Bulletin-screen train 200.txt'

    x=import_module('classify')
    config=x.config(model_name,embedding,train_path,dev_path)
    #print(config.class_list)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic=True

    if model_name !='Bert':
        start_time=time.time()
        print('Loading data...')
        vocab_dict,train_data=build_dataset(config, args.word, config.train_path)#train_data :[(sentence,label,seq_len)]
        train_iter=build_iter(train_data, config)
        dev_iter=copy.deepcopy(train_iter)
        time_diff=get_time_diff(start_time)
        print("Time Usage:",time_diff)

        config.n_vocab = len(vocab_dict)
        temp_model_class=getattr(x,model_name)
        model=temp_model_class(config).to(config.device)
        init_network(model)
        print(model.parameters)
        train(config,model,train_iter,dev_iter)#(config,model,train_iter,train_original)
    else:
        start_time = time.time()
        print('Basic bert model...')
        train(config)
        time_diff = get_time_diff(start_time)
        print("Time Usage:", time_diff)
        '''

