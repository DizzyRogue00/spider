import time
import numpy as np
import torch
from train import init_network, train
from importlib import import_module
import argparse
from word2vec import build_dataset,build_iter,get_time_diff
#python -m tensorboard.main --logdir=./data/log --port=6006

parser=argparse.ArgumentParser(description='Text Classification')
parser.add_argument('--model',type=str,required=True,help='Choose a model: TextCNN_trail,TextRNN,TextRNN_Attention,RCNN,DPCNN,Transformer,Bert') #python xx.py --model=
parser.add_argument('--embedding',default='pre_trained',type=str,help='random or pre_trained')
parser.add_argument('--word',default=False,type=bool,help='True for word, False for char')
args=parser.parse_args()

if __name__=='__main__':
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
        vocab_dict,train_data=build_dataset(config, args.word, config.train_path)
        train_iter=build_iter(train_data, config)
        time_diff=get_time_diff(start_time)
        print("Time Usage:",time_diff)

        config.n_vocab = len(vocab_dict)
        temp_model_class=getattr(x,model_name)
        model=temp_model_class(config).to(config.device)
        init_network(model)
        print(model.parameters)
        train(config,model,train_iter,train_iter)#(config,model,train_iter,train_original)
