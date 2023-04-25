import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from word2vec import get_time_diff
from tensorboardX import SummaryWriter

#initial default:xavier
def init_network(model,method='xavier',exclude='embedding',seed=1223):
    for name,w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method =='xavier':
                    nn.init.xavier_normal_(w)
                elif method=='kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w,0)
            else:
                pass

def train(config,model,train_iter):#(config,model,train_iter,dev_iter,test_iter)
    start_time=time.time()
    model.train()
    optimizer=torch.optim.Adam(model.parameters(),lr=config.learning_rate)
    total_batch=0
    #dev_best_loss=float('inf')
    last_improve=0
    flag=False
    writer=SummaryWriter(log_dir=config.log_path+'/'+time.strftime('%m-%d_%H.%M',time.localtime()))
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch+1,config.num_epochs))

