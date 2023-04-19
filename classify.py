import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class config(object):
    #
    def __init__(self,model_name,dataset,embedding):
        self.model_name = model_name
        #self.train_path = dataset + '/data/train.txt'                                # training set
        #self.dev_path = dataset + '/data/dev.txt'                                    # validation set
        #self.test_path = dataset + '/data/test.txt'                                  # test set
        #self.class_list = [x.strip() for x in open(
        #    dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
        #self.vocab_path = dataset + '/data/vocab.pkl'  # 词表
        #self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        #self.log_path = dataset + '/log/' + self.model_name
        #self.embedding_pretrained = torch.tensor(
        #    np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32')) \
        #    if embedding != 'random' else None  # 预训练词向量
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.train_path = '/data/train.txt'                                # training set
        self.dev_path = '/data/dev.txt'                                    # validation set
        self.test_path = '/data/test.txt'                                  # test set

        self.class_list=[0,1]   #class list
        self.vocab_path='/data/vocab.pkl' #word dictionary {word: word id} .pkl import pickle;open();load()
        self.save_path ='/data/' + self.model_name + '.ckpt'        # result of tranined model
        self.log_path = '/data/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load('/data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                       # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        #self.batch_size = 128                                           # mini-batch大小
        self.batch_size = 64  # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)

        #LSTM
        self.hidden_size=128    #lstm隐藏层
        self.num_layers=2 #lstm 层数

class TextCNN_trail(nn.Module):
    def __init__(self,config):
        super(TextCNN,self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding=nn.Embedding.from_pretrained(config.embedding_pretrained,freeze=False)
        else:
            self.embedding=nn.Embedding(config.n_vocab,config.embed,padding_idx=config.n_vocab-1)
        self.convs=nn.ModuleList(
            [nn.Conv2d(1,config.num_filters,(k,config.embed)) for k in config.filter_sizes]
        )
        self.dropout=nn.Dropout(config.dropout)
        self.classify=nn.Linear(config.num_filters*len(config.filter_sizes),config.num_classes)

    def conv_activate_and_pool(self,x,conv):
        x=F.relu(conv(x)).squeeze(3)
        x=F.max_pool1d(x,x.size(2)).squeeze(2)
        return x

    def forward(self,x):#include batch_size
        out=self.embedding(x[0])
        #out=out.unsqueeze(0)#if no batch_size
        out=out.unsqueeze(1)
        out=torch.cat([self.conv_activate_and_pool(out,conv) for conv in self.convs],dim=1)
        out=self.dropout(out)
        out=self.classify(out)
        return out

class TextRNN(nn.Module):
    def __init__(self,config):
        super(TextRNN,self).__init()
        if config.embedding_pretrained is not None:
            self.embedding=nn.Embedding.from_pretrained(config.embedding_pretrained,freeze=False)
        else:
            self.embedding=nn.Embedding(config.n_vocab,config.embed,padding_idx=config.n_vocab-1)
        self.lstm=nn.LSTM(config.embed,config.hidden_size,config.num_layers,
                          batch_first=True,dropout=config.dropout,bidirectional=True)
        self.classify=nn.Linear(config.hidden_size*2,config.num_classes)

    def forward(self,x):
        out=self.embedding(x[0]) #[batch_size,seq_len,embedding]
        out,_=self.lstm(out)
        out=self.classify(out[:,-1,:])
        return out

class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d,self).__init__()

    def forward(self,x):
        return F.max_pool1d(x,kernel_size=x.shape[2]) #shape:(batch_size,channel,1)

class TextCNN(nn.Module):
    def __init__(self,num_classes,num_embeddings=-1,embedding_dim=128,kernel_sizes=[3,4,5,6],
                 num_channels=[256,256,256,256],embeddings_pretrained=None):
        """
        :param num_classes:
        :param num_embeddings: size of the dictionary of embeddings, vocab_size
        :param embedding_dim: the size of each embedding vector
        :param kernel_sizes:
        :param num_channels:
        :param embeddings_pretrained:
        """
        super(TextCNN,self).__init__()
        self.num_classes=num_classes
        self.num_embeddings=num_embeddings
        #embedding
        if self.num_embeddings>0:
            #embedding shape:torch.Size([200,8,300])
            self.embedding=nn.Embedding(num_embeddings,embedding_dim)
            if embeddings_pretrained is not None:
                self.embedding=self.embedding.from_pretrained(embeddings_pretrained,freeze=False)
        #CNN
        self.cnn_layers=nn.ModuleList()
        for c,k in zip(num_channels,kernel_sizes):
            cnn=nn.Sequential(
                nn.Conv1d(in_channels=embedding_dim,
                          out_channels=c,
                          kernel_size=k),
                nn.BatchNorm1d(c),
                nn.ReLU(inplace=True),
            )
            self.cnn_layers.append(cnn)
        #Pooling
        self.pool=GlobalMaxPool1d()
        #output
        self.classify=nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(sum(num_channels),self.num_classes)
        )

    def forward(self,input):
        """
        :param input: batch_size,context_size,embedding_size(in_channels)
        :return:
        """
        if self.num_embeddings>0:
            # (b,context_size) --> (b,context_size,embedding_dim) b:batch
            input=self.embedding(input)
        # (batch_size,context_size,in-channel) --> (batch_size, channel, context_size)
        input=input.permute(0,2,1)
        y=[]
        for layer in self.cnn_layers:
            x=layer(input)
            x=self.pool(x).squeeze(-1)
            y.append(x)
        y=torch.cat(y,dim=1)
        output=self.classify(y)
        return output

"""
batch_size=4
num_classes=2
context_size=7
num_embeddings=1024
embedding_dim=6
kernel_sizes=[2,4]
num_channels=[4,5]
input=torch.ones(size=(batch_size,context_size)).long()
model=TextCNN(num_classes,num_embeddings,embedding_dim,kernel_sizes,
                 num_channels)
model.eval()
output=model(input)
print("-----"*10)
print(model)
print("-----"*10)
print("input.shape:{}".format(input.shape))
print("-----"*10)
print("output.shape:{}".format(output.shape))
"""
