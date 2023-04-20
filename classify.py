import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

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
        #LSTM+attention
        self.hidden_size2=64

        #transformer
        self.dim_model=300
        self.hidden=1024
        self.last_hidden=512
        self.num_head=5
        self.num_encoder=2

class TextCNN_trail(nn.Module):
    def __init__(self,config):
        super(TextCNN_trail,self).__init__()
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
        super(TextRNN,self).__init__()
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

class TextRNN_Attention(nn.Module):
    def __init__(self,config):
        super(TextRNN_Attention,self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding=nn.Embedding.from_pretrained(config.embedding_pretrained,freeze=False)
        else:
            self.embedding=nn.Embedding(config.n_vocab,config.embed,padding_idx=config.n_vocab-1)
        self.lstm=nn.LSTM(config.embed,config.hidden_size,config.num_layers,
                          batch_first=True,dropout=config.dropout,bidirectional=True)
        self.tanh1=nn.Tanh()
        self.w=nn.Parameter(torch.zeros(config.hidden_size*2))
        self.tanh2=nn.Tanh()
        self.classify1=nn.Linear(config.hidden_size*2,config.hidden_size2)
        self.classify2=nn.Linear(config.hidden_size2,config.num_classes)

    def forward(self,x):
        emb=self.embedding(x[0])#[64,32,300]
        H,_=self.lstm(emb) #[batch_size,seq_len,hidden_size*num_direction] [64,32,256]
        M=self.tanh1(H)#[64,32,256]
        alpha=F.softmax(torch.matmul(M,self.w),dim=1).unsqueeze(-1)#[64,32,1]
        out=H*alpha#[64,32,256]
        out=torch.sum(out,1)#[64,256]
        out=F.relu(out)
        out=self.classify1(out)
        out=self.classify2(out)
        return out

class RCNN(nn.Module):
    def __init__(self,config):
        super(RCNN,self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding=nn.Embedding.from_pretrained(config.embedding_pretrained,freeze=False)
        else:
            self.embedding=nn.Embedding(config.n_vocab,config.embed,padding_idx=config.n_vocab-1)
        self.lstm=nn.LSTM(config.embed,config.hidden_size,config.num_layers,
                          batch_first=True,dropout=config.dropout,bidirectional=True)
        self.maxpool=nn.MaxPool1d(config.pad_size)
        self.classify=nn.Linear(config.hidden_size*2+config.embed,config.num_classes)

    def forward(self,x):
        embed=self.embedding(x[0])#[batch_size,seq_len,embeding]=[64,32,300]
        out,_=self.lstm(embed)
        out=torch.cat((out,embed),2)
        out=F.relu(out)
        out=out.permute(0,2,1)
        out=self.maxpool(out).squeeze()
        out=self.classify(out)
        return out

class DPCNN(nn.Module):
    def __init__(self,config):
        super(DPCNN,self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding=nn.Embedding.from_pretrained(config.embedding_pretrained,freeze=False)
        else:
            self.embedding=nn.Embedding(config.n_vocab,config.embed,padding_idx=config.n_vocab-1)
        self.conv_init=nn.Conv2d(1,config.num_filters,(3,config.embed))
        self.conv=nn.Conv2d(config.num_filters,config.num_filters,(3,1))
        self.max_pool=nn.MaxPool2d((3,1),stride=2)
        self.padding1=nn.ZeroPad2d((0,0,1,1))
        self.padding2=nn.ZeroPad2d((0,0,0,1))
        self.relu=nn.ReLU()
        self.classify=nn.Linear(config.num_filters,config.num_classes)

    def forward(self,x):
        x=x[0]
        x=self.embedding(x)#[batch_size,seq_len,embed_size]
        x=x.unsqueeze(1)#[batch_size,1,seq_len,embed_size]
        x=self.conv_init(x)#[batch_size,out_channel(num_filters),seq_len+1-3,1]
        x=self.padding1(x)#[batch_size,out_channel,seq_len,1]
        x=self.relu(x)
        x=self.conv(x)
        x=self.padding1(x)
        x=self.relu(x)
        x=self.conv(x)#[batch_size,out_channel,seq_len-3+1,1]
        while x.size()[2]>2:
            x=self._block(x)
        x=x.squeeze()#[batch_size,num_filters]
        out=self.classify(x)
        return out

    def _block(self,x):
        x=self.padding2(x)
        x_temp=self.max_pool(x)
        x=self.padding1(x_temp)
        x=F.relu(x)
        x=self.conv(x)

        x=self.padding1(x)
        x=F.relu(x)
        x=self.conv(x)
        x=x+x_temp
        return x

class Transformer(nn.Module):
    def __init__(self,config):
        super(Transformer,self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding=nn.Embedding.from_pretrained(config.embedding_pretrained,freeze=False)
        else:
            self.embedding=nn.Embedding(config.n_vocab,config.embed,padding_idx=config.n_vocab-1)

class Scaled_Dot_Product_Attention(nn.Module):
    def __init__(self):
        super(Scaled_Dot_Product_Attention,self).__init__()

    def forward(self,Q,K,V,scale=None):
        '''
        :param Q: [batch_size,len_Q,dim_Q]
        :param K: [batch_size,len_K,dim_K]
        :param V: [batch_size,len_V,dim_V]
        :param scale:
        :return:
        '''
        attention=torch.matmul(Q,K.permute(0,2,1))
        if scale:
            attention=attention*scale
        #if mask:
        #   attention=attention.masked_fill_(mask==0,-1e9)
        attention=F.softmax(attention,dim=-1)
        context=torch.matmul(attention,V)
        return context

class Multi_Head_Attention(nn.Module):
    def __init__(self,dim_model,num_head,dropout=0.0):
        super(Multi_Head_Attention,self).__init__()
        self.num_head=num_head
        assert dim_model%num_head==0
        self.dim_head=dim_model//self.num_head
        self.fc_Q=nn.Linear(dim_model,num_head*self.dim_head)
        self.fc_K=nn.Linear(dim_model,num_head*self.dim_head)
        self.fc_V=nn.Linear(dim_model,num_head*self.dim_head)
        self.attention=Scaled_Dot_Product_Attention()
        self.fc=nn.Linear(num_head*self.dim_head,dim_model)
        self.dropout=nn.Dropout(dropout)
        self.layer_norm=nn.LayerNorm(dim_model)
    def forward(self,x):
        batch_size=x.size(0)
        Q=self.fc_Q(x)
        K=self.fc_K(x)
        V=self.fc_V(x)
        Q=Q.view(batch_size*self.num_head,-1,self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        scale=K.size(-1)**(-0.5)
        context=self.attention(Q,K,V,scale)
        context=context.view(batch_size,-1,self.dim_head*self.num_head)
        out=self.fc(context)
        out=self.dropout(out)
        out=x+out
        out=self.layer_norm(out)
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
