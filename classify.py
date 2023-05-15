import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from torch.nn.init import normal_
import six
import logging
import json
import os
from copy import deepcopy
from utils_bert import logger_init

class config(object):
    #
    def __init__(self,model_name,embedding='random',train_path=None,dev_path=None,test_path=None):
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
        self.project_dir=os.path.dirname(os.path.abspath(__file__))
        self.pretrained_model_dir=os.path.join('bert_pretrained','bert-base-chinese')
        self.bert_vocab_path=os.path.join(self.pretrained_model_dir,'vocab.txt')
        self.log_save_dir=os.path.join(self.project_dir,'data','log')
        logger_init(log_file_name='Bert',log_level=logging.INFO,log_dir=self.log_save_dir)

        self.train_path = train_path                               # training set
        self.dev_path = dev_path                                  # validation set
        self.test_path = test_path                                 # test set

        self.class_list=['0','1']   #class list
        self.vocab_path='data/vocab.pkl' #word dictionary {word: word id} .pkl import pickle;open();load()
        self.save_path ='data/' + self.model_name + '.ckpt'        # result of tranined model
        self.log_path = 'data/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(np.load('data/' + embedding)["embeddings"].astype('float32')) if embedding != 'random' else None # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs =20 #Transformer 100                                        # epoch数
        #self.batch_size = 128                                           # mini-batch大小
        self.batch_size = 16  # mini-batch大小
        self.pad_size = 32                                             # 每句话处理成的长度(短填长切)
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

        #Bert
        self.bert_hidden_size=768
        self.bert_vocab_size=21128
        self.bert_pad_token_id=0
        self.bert_initializer_range=0.02
        self.bert_max_position_embeddings=512
        self.bert_type_vocab_size=2
        self.bert_embedding_dropout=0.1
        self.bert_num_head=12 #12
        self.bert_attention_dropout=0.1
        self.bert_hidden_dropout=0.1
        self.bert_intermediate_size=3072
        self.bert_hidden_act='gelu'
        self.bert_num_hidden_layers=12
        self.bert_pooler_type='first_token_transform'
        self.bert_num_epochs = 100
        #"directionality": "bidi",
        #"pooler_fc_size": 768,
        #"pooler_num_attention_heads": 12,
        #"pooler_num_fc_layers": 3,
        #"pooler_size_per_head": 128,
        logging.info('current configuration')
        for key,value in self.__dict__.items():
            logging.info(f'{key}={value}')

    @classmethod
    def from_dict(cls,json_object):
        "Constructs from a dictionary of parameters"
        config1=config()
        for (key,value) in six.iteritems(json_object):
            config1.__dict__[key]=value
        return config1

    @classmethod
    def from_json_file(cls,json_file):
        "Constructs from a json file"
        with open(json_file,'r') as reader:
            text=reader.read()
        logging.info(f'Successfully load configuration file {json_file}')
        return cls.from_dict(json.loads(text))

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
        x=F.relu(conv(x)).squeeze(3)#[batch_size,out_channel,seq_len]
        x=F.max_pool1d(x,x.size(2)).squeeze(2)#[batch_size,out_channel]
        return x

    def forward(self,x):#include batch_size
        out=self.embedding(x[0])
        #out=out.unsqueeze(0)#if no batch_size
        out=out.unsqueeze(1)#[batch_size,1,seq_len,embedding_dim]
        out=torch.cat([self.conv_activate_and_pool(out,conv) for conv in self.convs],dim=1)#[batch_size,num_filters*out_channels]
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
        self.position_embedding=Positional_Encoding(config.embed,config.pad_size,config.dropout,config.device)
        self.encoder=Encoder(config.dim_model,config.num_head,config.hidden,config.dropout)
        self.encoders=nn.ModuleList(
            [copy.deepcopy(self.encoder) for item in range(config.num_encoder)]
        )
        self.classify=nn.Linear(config.pad_size*config.dim_model,config.num_classes)

    def forward(self,x):
        out=self.embedding(x[0])
        out=self.position_embedding(out)
        for encoder in self.encoders:
            out=encoder(out)
        out=out.view(out.size(0),-1)
        out=self.classify(out)
        return out

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

class Position_Wise_Feed_Forward(nn.Module):
    def __init__(self,dim_model,hidden,dropout=0.0):
        super(Position_Wise_Feed_Forward,self).__init__()
        self.l1=nn.Linear(dim_model,hidden)
        self.l2=nn.Linear(hidden,dim_model)
        self.dropout=nn.Dropout(dropout)
        self.layer_norm=nn.LayerNorm(dim_model)

    def forward(self,x):
        out=self.l1(x)
        out=F.relu(out)
        out=self.l2(out)
        out=self.dropout(out)
        out=x+out
        out=self.layer_norm(out)
        return out

class Positional_Encoding(nn.Module):
    def __init__(self,embed,pad_size,dropout,device):
        super(Positional_Encoding,self).__init__()
        self.device=device
        self.pe=torch.tensor([[pos/10000.0**(i//2*2.0/embed) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:,0::2]=np.sin(self.pe[:,0::2])
        self.pe[:,1::2]=np.cos(self.pe[:,1::2])
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        out=x+nn.Parameter(self.pe,requires_grad=False).to(self.device)
        out=self.dropout(out)
        return out

class Encoder(nn.Module):
    def __init__(self,dim_model,num_head,hidden,dropout):
        super(Encoder,self).__init__()
        self.attention=Multi_Head_Attention(dim_model,num_head,dropout)
        self.feed_forward=Position_Wise_Feed_Forward(dim_model,hidden,dropout)

    def forward(self,x):
        out=self.attention(x)
        out=self.feed_forward(out)
        return out

def get_activation(act_str):
    act=act_str.lower()
    if act=='linear':
        return None
    elif act=='relu':
        return nn.ReLU()
    elif act=="gelu":
        return nn.GELU()
    elif act=='tanh':
        return nn.Tanh()
    else:
        raise ValueError("Unsupported activation:%s"%act)

class TokenEmbedding(nn.Module):
    def __init__(self,vocab_size,hidden_size,pad_token_id=0,initializer_range=0.02):
        super(TokenEmbedding,self).__init__()
        self.embedding=nn.Embedding(vocab_size,hidden_size,padding_idx=pad_token_id)
        self._reset_parameters(initializer_range)

    def forward(self,input_ids):
        '''
        :param input_ids: [batch_size,seq_len]
        :return: [batch_size,seq_len,hidden_size]
        '''
        return self.embedding(input_ids)

    def _reset_parameters(self,initializer_range):
        for p in self.parameters():
            if p.dim()>1:
                normal_(p,mean=0.0,std=initializer_range)

class PositionalEmbedding(nn.Module):
    def __init__(self,hidden_size,max_position_embeddings=512,initializer_range=0.02):
        super(PositionalEmbedding,self).__init__()
        self.embedding=nn.Embedding(max_position_embeddings,hidden_size)
        self._reset_parameters(initializer_range)

    def forward(self,position_ids):
        '''
        :param position_ids: [1,position_ids_len]
        :return: [1,postion_ids_len,hidden_size]
        '''
        return self.embedding(position_ids)

    def _reset_parameters(self,initializer_range):
        for p in self.parameters():
            if p.dim()>1:
                normal_(p,mean=0.0,std=initializer_range)

class SegementEmbedding(nn.Module):
    def __init__(self,type_vocab_size,hidden_size,initializer_range=0.02):
        super(SegementEmbedding,self).__init__()
        self.embedding=nn.Embedding(type_vocab_size,hidden_size)
        self._reset_parameters(initializer_range)

    def forward(self,token_type_ids):
        '''
        :param token_type_ids: [batch_size,token_type_ids_len]
        :return: [batch_size,token_type_ids_len,hidden_size]
        '''
        return self.embedding(token_type_ids)

    def _reset_parameters(self,initializer_range):
        for p in self.parameters():
            if p.dim()>1:
                normal_(p,mean=0.0,std=initializer_range)

class BertEmbedding(nn.Module):
    def __init__(self,config):
        super(BertEmbedding,self).__init__()
        self.word_embeddings=TokenEmbedding(
            vocab_size=config.bert_vocab_size,
            hidden_size=config.bert_hidden_size,
            pad_token_id=config.bert_pad_token_id,
            initializer_range=config.bert_initializer_range
        )
        self.position_embeddings=PositionalEmbedding(
            max_position_embeddings=config.bert_max_position_embeddings,
            hidden_size=config.bert_hidden_size,
            initializer_range=config.bert_initializer_range
        )
        self.token_type_embeddings=SegementEmbedding(
            type_vocab_size=config.bert_type_vocab_size,
            hidden_size=config.bert_hidden_size,
            initializer_range=config.bert_initializer_range
        )
        self.LayerNorm=nn.LayerNorm(config.bert_hidden_size)
        self.dropout=nn.Dropout(config.bert_embedding_dropout)
        self.register_buffer('position_ids',torch.arange(config.bert_max_position_embeddings).expand(1,-1))
        self.device=config.device

    def forward(self,input_ids=None,position_ids=None,token_type_ids=None):
        '''
        :param input_ids: [batch_size,src_len]
        :param position_ids: [0,1,2,...,src_len-1] shape: [1,src_len]
        :param token_type_ids: [0,0,0,0,1,1,1] shape: [batch_size,src_len]
        :return: [batch_size,src_len,hidden_size]
        '''
        src_len=input_ids.size(1)
        token_embedding=self.word_embeddings(input_ids)
        if position_ids is None:
            position_ids=self.position_ids[:,:src_len]
        positional_embedding=self.position_embeddings(position_ids)
        if token_type_ids is None:
            token_type_ids=torch.zeros_like(input_ids,device=self.device)
        segement_embedding=self.token_type_embeddings(token_type_ids)
        embeddings=token_embedding+positional_embedding+segement_embedding
        embeddings=self.LayerNorm(embeddings)
        embeddings=self.dropout(embeddings)
        return embeddings

class MymultiheadAttention(nn.Module):
    def __init__(self,embed_dim,num_heads,dropout=0,bias=True):
        '''
        :param embed_dim: word_dimenssions
        :param num_heads:
        :param dropout:
        :param bias:
        '''
        super().__init__()
        self.embed_dim=embed_dim
        self.head_dim=embed_dim//num_heads #d_k,d_v
        self.kdim=self.head_dim
        self.vdim=self.head_dim
        self.num_heads=num_heads
        self.dropout=dropout
        assert self.head_dim*num_heads==self.embed_dim

        self.q=nn.Linear(embed_dim,embed_dim,bias=bias)
        self.k = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out=nn.Linear(embed_dim,embed_dim,bias=bias)

    def forward(self,query,key,value,attn_mask=None,key_padding_mask=None):#attn_mask=None,#key_padding_mask=None):
        return multi_head_attention_forward(query,key,value,self.num_heads,self.dropout,
                                            out=self.out,
                                            training=True,
                                            key_padding_mask=key_padding_mask,
                                            q_matrix=self.q,
                                            k_matrix=self.k,
                                            v_matrix=self.v,
                                            attn_mask=attn_mask
                                            )

def multi_head_attention_forward(query,#[batch_size,target_len,embed_dim]
                                 key,#[batch_size,src_len,embed_dim]
                                 value,#[batch_size,src_len,embed_dim]
                                 num_heads,
                                 dropout,
                                 out,
                                 training=True,
                                 #key_padding_mask=None, #[batch_size,src_len/target_len]
                                 key_padding_mask=None,
                                 q_matrix=None,
                                 k_matrix=None,
                                 v_matrix=None,
                                 attn_mask=None#attn_mask=None,#[target_len,src_len] or [num_heads*batch_size,target_len,src_len]
                                 ):
    q=q_matrix(query)
    #[batch_size,target_len,kdim*num_heads]
    k=k_matrix(key)
    v=v_matrix(value)
    batch_size,target_len,embed_dim=query.size()
    src_len=key.size(1)
    head_dim=embed_dim//num_heads
    scaling=float(head_dim)**-0.5
    q=q*scaling
    if attn_mask is not None:#[tgt_len,src_len] or [num_heads*batch_size,tgt_len,src_len]
        if attn_mask.dim()==2:
            attn_mask=attn_mask.unsqueeze(0)#[1,tgt_len,src_len]
            if list(attn_mask.size())!=[1,query.size(1),key.size(1)]:
                raise RuntimeError('The size of the 2D attn_mask is not correct.')
        elif attn_mask.dim()==3:
            if list(attn_mask.size())!=[batch_size*num_heads,query.size(1),key.size(1)]:
                raise RuntimeError('The size of the 3D attn_mask is not correct.')
    q=q.contiguous().view(batch_size*num_heads,-1,head_dim)
    k = k.contiguous().view(batch_size * num_heads, -1, head_dim)
    v = v.contiguous().view(batch_size * num_heads, -1, head_dim)
    attn_output_weights=torch.bmm(q,k.transpose(1,2))#[batch_size*num_heads,target_len,src_len]
    if attn_mask is not None:
        attn_output_weights+=attn_mask
    if key_padding_mask is not None:
        attn_output_weights=attn_output_weights.view(batch_size,num_heads,target_len,src_len)
        attn_output_weights=attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf')
        )
        attn_output_weights=attn_output_weights.view(batch_size*num_heads,target_len,src_len)

    attn_output_weights=F.softmax(attn_output_weights,dim=-1)#[batch_size*num_heads,target_len,src_len]
    attn_output_weights=F.dropout(attn_output_weights,p=dropout,training=training)
    attn_output=torch.bmm(attn_output_weights,v)#[batch_size*num_heads,target_len,vdim]
    attn_output=attn_output.contiguous().view(batch_size,-1,embed_dim)
    attn_output_weights=attn_output_weights.view(batch_size,num_heads,target_len,src_len)
    Z=out(attn_output)
    return Z,attn_output_weights.sum(dim=1)/num_heads

class BertSelfAttention(nn.Module):
    def __init__(self,config):
        super(BertSelfAttention,self).__init__()
        if 'use_torch_multi_head' in config.__dict__ and config.use_torch_multi_head:
            MultiHeadAttention=nn.MultiheadAttention
        else:
            MultiHeadAttention=MymultiheadAttention
        self.multi_head_attention=MultiHeadAttention(embed_dim=config.bert_hidden_size,
                                                       num_heads=config.bert_num_head,
                                                       dropout=config.bert_attention_dropout)

    def forward(self,query,key,value,attn_mask=None,key_padding_mask=None):
        return self.multi_head_attention(query,key,value,attn_mask=attn_mask,key_padding_mask=key_padding_mask)

class BertSelfOutput(nn.Module):
    def __init__(self,config):
        super(BertSelfOutput,self).__init__()
        self.LayerNorm=nn.LayerNorm(config.bert_hidden_size,eps=1e-12)
        self.dropout=nn.Dropout(config.bert_hidden_dropout)

    def forward(self,hidden_states,input_tensor):
        '''
        :param hidden_states: [batch_size,src_len,hidden_size]
        :param input_tensor: [batch_size,src_len,hidden_size]
        :return: [batch_size,src_len,hidden_size]
        '''
        hidden_states=self.dropout(hidden_states)
        hidden_states=self.LayerNorm(hidden_states+input_tensor)
        return hidden_states

class BertAttention(nn.Module):
    def __init__(self,config):
        super(BertAttention,self).__init__()
        self.attention=BertSelfAttention(config)
        self.output=BertSelfOutput(config)

    def forward(self,hidden_states,attn_mask=None):
        '''
        :param hidden_states: [batch_size,src_len,hidden_size]
        :return: [batch_size,src_len,hidden_size]
        '''
        self.outputs=self.attention(hidden_states,hidden_states,hidden_states,attn_mask=None,key_padding_mask=attn_mask)
        self.outputs=self.output(self.outputs[0],hidden_states)
        return self.outputs

class BertIntermediateLayer(nn.Module):
    def __init__(self,config):
        super(BertIntermediateLayer,self).__init__()
        self.dense=nn.Linear(config.bert_hidden_size,config.bert_intermediate_size)
        if isinstance(config.bert_hidden_act,str):
            self.intermediate_act=get_activation(config.bert_hidden_act)
        else:
            self.intermediate_act=nn.ReLU()

    def forward(self,hidden_states):
        hidden_states=self.dense(hidden_states)
        if self.intermediate_act is None:
            hidden_states=hidden_states
        else:
            hidden_states=self.intermediate_act(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self,config):
        super(BertOutput,self).__init__()
        self.dense=nn.Linear(config.bert_intermediate_size,config.bert_hidden_size)
        self.LayerNorm=nn.LayerNorm(config.bert_hidden_size,eps=1e-12)
        self.dropout=nn.Dropout(config.bert_hidden_dropout)

    def forward(self,hidden_states,input_tensor):
        '''
        :param hidden_states: [batch_size,src_len,intermediate_size]
        :param input_tensor: [batch_size,src_len,hidden_size]
        :return: [batch_size,src_len.hidden_size]
        '''
        hidden_states=self.dense(hidden_states)
        hidden_states=self.dropout(hidden_states)
        hidden_states=self.LayerNorm(hidden_states+input_tensor)
        return hidden_states

class BertLayer(nn.Module):
    def __init__(self,config):
        super(BertLayer,self).__init__()
        self.bert_attention=BertAttention(config)
        self.bert_intermediate=BertIntermediateLayer(config)
        self.bert_output=BertOutput(config)

    def forward(self,hidden_states,attention_mask=None):
        '''
        :param hidden_states: [batch_size,src_len,embed_dim]
        :return:
        '''
        hidden_states=self.bert_attention(hidden_states,attention_mask)
        intermediate_states=self.bert_intermediate(hidden_states)
        layer_output=self.bert_output(intermediate_states,hidden_states)
        return layer_output

class BertEncoder(nn.Module):
    def __init__(self,config):
        super(BertEncoder,self).__init__()
        self.config=config
        self.bert_layers=nn.ModuleList(
            [BertLayer(config) for _ in range(config.bert_num_hidden_layers)]
        )

    def forward(self,hidden_states,attention_mask=None):
        all_encoder_layers=[]
        layer_output=hidden_states
        for i,layer_module in enumerate(self.bert_layers):
            layer_output=layer_module(layer_output,attention_mask)
            all_encoder_layers.append(layer_output)
        return all_encoder_layers

class BertPooler(nn.Module):
    def __init__(self,config):
        super(BertPooler,self).__init__()
        self.dense=nn.Linear(config.bert_hidden_size,config.bert_hidden_size)
        self.activation=nn.Tanh()
        self.config=config

    def forward(self,hidden_states):
        '''
        :param hidden_states: [batch_size,src_len,hidden_size]
        :return: [batch_size,hidden_size]
        '''
        if self.config.bert_pooler_type=='first_token_transform':
            token_tensor=hidden_states[:,0].reshape(-1,self.config.bert_hidden_size)
        elif self.config.bert_pooler_type=='all_token_average':
            token_tensor=torch.mean(hidden_states,dim=1)
        pool_output=self.dense(token_tensor)
        pool_output=self.activation(pool_output)
        return pool_output

class BertModel(nn.Module):
    def __init__(self,config):
        super(BertModel,self).__init__()
        self.bert_embedding=BertEmbedding(config)
        self.bert_encoder=BertEncoder(config)
        self.bert_pooler=BertPooler(config)
        self.config=config
        self._reset_parameters()

    def forward(self,input_ids=None,attention_mask=None,token_type_ids=None,position_ids=None):
        embedding_output=self.bert_embedding(input_ids=input_ids,
                                             position_ids=position_ids,
                                             token_type_ids=token_type_ids)
        all_encoder_outputs=self.bert_encoder(embedding_output,attention_mask=attention_mask)
        sequence_output=all_encoder_outputs[-1]
        pooled_output=self.bert_pooler(sequence_output)
        return pooled_output,all_encoder_outputs

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim()>1:
                normal_(p,mean=0.0,std=self.config.bert_initializer_range)

    @classmethod
    def from_pretrained(cls,config,pretrained_model_dir=None):
        model=cls(config)
        pretrained_model_path=os.path.join(pretrained_model_dir,'pytorch_model.bin')
        if not os.path.exists(pretrained_model_path):
            raise ValueError(f"<path：No such file in the directory {pretrained_model_path}, please check carefully！>\n"
                             f"Chinese version model download：https://huggingface.co/bert-base-chinese/tree/main\n"
                             f"English version model download：https://huggingface.co/bert-base-uncased/tree/main\n")
        load_params=torch.load(pretrained_model_path)
        state_dict=deepcopy(model.state_dict())
        load_params_names=list(load_params.keys())[:-8]
        state_dict_names=list(state_dict.keys())[1:]
        for i in range(len(load_params_names)):
            state_dict[state_dict_names[i]]=load_params[load_params_names[i]]
            logging.info(f'successfully assign parameters from {load_params_names[i]} to {state_dict_names[i]}')
        model.load_state_dict(state_dict)
        return model
    #BertModel.frompretrained(config,pretrained_model_dir)
#test
# from classify import config,BertModel
# config_model=config('Bert')
# bert_model=BertModel(config_model)
# print('Bert parameters')
# print(len(bert_model.state_dict()))
# for param in bert_model.state_dict():
#     print(param,'\t',bert_model.state_dict()[param].size())
#load_paras=torch.load('bert_pretrained/bert-base-chinese/pytorch_model.bin')
# print(type(load_paras))
# print(len(list(load_paras.keys())))
# for name in load_paras.keys():
#     print(name,'\t',load_paras[name].size())

class BertForSentenceClassification(nn.Module):
    def __init__(self,config,bert_pretrained_model_dir=None):
        super().__init__()
        self.num_labels=config.num_classes
        if bert_pretrained_model_dir is not None:
            self.bert=BertModel.from_pretrained(config,bert_pretrained_model_dir)
        else:
            self.bert=BertModel(config)
        self.dropout=nn.Dropout(config.bert_hidden_dropout)
        self.classifier=nn.Linear(config.bert_hidden_size,self.num_labels)

    def forward(self,input_ids=None,attention_mask=None,token_type_ids=None,position_ids=None,labels=None):
        '''
        :param input_ids: [batch_size,src_len]
        :param token_type_ids: [batch_size,src_len]
        :param position_ids: [1,src_len]
        :param labels: [batch_size,]
        :return:
        '''
        pooled_output,_=self.bert(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,position_ids=position_ids)
        pooled_output=self.dropout(pooled_output)
        logits=self.classifier(pooled_output)#[batch_size,num_classes]
        if labels is not None:
            loss_fct=nn.CrossEntropyLoss()
            loss=loss_fct(logits.view(-1,self.num_labels),labels.view(-1))
            return loss, logits
        else:
            return logits

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
