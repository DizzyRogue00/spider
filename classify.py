import torch
import torch.nn as nn
import torch.nn.functional as F

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
                          kernel_sizes=k),
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
            # (batch_size,context_size,in-channel) --