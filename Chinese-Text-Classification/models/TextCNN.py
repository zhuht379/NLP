import  torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Config():
    def __init__(self,dataset,embedding):
        self.model_name="TextCNN"
        self.train_path=dataset+'/data/train.txt'
        self.dev_path=dataset+'/data/dev.txt'
        self.test_path=dataset+'/data/test.txt'
        self.vocab_path=dataset+'/data/vocab.pkl'
        self.class_list=[x.strip() for x in open(dataset+'/data/class.txt',encoding='utf-8').readlines()]
        self.save_path=dataset+'/saved_dict/'+self.model_name+'.ckpt'
        self.log_path=dataset+'log'+self.model_name
        self.embedding_pretrained=torch.tensor(np.load(dataset+'/data/'+embedding)['embeddings'].astype('float32')) if embedding !='random' else None
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dropout=0.5
        self.require_improvement=1000
        self.num_classes=len(self.class_list)
        self.n_vocab=0                      #词表大小，在运行时赋值
        self.num_epochs=20
        self.batch_size=128
        self.pad_size=32
        self.learning_rate=1e-3
        self.embed=self.embedding_pretrained.size(1) if self.embedding_pretrained is not None else 300
        self.filter_sizes=(2,3,4)
        self.num_filters=256



class Model(nn.Module):
    def __init__(self,config):
        super(Model,self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding=nn.Embedding.from_pretrained(config.embedding_pretrained,freeze=False)
        else:
            """  
            自然语言中使用批处理时候, 每个句子的长度并不一定是等长的, 这时候就需要对较短的句子进行padding, 
            填充的数据一般是0, 这个时候, 在进行词嵌入的时候就会进行相应的处理, nn.embedding会将填充的映射为0
            """
            self.embedding=nn.Embedding(config.n_vocab,config.embed,padding_idx=config.n_vocab-1)           #词嵌入，就是把一个词典，随机初始化映射为一个向量矩阵。
            # padding_idx：表示用于填充的参数索引，比如用3填充padding，嵌入向量索引为3的向量设置为0
        self.convs=nn.ModuleList([nn.Conv2d(1,config.num_filters,(k,config.embed)) for k in config.filter_sizes])
        """
        1. torch.nn.Sequential()
        2. torch.nn.ModuleList()
        
        """
        self.dropout=nn.Dropout(config.dropout)
        self.fc=nn.Linear(config.num_filters*len(config.filter_sizes),config.num_classes)

    def conv_and_pool(self,x,conv):
        x=F.relu(conv(x)).squeeze(3)
        x=F.max_pool1d(x,x.size(2)).squeeze(2)
        return x

    """
    1. torch.squeeze()这个函数对数据的维度进行压缩，去掉维数为1的维度
    2. torch.unsqueeze()对数据维度进行扩充，给指定的位置加上维数为1的维度
    """

    def forward(self,x):
        out=self.embedding(x[0])
        out=out.unsqueeze(1)
        out=torch.cat([self.conv_and_pool(out,conv) for conv in self.convs],1)   # 1 按照列拼接，横向拼接
        out=self.fc(out)
        return out