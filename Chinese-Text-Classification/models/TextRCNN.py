import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config():
    def __init__(self, dataset, embedding):
        self.model_name = "TextCNN"
        self.train_path = dataset + '/data/train.txt'
        self.dev_path = dataset + '/data/dev.txt'
        self.test_path = dataset + '/data/test.txt'
        self.vocab_path = dataset + '/data/vocab.pkl'
        self.class_list = [x.strip() for x in open(dataset + '/data/class.txt', encoding='utf-8').readlines()]
        self.save_path = dataset + '/save_dict/' + self.model_name + '.ckpt'
        self.log_path = dataset + 'log' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)['embedding'].astype('float32')) if embedding != 'random' else None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dropout = 1.0
        self.require_improvement = 1000
        self.num_classes = len(self.class_list)
        self.n_vocab = 0  # 词表大小，在运行时赋值
        self.num_epochs = 20
        self.batch_size = 128
        self.pad_size = 32
        self.learning_rate = 1e-3
        self.embed = self.embedding_pretrained.size(1) if self.embedding_pretrained is not None else 300
        self.hidden_size = 256
        self.num_layers = 1


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed,
                                          padding_idx=config.n_vocab - 1)  # 词嵌入，就是把一个词典，随机初始化映射为一个向量矩阵。
            # padding_idx：表示用于填充的参数索引，比如用3填充padding，嵌入向量索引为3的向量设置为0

        self.lstm=nn.LSTM(config.embed,config.hidden_size,config.num_layers,
                          bidirectional=True,batch_first=True,dropout=config.dropout)
        self.maxpool=nn.MaxPool1d(config.pad_size)
        self.fc=nn.Linear(config.hidden_size*2+config.embed,config.num_classes)


    def forward(self, x):
        x,_=x
        embed=self.embedding(x)
        out,_=self.lstm(embed)
        out=torch.cat((embed,out),2)
        out=F.relu(out)
        out=out.permute(0,2,1)
        out=self.maxpool(out).squeeze()
        out=self.fc(out)
        return out