import  torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Config():
    # 配置参数
    def __init__(self,dataset,embedding):
        # 模型名称 model_name
        self.model_name="FastText"

        #数据源 ：train_path，test_path，dev_path,vocab_path，pretrainedVec,labels_list
        self.train_path=dataset+'/data/train.txt'
        self.dev_path=dataset+'/data/dev.txt'
        self.test_path=dataset+'/data/test.txt'
        self.vocab_path=dataset+'/data/vocab.pkl'
        self.embedding_pretrained=torch.tensor(
            np.load(dataset+'/data/'+embedding)['embedding'].astype('float32')) if embedding != 'random' else None
        self.class_list = [x.strip() for x in open(dataset + '/data/class.txt', encoding='utf-8').readlines()]

        #输出数据：model_save，log_save
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'
        self.log_path = dataset + '/log/' + self.model_name

        #m模型参数：num_epochs,batch_size, pad_size,embed_dim
        self.num_epochs = 20
        self.batch_size = 128
        self.pad_size = 32
        self.embed = self.embedding_pretrained.size(1) if self.embedding_pretrained is not None else 300


        #模型参数： hidden_size,drouout,hidden_size, require_improvement(early_stopping),num_classes，learning_rate
        self.hidden_size = 256
        self.dropout=0.5
        self.require_improvement=1000
        self.learning_rate = 1e-3
        self.num_classes=len(self.class_list)
        self.n_vocab=0
        self.n_gram_vocab=250499

        #设备参数
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Model(nn.Module):
    def __init__(self,config):
        super(Model,self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding=nn.Embedding.from_pretrained(config.embedding_pretrained,freeze=False)
        else:
            self.embedding=nn.Embedding(config.n_vocab,config.embed,padding_idx=config.n_vocab -1)
        self.embedding_ngram2=nn.Embedding(config.n_gram_vocab,config.embed)
        self.embedding_ngram3=nn.Embedding(config.n_gram_vocab,config.embed)
        self.dropout=nn.Dropout(config.dropout)
        self.fc1=nn.Linear(config.embed*3,config.hidden_size)
        self.fc2=nn.Linear(config.hidden_size,config.num_classes)


    def forward(self,x):
        print(x[0])
        out_word=self.embedding(x[0])
        out_bigram=self.embedding_ngram2(x[2])
        out_trigram=self.embedding_ngram3(x[3])
        out=torch.cat((out_word,out_bigram,out_trigram),-1)
        out=out.mean(dim=1)
        out=self.dropout(out)
        out=self.fc1(out)
        out=F.relu(out)
        out=self.fc2(out)
        return out


