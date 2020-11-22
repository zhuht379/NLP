import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config():
    def __init__(self, dataset, embedding):
        self.model_name = "DPCNN"
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

        self.dropout = 0.5
        self.require_improvement = 1000
        self.num_classes = len(self.class_list)
        self.n_vocab = 0  # 词表大小，在运行时赋值
        self.num_epochs = 20
        self.batch_size = 128
        self.pad_size = 32
        self.learning_rate = 1e-3
        self.embed = self.embedding_pretrained.size(1) if self.embedding_pretrained is not None else 300
        self.num_filters = 250


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            """  
            自然语言中使用批处理时候, 每个句子的长度并不一定是等长的, 这时候就需要对较短的句子进行padding, 
            填充的数据一般是0, 这个时候, 在进行词嵌入的时候就会进行相应的处理, nn.embedding会将填充的映射为0
            """
            self.embedding = nn.Embedding(config.n_vocab, config.embed,
                                          padding_idx=config.n_vocab - 1)  # 词嵌入，就是把一个词典，随机初始化映射为一个向量矩阵。
            # padding_idx：表示用于填充的参数索引，比如用3填充padding，嵌入向量索引为3的向量设置为0
        self.conv_region=nn.Con2d(1,config.num_filters,(3,config.embed),stride=1)        # (in_channels,out_channels,kernel_size,stride)
        self.conv=nn.Conv2d(config.num_filters,config.num_filters,(3,1),stride=2)
        self.max_pool=nn.MaxPool2d(kernel_size=(3,1),stride=2)
        self.padding1=nn.ZeroPad2d((0,0,1,1))  #top bottom
        self.padding2=nn.ZeroPad2d((0,0,0,1))  # bottom
        self.relu=nn.Relu()

        """
        1. torch.nn.Sequential()
        2. torch.nn.ModuleList()

        """

        self.fc = nn.Linear(config.num_filters , config.num_classes)

    """
    1. torch.squeeze()这个函数对数据的维度进行压缩，去掉维数为1的维度
    2. torch.unsqueeze()对数据维度进行扩充，给指定的位置加上维数为1的维度
    """

    def forward(self, x):
        x=x[0]
        x=self.embedding(x)
        x=x.unsqueeze(1)    #[batch_size,250,seq_len,1]
        x=self.conv_region(x)    # [batch_size, 250, seq_len-3+1, 1]

        x=self.padding1(x)         # [batch_size, 250, seq_len, 1]
        x=self.relu(x)
        x=self.conv(x)
        x=self.padding1(x)
        x=self.relu(x)
        x=x.conv(x)
        while  x.size()[2] >2:
            x=self._block(x)
        x=x.squeeze()
        x=self.fc(x)
        return x

    def _block(self,x):
        x=self.padding2(x)
        px=self.max_pool(x)

        x=self.padding1(px)
        x=F.relu(x)
        x=self.conv(x)
        # short cut
        x=x+px
        return x

