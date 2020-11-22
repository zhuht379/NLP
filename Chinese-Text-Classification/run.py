import time
import torch
import numpy as np
from train_eval import train,init_network
from importlib import import_module
"""
一个函数运行需要根据不同项目的配置，动态导入对应的配置文件运行,多个模型根据model_name进行动态的导入脚本中的对象，
动态的取不同脚本中的对象

"""
import argparse

parser=argparse.ArgumentParser(description='Chinese Text Classification')

parser.add_argument('--model',default="TextCNN",type=str,help='choose a model: TextCNN,TextRNN,FastText,TextRCNN,TextRNN_Att,DPCNN,Transformer')
parser.add_argument('--embedding',default='pre_trained',type=str,help=" random or pre_trained")
parser.add_argument('--word',default=False,type=bool,help='True for word ,False for char')
args=parser.parse_args()

if __name__ == '__main__':

    dataset='THUCNews'
    #embedding='embedding_SougouNews.npz'
    embedding='embedding_Tencent.npz'
    if args.embedding=='random':
        embedding='random'
    model_name=args.model
    if model_name=="FastText":
        from utils_fasttext import build_dataset,build_iterator,get_time_dif
        embedding='random'
    else:
        from utils import build_dataset,build_iterator,get_time_dif

    X=import_module('models.'+model_name)    # models. 文件夹下的model_name python脚本文件
    config=X.Config(dataset,embedding)
    """
    保证每次训练时的初始化时确定的。
    在神经网络中，参数默认是进行随机初始化的，不同的初始化参数往往导致不同的结果，当得到比较好的结果时，
    我们通常希望这个结果是可以复现的，在pytorch中，通过设置随机数种子也可以达到这样的目的

    """
    np.random.seed(1)                # 每次生成相同的随机数
    torch.manual_seed(1)             # 为cpu设置随机种子，保证每次神经网络相同的初始化初始化
    torch.cuda.manual_seed_all(1)    #为当前GPU设置随机种子
    torch.backends.cudnn.deterministic=True # 保证每次结果一样

    start_time=time.time()
    print("Loading data ...")
    vocab,train_data,dev_data,test_data=build_dataset(config,args.word)
    train_iter=build_iterator(train_data,config)
    dev_iter=build_iterator(dev_data,config)
    test_iter=build_iterator(test_data,config)
    time_dif=get_time_dif(start_time)
    print("Time Usage:",time_dif)


    # train
    config.n_vocab=len(vocab)
    model=X.Model(config).to(config.device)
    if model_name !='Transformer':
        init_network(model)
    print(model.parameters)
    train(config,model,train_iter,dev_iter,test_iter)

    #embedding = 'THUCNews/data/embedding_Tencent.npz'
    #print(np.load(embedding))