# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from tensorboardX import SummaryWriter

"""
1、model.named_parameters(),迭代打印model.named_parameters()将会打印每一次迭代元素的名字和param
for name,param in model.named_parameters():
    print(name,param.reqquires_grad()
    param.requires_grad=False

2、model.parameters(),迭代打印model.parameters()将会打印每一次迭代元素的param，而不会打印名字，都可以改变param.requires_grad()
for  param in model.parameters():
	print(param.requires_grad)
	param.requires_grad=False
3. model.state_dict().items() 每次迭代打印该选项的话，会打印所有的name和param，但是这里的所有的param都是requires_grad=False,
没有办法改变requires_grad的属性，所以改变requires_grad的属性只能通过上面的两种方式。
for name, param in model.state_dict().items():
	print(name,param.requires_grad=True)    # True 表示可以参与求导，向后传播
4、 改变了requires_grad之后要修改optimizer的属性
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),   #只更新requires_grad=True的参数
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )

5、随机参数初始化
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight.data)
model.apply(init_weights)
"""


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':      #https://blog.csdn.net/weixin_39653948/article/details/107950764
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()                        #eval（）时，框架会自动把BN和DropOut固定住，不会取平均，而是用训练好的值，

    """
        为了使用torch.optim,需要构造一个优化器对象Optimizer，用来保存当前的状态，并保存当前的状态，并能够根据计算得到的梯度更新参数
        必须给它一个可进行迭代优化的包含了所有参数（所有的参数必须是变量s）的列表。 然后，可以指定程序优化特定的选项，例如学习速率，权重衰减等
        optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
        optimizer = optim.Adam([var1, var2], lr = 0.0001)
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        Optimizer还支持指定每个参数选项。 只需传递一个可迭代的dict来替换先前可迭代的Variable。dict中的每一项都可以定义为一个单独的参数组，
        参数组用一个params键来包含属于它的参数列表。其他键应该与优化器接受的关键字参数相匹配，才能用作此组的优化选项。
        optim.SGD([
                    {'params': model.base.parameters()},
                    {'params': model.classifier.parameters(), 'lr': 1e-3}
                ], lr=1e-2, momentum=0.9)
        如上，model.base.parameters()将使用1e-2的学习率，model.classifier.parameters()将使用1e-3的学习率。0.9的momentum作用于所有的parameters
        优化步骤：
        所有的优化器Optimizer都实现了step()方法来对所有的参数进行更新，它有两种调用方法：
        optimizer.step()
        这是大多数优化器都支持的简化版本，使用如下的backward()方法来计算梯度的时候会调用它
        for input, target in dataset:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        """

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        for i, (trains, labels) in enumerate(train_iter):         #iterator存放的所有的batch
            outputs = model(trains)
            model.zero_grad()

            """
                        根据pytorch 中的backward（）函数计算，当网络参量进行反馈时，梯度时被累积的为不是被替换掉的，因此需要每一个batch 设置一遍zero_grad
                        model.zero_grad() 
                        optimizer.zero_grad()都是把模型中参数的梯度设置为0
                        当optimizer=optim.Optimizer(model.parameters()) 二者等效
                        如果想把某一个Variable的梯度设置为0，只需要  Variable.grad.data.zero_()

            """
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            """
                       optimizer.step()通常用在每个mini-batch之中，而scheduler.step()通常用在epoch里面,但是不绝对，可以根据具体的需求来做。
                       只有用了optimizer.step()，模型才会更新，而scheduler.step()是对lr进行调整

            """
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()                       #模型训练在GPU上，模型评估在CPU上
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)    # 保存更新最优的模型结构
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                #“<”左对齐，“>”右对齐，“^”居中
                # 字符串默认左对齐，数字默认右对齐
                # “:”相当自带索引，可重复添加
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:   #100----1100可能100 --200 之间出现降低，但是200词的时候loss又上去了
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break                 # batch上终止
        if flag:
            break      # epoch上停止循环
    writer.close()
    test(config, model, test_iter)           #在测试集上解释模型结果


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():              #是一个上下文管理器，被该局wrap起来的部分将不会track 梯度，也不会进行反向传播
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)