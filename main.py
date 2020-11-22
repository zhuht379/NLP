import argparse
import torchtext.data as data
from torchtext.vocab import Vectors

import model
import train
import dataset

parser = argparse.ArgumentParser(description='TextCNN text classifier')
# learning
parser.add_argument('-lr', type=float, default=0.001)
parser.add_argument('-epochs', type=int, default=256)
parser.add_argument('-batch-size', type=int, default=128)
parser.add_argument('-log-interval', type=int, default=1)
parser.add_argument('-test-interval', type=int, default=100)
parser.add_argument('-save-dir', type=str, default='snapshot')
parser.add_argument('-early-stopping', type=int, default=1000)
parser.add_argument('-save-best', type=bool, default=True)
# model
parser.add_argument('-dropout', type=float, default=0.5)
parser.add_argument('-max-norm', type=float, default=3.0)
parser.add_argument('-embedding-dim', type=int, default=128)
parser.add_argument('-filter-num', type=int, default=100)
parser.add_argument('-filter-sizes', type=str, default='3,4,5')

# device
parser.add_argument('-device', type=int, default=-1)


args = parser.parse_args()


def load_word_vectors(model_name, model_path):
    vectors = Vectors(name=model_name, cache=model_path)
    return vectors







def load_dataset(text_field, label_field, args):
    train, test = dataset.get_dataset('/users/test/Desktop/', text_field, label_field)
    text_field.build_vocab(train, test)         #构建词表
    label_field.build_vocab(train, test)        #构建标签表

    train_iter, test_iter = data.BucketIterator.splits(             # 构造迭代器
        (train, test),
        batch_sizes=(args.batch_size, len(test)),        # 每次batch大小， 未将train划分训练和验证集
        sort_key=lambda x: len(x.text_a),
        sort_within_batch=False,
        device=-1, repeat=False
       )

    return train_iter, test_iter


"""
torch处理数据步骤，field  dataset  和迭代器三部分

"""

print('Loading data...')
#构建field 对象
text_field = data.Field(lower=True)
label_field = data.Field(sequential=False, use_vocab=True)
# 生成dataset数据集
train_iter, test_iter = load_dataset(text_field, label_field, args)

# for i in  train_iter:
#     print(i)





args.vocabulary_size = len(text_field.vocab)

args.class_num = len(label_field.vocab)
# print(args.class_num)


args.filter_sizes = [int(size) for size in args.filter_sizes.split(',')]

print('Parameters:')
for attr, value in sorted(args.__dict__.items()):
    if attr in {'vectors'}:
        continue
    print('\t{}={}'.format(attr.upper(), value))

text_cnn = model.TextCNN(args)
#print(text_cnn)


#print(train_iter.__dict__.keys())


try:
    train.train(train_iter, test_iter, text_cnn, args)
except KeyboardInterrupt:
    print('Exiting from training early')


