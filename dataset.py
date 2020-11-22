import re
from torchtext import data
import jieba


regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9]')

def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

# 加载停用词
stopwords = stopwordslist('/Users/test/Desktop/hit_stopwords.txt')


def word_cut(text):
    text = regex.sub(' ', text)
    return [word for word in jieba.cut(text) if word not in stopwords]


def get_dataset(path, text_field,label_field):
    text_field.tokenize = word_cut


    train, test = data.TabularDataset.splits(
        path=path, format='tsv', skip_header=True,
        train='train.tsv', validation='test.tsv',
        fields=[
            #(None, None),         # 根据原文格式选择field
            ('label', label_field),
            ('text_a', text_field)
        ]
    )
    return train, test


if __name__ == '__main__':
    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False,use_vocab=True)
    train, val = get_dataset('/Users/test/Desktop/', text_field, label_field)
    text_field.build_vocab(train)
    label_field.build_vocab(train)

    print(len(label_field.vocab))















