# -*- coding: utf-8 -*-

import sys
class DefaultConfig():

    try:
        model_name = sys.argv[1]
    except:
        print("use default model VGG16, see config.py")
        model_name = "VGG16"

    train_data_path = 'dataset/train'
    test_data_path = 'dataset/test'
    checkpoints = './checkpoints/'

    normal_size = 224
    epochs = 1
    batch_size = 2
    classNumber = 2 # see dataset
    channles = 3  # or 3 or 1
    lr = 0.001

    lr_reduce_patience = 5  # 需要降低学习率的训练步长
    early_stop_patience = 10  # 提前终止训练的步长

    data_augmentation = False
    monitor = 'val_loss'
    cut = False
    rat = 0.1

config = DefaultConfig()
