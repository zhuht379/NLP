#!/usr/bin/env python
#-*- coding:utf-8 -*-


from __future__ import print_function
import  tensorflow as tf
from MODEL import MODEL,ResnetBuilder
import sys
sys.setrecursionlimit(10000)

# from keras import backend as K
# import densenet   #取消densenet模型

class Build_model(object):
    def __init__(self,config):
        self.train_data_path = config.train_data_path
        self.checkpoints = config.checkpoints
        self.normal_size = config.normal_size
        self.channles = config.channles
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.classNumber = config.classNumber
        self.model_name = config.model_name
        self.lr = config.lr
        self.config = config
        # self.default_optimizers = config.default_optimizers
        self.rat = config.rat
        self.cut = config.cut

    def model_confirm(self,choosed_model):
        if choosed_model == 'VGG16':
            model = MODEL(self.config).VGG16()
        elif choosed_model == 'VGG19':
            model = MODEL(self.config).VGG19()
        elif choosed_model == 'AlexNet':
            model = MODEL(self.config).AlexNet()
        elif choosed_model == 'LeNet':
            model = MODEL(self.config).LeNet()
        elif choosed_model == 'ZF_Net':
            model = MODEL(self.config).ZF_Net()
        elif choosed_model == 'ResNet18':
            model = ResnetBuilder().build_resnet18(self.config)
        elif choosed_model == 'ResNet34':
            model = ResnetBuilder().build_resnet34(self.config)
        elif choosed_model == 'ResNet101':
            model = ResnetBuilder().build_resnet101(self.config)
        elif choosed_model == 'ResNet152':
            model = ResnetBuilder().build_resnet152(self.config)
        elif choosed_model =='mnist_net':
            model = MODEL(self.config).mnist_net()
        elif choosed_model == 'TSL16':
            model = MODEL(self.config).TSL16()
        elif choosed_model == 'ResNet50':
            model = tf.keras.applications.ResNet50(include_top=True,
                                                   weights=None,
                                                   input_tensor=None,
                                                   input_shape=(self.normal_size,self.normal_size,self.channles),
                                                   pooling='max',
                                                   classes=self.classNumber)
        elif choosed_model == 'InceptionV3':
            model = tf.keras.applications.InceptionV3(include_top=True,
                                                   weights=None,
                                                   input_tensor=None,
                                                   input_shape=(self.normal_size,self.normal_size,self.channles),
                                                   pooling='max',
                                                   classes=self.classNumber)

        elif choosed_model == 'Xception':
            model = tf.keras.applications.Xception(include_top=True,
                                                weights=None,
                                                input_tensor=None,
                                                input_shape=(self.normal_size,self.normal_size,self.channles),
                                                pooling='max',
                                                classes=self.classNumber)
        elif choosed_model == 'MobileNet':
            model = tf.keras.applications.MobileNet(include_top=True,
                                                 weights=None,
                                                 input_tensor=None,
                                                 input_shape=(self.normal_size,self.normal_size,self.channles),
                                                 pooling='max',
                                                 classes=self.classNumber)

        return model

    def model_compile(self,model):
        adam = tf.keras.optimizers.Adam(lr=self.lr)
        model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])  # compile之后才会更新权重和模型
        return model

    def build_model(self):
        model = self.model_confirm(self.model_name)
        model = self.model_compile(model)
        return model