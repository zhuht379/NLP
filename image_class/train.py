
from __future__ import print_function
from sklearn.model_selection import train_test_split
import tensorflow as tf
from config import config
from data_oprt2 import *
from Build_model import Build_model

import sys
sys.setrecursionlimit(10000)



class Train(Build_model):
    def __init__(self,config):
        Build_model.__init__(self,config)

    def get_file(self,path):
        ends = os.listdir(path)[0].split('.')[-1]
        img_list = glob.glob(os.path.join(path,'*.'+ends))
        return img_list

    def load_data(self):
        images_data, labels = instance_get_separate_imgdataset()
        images_data = np.array(images_data, dtype='float32') / 255.0
        X_train, X_test, y_train, y_test = train_test_split(images_data,labels)
        return X_train, X_test, y_train, y_test

    def mkdir(self,path):
        if not os.path.exists(path):
            return os.mkdir(path)
        return path

    def train(self,X_train, X_test, y_train, y_test,model):
        print("*"*50)
        print("-"*20+"train",config.model_name+"-"*20)
        print("*"*50)

        tensorboard=tf.keras.callbacks.TensorBoard(log_dir=self.mkdir(os.path.join(self.checkpoints,self.model_name) ))

        lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor=config.monitor,
                                                      factor=0.1,
                                                      patience=config.lr_reduce_patience,
                                                      verbose=1,
                                                      mode='auto',
                                                      cooldown=0)
        early_stop = tf.keras.callbacks.EarlyStopping(monitor=config.monitor,
                                                   min_delta=0,
                                                   patience=config.early_stop_patience,
                                                   verbose=1,
                                                   mode='auto')
        checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(self.mkdir( os.path.join(self.checkpoints,self.model_name) ),self.model_name+'.h5'),
                                                     monitor=config.monitor,
                                                     verbose=1,
                                                     save_best_only=True,
                                                     save_weights_only=True,
                                                     mode='auto',
                                                     period=1)

        model.fit(x=X_train,y=y_train,
                  batch_size=self.batch_size,
                  validation_data=(X_test,y_test),
                  epochs=self.epochs,
                  callbacks=[early_stop,checkpoint,lr_reduce,tensorboard],
                  shuffle=True,
                  verbose=1)

    def start_train(self):
        X_train, X_test, y_train, y_test=self.load_data()
        model = Build_model(config).build_model()
        self.train(X_train, X_test, y_train, y_test,model)

    def remove_logdir(self):
        self.mkdir(self.checkpoints)
        self.mkdir(os.path.join(self.checkpoints,self.model_name))
        events = os.listdir(os.path.join(self.checkpoints,self.model_name))
        for evs in events:
            if "events" in evs:
                os.remove(os.path.join(os.path.join(self.checkpoints,self.model_name),evs))

    def mkdir(self,path):
        if os.path.exists(path):
            return path
        os.mkdir(path)
        return path


def main():
    train = Train(config)
    train.remove_logdir()
    train.start_train()
    print('Done')

if __name__=='__main__':
    main()
