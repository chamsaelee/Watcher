import os, sys, time
import tkinter as tk
import matplotlib.pyplot as plt
from datetime import datetime
from RunTabnetFunction import showmethepredict

import tensorflow as tf
import tabnet
import pandas as pd
import numpy as np
import csv
from tqdm import tqdm
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow.keras.models
from typing import Optional, Union, Tuple
import tensorflow_addons as tfa

import keras
import h5py
import warnings
warnings.filterwarnings(action="ignore")

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
except:
    print("error")
    os.system("pip install watchdog")

class TN:
    def make_data(data_path):
            data_condition = pd.read_excel(data_path,header=None)
            y_90 = data_condition.iloc[:1,1:].T
            x_90 = data_condition.iloc[1:,1:].T
            y_90.columns = ["y"]
            x_90.columns = data_condition.iloc[1:,:1][0].tolist()
            total_90 = pd.concat([x_90,y_90],axis=1)
            return total_90

    def get_feature(x: pd.DataFrame
                    , dimension=1) -> Union[tf.feature_column.numeric_column
                                            , tf.feature_column.embedding_column]:
        if x.dtype == np.float32:
            return tf.feature_column.numeric_column(str(x.name))
        else:
            return tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_identity(x.name, num_buckets=x.max()+2
                                                            , default_value=0),
            dimension=dimension)
        
    def df_to_dataset(X: pd.DataFrame, y: pd.Series, shuffle=False
                    , batch_size=4) -> tf.data.Dataset.from_tensor_slices:
        ds = tf.data.Dataset.from_tensor_slices((dict(X.copy()), y.copy()))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(X))
        ds = ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        return ds

    def build_model():
        # weight, data
        data = TN.make_data("C:/Users/BRAIN/OneDrive - (주)에이치엠이스퀘어/업무_공유/1. 개발/2. DATA/1. Data Excel 모음/905nm/905nm_1214.xlsx")
        weight = "C:/Users/BRAIN/OneDrive - (주)에이치엠이스퀘어/업무_공유/1. 개발/2. DATA/2. Data (raw)/testing/905nm_1214_6%_12%.h5"
        # parameters
        optimizer = tf.keras.optimizers.Adam(0.005)
        lr = 0.1e-7

        data = data*(data>0)
        data.columns = [str(i) for i in data.columns]
        X_col_name = data.columns[:-1]
        CATEGORICAL_COLUMNS = []
        NUMERIC_COLUMNS = X_col_name
        data_tabnet = data.copy()
        data_tabnet.loc[:, NUMERIC_COLUMNS] = data_tabnet.loc[:, NUMERIC_COLUMNS].astype(np.float32).pipe(
            lambda data_tabnet: np.log(data_tabnet)).pipe(
            lambda data_tabnet: abs(data_tabnet))

        X = data_tabnet.loc[:, list(NUMERIC_COLUMNS)].astype('float32')
        y = data_tabnet[["y"]]
        data_tabnet = data_tabnet.loc[:, list(NUMERIC_COLUMNS)].astype('float32')

        X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=int(len(data_tabnet)*(0.8))
                                                , test_size=int(len(data_tabnet)*(0.2)),random_state=7777)

        columns = [TN.get_feature(f) for k, f in data_tabnet[X_col_name].iteritems()]
        feature_column = tf.keras.layers.DenseFeatures(columns, trainable=True)

        train, valid = TN.df_to_dataset(X_train, y_train), TN.df_to_dataset(X_valid, y_valid)
        tf.keras.backend.clear_session

        model = tabnet.TabNetRegression(columns, 
                                        num_regressors=1,
                                        feature_dim=8, # 데이터 양이 증가하면 조절해봐야할 파라미터
                                        output_dim=4, # 데이터 양이 증가하면 조절해봐야할 파라미터
                                        num_decision_steps=1 # 데이터 양이 증가하면 조절해봐야할 파라미터
                                )

        model.compile(optimizer, 
                    loss = tfa.losses.pinball_loss,
                    metrics=[tf.keras.metrics.MAE,tf.keras.metrics.MAPE])

        callbacks = ModelCheckpoint("t.h5" ## 채택
                                        , monitor='val_mean_absolute_percentage_error', 
                                    verbose=0, save_best_only=True, save_weights_only=True)
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                    cooldown=0,
                                    patience=5,
                                    min_lr=lr)
        model.fit(train, epochs=1, validation_data=valid
                , verbose=0, batch_size=1,callbacks=[callbacks,lr_reducer])

        model.load_weights(weight)
        return model

model = TN.build_model()

class Handler(FileSystemEventHandler):
    def on_created(self, event):
        print(f'event type : {event.event_type}\n'
              f'event src_path : {event.src_path}')
        if event.is_directory:
            pass
        else:
            Fname, Extension = os.path.splitext(os.path.basename(event.src_path))
            if Extension =='.csv': 
                fname = str(fpath) + '/' + str(Fname) + '.csv'
                text = {}
                print("predicting ... !")
                text["value"] = showmethepredict(model, fname)
                now = datetime.now()
                date = str(now.strftime("%m")) + "월 "+str(now.strftime("%d")) + "일   "+str(now.strftime("%H"))+" : "+str(now.strftime("%M"))
                text["time"] = date
                add_result = pd.DataFrame(text, index=[0])
                result = pd.read_excel("result.xlsx")
                result = pd.concat([result, add_result], axis=0)
                result.to_excel("result.xlsx", index=False)


class App:
    def __init__(self, path):
        print("detecting ... ") 
        self.event_handler = None
        self.observer = Observer()
        self.target_directory = path
        self.currentDirectorySetting()
    
    def currentDirectorySetting(self):
        print("======================================")
        print("cwd : ",end=" ")
        os.chdir(self.target_directory)
        print("{cwd}".format(cwd=os.getcwd()))
        print("======================================")

    def run(self):
        self.event_handler = Handler()
        self.observer.schedule(
            self.event_handler,
            self.target_directory,
            recursive=False
        )
        self.observer.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt as e:
            print("stop detecting ... ")
            self.observer.stop()


fpath = "C:/Users/BRAIN/OneDrive - (주)에이치엠이스퀘어/업무_공유/1. 개발/2. DATA/2. Data (raw)/testing"
myWatcher = App(fpath)
myWatcher.run()