# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 05:05:28 2021

@author: liang
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 15:29:14 2021

@author: liang.zhao@mail.ccnu.edu.cn
"""

# =============================================================================
# def create_model():
#     #keras.backend.clear_session()
#     #inputs        = Input(shape=(nb_time_steps, nb_input_vector,))
#     model = tf.keras.Sequential([
#         Conv1D(filters=32, kernel_size=40, strides=1, padding='valid', activation=None),  #CNN1
#         #1800*32
#         BatchNormalization(),  # BN层
#         Activation('relu'),    # 激活层
#         MaxPooling1D(pool_size=4, strides=4, padding='valid'),  # max池化   #pooling size 8, strides = 20
#         Dropout(0.1),
#         #450*32
#
#         Conv1D(filters=32, kernel_size=40, strides=1, padding='valid', activation=None),#CNN2
#         BatchNormalization(),  # BN层
#         Activation('relu'),    # 激活层
#         MaxPooling1D(pool_size=4, strides=4, padding='valid'),  # max池化
#         Dropout(0.1),
#         #450*32*batch size
#
#         #concact#Dense()
#         #每个特征求mean
#
#         #每一步维度32  投影到128
#         LSTM(128, return_sequences=True, activation='tanh'), #LSTM1   #2万，32*128*4
#         Dropout(0.1),
#         #450*128    *BATCH SIZE
#
#         LSTM(128, return_sequences=False,  activation='tanh'),#LSTM2  #8万，128*128*4
#         Dropout(0.1),
#         #450*128    *BATCH SIZE
#
#         #Dense(1),#全连接层
#         ##Flatten(),
#         Dense(1),#全连接层
#         #Dense(72),#全连接层
#     ])
#     model.compile(optimizer=tf.keras.optimizers.Adam(0.8),loss='mean_squared_error',metrics=['mse'])  # 损失函数用均方误差
#     #mean error, batch error
#     #adamoptimizer = tf.keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.00001)
#     #model.compile(optimizer='rmsprop', loss='mse',metrics=["accuracy"] )
#     return model
# =============================================================================

import numpy as np


def gen_gaussian_noise(signal, SNR):
    """
    :param signal: 原始信号
    :param SNR: 添加噪声的信噪比
    :return: 生成的噪声
    """
    noise = np.random.randn(*signal.shape)  # *signal.shape 获取样本序列的尺寸
    noise = noise - np.mean(noise)
    signal_power = (1 / signal.shape[0]) * np.sum(np.power(signal, 2))
    noise_variance = signal_power / np.power(10, (SNR / 10))
    noise = (np.sqrt(noise_variance) / np.std(noise)) * noise
    signal = signal + noise
    return signal


def create_model_3():
    keras.backend.clear_session()
    v_alpha = 0.3  # 0.0005
    v_dropout = 0.3  # 0.2

    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=64, strides=2, padding='same', activation=keras.layers.LeakyReLU(),
                     input_shape=(nb_time_steps, nb_input_vector,)))
    model.add(Activation(keras.layers.LeakyReLU(alpha=v_alpha)))  # 激活层
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))  # max池化
    model.add(Dropout(v_dropout))

    model.add(Conv1D(filters=64, kernel_size=32, strides=1, padding='same', activation=keras.layers.LeakyReLU()))
    model.add(BatchNormalization())  # BN层
    model.add(Activation(keras.layers.LeakyReLU(alpha=v_alpha)))  # 激活层
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))  # max池化
    model.add(Dropout(v_dropout))

    for i in range(5):  # i+5;10-i
        model.add(layers.Conv1D(filters=2 * (5 + i), kernel_size=3, strides=1, padding='same',
                                activation=keras.layers.LeakyReLU()))
        # if i<=2:
        model.add(layers.BatchNormalization())
        model.add(Activation(keras.layers.LeakyReLU(alpha=v_alpha)))
        # model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
        model.add(Dropout(v_dropout))

    model.add(
        Conv1D(filters=32, kernel_size=64, strides=2, padding='same', activation=keras.layers.LeakyReLU(alpha=v_alpha)))
    model.add(
        Conv1D(filters=32, kernel_size=64, strides=2, padding='same', activation=keras.layers.LeakyReLU(alpha=v_alpha)))
    model.add(layers.BatchNormalization())
    model.add(Activation(keras.layers.LeakyReLU(alpha=v_alpha)))

    # model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))
    model.add(Dropout(v_dropout))
    model.add(GlobalAveragePooling1D())

    # model.add(Dense(32))#, kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01))
    # model.add(AveragePooling1D(pool_size=2,strides=2,padding='valid'))

    model.add(Dense(1, activation=keras.layers.LeakyReLU(alpha=v_alpha)))  # alpha=v_alpha#83,9,7
    # model.add(Dense(1,kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01)))
    model.compile(
        optimizer=tf.keras.optimizers.Nadam(lr=0.1, beta_1=0.9, beta_2=0.999, schedule_decay=0.4, epsilon=1e-08),
        # ,kappa=1-1e-8
        loss=tf.keras.losses.Huber(delta=0.2, reduction="auto", name="huber_loss"),
        metrics=['mse', 'mae'])  # with bathnomolization
    # =============================================================================
    #     model.compile(optimizer=keras.optimizers.RMSprop(lr=0.1,rho=0.9,momentum=0.0,epsilon=1e-7,centered=False,decay=0.00001),  # =**kwargs"clipnorm"    #,128  #趋势对，但测试集波动大  （200,88,9,7）
    #                   loss=tf.keras.losses.Huber(delbta=5.0,reduction="auto",name="huber_loss"),metrics=['mse','mae'])  # 损失函数用均方误差,#metrics=['mse']
    # =============================================================================
    # =============================================================================
    #     model.compile(optimizer=tf.keras.optimizers.Nadam(lr=0.1, beta_1=0.9, beta_2=0.999, schedule_decay=0.4, epsilon=1e-08),       #,kappa=1-1e-8
    #                   loss='mse',metrics=['mae'])
    # =============================================================================
    # =============================================================================
    #     model.compile(optimizer=tf.keras.optimizers.Nadam(lr=0.1, beta_1=0.9, beta_2=0.999, schedule_decay=0.2, epsilon=1e-08),       #,,clipnorm=5.kappa=1-1e-8
    #                   loss=tf.keras.losses.Huber(delta=0.1,reduction="auto",name="huber_loss"),metrics=['mse','mae'])  #;, ,clipvalue=0.5
    # =============================================================================
    # =============================================================================
    #     model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, decay=0.4, epsilon=1e-08),       #,kappa=1-1e-8
    #                   loss='mse',metrics=['mae'])
    # =============================================================================

    model.save('time_cnn.h5')
    return model


def create_model_4():
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
    # https://github.com/xiaosongshine/ECG_challenge_baseline_keras
    # GRU, Attention, Resnet
    keras.backend.clear_session()
    v_alpha = 0.0005  # 0.0005,66,8,6;0.0001,100,10,8;0.001,94,9,7;0.0003,10,8
    v_dropout = 0.3  # 0.2

    model = Sequential()
    # model.add(Embedding(64, 32, input_length = nb_time_steps,input_shape=(nb_time_steps, nb_input_vector,)))
    model.add(Conv1D(32, 64, strides=2, activation=keras.layers.LeakyReLU(alpha=v_alpha), padding="same",
                     input_shape=(nb_time_steps, nb_input_vector,)))  # ,input_shape=(nb_time_steps, nb_input_vector,)
    model.add(Conv1D(32, 64, strides=2, activation=keras.layers.LeakyReLU(alpha=v_alpha), padding="same"))
    model.add(BatchNormalization())  # BN层
    model.add(Activation(keras.layers.LeakyReLU(alpha=v_alpha)))
    model.add(MaxPooling1D(2))
    model.add(Dropout(v_dropout))

    model.add(Conv1D(16, 16, strides=2, activation=keras.layers.LeakyReLU(alpha=v_alpha), padding="same"))
    model.add(Conv1D(16, 16, strides=2, activation=keras.layers.LeakyReLU(alpha=v_alpha), padding="same"))
    model.add(BatchNormalization())  # BN层
    model.add(Activation(keras.layers.LeakyReLU(alpha=v_alpha)))  # model.add(Activation('relu'))   # 激活层
    model.add(MaxPooling1D(2))
    model.add(Dropout(v_dropout))

    model.add(Conv1D(64, 8, strides=1, activation=keras.layers.LeakyReLU(alpha=v_alpha), padding="same"))
    model.add(Conv1D(64, 8, strides=1, activation=keras.layers.LeakyReLU(alpha=v_alpha), padding="same"))
    model.add(BatchNormalization())  # BN层
    model.add(Activation(
        keras.layers.LeakyReLU(alpha=v_alpha)))  # 激活层  alpha=v_alpha  #model.add(Activation('relu'))   # 激活层
    model.add(MaxPooling1D(2))
    model.add(Dropout(v_dropout))

    model.add(Conv1D(128, 4, strides=1, activation=keras.layers.LeakyReLU(alpha=v_alpha), padding="same"))
    model.add(Conv1D(128, 4, strides=1, activation=keras.layers.LeakyReLU(alpha=v_alpha), padding="same"))
    # model.add(BatchNormalization())  # BN层
    model.add(Activation(keras.layers.LeakyReLU(alpha=v_alpha)))
    # model.add(MaxPooling1D(2));model.add(Dropout(v_dropout));model.add(Flatten());model.add(Dense(128, activation=keras.layers.LeakyReLU(alpha=v_alpha)))#alpha=v_alpha
    model.add(GlobalAveragePooling1D())  # pooling_size=2,strides=None

    # model.add(AveragePooling1D(pool_size=2,strides=2,padding='valid'))
    model.add(Dense(1, activation=keras.layers.LeakyReLU(alpha=v_alpha)))  # alpha=v_alpha#83,9,7
    # model.add(Dense(1,kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01)))
    # =============================================================================
    #     model.compile(optimizer=keras.optimizers.RMSprop(lr=0.1,rho=0.9,momentum=0.0,epsilon=1e-7,centered=False,decay=0.00001),  # =**kwargs"clipnorm"    #,128  #趋势对，但测试集波动大  （200,88,9,7）
    #                   loss=tf.keras.losses.Huber(delta=5.0,reduction="auto",name="huber_loss"),metrics=['mse','mae'])  # 损失函数用均方误差,#metrics=['mse']
    # =============================================================================

    # =============================================================================
    #     model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, decay=0.2, epsilon=1e-08),
    #                   loss='mse',metrics=['mae'])  # 损失函数用均方误差,#metrics=['mse']metrics=['mse','mae']
    # =============================================================================
    model.compile(optimizer=keras.optimizers.Nadam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.4),
                  loss=tf.keras.losses.Huber(delta=5.0, reduction="auto", name="huber_loss"), metrics=['mse', 'mae'])  #
    # =============================================================================
    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss='mean_squared_error', optimizer=sgd)
    # =============================================================================
    # =============================================================================
    #     model.compile(optimizer=keras.optimizers.RMSprop(lr=0.0001,rho=0.9,epsilon=1e-8),       #,32，decay=0.001  #趋势对，但测试集波动大  （200,86,9,7）
    #                   loss=tf.keras.losses.MeanSquaredLogarithmicError() ,metrics=['mae'])  # 损失函数用均方误差,#metrics=['mse']
    # =============================================================================
    # =============================================================================
    #     model.compile(optimizer=keras.optimizers.RMSprop(lr=0.001,rho=0.9,epsilon=1e-8),       #,128  #趋势对，但测试集波动大  （200,88,9,7）
    #                   loss='mean_squared_error')  # 损失函数用均 方误差,#metrics=['mse']
    # =============================================================================
    # =============================================================================
    #     model.compile(optimizer=keras.optimizers.RMSprop(lr=0.0001,rho=0.9,epsilon=1e-8, decay=0.0),       #,128  #趋势对，但测试集波动大  （200,88,9,7）
    #                   loss=tf.keras.losses.CosineSimilarity(axis=1),metrics=['mse','mae'])  # 损失函数用均方误差,#metrics=['mse']
    # =============================================================================
    # =============================================================================
    #     model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.04, epsilon=1e-08),
    #                   loss=tf.keras.losses.Huber(),metrics=['mse','mae'])  # 损失函数用均方误差,#metrics=['mse']
    # =============================================================================
    # tf.keras.losses.CosineSimilarity(axis=1);epsilon=1e-8
    # tf.keras.losses.MeanAbsoluteError()
    # tf.keras.losses.MeanSquaredLogarithmicError()
    # tf.keras.losses.Huber()
    # 'mean_squared_error'
    # tf.keras.losses.MeanSquaredLogarithmicError()
    # tf.keras.losses.log_cosh()
    # keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    # =============================================================================
    #     model.compile(optimizer=keras.optimizers.Adadelta(lr=0.0001, rho=0.95, epsilon=1e-8),
    #                   loss='mean_squared_error')  # 损失函数用均方误差,#metrics=['mse']
    # =============================================================================
    # =============================================================================300,训：好；测：100,10,8
    #     model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.04, epsilon=1e-08),
    #                   loss='mean_squared_error')  # 损失函数用均方误差,#metrics=['mse']
    # =============================================================================
    # =============================================================================300,训：中好；测：99,9.98,8.16，中上心率=中，截断
    #     model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.05, beta_1=0.9, beta_2=0.999, decay=0.2, epsilon=1e-08),
    #                   loss='mean_squared_error')  # 损失函数用均方误差,#metrics=['mse']
    # =============================================================================

    # =============================================================================117,10.8,9
    #     model.compile(optimizer=keras.optimizers.Adamax(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.04),
    #                   loss='mean_squared_error')  # 损失函数用均方误差,#metrics=['mse']
    # =============================================================================

    # =============================================================================
    #     model.compile(optimizer=keras.optimizers.RMSprop(lr=0.1,rho=0.9,epsilon=1e-8,decay=0),
    #                   loss='mean_squared_error')  # 损失函数用均方误差,#metrics=['mse']
    # =============================================================================
    return model


def get_ECGSegments(x, fs):
    N_col = x.shape[1]
    N_row = x.shape[0]
    N_sample_perStep = 2 * fs  # 30s, step
    N_sample_perWindow = 8 * fs  # 30s, window size
    win_num = N_col - N_sample_perWindow
    s_win = np.arange(0, win_num + N_sample_perStep, N_sample_perStep)
    N_s = len(s_win)

    N_sample_X = N_s * N_sample_perWindow
    X = np.zeros((N_row, N_sample_X))
    for kk in range(0, N_s, 1):
        ind_start = s_win[kk]
        ind_end = ind_start + N_sample_perWindow
        ss = x[:, ind_start:ind_end]

        iind_start = kk * N_sample_perWindow
        iind_end = iind_start + N_sample_perWindow
        X[:, iind_start:iind_end] = np.array(ss)
    X = np.array(X)
    return (X)


# =============================================================================
#     numl = data.shape[0]
#     d    = np.zeros((numl, 12, 1025, 1))
#     la = []
#     for i in range(0, numl, 12):
#         temp = data[i:i+12, :]
#         la.append(label[i+3])
#         d[k, :, :, 0] = temp
#         k = k+1
#     d = d[0:k, :, :, :]
#     la = np.array(la)
#     return d, la
# =============================================================================


# https://blog.csdn.net/weixin_39653948/article/details/105446709
import os, keras, csv, scipy.stats, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import stats
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, MaxPooling1D, Dropout, Flatten, Dense, LSTM, \
    AveragePooling1D
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, f_classif, chi2
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.utils import shuffle
import keras.backend as K
from keras import regularizers
from keras import layers, optimizers
from keras.callbacks import ReduceLROnPlateau
from keras.utils import to_categorical, np_utils
from keras.models import Sequential, Model
from keras.constraints import max_norm
from keras.callbacks import Callback
from keras.datasets import mnist
from keras.layers import Dense, LSTM, Bidirectional, Dropout, Flatten, Embedding  # RepeatVect
from keras.layers import Activation, AveragePooling2D, Reshape, Conv1D, BatchNormalization
from keras.layers import LeakyReLU, Bidirectional, Dense, LSTM, Concatenate
from keras.layers import GlobalAveragePooling1D, MaxPooling1D
from keras.layers.core import Masking
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import to_categorical, np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers import scikit_learn
from keras.layers import *
from keras.models import *

# ---------------------------------------------------------------------------------------------
# Step 0 parameter setting
fs = 61
flag_GaussianNoise = 1
# ---------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------
# Step 1 load data
# (1.1)data
# df_ippg = pd.read_excel('ippg_1.xlsx')  # good_30s.xlsx,8s_ippg_update_v.xlsx
# x = df_ippg.iloc[:, 5:].values
#
# y = df_ippg.loc[:, 'ECG'].values
# print(type(x))
# print(type(y))
# for i in range(2,25):
#     file_name="ippg_"+str(i)+".xlsx"
#     df_ippg = pd.read_excel(file_name)
#     x = np.vstack((x, df_ippg.iloc[:, 5:].values))
#     y = np.append(y, df_ippg.loc[:, 'ECG'].values)
# print(type(x))
# print(type(y))
# print(x.shape)
# print(y.shape)

df_ippg = pd.read_excel('good_30s.xlsx')  # good_30s.xlsx,8s_ippg_update_v.xlsx
x = df_ippg.iloc[:, 4:].values
y = df_ippg.loc[:, 'ECG'].values

# (1.2)x,scaler&gaussian noise
# =============================================================================
# if flag_GaussianNoise==1:
#     xx       = np.array([gen_gaussian_noise(x[i,:],-1) for i in range(Nx_row)])
# else:
#     xx       = x
# =============================================================================

sc = MinMaxScaler(feature_range=(-1, 1))
# =============================================================================
# xx           = np.transpose(sc.fit_transform(np.transpose(xx))) # 求得训练集的最大值，最小值这些训练集固有的属性，并在训练集上进行归一化
# xx           = xx.astype('float64')
# =============================================================================
x = np.transpose(sc.fit_transform(np.transpose(x)))  # 求得训练集的最大值，最小值这些训练集固有的属性，并在训练集上进行归一化
x = x.astype('float64')
Nx_row = x.shape[0]  # 返回行,样本个数
Nx_col = x.shape[1]  # 返回列
if flag_GaussianNoise == 1:
    xx = np.array([gen_gaussian_noise(x[i, :], 5) for i in range(Nx_row)])
else:
    xx = x

# (1.3)y,scaler
scaler_y = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(y.reshape(-1, 1))
y = scaler_y.transform(y.reshape(-1, 1))
y = y.astype('float')
y = y.reshape(-1, )

# (1.4)train_test_split
p_test_size = 0.1
# N_SegmentPerVideo = 23
# # 按video划分训练集、测试集
# s_video = np.unique(df_ippg.loc[:, 'filename'].values)
# indices = np.arange(s_video.shape[0])
# # indices                 = shuffle(indices, random_state=1337)
# s1, s2, s1, s2, idx1, idx2 = train_test_split(s_video, s_video, indices, test_size=p_test_size, random_state=100)
#
# index_train = [np.arange(i * N_SegmentPerVideo, (i + 1) * N_SegmentPerVideo, 1) for i in idx1]
# index_train = np.array(index_train).reshape(-1, )
# index_test = [np.arange(i * N_SegmentPerVideo, (i + 1) * N_SegmentPerVideo, 1) for i in idx2]
# index_test = np.array(index_test).reshape(-1, )
# =============================================================================
# from sklearn.utils import shuffle
# xx,y = shuffle(xx,y, random_state=1337)
# s1,s2,s1,s2,idx1, idx2  = train_test_split(s_video,s_video,indices,test_size = p_test_size, random_state = 100)
# df_out      = df_ippg_1[df_ippg_1[s1].isin(s_video)]
# df_out      = pd.DataFrame(df_out)
# =============================================================================
x_train, x_test, y_train, y_test = train_test_split(xx, y, test_size=p_test_size)
x_train, y_train = np.array(x_train), np.array(y_train)
x_test, y_test = np.array(x_test), np.array(y_test)


# (1.5)reshape
nb_time_steps, nb_input_vector = fs * 30, 1
x_train = np.reshape(x_train, (x_train.shape[0], nb_time_steps, nb_input_vector))
x_test = np.reshape(x_test, (x_test.shape[0], nb_time_steps, nb_input_vector))
# ---------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------
# Step2  regression model
model = create_model_3()
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                 patience=4,
                                                 verbose=1,
                                                 factor=0.9,
                                                 min_lr=0.00001)
model_lstm = model.fit(x_train, y_train,
                       batch_size=32, epochs=200, validation_data=(x_test, y_test),
                       validation_freq=1, callbacks=[reduce_lr], shuffle=True)  #
model.summary()
# ---------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------
# Step3 visulization
# (1)loss
loss = model_lstm.history['loss']
val_loss = model_lstm.history['val_loss']

# (2)regression
yy_train = model.predict(x_train)
yy_pred = model.predict(x_test)
y_train = scaler_y.inverse_transform(y_train.reshape(-1, 1))
y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1))
yy_pred = scaler_y.inverse_transform(yy_pred.reshape(-1, 1))
yy_train = scaler_y.inverse_transform(yy_train.reshape(-1, 1))

# (3)figure plot
ind_plot_train = np.argsort(y_train.reshape(-1, ))
ind_plot_test = np.argsort(y_test.reshape(-1, ))

plt.subplot(221)
#xnoised= np.array(gen_gaussian_noise(x[0, :], -1))
plt.plot(xx[0, 0:256], label='data+noise');
plt.plot(x[0, 0:256], label='noise,snr=-2')

plt.subplot(222)
plt.plot(scipy.log(loss), label='log10, Training Loss')
plt.plot(scipy.log(val_loss), label='log10, Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

plt.subplot(223)
plt.plot(y_train[ind_plot_train], color='red', label='real_HR')
plt.plot(yy_train[ind_plot_train], color='blue', label='Trained HR')
plt.title('HR Trained')
plt.xlabel('sample')
plt.ylabel('real HR')
plt.legend()

plt.subplot(224)
plt.plot(y_test[ind_plot_test], color='red', label='real_HR')
plt.plot(yy_pred[ind_plot_test], color='blue', label='Predicted HR')
plt.title('HR Prediction')
plt.xlabel('sample')
plt.ylabel('real HR')
plt.legend()
plt.show()

# (4)results
mse = mean_squared_error(yy_pred, y_test)  # calculate MSE 均方误差 ----->E[(预测值-真实值)^2] (预测值减真实值求平方后求均值)
rmse = math.sqrt(mean_squared_error(yy_pred, y_test))  # calculate RMSE 均方根误差--->sqrt[MSE]           (对均方误差开方)
mae = mean_absolute_error(yy_pred, y_test)  # calculate MAE 平均绝对误差-->E[|预测值-真实值|]    (预测值减真实值求绝对值后求均值）
print('mse:  %.3f' % mse)
print('rmse:  %.3f' % rmse)
print('mae :  %.3f' % mae)
# ---------------------------------------------------------------------------------------------

