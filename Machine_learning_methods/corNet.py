# =============================================================================
def create_model():
    # keras.backend.clear_session()
    # inputs        = Input(shape=(nb_time_steps, nb_input_vector,))
    model = tf.keras.Sequential([
        Conv1D(filters=32, kernel_size=40, strides=1, padding='valid', activation=None),  # CNN1
        # 1800*32
        BatchNormalization(),  # BN层
        Activation('relu'),  # 激活层
        MaxPooling1D(pool_size=4, strides=4, padding='valid'),  # max池化   #pooling size 8, strides = 20
        Dropout(0.1),
        # 450*32

        Conv1D(filters=32, kernel_size=40, strides=1, padding='valid', activation=None),  # CNN2
        BatchNormalization(),  # BN层
        Activation('relu'),  # 激活层
        MaxPooling1D(pool_size=4, strides=4, padding='valid'),  # max池化
        Dropout(0.1),
        # 450*32*batch size

        # concact#Dense()
        # 每个特征求mean

        # 每一步维度32  投影到128
        LSTM(128, return_sequences=True, activation='tanh'),  # LSTM1   #2万，32*128*4
        Dropout(0.1),
        # 450*128    *BATCH SIZE

        LSTM(128, return_sequences=False, activation='tanh'),  # LSTM2  #8万，128*128*4
        Dropout(0.1),
        # 450*128    *BATCH SIZE

        # Dense(1),#全连接层
        ##Flatten(),
        Dense(1),  # 全连接层
        # Dense(72),#全连接层
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.8), loss='mean_squared_error', metrics=['mse'])  # 损失函数用均方误差
    # mean error, batch error
    # adamoptimizer = tf.keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.00001)
    # model.compile(optimizer='rmsprop', loss='mse',metrics=["accuracy"] )
    # model.compile(loss='categorical_crossentropy',
    #                  optimizer='rmsprop',
    #                  metrics=['accuracy'])
    return model


# =============================================================================
from keras import regularizers


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


def create_model_2():
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
    keras.backend.clear_session()

    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=90, strides=1, padding='valid', activation=None,
                     input_shape=(nb_time_steps, nb_input_vector)))  # CNN1
    # 1800*32
    model.add(BatchNormalization())  # BN层
    # model.add(Activation('tanh'))   # 激活层
    # model.add(Activation('relu'))    # 激活层
    model.add(Activation(keras.layers.LeakyReLU(alpha=0.3)))  # 激活层
    # model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling1D(pool_size=4, strides=4, padding='valid'))  # max池化   #pooling size 8, strides = 20
    model.add(Dropout(0.1))
    # 450*32

    model.add(Conv1D(filters=64, kernel_size=90, strides=1, padding='valid', activation=None))  # CNN2
    model.add(BatchNormalization())  # BN层
    # model.add(Activation('tanh'))   # 激活层
    # model.add(Activation('relu'))    # 激活层
    model.add(Activation(keras.layers.LeakyReLU(alpha=0.3)))  # 激活层
    # model.add(LeakyReLU(alpha=0.5))
    model.add(MaxPooling1D(pool_size=4, strides=4, padding='valid'))  # max池化
    model.add(Dropout(0.1))
    # 450*32*batch size
    # concact#Dense()
    # 每个特征求mean

    # 每一步维度32  投影到128
    # model.add(LSTM(128, return_sequences=False, activation='tanh'))
    model.add(Bidirectional(LSTM(128, return_sequences=True, activation='tanh')))  # LSTM1   #2万，32*128*4;'tanh'
    model.add(BatchNormalization())  # BN层
    model.add(Dropout(0.1))  # 450*128    *BATCH SIZE
    #
    # model.add(LSTM(128, return_sequences=False, activation='tanh'))
    model.add(Bidirectional(LSTM(128, return_sequences=False, activation='tanh')))  # LSTM2  #8万，128*128*4;'tanh'
    model.add(BatchNormalization())  # BN层
    model.add(Dropout(0.1))  # 450*128    *BATCH SIZE

    # =============================================================================
    # ee
    # =============================================================================

    # model.add(BatchNormalization())  # BN层
    # model.add(LeakyReLU(alpha=0.05))   # 激活层
    model.add(Flatten())
    # model.add(Activation('relu'))   # 激活层
    # model.add(Dense(128))#全连接层
    # model.add(Dropout(0.1))
    model.add(Dense(1))  # 全连接层

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.02, epsilon=1e-08),
                  # optimizer=keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004),
                  loss='mean_squared_error')  # 损失函数用均方误差,#metrics=['mse']
    # lr=0.0005, beta_1=0.9, beta_2=0.999, decay=0.002,
    # optimizer=tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.004, epsilon=1e-08)
    # model.compile(loss='categorical_crossentropy',
    #                  optimizer='rmsprop',
    #                  metrics=['accuracy'])
    return model


from keras.callbacks import ReduceLROnPlateau

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, MaxPooling1D, Dropout, Flatten, Dense, LSTM
import matplotlib.pyplot as plt
import os, math, keras
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical, np_utils
from keras.models import Sequential, Model
from keras.layers import LeakyReLU, Bidirectional, Dense, LSTM

# Step1 读取数据 分割训练集和测试集
df_ippg = pd.read_excel('30s_ippg.xlsx')
x = df_ippg.iloc[:, 3:].values
y = df_ippg.loc[:, 'ECG'].values

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


# df_ippg = pd.read_excel('SPLIT(1).xlsx',header=None)
# x = df_ippg.iloc[:, 0:].values
# df_ippg_y = pd.read_excel('output_data.xlsx')
# temp = df_ippg_y.loc[:, 'ECG'].values
# y=temp
# for i in range(0,22):
#     y = np.append(y, temp)
# x = np.vstack((x, x[:, ::-1]))
# y = np.append(y, y)
# sigma = 1e-9
# noise1 = sigma * np.random.randn(len(x), len(x[0]))
# x2 = x + noise1
# x = np.vstack((x, x2))
# y = np.append(y, y)
# x = x2
# y = y
# 归一化
# sc = MinMaxScaler(feature_range=(-1, 1))
# x = np.transpose(sc.fit_transform(np.transpose(x)))  # 求得训练集的最大值，最小值这些训练集固有的属性，并在训练集上进行归一化
# x = x.astype('float64')
# y = y.astype('int64')
# y = y.reshape(-1, )
# =============================================================================
# nn_y_1        = len(set(y))
# nn_y_2        = max(y)-min(y)
# y_encoder     = keras.utils.to_categorical(y, num_classes=nn_y_2)
# num_labels    = len(y)#len(labels)
# num_classes   = nn_y_1
# 
# #生成值全为0的独热编码的矩阵
# y_encoder     = np.zeros((num_labels, num_classes))
# #计算向量中每个类别值在最终生成的矩阵“压扁”后的向量里的位置
# index_offset  = np.arange(num_labels) * num_classes
# #遍历矩阵，为每个类别的位置填充1
# y_encoder.flat[index_offset + y] = 1
# =============================================================================

p_test_size = 0.1
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=p_test_size, random_state=2)
x_train, y_train = np.array(x_train), np.array(y_train)
x_test, y_test = np.array(x_test), np.array(y_test)
#
# y_axis = np.linspace(0, len(x[0]), num=len(x[0]))
# plt.plot(y_axis, x_train[0], color='red', linestyle='', marker='.')
#
# sigma = 1e-7
# noise1 = sigma * np.random.randn(len(x_train), len(x_train[0]))
# x2 = x_train + noise1
# x_train = np.vstack((x_train, x2))
# y_train = np.append(y_train, y_train)

# x_train = x2
# y_train = y_train
# plt.plot(y_axis, x_train[792], color='blue', linestyle='', marker='.')
# plt.show()

sc = MinMaxScaler(feature_range=(-1, 1))
x_train = np.transpose(sc.fit_transform(np.transpose(x_train)))  # 求得训练集的最大值，最小值这些训练集固有的属性，并在训练集上进行归一化
x_train = x_train.astype('float64')
y_train = y_train.astype('int64')
y_train = y_train.reshape(-1, )

x_test = np.transpose(sc.fit_transform(np.transpose(x_test)))  # 求得训练集的最大值，最小值这些训练集固有的属性，并在训练集上进行归一化
x_test = x_test.astype('float64')
y_test = y_test.astype('int64')
y_test = y_test.reshape(-1, )

# x_train = np.array([gen_gaussian_noise(x_train[i, :], 0) for i in range(x_train.shape[0])])
# x_test = np.array([gen_gaussian_noise(x_test[i, :], 0) for i in range(x_test.shape[0])])

fs = 61
nb_time_steps, nb_input_vector = fs * 30, 1
x_train = np.reshape(x_train, (x_train.shape[0], nb_time_steps, nb_input_vector))
x_test = np.reshape(x_test, (x_test.shape[0], nb_time_steps, nb_input_vector))

# Step2 定义网络结构
model = create_model_2()

# Step3断点续训 保存模型，每次运行再上次基础上继续训练
# 模型保存在 checkpoint文件夹和weight文件
checkpoint_save_path = "checkpoint/corNet_1.ckpt"  # 模型保存路径
if os.path.exists(checkpoint_save_path + '_1.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 monitor='val_loss')
model_lstm = model.fit(x_train, y_train, batch_size=32, epochs=200, validation_data=(x_test, y_test), validation_freq=1,
                       callbacks=[cp_callback])
# 32

# Step5模型结构和训练效果的可视化
model.summary()
file = open('weights_1.txt', 'w')  # 参数提取
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

loss = model_lstm.history['loss']
val_loss = model_lstm.history['val_loss']

plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Step6 测试集输入模型进行预测
yy_train = model.predict(x_train)
yy_pred = model.predict(x_test)

print(yy_train)
print(yy_pred)
# 可能的步骤

# Step7 画出真实数据和预测数据的对比曲线 
ind_plot_train = np.argsort(y_train)
# error_plot = yy_pred[ind_plot]-y_test[ind_plot]
plt.plot(y_train[ind_plot_train], color='red', label='real_HR')
plt.plot(yy_train[ind_plot_train], color='blue', label='Trained HR')
# plt.plot(error_plot, color='black', label='error')
plt.title('HR Trained')
plt.xlabel('sample')
plt.ylabel('real HR')
plt.legend()
plt.show()

ind_plot_test = np.argsort(y_test)
# error_plot = yy_pred[ind_plot]-y_test[ind_plot]
plt.plot(y_test[ind_plot_test], color='red', label='real_HR')
plt.plot(yy_pred[ind_plot_test], color='blue', label='Predicted HR')
# plt.plot(error_plot, color='black', label='error')
plt.title('HR Prediction')
plt.xlabel('sample')
plt.ylabel('real HR')
plt.legend()
plt.show()

# calculate MSE 均方误差 ---> E[(预测值-真实值)^2] (预测值减真实值求平方后求均值)
mse = mean_squared_error(yy_pred, y_test)
mse_t = mean_squared_error(yy_train, y_train)
# calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
rmse = math.sqrt(mean_squared_error(yy_pred, y_test))
rmse_t = math.sqrt(mean_squared_error(yy_train, y_train))
# calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
mae = mean_absolute_error(yy_pred, y_test)
mae_t = mean_absolute_error(yy_train, y_train)
print('test set')
print('mse:     %.6f' % mse)
print('rmse:   %.6f' % rmse)
print('mae: %.6f' % mae)
print('train set')
print('mse:     %.6f' % mse_t)
print('rmse:   %.6f' % rmse_t)
print('mae: %.6f' % mae_t)
