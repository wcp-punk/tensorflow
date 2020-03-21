'''
version 3
功能：添加了按照时间创建保存模型文件夹的功能
版本：tf1.5-gpu python3.7
'''
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os, time
from datetime import datetime

need_train = False

#----------------------------------------------------------------
#创建保存模型用的文件夹
# if need_train:
#     strat_time = time.time()
#     saveDir = 'savedModels/'
#     cwd = os.getcwd()
#     directory = saveDir + datetime.now().strftime("%b_%d_%I_%M%p_") + 'bike_wcp'
    
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#     sessFileName = directory + '/model'
# else:
#     directory='savedModels/Mar_21_10_59AM_bike_wcp'
#----------------------------------------------------------------
#准备数据
data_path = './bike-sharing-dataset/hour.csv'
rides = pd.read_csv(data_path)

dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides = pd.concat([rides, dummies], axis=1)
    
fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis=1)

quant_feature = ['cnt', 'temp', 'hum', 'windspeed']

scaled_features = {}
for each in quant_feature:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean) / std

test_data = data[-21 * 24:]
train_data = data[:-21 * 24]

target_fields = ['cnt', 'casual', 'registered']

features, targets = train_data.drop(target_fields, axis=1), train_data[target_fields]

test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]

X = features.values

Y = targets['cnt'].values
Y = Y.astype(float)
Y = np.reshape(Y, [len(Y), 1])

#测试数据----------------------------------------------------------------------------------
targets = test_targets['cnt']  #读取测试集的cnt数值
targets = targets.values.reshape([len(targets), 1])  #将数据转换成合适的tensor形式
targets = targets.astype(float)  #保证数据为实数
#-----------------------------------------------------------------------------------------
input_size = features.shape[1]
hidden_size = 10
output_size = 1
batch_size = 128

#------------------------------------------------------------------------
if need_train:
    #build_new_dir-----------------------------------------------------------------------------
    strat_time = time.time()
    saveDir = 'savedModels/'
    cwd = os.getcwd()
    directory = saveDir + datetime.now().strftime("%b_%d_%I_%M%p_") + 'bike_wcp'
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    sessFileName = directory + '/model'
    #build_new_model---------------------------------------------------------------------------
    tf_x = tf.placeholder(tf.float32, shape=(None, input_size), name='x_input')  # input x
    tf_y = tf.placeholder(tf.float32, shape=(None, output_size), name='y_input')  # input y

    # neural network layers
    l1 = tf.layers.dense(tf_x, hidden_size, tf.nn.sigmoid)  # hidden layer
    output = tf.layers.dense(l1, output_size)  # output layer

    loss = tf.compat.v1.losses.mean_squared_error(tf_y, output)  # compute cost
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss)
    # train------------------------------------------------------------------------------------
    saver = tf.compat.v1.train.Saver()  # define a saver for saving and restoring
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())  # initialize var in graph

        for step in range(1000):
            # training
            for start in range(0, len(X), batch_size):
                end = start + batch_size if start + batch_size < len(X) else len(X)
                _, loss_value = sess.run([train_op, loss], feed_dict={tf_x: X[start:end], tf_y: Y[start:end]})
            if step % 100 == 0:
                print("steps:", step, "loss:", loss_value)
        
        saver.save(sess, sessFileName)  # meta_graph is not recommended
else:
    # test--------------------------------------------------------------------------------------
    # # destroy previous net
    tf.reset_default_graph()
    #build_model--------------------------------------------------------------------------------
    tf_x = tf.placeholder(tf.float32, shape=(None, input_size), name='x_input')  # input x
    tf_y = tf.placeholder(tf.float32, shape=(None, output_size), name='y_input')  # input y

    # neural network layers
    l1 = tf.layers.dense(tf_x, hidden_size, tf.nn.sigmoid)  # hidden layer
    output = tf.layers.dense(l1, output_size)  # output layer

    loss = tf.compat.v1.losses.mean_squared_error(tf_y, output)  # compute cost
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss)
    #choose_dir---------------------------------------------------------------------------------
    directory = 'savedModels/Mar_21_11_24AM_bike_wcp'
    #-------------------------------------------------------------------------------------------
    with tf.compat.v1.Session() as sess:
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, directory + '/model')
        predict = sess.run(output, feed_dict={tf_x: test_features.values, tf_y: targets})
        
    fig, ax = plt.subplots(figsize=(10, 7))
    mean, std = scaled_features['cnt']
    ax.plot(predict * std + mean, label='Prediction', linestyle='--')
    ax.plot(targets * std + mean, label='Data', linestyle='-')
    ax.legend()
    ax.set_xlabel('Data-time')
    ax.set_ylabel('Counts')
    dates=pd.to_datetime(rides.loc[test_data.index]['dteday'])
    dates=dates.apply(lambda d: d.strftime('%b %d'))
    ax.set_xticks(np.arange(len(dates))[12::24])
    _=ax.set_xticklabels(dates[12::24],rotation=45)





















