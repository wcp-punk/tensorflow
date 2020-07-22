import tensorflow as tf 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import os
import glob
import datetime

from tensorflow.keras import optimizers,metrics
from d_o_c_model import d_o_c_conv
from dataset import load_dog_or_cat_data, normalize

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
batch_size = 15

def preprocess(x,y):
    # x: 图片的路径List，y：图片的数字编码List
    x = tf.io.read_file(x) # 根据路径读取图片
    x = tf.image.decode_jpeg(x, channels=3) # 图片解码
    x = tf.image.resize(x, IMAGE_SIZE) # 图片缩放

    # 数据增强
    # x = tf.image.random_flip_up_down(x)
    x= tf.image.random_flip_left_right(x) # 左右镜像
    # 转换成张量
    # x: [0,255]=> 0~1
    x = tf.cast(x, dtype=tf.float32) / 255.
    # 0~1 => D(0,1)
    x = normalize(x) # 标准化
    y = tf.convert_to_tensor(y) # 转换成张量

    return x, y

# 创建训练集Datset对象
images, labels, table = load_dog_or_cat_data(r'D:\data\kaggle\dogs-vs-cats\train', mode='train')
db_train = tf.data.Dataset.from_tensor_slices((images, labels))
db_train = db_train.shuffle(1000).map(preprocess).batch(batch_size)
# 创建验证集Datset对象
images2, labels2, table = load_dog_or_cat_data(r'D:\data\kaggle\dogs-vs-cats\train', mode='val')
db_val = tf.data.Dataset.from_tensor_slices((images2, labels2))
db_val = db_val.map(preprocess).batch(batch_size)

#设置记录-------------------------------------------------------------------------
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)
#-----------------------------------------------------------

model = d_o_c_conv()
model.build(input_shape=(None, 128, 128, 3))
model.summary()

optimizer = optimizers.RMSprop(learning_rate=0.001)
@tf.function
def train_step(x,y):
    with tf.GradientTape() as tape:
        out = model(x)
        loss_1 = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        loss = loss_1(y, out)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    loss_meter.update_state(loss)

@tf.function
def val_step(x,y):
    out = model(x, training=False) 
    pred = tf.argmax(out, axis=1) 
    pred = tf.one_hot(pred, depth=2)
    acc_meter.update_state(y, pred)

#meter----------------------------------------------------------------------------
acc_meter = metrics.Accuracy()
loss_meter = metrics.Mean()

for epoch in range(30):
    for step, (x, y) in enumerate(db_train):
        y_onehot = tf.one_hot(y, depth=2)
        train_step(x,y_onehot)
    
    print(epoch,'loss:', loss_meter.result().numpy())
    with summary_writer.as_default():
        tf.summary.scalar('train-loss', loss_meter.result().numpy(), step=epoch)
    loss_meter.reset_states()

    acc_meter.reset_states()

    for step_2, (x, y) in enumerate(db_val):
        y_onehot = tf.one_hot(y, depth=2)
        val_step(x, y_onehot)
    print(epoch, 'Evaluate Acc:', acc_meter.result().numpy())
    with summary_writer.as_default():
        tf.summary.scalar('val-acc', acc_meter.result().numpy(), step=epoch)  

    #一次epoch保存一次模型   
    sessFileName = 'model_weight/model_%03d.hdf5' % (epoch + 1)
    model.save_weights(sessFileName)  
        
        
        
        
        
        
        
        
        
        
        
        
        
        