import tensorflow as tf 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import os
import glob
import datetime

from tensorflow.keras import optimizers,metrics
from sklearn.model_selection import train_test_split
from d_o_c_model import d_o_c_conv
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img


tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
batch_size = 15

filenames = os.listdir("D:/data/kaggle/dogs-vs-cats/train")
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)
        
df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})

model = d_o_c_conv()
model.build(input_shape=(None, 128, 128, 3))
model.summary()

df["category"] = df["category"].replace({0: 'cat', 1: 'dog'})
train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
total_train = train_df.shape[0]
total_validate = validate_df.shape[0]

train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1.0 / 255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "D:/data/kaggle/dogs-vs-cats/train/", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "D:/data/kaggle/dogs-vs-cats/train/", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)

example_df = train_df.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(
    example_df, 
    "D:/data/kaggle/dogs-vs-cats/train/", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical'
)

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
    for step,(x,y) in enumerate(train_generator):
        train_step(x, y)     
        
        if step == total_train//batch_size:
            break

    print(epoch,'loss:', loss_meter.result().numpy())
    with summary_writer.as_default():
        tf.summary.scalar('train-loss', loss_meter.result().numpy(), step=epoch)
    loss_meter.reset_states()
    
    acc_meter.reset_states()
    
    for step_2,(x,y) in enumerate(validation_generator):
        val_step(x,y)
        
        if step_2== total_validate//batch_size:
            break
        
    print(epoch, 'Evaluate Acc:', acc_meter.result().numpy())
    with summary_writer.as_default():
        tf.summary.scalar('val-acc', acc_meter.result().numpy(), step=epoch)

    #一次epoch保存一次模型   
    sessFileName = 'model_weight/model_%03d.hdf5' % (epoch + 1)
    model.save_weights(sessFileName)


# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# earlystop = EarlyStopping(patience=10)
# learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
#                                             patience=2, 
#                                             verbose=1, 
#                                             factor=0.5, 
#                                             min_lr=0.00001)
# callbacks = [earlystop, learning_rate_reduction]           
# epochs=3         
# history = model.fit(
#     train_generator, 
#     epochs=epochs,
#     validation_data=validation_generator,
#     validation_steps=total_validate//batch_size,
#     steps_per_epoch=total_train//batch_size,
#     callbacks=callbacks
# )
# model = d_o_c_conv()
# model.build(input_shape=(None, 128, 128, 3))
# model.summary()

# from tensorflow.keras.callbacks import TensorBoard
# current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard = TensorBoard(log_dir='logs/{}'.format(current_time))

# model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001),
#               loss=tf.losses.CategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
# model.fit(train_generator,
#           epochs=30,
#           validation_data=validation_generator,
#           validation_steps=total_validate//batch_size,
#           steps_per_epoch=total_train//batch_size,
#           callbacks=[tensorboard]
#           )




