import tensorflow as tf
import numpy as np
from tensorflow.keras import optimizers, datasets, metrics
import os
import matplotlib.pyplot as plt
from vgg16 import VGG16

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'
# tf.random.set_seed(2345)
np.set_printoptions(threshold=np.inf)

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
y_train = tf.squeeze(y_train, axis=1)
y_test = tf.squeeze(y_test, axis=1)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_db = train_db.shuffle(1000).map(preprocess).batch(32)

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).batch(32)

sample = next(iter(train_db))
print('sample:', sample[0].shape, sample[1].shape,
tf.reduce_min(sample[0]), tf.reduce_max(sample[0]))

model = VGG16()
model.build(input_shape=(None, 32, 32, 3))
model.summary()

optimizer = optimizers.Adam()

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        out = model(x, training=True)
        loss_1 = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        loss = loss_1(y, out)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    loss_meter.update_state(loss)

@tf.function
def val_step(x, y):
    out = model(x, training=False)
    pred = tf.argmax(out, axis=1)
    acc_meter.update_state(y, pred)
#meter----------------------------------------------------------------------------
acc_meter = metrics.Accuracy()
loss_meter = metrics.Mean()

losses = []
val_acc = []
for epoch in range(5):
    for step, (x, y) in enumerate(train_db):
        train_step(x, y)
    print(epoch, 'loss:', loss_meter.result().numpy())
    losses.append(loss_meter.result().numpy())
    loss_meter.reset_states()

    acc_meter.reset_states()

    for step_2, (x, y) in enumerate(test_db):
        val_step(x, y)
    print(epoch, 'Evaluate Acc:', acc_meter.result().numpy())
    val_acc.append(acc_meter.result().numpy())

model.save_weights('model.hdf5')  

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_acc)
plt.title('Validation Accuracy')
plt.legend()
plt.show()




