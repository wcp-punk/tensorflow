import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#定义模型----------------------------------------------------------------------------
class AlexNet8(keras.Model):
    def __init__(self):
        super(AlexNet8, self).__init__()
        
        self.c1 = layers.Conv2D(filters=96, kernel_size=[3, 3], strides=1, padding='valid', name='c1')
        self.b1 = layers.BatchNormalization(name='b1')
        self.a1 = layers.Activation('relu', name='a1')
        self.p1 = layers.MaxPool2D(pool_size=[3, 3], strides=2, name='p1')

        self.c2 = layers.Conv2D(filters=256, kernel_size=[3, 3], strides=1, padding='valid', name='c2')
        self.b2 = layers.BatchNormalization(name='b2')
        self.a2 = layers.Activation('relu', name='a2')
        self.p2 = layers.MaxPool2D(pool_size=[3, 3], strides=2, name='p2')
        
        self.c3 = layers.Conv2D(filters=384, kernel_size=[3, 3], strides=1, padding='same', activation='relu', name='c3')
        self.c4 = layers.Conv2D(filters=384, kernel_size=[3, 3], strides=1, padding='same', activation='relu', name='c4')
        self.c5 = layers.Conv2D(filters=256, kernel_size=[3, 3], strides=1, padding='same', activation='relu', name='c5')

        self.p3 = layers.MaxPool2D(pool_size=[3, 3], strides=2, name='p3')
        
        self.flatten = layers.Flatten()

        self.f1 = layers.Dense(2048, activation='relu', name='f1')
        self.d1 = layers.Dropout(0.5)
        self.f2 = layers.Dense(2048, activation='relu', name='f2')
        self.d2 = layers.Dropout(0.5)
        self.f3 = layers.Dense(10, activation='softmax', name='f3')
        
    def call(self, inputs, training=None):
        x = self.c1(inputs)
        x = self.b1(x, training=training)
        x = self.a1(x)
        x = self.p1(x)

        x = self.c2(x)
        x = self.b2(x, training=training)
        x = self.a2(x)
        x = self.p2(x)

        x = self.c3(x)

        x = self.c4(x)

        x = self.c5(x)
        x = self.p3(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d1(x, training=training)
        x = self.f2(x)
        x = self.d2(x, training=training)
        y = self.f3(x)
        return y
# #------------------------------------------------------------------------------------