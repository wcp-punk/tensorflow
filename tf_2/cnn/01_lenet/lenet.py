import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#定义模型----------------------------------------------------------------------------
class LeNet5(keras.Model):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.c1 = layers.Conv2D(filters=6, kernel_size=(5, 5),
                         activation='sigmoid')
        self.p1 = layers.MaxPool2D(pool_size=(2, 2), strides=2)

        self.c2 = layers.Conv2D(filters=16, kernel_size=(5, 5),
                         activation='sigmoid')
        self.p2 = layers.MaxPool2D(pool_size=(2, 2), strides=2)

        self.flatten = layers.Flatten()
        self.f1 = layers.Dense(120, activation='sigmoid')
        self.f2 = layers.Dense(84, activation='sigmoid')
        self.f3 = layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.p1(x)

        x = self.c2(x)
        x = self.p2(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.f2(x)
        y = self.f3(x)
        return y
# class LeNet5(keras.Model):
#     def __init__(self):
#         super(LeNet5, self).__init__()
        
#         self.c1 = layers.Conv2D(filters=6, kernel_size=[5, 5], strides=1, padding='valid', activation='sigmoid', name='c1')

#         self.p1 = layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='valid', name='p1')
        
#         self.c2 = layers.Conv2D(filters=16, kernel_size=[5, 5], strides=1, padding='valid', activation='sigmoid', name='c2')

#         self.p2 = layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='valid', name='p2')
        
#         self.flatten = layers.Flatten()

#         self.f1 = layers.Dense(120, activation='sigmoid', name='f1')
#         self.f2 = layers.Dense(84, activation='sigmoid', name='f2')
#         self.f3 = layers.Dense(10, activation='softmax', name='f3')
        
#     def call(self, inputs, training=None):
#         x = self.c1(inputs)
#         x = self.p1(x)

#         x = self.c2(x)
#         x = self.p2(x)

#         x = self.flatten(x)
#         x = self.f1(x)
#         x = self.f2(x)
#         y = self.f3(x)
#         return y
# #------------------------------------------------------------------------------------