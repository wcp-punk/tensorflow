import tensorflow
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

class dncnn_conv(keras.Model):
    def __init__(self):
        super(dncnn_conv, self).__init__()
        
        self.c1 = layers.Conv2D(filters=64, kernel_size=[3, 3], strides=1, kernel_initializer='Orthogonal', padding='same', name='c1')
        self.a1 = layers.Activation('relu', name='a1')

        self.c2 = layers.Conv2D(filters=64, kernel_size=[3, 3], strides=1, kernel_initializer='Orthogonal', padding='same', use_bias=False, name='c2')
        self.b2 = layers.BatchNormalization(axis=3, momentum=0.0, epsilon=0.0001, name='b2')
        self.a2 = layers.Activation('relu', name='a2')
        
        self.c3 = layers.Conv2D(filters=64, kernel_size=[3, 3], strides=1, kernel_initializer='Orthogonal', padding='same', use_bias=False, name='c3')
        self.b3 = layers.BatchNormalization(axis=3, momentum=0.0, epsilon=0.0001, name='b3')
        self.a3 = layers.Activation('relu', name='a3')
        
        self.c4 = layers.Conv2D(filters=64, kernel_size=[3, 3], strides=1, kernel_initializer='Orthogonal', padding='same', use_bias=False, name='c4')
        self.b4 = layers.BatchNormalization(axis=3, momentum=0.0, epsilon=0.0001, name='b4')
        self.a4 = layers.Activation('relu', name='a4')

        self.c5 = layers.Conv2D(filters=64, kernel_size=[3, 3], strides=1, kernel_initializer='Orthogonal', padding='same', use_bias=False, name='c5')
        self.b5 = layers.BatchNormalization(axis=3, momentum=0.0, epsilon=0.0001, name='b5')
        self.a5 = layers.Activation('relu', name='a5')
        
        self.c6 = layers.Conv2D(filters=64, kernel_size=[3, 3], strides=1, kernel_initializer='Orthogonal', padding='same', use_bias=False, name='c6')
        self.b6 = layers.BatchNormalization(axis=3, momentum=0.0, epsilon=0.0001, name='b6')
        self.a6 = layers.Activation('relu', name='a6')
        
        self.c7 = layers.Conv2D(filters=64, kernel_size=[3, 3], strides=1, kernel_initializer='Orthogonal', padding='same', use_bias=False, name='c7')
        self.b7 = layers.BatchNormalization(axis=3, momentum=0.0, epsilon=0.0001, name='b7')
        self.a7 = layers.Activation('relu', name='a7')

        self.c8 = layers.Conv2D(filters=64, kernel_size=[3, 3], strides=1, kernel_initializer='Orthogonal', padding='same', use_bias=False, name='c8')
        self.b8 = layers.BatchNormalization(axis=3, momentum=0.0, epsilon=0.0001, name='b8')
        self.a8 = layers.Activation('relu', name='a8')
        
        self.c9 = layers.Conv2D(filters=64, kernel_size=[3, 3], strides=1, kernel_initializer='Orthogonal', padding='same', use_bias=False, name='c9')
        self.b9 = layers.BatchNormalization(axis=3, momentum=0.0, epsilon=0.0001, name='b9')
        self.a9 = layers.Activation('relu', name='a9')
        
        self.c10 = layers.Conv2D(filters=64, kernel_size=[3, 3], strides=1, kernel_initializer='Orthogonal', padding='same', use_bias=False, name='c10')
        self.b10 = layers.BatchNormalization(axis=3, momentum=0.0, epsilon=0.0001, name='b10')
        self.a10 = layers.Activation('relu', name='a10')

        self.c11 = layers.Conv2D(filters=64, kernel_size=[3, 3], strides=1, kernel_initializer='Orthogonal', padding='same', use_bias=False, name='c11')
        self.b11 = layers.BatchNormalization(axis=3, momentum=0.0, epsilon=0.0001, name='b11')
        self.a11 = layers.Activation('relu', name='a11')
        
        self.c12 = layers.Conv2D(filters=64, kernel_size=[3, 3], strides=1, kernel_initializer='Orthogonal', padding='same', use_bias=False, name='c12')
        self.b12 = layers.BatchNormalization(axis=3, momentum=0.0, epsilon=0.0001, name='b12')
        self.a12 = layers.Activation('relu', name='a12')
        
        self.c13 = layers.Conv2D(filters=64, kernel_size=[3, 3], strides=1, kernel_initializer='Orthogonal', padding='same', use_bias=False, name='c13')
        self.b13 = layers.BatchNormalization(axis=3, momentum=0.0, epsilon=0.0001, name='b13')
        self.a13 = layers.Activation('relu', name='a13')

        self.c14 = layers.Conv2D(filters=64, kernel_size=[3, 3], strides=1, kernel_initializer='Orthogonal', padding='same', use_bias=False, name='c14')
        self.b14 = layers.BatchNormalization(axis=3, momentum=0.0, epsilon=0.0001, name='b14')
        self.a14 = layers.Activation('relu', name='a14')

        self.c15 = layers.Conv2D(filters=64, kernel_size=[3, 3], strides=1, kernel_initializer='Orthogonal', padding='same', use_bias=False, name='c15')
        self.b15 = layers.BatchNormalization(axis=3, momentum=0.0, epsilon=0.0001, name='b15')
        self.a15 = layers.Activation('relu', name='a15')

        self.c16 = layers.Conv2D(filters=64, kernel_size=[3, 3], strides=1, kernel_initializer='Orthogonal', padding='same', use_bias=False, name='c16')
        self.b16 = layers.BatchNormalization(axis=3, momentum=0.0, epsilon=0.0001, name='b16')
        self.a16 = layers.Activation('relu', name='a16')

        self.c17 = layers.Conv2D(filters=1, kernel_size=[3, 3], strides=1, kernel_initializer='Orthogonal', padding='same', use_bias=False, name='c17')
        self.sub = layers.Subtract(name='subtract')        
        

    def call(self, x_in, training=None):
        r = self.c1(x_in)
        r = self.a1(r)

        r = self.c2(r)
        r = self.b2(r, training=training)
        r = self.a2(r)
        r = self.c3(r)
        r = self.b3(r, training=training)
        r = self.a3(r)
        r = self.c4(r)
        r = self.b4(r, training=training)
        r = self.a4(r)
        r = self.c5(r)
        r = self.b5(r, training=training)
        r = self.a5(r)
        r = self.c6(r)
        r = self.b6(r, training=training)
        r = self.a6(r)

        r = self.c7(r)
        r = self.b7(r, training=training)
        r = self.a7(r)
        r = self.c8(r)
        r = self.b8(r, training=training)
        r = self.a8(r)
        r = self.c9(r)
        r = self.b9(r, training=training)
        r = self.a9(r)
        r = self.c10(r)
        r = self.b10(r, training=training)
        r = self.a10(r)
        r = self.c11(r)
        r = self.b11(r, training=training)
        r = self.a11(r)

        r = self.c12(r)
        r = self.b12(r, training=training)
        r = self.a12(r)
        r = self.c13(r)
        r = self.b13(r, training=training)
        r = self.a13(r)
        r = self.c14(r)
        r = self.b14(r, training=training)
        r = self.a14(r)
        r = self.c15(r)
        r = self.b15(r, training=training)
        r = self.a15(r)
        r = self.c16(r)
        r = self.b16(r, training=training)
        r = self.a16(r)

        r = self.c17(r)
        r = self.sub([x_in, r])

        return r
         


