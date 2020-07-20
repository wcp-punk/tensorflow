import tensorflow
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

class d_o_c_conv(keras.Model):
    def __init__(self):
        super(d_o_c_conv, self).__init__()
        
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPooling2D(pool_size=(2, 2))
        self.Dropout1 = layers.Dropout(0.25)
        
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu')
        self.bn2 = layers.BatchNormalization()
        self.pool2 = layers.MaxPooling2D(pool_size=(2, 2))
        self.Dropout2 = layers.Dropout(0.25)

        self.conv3 = layers.Conv2D(128, (3, 3), activation='relu')
        self.bn3 = layers.BatchNormalization()
        self.pool3 = layers.MaxPooling2D(pool_size=(2, 2))
        self.Dropout3 = layers.Dropout(0.25)

        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(512, activation='relu')
        self.bn4 = layers.BatchNormalization()
        self.Dropout4 = layers.Dropout(0.5)
        self.dense2 = layers.Dense(2)
        

    def call(self, x_in, training=None):
        r = self.conv1(x_in)
        r = self.bn1(r, training=training)
        r = self.pool1(r)
        r = self.Dropout1(r, training=training)

        r = self.conv2(r)
        r = self.bn2(r, training=training)
        r = self.pool2(r)
        r = self.Dropout2(r, training=training)

        r = self.conv3(r)
        r = self.bn3(r, training=training)
        r = self.pool3(r)
        r = self.Dropout3(r, training=training)

        r = self.flatten(r)
        r = self.dense1(r)
        r = self.bn4(r, training=training)
        r = self.Dropout4(r, training=training)
        r = self.dense2(r)

        return r
         

# model = Sequential()

# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))
# model.add(Dense(2, activation='softmax')) # 2 because we have cat and dog classes

# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# model.summary()
