import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '15'
class BasicBlock(layers.Layer):

    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()

        if stride!=1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride))
        else:
            self.downsample = lambda x : x


    def call(self, inputs, training=None):

        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(inputs)

        output = layers.add([out, identity])

        return tf.nn.relu(output)

class ResNet(keras.Model):
    def __init__(self, layer_dims, num_classes):
        super(ResNet, self).__init__()
        self.stem = Sequential([layers.Conv2D(64, (3,3), strides=(1, 1)),
                                layers.BatchNormalization(),
                                layers.Activation('relu'),
                                layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')])
        self.layer1 = self.build_resblock(64, layer_dims[0])
        self.layer2 = self.build_resblock(64, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(128, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(256, layer_dims[3], stride=2)

        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes)



    def call(self, inputs, training=None):

        x = self.stem(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        logits = self.fc(x)
        return tf.nn.softmax(logits)


    def build_resblock(self, filter_num, blocks, stride=1):

        res_block = Sequential()
        res_block.add(BasicBlock(filter_num, stride))
        for _ in range(1, blocks):
            res_block.add(BasicBlock(filter_num, stride=1))
        return res_block


def preprocess(x, y):
    # [-1~1]
    x = tf.cast(x, dtype=tf.float32) / 127.5  - 1 
    y = tf.cast(y, dtype=tf.int32)
    return x,y


def main():
    (x, y), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x, y = preprocess(x, y)
    x_test, y_test = preprocess(x_test, y_test)
    preprocess(x_test, y_test)
    y = tf.squeeze(y, axis=1)
    y_test = tf.squeeze(y_test, axis=1)
    y = tf.one_hot(y, depth=10)
    y_test = tf.one_hot(y_test, depth=10)

    resnet18 = ResNet([2, 2, 2, 2], 10)

    resnet18.build(input_shape=(None, 32, 32, 3))
    resnet18.summary()
    resnet18.compile(optimizer=keras.optimizers.Adam(0.00001),
                     loss=tf.losses.categorical_crossentropy,
                     metrics=['accuracy']
                     )

    resnet18.fit(x, y, epochs=20, batch_size=64,validation_data=(x_test, y_test))

def main2():
    (x, y), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x, y = preprocess(x, y)
    x_test, y_test = preprocess(x_test, y_test)
    y = tf.squeeze(y, axis=1)
    y_test = tf.squeeze(y_test, axis=1)
    y = tf.one_hot(y, depth=10)
    y_test = tf.one_hot(y_test, depth=10)
    myModel2 = ResNet([2, 2, 2, 2], 10)
    # myModel.build(input_shape=(None, 32, 32, 3))
    datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)
    for i in range(5):
        datagen.fit(x)
        myModel2.compile(optimizer=optimizers.Adam(0.00001), loss=tf.losses.categorical_crossentropy,
                        metrics=['accuracy'])

        filepath = './checkpoints%d/net.ckpt' % ((i+6) * 200)
        # checkpoints = keras.callbacks.ModelCheckpoint(filepath, mode='max', monitor='val_acc', save_best_only=True,
        #                                               verbose=1)
        #
        myModel2.fit_generator(datagen.flow(x, y, batch_size=64), steps_per_epoch=250, epochs=200,
                            validation_data=(x_test, y_test), validation_freq=1, validation_steps=200,
                            verbose=1)
        myModel2.save_weights(filepath)


def main3():
    (x, y), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x, y = preprocess(x, y)
    x_test, y_test = preprocess(x_test, y_test)
    y = tf.squeeze(y, axis=1)
    y_test = tf.squeeze(y_test, axis=1)
    y = tf.one_hot(y, depth=10)
    y_test = tf.one_hot(y_test, depth=10)
    filepath = './checkpoints1000/net.ckpt'

    myModel = ResNet([2, 2, 2, 2], 10)
    myModel.compile(optimizer=optimizers.Adam(0.00001), loss=tf.losses.categorical_crossentropy,
                     metrics=['accuracy'])
    myModel.load_weights(filepath)
    myModel.evaluate(x_test, y_test)

if __name__ == '__main__':

    #main()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:

            tf.config.experimental.set_virtual_device_configuration(gpu,\
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5200)])


    #main2()

    main3()
