from __future__ import print_function
import os
import sys

print("sys.args: ", sys.argv)
os.environ['CUDA_VISIBLE_DEVICES'] = str(sys.argv[5])  # set GPU Limits

import tensorflow as tf
import keras.backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, Callback
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10, fashion_mnist, mnist
import numpy as np
import os

import spacial_transformation as tr


dataset = str(sys.argv[1])
trans = str(sys.argv[2])
data_aug = str(sys.argv[3])
data_aug_adv = str(sys.argv[4])

if data_aug == "True":
    data_aug = True
else:
    data_aug = False
if data_aug_adv == "True":
    data_aug_adv = True
else:
    data_aug_adv = False
    

# Training parameters
batch_size = 32  # orig paper trained all networks with batch_size=128
# epochs = 200
epochs = 500

# Subtracting pixel mean improves accuracy
subtract_pixel_mean = False

# Model parameter
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------
n = 3

# Model version
# Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
version = 1

# Computed depth from supplied model parameter n
if version == 1:
    depth = n * 6 + 2
elif version == 2:
    depth = n * 9 + 2

# Model name, depth and version
model_type = 'ResNet%dv%d' % (depth, version)

if dataset == "cifar10":
    # Load the CIFAR10 data.
    (x_train_total, y_train_total), (x_test, y_test) = cifar10.load_data()
    num_train = 40000 # 40000:10000:10000
    num_classes = 10
else:
    if dataset == "mnist":
        (x_train_total, y_train_total), (x_test, y_test) = mnist.load_data()
    else:
        (x_train_total, y_train_total), (x_test, y_test) = fashion_mnist.load_data()
    x_train_total = x_train_total.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    num_train = 50000
    num_classes = 10

if data_aug:
    path_name = dataset + "/resnet"
    degrees_train_tot = np.load("./data/" + path_name + "/degree/" + trans + "_degrees_train.npy")
    degrees_test = np.load("./data/" + path_name + "/degree/" + trans + "_degrees_test.npy")
    print("num of -1 in train: ", np.sum(degrees_train_tot[:num_train] == -1))
    print("num of -1 in valid: ", np.sum(degrees_train_tot[num_train:] == -1))
    print("num of -1 in test: ", np.sum(degrees_test == -1))
    if trans == "contrast":
        max_degree = np.percentile(degrees_train_tot[np.where(degrees_train_tot != -1)[0]], 1)
        mean_degree = np.mean(degrees_train_tot[np.where((degrees_train_tot != -1) & (degrees_train_tot != 1))[0]])
    else:
        max_degree = np.percentile(degrees_train_tot, 99)
        mean_degree = np.mean(degrees_train_tot[np.where((degrees_train_tot != -1) & (degrees_train_tot != 0))[0]])

    dispatcher = {'blur': tr.image_blur, "contrast": tr.image_contrast, "bright": tr.image_brightness, "zoom": tr.image_zoom, "translation": tr.image_translation_cropped, "shear": tr.image_shear_cropped}

    x_train_tr = []
    for idx in range(x_train_total.shape[0]):
        if trans != "contrast" and degrees_train_tot[idx] != -1 and degrees_train_tot[idx] != 0:
            image = dispatcher[trans](x_train_total[idx], degrees_train_tot[idx])
            x_train_tr.append(image.astype(np.float))
        elif trans == "contrast" and degrees_train_tot[idx] != -1 and degrees_train_tot[idx] != 1:
            image = dispatcher[trans](x_train_total[idx], degrees_train_tot[idx])
            x_train_tr.append(image.astype(np.float))
        else:
            # image = dispatcher[trans](x_train_total[idx], max_degree)
            image = dispatcher[trans](x_train_total[idx], mean_degree)
            x_train_tr.append(image.astype(np.float))
    x_train_tr = np.array(x_train_tr)
    if len(x_train_tr.shape) == 3:
        x_train_tr = np.expand_dims(x_train_tr, axis=-1)

# Input image dimensions.
input_shape = x_train_total.shape[1:]

# Normalize data.
x_train_total = x_train_total.astype('float32') / 255
x_test = x_test.astype('float32') / 255
if data_aug:
    x_train_tr = x_train_tr.astype("float32")
    x_train_tr = x_train_tr / 255
    if data_aug_adv:
        x_train_adv = np.load("./data/" + path_name + "/degree/" + "x_train_adv.npy")
        x_tr_train = x_train_tr[:num_train]
        x_tr_val = x_train_tr[num_train:]
        x_adv_train = x_train_adv[:num_train]
        x_adv_val = x_train_adv[num_train:]
        x_train_aug = np.concatenate((x_tr_train[:int(num_train / 2)], x_adv_train[int(num_train / 2):]), axis=0)
        x_val_aug = np.concatenate((x_tr_val[:5000], x_adv_val[5000:]), axis=0)

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train_total, axis=0)
    x_train_total -= x_train_mean
    x_test -= x_train_mean

print('x_train shape:', x_train_total.shape)
print(x_train_total.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train_total.shape)

# Convert class vectors to binary class matrices.
y_train_total = keras.utils.to_categorical(y_train_total, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# split original training dataset into training and validation dataset
x_train = x_train_total[:num_train]
x_valid = x_train_total[num_train:]
y_train = y_train_total[:num_train]
y_valid = y_train_total[num_train:]

if data_aug and not data_aug_adv:
    x_train = np.concatenate((x_train, x_train_tr[:num_train]), axis=0)
    x_valid = np.concatenate((x_valid, x_train_tr[num_train:]), axis=0)
if data_aug and data_aug_adv:
    x_train = np.concatenate((x_train, x_train_aug), axis=0)
    x_valid = np.concatenate((x_valid, x_val_aug), axis=0)
if data_aug:
    y_train = np.concatenate((y_train, y_train), axis=0)
    y_valid = np.concatenate((y_valid, y_valid), axis=0)


class SGDLearningRateTracker(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(K.eval(self.model.optimizer.lr))


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=10):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    # x = AveragePooling2D(pool_size=8)(x) # cifar10
    x = AveragePooling2D(pool_size=7)(x) # fmnist
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_v2(input_shape, depth, num_classes=10):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


if version == 2:
    model = resnet_v2(input_shape=input_shape, depth=depth, num_classes=num_classes)
else:
    model = resnet_v1(input_shape=input_shape, depth=depth, num_classes=num_classes)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])
model.summary()
print(model_type)

# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'origin')
model_name = dataset + '_resnet' + '.h5'
if data_aug:
    model_name = dataset + '_resnet' + '_' + str(trans) + '.h5'
    if data_aug_adv:
        model_name = dataset + '_resnet' + '_' + str(trans) + '_adv.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)
print(filepath)

# Prepare callbacks for model saving and for learning rate adjustment.
EarlyStop = EarlyStopping(monitor='val_accuracy',
                              patience=20, verbose=1, mode='auto')

checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

# callbacks = [checkpoint, lr_reducer, lr_scheduler, EarlyStop]
callbacks = [checkpoint, lr_reducer, EarlyStop, SGDLearningRateTracker()]

# Run training, with or without data augmentation.
print('Not using data augmentation.')
model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_valid, y_valid),
            shuffle=True, verbose=2,
            callbacks=callbacks)


# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

if data_aug:
    x_test_tr = []
    for idx in range(x_test.shape[0]):
        if trans != "contrast" and degrees_test[idx] != -1 and degrees_test[idx] != 0:
            image = dispatcher[trans](x_test[idx], degrees_test[idx])
            x_test_tr.append(image.astype(np.float))
        elif trans == "contrast" and degrees_test[idx] != -1 and degrees_test[idx] != 1:
            image = dispatcher[trans](x_test[idx], degrees_test[idx])
            x_test_tr.append(image.astype(np.float))
        else:
            # image = dispatcher[trans](x_test[idx], max_degree)
            image = dispatcher[trans](x_test[idx], mean_degree)
            x_test_tr.append(image.astype(np.float))
    x_test_tr = np.array(x_test_tr)
    if len(x_test_tr.shape) == 3:
        x_test_tr = np.expand_dims(x_test_tr, axis=-1)

    x_test_tr = x_test_tr.astype("float32")
    x_test_tr = x_test_tr / 255.0

    scores_tr = model.evaluate(x_test_tr, y_test, verbose=1)
    print('Test tr loss:', scores_tr[0])
    print('Test tr accuracy:', scores_tr[1])
    print('total tr accuracy:', (scores[1] + scores_tr[1]) / 2.0)