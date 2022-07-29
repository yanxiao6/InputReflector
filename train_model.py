import sys
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # set GPU Limits

import argparse

import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.datasets import mnist, fashion_mnist, cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler, ModelCheckpoint, Callback
from keras.optimizers import Adam
import spacial_transformation as tr


parser = argparse.ArgumentParser()
parser.add_argument("-gpu", type=str, default='0')
parser.add_argument("-dataset", type=str, default='cifar10')
parser.add_argument("-model", type=str, default='vgg16')
parser.add_argument("-trans", type=str, default='blur')
parser.add_argument('-data_aug', help='use data augmentation', dest='data_aug', type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False)
parser.add_argument('-data_aug_adv', help='use data augmentation with adversarial examples', dest='data_aug_adv',
                        type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False)
# parser.add_argument("-epoch", type=int, default=300)
parser.add_argument("-epoch", type=int, default=500)
parser.add_argument("-batch", type=int, default=128)
args = parser.parse_args()
assert args.dataset in ["mnist", "fmnist", "cifar10", "svhn"], "Dataset should be either 'mnist' or 'cifar'"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # set GPU Limits

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


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


def train(args):
    if "mnist" in args.dataset:
        if args.dataset == "mnist":
            (x_train_total, y_train_total), (x_test, y_test) = mnist.load_data()
        else:
            (x_train_total, y_train_total), (x_test, y_test) = fashion_mnist.load_data()
        num_train = 50000
        x_train_total = x_train_total.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        input_shape = (28, 28, 1)
        num_classes = 10
    else:
        num_train = 40000  # 40000:10000:10000
        (x_train_total, y_train_total), (x_test, y_test) = cifar10.load_data()
        y_train_total = y_train_total.reshape([y_train_total.shape[0]])
        y_test = y_test.reshape([y_test.shape[0]])
        input_shape = (32, 32, 3)
        num_classes = 10

    if args.data_aug:
        path_name = args.dataset + "/" + args.model
        degrees_train_tot = np.load("./data/" + path_name + "/degree/" + args.trans + "_degrees_train.npy")
        degrees_test = np.load("./data/" + path_name + "/degree/" + args.trans + "_degrees_test.npy")
        print("num of -1 in train: ", np.sum(degrees_train_tot[:num_train] == -1))
        print("num of -1 in valid: ", np.sum(degrees_train_tot[num_train:] == -1))
        print("num of -1 in test: ", np.sum(degrees_test == -1))
        if args.trans == "contrast":
            max_degree = np.percentile(degrees_train_tot[np.where(degrees_train_tot != -1)[0]], 1)
            mean_degree = np.mean(degrees_train_tot[np.where((degrees_train_tot != -1) & (degrees_train_tot != 1))[0]])
        else:
            max_degree = np.percentile(degrees_train_tot, 99)
            mean_degree = np.mean(degrees_train_tot[np.where((degrees_train_tot != -1) & (degrees_train_tot != 0))[0]])

        dispatcher = {'blur': tr.image_blur, "contrast": tr.image_contrast, "bright": tr.image_brightness, "zoom": tr.image_zoom, "translation": tr.image_translation_cropped, "shear": tr.image_shear_cropped}

        x_train_tr = []
        for idx in range(x_train_total.shape[0]):
            if args.trans != "contrast" and degrees_train_tot[idx] != -1 and degrees_train_tot[idx] != 0:
                image = dispatcher[args.trans](x_train_total[idx], degrees_train_tot[idx])
                x_train_tr.append(image.astype(np.float))
            elif args.trans == "contrast" and degrees_train_tot[idx] != -1 and degrees_train_tot[idx] != 1:
                image = dispatcher[args.trans](x_train_total[idx], degrees_train_tot[idx])
                x_train_tr.append(image.astype(np.float))
            else:
                # image = dispatcher[args.trans](x_train_total[idx], max_degree)
                image = dispatcher[args.trans](x_train_total[idx], mean_degree)
                x_train_tr.append(image.astype(np.float))
        x_train_tr = np.array(x_train_tr)
        if len(x_train_tr.shape) == 3:
            x_train_tr = np.expand_dims(x_train_tr, axis=-1)

    x_train_total = x_train_total.astype("float32")
    x_test = x_test.astype("float32")
    x_train_total = x_train_total / 255.0
    x_test = x_test / 255.0

    if args.data_aug:
        x_train_tr = x_train_tr.astype("float32")
        x_train_tr = x_train_tr / 255.0
        if args.data_aug_adv:
            x_train_adv = np.load("./data/" + path_name + "/degree/" + "x_train_adv.npy")
            x_tr_train = x_train_tr[:num_train]
            x_tr_val = x_train_tr[num_train:]
            x_adv_train = x_train_adv[:num_train]
            x_adv_val = x_train_adv[num_train:]
            x_train_aug = np.concatenate((x_tr_train[:int(num_train/2)], x_adv_train[int(num_train/2):]), axis=0)
            x_val_aug = np.concatenate((x_tr_val[:5000], x_adv_val[5000:]), axis=0)

    # split original training dataset into training and validation dataset
    x_train = x_train_total[:num_train]
    x_valid = x_train_total[num_train:]
    y_train = y_train_total[:num_train]
    y_valid = y_train_total[num_train:]

    # one-hot
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_valid = np_utils.to_categorical(y_valid, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    if args.data_aug and not args.data_aug_adv:
        x_train = np.concatenate((x_train, x_train_tr[:num_train]), axis=0)
        x_valid = np.concatenate((x_valid, x_train_tr[num_train:]), axis=0)
    if args.data_aug and args.data_aug_adv:
        x_train = np.concatenate((x_train, x_train_aug), axis=0)
        x_valid = np.concatenate((x_valid, x_val_aug), axis=0)
    if args.data_aug:
        y_train = np.concatenate((y_train, y_train), axis=0)
        y_valid = np.concatenate((y_valid, y_valid), axis=0)

    if args.model == "conv":
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(1024, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        model.add(Dense(512, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation("softmax"))
    else:
        model = Sequential()
        weight_decay = 0.0005

        model.add(Conv2D(64, (3, 3), padding='same',
                         input_shape=input_shape, kernel_regularizer=l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        # model.add(MaxPooling2D(pool_size=(2, 2))) # cifar10
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512, kernel_regularizer=l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

    model.build()
    model.summary()

    model.compile(
        loss="categorical_crossentropy", optimizer=Adam(lr=lr_schedule(0)), metrics=["accuracy"]
    )

    EarlyStop = EarlyStopping(monitor='val_accuracy',
                              patience=20, verbose=1, mode='auto')

    # Prepare model model saving directory.
    save_dir = os.path.join(os.getcwd(), 'origin')
    model_name = args.dataset + '_' + str(args.model) + '.h5'
    if args.data_aug:
        model_name = args.dataset + '_' + str(args.model) + '_' + str(args.trans) + '.h5'
        if args.data_aug_adv:
            model_name = args.dataset + '_' + str(args.model) + '_' + str(args.trans) + '_adv.h5'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)
    print(filepath)

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_accuracy',
                                 verbose=1,
                                 save_best_only=True)

    lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                                   factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   mode='auto',
                                   min_lr=0.5e-20)

    callbacks = [checkpoint, lr_reducer, EarlyStop, SGDLearningRateTracker()]

    model.fit(
        x_train,
        y_train,
        epochs=args.epoch,
        batch_size=args.batch,
        shuffle=True,
        verbose=2,
        validation_data=(x_valid, y_valid),
        callbacks=callbacks
    )

    model.save(filepath, True)

    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    if args.data_aug:
        x_test = x_test * 255.0
        x_test_tr = []
        for idx in range(x_test.shape[0]):
            if args.trans != "contrast" and degrees_test[idx] != -1 and degrees_test[idx] != 0:
                image = dispatcher[args.trans](x_test[idx], degrees_test[idx])
                x_test_tr.append(image.astype(np.float))
            elif args.trans == "contrast" and degrees_test[idx] != -1 and degrees_test[idx] != 1:
                image = dispatcher[args.trans](x_test[idx], degrees_test[idx])
                x_test_tr.append(image.astype(np.float))
            else:
                # image = dispatcher[args.trans](x_test[idx], max_degree)
                image = dispatcher[args.trans](x_test[idx], mean_degree)
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


if __name__ == "__main__":

    train(args)
