import sys
import matplotlib.pyplot as plt
import numpy as np
import os

import tensorflow as tf

from keras import layers
from keras import losses
from keras import optimizers
from keras import metrics
from keras.models import load_model, Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler, ModelCheckpoint, Callback
import keras.backend as K

from keras.datasets import cifar10, mnist, fashion_mnist
import scipy.io as sio
import spacial_transformation as tr
from triplet_loss import *

import argparse

# add args
parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-data', '--data',
                    type=str,
                    default='mnist',
                    help='Dataset: mnist/fmnist/cifar10')
parser.add_argument('-model', '--model',
                    type=str,
                    default='conv',
                    help='Dataset: conv/vgg16/resnet')
parser.add_argument('-trans', '--trans',
                    type=str,
                    default='zoom',
                    help='Augmentation method. blur/contrast/bright/zoom/shear/translation')
parser.add_argument('-batch_size', help='batch size', dest='batch_size', type=int, default=32)
parser.add_argument('-epochs', help='epochs', dest='epochs', type=int, default=200)
parser.add_argument('-r', help='random state', dest='random_state', type=int, default=0)   # random seeds
parser.add_argument('-stage', '--stage', type=str, default='sia', help='stages: sia/quad')
parser.add_argument('-data_aug', help='use data augmentation', dest='data_aug', type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=True)
parser.add_argument('-data_aug_adv', help='use data augmentation with adversarial examples', dest='data_aug_adv',
                    type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False)
parser.add_argument("-gpu", type=str, default='0')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # set GPU Limits
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

print(args)


class SGDLearningRateTracker(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(K.eval(self.model.optimizer.lr))


def load_dataset(args):
    x_train_total = y_train_total = x_test_in = y_test_in = num_train = target_shape = None
    # load original data
    if args.data == "mnist":
        (x_train_total, y_train_total), (x_test_in, y_test_in) = mnist.load_data()
        num_train = 50000
        x_train_total = x_train_total.reshape(-1, 28, 28, 1).astype("float32")
        x_test_in = x_test_in.reshape(-1, 28, 28, 1).astype("float32")
        target_shape = (28, 28, 1)

    if args.data == "fmnist":
        (x_train_total, y_train_total), (x_test_in, y_test_in) = fashion_mnist.load_data()
        num_train = 50000
        x_train_total = x_train_total.reshape(-1, 28, 28, 1).astype("float32")
        x_test_in = x_test_in.reshape(-1, 28, 28, 1).astype("float32")
        target_shape = (28, 28, 1)

    if args.data == "cifar10":
        (x_train_total, y_train_total), (x_test_in, y_test_in) = cifar10.load_data()
        num_train = 40000
        y_train_total = y_train_total.reshape([y_train_total.shape[0]])
        y_test_in = y_test_in.reshape([y_test_in.shape[0]])
        target_shape = (32, 32, 3)

    return x_train_total, y_train_total, x_test_in, y_test_in, num_train, target_shape


# triplet loss
def Quadruplet_loss(margin1, margin2, margin3, embedding_dim, args):
    def get_loss(y_true, y_pred):
        # getting output embedding dimension of positive, negative and anchor images
        anchor_output = y_pred[:, : embedding_dim]
        positive_output = y_pred[:, embedding_dim: 2 * embedding_dim]
        negative_output = y_pred[:, 2 * embedding_dim:]

        labels_input = tf.reshape(tf.cast(y_true, tf.int64), [-1])

        if args.stage == "sia":
            # cc'c''
            quadruplet_loss = batch_hard_triplet_loss_c1c2_c1c(labels_input, anchor_output, anchor_output,
                                                               positive_output,
                                                               margin=margin1,
                                                               squared=False) + batch_hard_triplet_loss_c1c2_c1c(
                labels_input, anchor_output, anchor_output, negative_output, margin=margin2,
                squared=False) + batch_hard_triplet_loss_c1c_c1c2(
                labels_input, anchor_output, positive_output, anchor_output, margin=margin2, squared=False)
        elif args.stage == "quad" and args.data_aug:
            # cc'd
            quadruplet_loss = batch_hard_triplet_loss_new(labels_input, anchor_output, positive_output, anchor_output,
                                                          margin=margin1, squared=False) + batch_hard_triplet_loss(
                labels_input, anchor_output, anchor_output, anchor_output, margin=margin2, squared=False)
        else:
            # cde
            quadruplet_loss = batch_hard_triplet_loss(labels_input, anchor_output, anchor_output, anchor_output,
                                                      margin=margin1, squared=False) + batch_hard_triplet_loss_cde(
                labels_input, anchor_output, anchor_output, anchor_output, margin=margin2, squared=False)

        return quadruplet_loss

    return get_loss


def train(args):
    np.random.seed(args.random_state)  # default as 0

    x_train_total, y_train_total, x_test_in, y_test_in, num_train, target_shape = load_dataset(args)

    path_name = args.data + "/" + args.model
    degrees_train_tot = np.load("./data/" + path_name + "/degree/" + args.trans + "_degrees_train.npy")
    degrees_test = np.load("./data/" + path_name + "/degree/" + args.trans + "_degrees_test.npy")
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

    x_train_out = []
    for idx in range(x_train_total.shape[0]):
        if args.trans != "contrast":
            if degrees_train_tot[idx] != -1 and degrees_train_tot[idx] != 0:
                image = dispatcher[args.trans](x_train_total[idx], degrees_train_tot[idx] + max_degree)
                x_train_out.append(image.astype(np.float))
            else:
                image = dispatcher[args.trans](x_train_total[idx], max_degree * 2)
                x_train_out.append(image.astype(np.float))
        if args.trans == "contrast":
            if degrees_train_tot[idx] != -1 and degrees_train_tot[idx] != 1:
                image = dispatcher[args.trans](x_train_total[idx], degrees_train_tot[idx] / 10.0)
                x_train_out.append(image.astype(np.float))
            else:
                image = dispatcher[args.trans](x_train_total[idx], max_degree / 10.0)
                x_train_out.append(image.astype(np.float))
    x_train_out = np.array(x_train_out)
    if len(x_train_out.shape) == 3:
        x_train_out = np.expand_dims(x_train_out, axis=-1)

    x_train_total = x_train_total / 255.0
    x_train_tr = x_train_tr / 255.0
    x_train_out = x_train_out / 255.0

    if args.data_aug_adv:
        x_train_tr_adv = np.load("./data/" + path_name + "/degree/" + "x_train_adv.npy")

        x_tr_train = x_train_tr[:num_train]
        x_tr_val = x_train_tr[num_train:]
        x_adv_train = x_train_tr_adv[:num_train]
        x_adv_val = x_train_tr_adv[num_train:]
        x_train_aug = np.concatenate((x_tr_train[:int(num_train / 2)], x_adv_train[int(num_train / 2):]), axis=0)
        x_val_aug = np.concatenate((x_tr_val[:5000], x_adv_val[5000:]), axis=0)
        x_train_tr_new = np.concatenate((x_train_aug, x_val_aug), axis=0)

    orig_model = load_model("./origin/" + args.data + "_" + args.model + ".h5")
    orig_model.summary()
    if args.model == "conv":
        outputs = orig_model.get_layer("activation_6").output
    if args.model == "vgg16":
        outputs = orig_model.get_layer("activation_13").output
    if args.model == "resnet":
        outputs = orig_model.get_layer("activation_17").output
    temp_model = Model(inputs=orig_model.input, outputs=outputs)

    anchor_images = x_train_total
    positive_images = x_train_tr
    negative_images = x_train_out
    if args.data_aug_adv:
        positive_images = x_train_tr_new

    base_cnn = temp_model
    flatten = layers.Flatten()(base_cnn.output)

    dense1 = layers.Dense(512, activation="relu", name="Sia_act1")(flatten)
    dense1 = layers.BatchNormalization(name="Sia_BN1")(dense1)
    dense2 = layers.Dense(256, activation="relu", name="Sia_act2")(dense1)
    dense2 = layers.BatchNormalization(name="Sia_BN2")(dense2)
    output = layers.Dense(128, name="Sia_dense")(dense2)

    embedding = Model(base_cnn.input, output, name="Embedding")

    for layer in base_cnn.layers:
        layer.trainable = True

    embedding_dim = embedding.output_shape[1]

    # build siamese model
    anchor_input = layers.Input(name="anchor", shape=target_shape)
    positive_input = layers.Input(name="positive", shape=target_shape)
    negative_input = layers.Input(name="negative", shape=target_shape)

    embedding_anchor = embedding(anchor_input)
    embedding_positive = embedding(positive_input)
    embedding_negative = embedding(negative_input)

    # concatenating output of each input
    final_output = layers.concatenate([embedding_anchor, embedding_positive, embedding_negative], axis=1)

    # final model
    model = Model([anchor_input, positive_input, negative_input], final_output)

    model.compile(optimizer=optimizers.Adam(0.0001),
                  loss=Quadruplet_loss(margin1=0.5, margin2=1.0, margin3=1.5, embedding_dim=embedding_dim, args=args))

    EarlyStop = EarlyStopping(monitor='val_loss',
                              patience=20, verbose=1, mode='auto')

    # Prepare model model saving directory.
    save_dir = os.path.join(os.getcwd(), args.stage + '_models')
    if not args.data_aug:
        save_dir = os.path.join(os.getcwd(), args.stage + '_noaug_models')
    if args.data_aug_adv:
        save_dir = os.path.join(os.getcwd(), args.stage + '_adv_models')
    model_name = args.data + '_' + args.model + '.h5'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)
    print(filepath)

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True)

    # lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                                   factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   mode='auto',
                                   min_lr=0.5e-20)

    callbacks = [checkpoint, lr_reducer, EarlyStop, SGDLearningRateTracker()]

    # training over dataset
    history = model.fit(x=[anchor_images, positive_images, negative_images],
                        y=y_train_total,
                        batch_size=args.batch_size,
                        validation_split=(anchor_images.shape[0] - num_train) * 1.0 / anchor_images.shape[0],
                        epochs=args.epochs, verbose=2, callbacks=callbacks)

    model.save(filepath, overwrite=True)


if __name__ == '__main__':
    
    train(args)
