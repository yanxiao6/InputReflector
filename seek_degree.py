import numpy as np
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # set GPU Limits
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import tensorflow as tf

from keras.models import load_model, Model
from keras.datasets import cifar10, mnist, fashion_mnist, cifar100
import scipy.io as sio
import spacial_transformation as tr
import argparse

parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-d', '--data',
                    type=str,
                    default='mnist',
                    help='Dataset: mnist/fmnist/cifar10')
parser.add_argument('-m', '--model',
                    type=str,
                    default='conv',
                    help='Dataset: conv/vgg16/resnet')
parser.add_argument('-a', '--aug',
                    type=str,
                    default='zoom',
                    help='Augmentation method. blur/contrast/bright/zoom/shear/translation')
parser.add_argument("-gpu", type=str, default='0')

args = parser.parse_args()
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # set GPU Limits
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


def load_dataset(args):
    x_train_total = y_train_total = x_test = y_test = num_train = None
    # load original data
    if args.data == "mnist":
        (x_train_total, y_train_total), (x_test, y_test) = mnist.load_data()
        num_train = 50000
        x_train_total = x_train_total.reshape(-1, 28, 28, 1).astype("float32")
        x_test = x_test.reshape(-1, 28, 28, 1).astype("float32")

    if args.data == "fmnist":
        (x_train_total, y_train_total), (x_test, y_test) = fashion_mnist.load_data()
        num_train = 50000
        x_train_total = x_train_total.reshape(-1, 28, 28, 1).astype("float32")
        x_test = x_test.reshape(-1, 28, 28, 1).astype("float32")

    if args.data == "cifar10":
        (x_train_total, y_train_total), (x_test, y_test) = cifar10.load_data()
        num_train = 40000
        y_train_total = y_train_total.reshape([y_train_total.shape[0]])
        y_test = y_test.reshape([y_test.shape[0]])

    return x_train_total, y_train_total, x_test, y_test, num_train


def data_aug(x, blur_value, aug_fun):
    x = x * 255.0
    if len(x.shape) == 3:
        x = np.expand_dims(x, axis=0)
    x_tr = []
    for idx in range(x.shape[0]):
        img = aug_fun(x[idx], blur_value)
        x_tr.append(img.astype(np.float) / 255.0)
    x_tr = np.array(x_tr)
    if len(x_tr.shape) == 3:
        x_tr = np.expand_dims(x_tr, axis=-1)
    return x_tr


if __name__ == '__main__':

    path_name = args.data + "/" + args.model
    fileName = "./data/" + path_name + "/degree/"
    dir = os.path.dirname(fileName)
    if not os.path.exists(dir):
        os.makedirs(dir)

    blur_values = np.around(np.linspace(0, 5, 500, endpoint=False), 2)
    contrast_values = np.around(np.linspace(1, 0, 200, endpoint=False), 2)
    bright_values = np.around(np.linspace(0, 255, 255, endpoint=False), 2)
    zoom_values = np.around(np.linspace(1, 5, 400, endpoint=False), 2)
    translation_values = np.around(np.linspace(0, 32, 256, endpoint=False), 2)
    shear_values = np.around(np.linspace(0, 1, 200, endpoint=False), 2)

    dispatcher = {'blur': tr.image_blur, "contrast": tr.image_contrast, "bright": tr.image_brightness, "zoom": tr.image_zoom, "translation": tr.image_translation_cropped, "shear": tr.image_shear_cropped}
    aug_vals = {'blur': blur_values, "contrast": contrast_values, "bright": bright_values, "zoom": zoom_values, "translation": translation_values, "shear": shear_values}


    aug_fun = dispatcher[args.aug]
    aug_val = aug_vals[args.aug]

    x_train_total, y_train_total, x_test, y_test, num_train = load_dataset(args)
    x_train_total = x_train_total / 255.0
    x_test = x_test / 255.0

    model = load_model("./origin/" + args.data + "_" + args.model + ".h5")
    model.summary()

    degrees_test = np.zeros(x_test.shape[0]) - 1
    for idx, val in enumerate(aug_val):
        count_1 = np.sum(degrees_test == -1)
        if count_1 != 0:
            remain_idx = np.where(degrees_test == -1)[0]
            x_aug = data_aug(x_test[remain_idx], val, aug_fun)
            preds = np.argmax(model.predict(x_aug), axis=1)
            right_count = np.sum(preds == y_test[remain_idx])
            print("Right count: {} of total: {} in degree: {}".format(right_count, remain_idx.shape[0], val))
            if right_count == count_1:
                continue
            degrees_test[remain_idx[preds != y_test[remain_idx]]] = val

    print("num of -1 in test: ", np.sum(degrees_test == -1))
    np.save(fileName + args.aug + "_degrees_test.npy", degrees_test)

    degrees_train = np.zeros(x_train_total.shape[0]) - 1
    for idx, val in enumerate(aug_val):
        count_1 = np.sum(degrees_train == -1)
        if count_1 != 0:
            remain_idx = np.where(degrees_train == -1)[0]
            x_aug = data_aug(x_train_total[remain_idx], val, aug_fun)
            preds = np.argmax(model.predict(x_aug), axis=1)
            right_count = np.sum(preds == y_train_total[remain_idx])
            print("Right count: {} of total: {} in degree: {}".format(right_count, remain_idx.shape[0], val))
            if right_count == count_1:
                continue
            degrees_train[remain_idx[preds != y_train_total[remain_idx]]] = val

    print("num of -1 in train: ", np.sum(degrees_train == -1))
    np.save(fileName + args.aug + "_degrees_train.npy", degrees_train)

    print("done!")