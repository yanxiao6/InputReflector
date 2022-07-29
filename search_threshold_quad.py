import sys
import os
import numpy as np
from datetime import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # set GPU Limits

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

from tensorflow.keras.models import load_model, Model
from keras import metrics
from keras.datasets import cifar10, mnist, fashion_mnist
import scipy.io as sio
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import seaborn as sns

import spacial_transformation as tr

from time import *


def load_dataset(data):
    x_train_total = y_train_total = x_test_in = y_test_in = num_train = target_shape = None
    # load original data
    if data == "mnist":
        (x_train_total, y_train_total), (x_test_in, y_test_in) = mnist.load_data()
        num_train = 50000
        x_train_total = x_train_total.reshape(-1, 28, 28, 1).astype("float32")
        x_test_in = x_test_in.reshape(-1, 28, 28, 1).astype("float32")
        target_shape = (28, 28, 1)

    if data == "fmnist":
        (x_train_total, y_train_total), (x_test_in, y_test_in) = fashion_mnist.load_data()
        num_train = 50000
        x_train_total = x_train_total.reshape(-1, 28, 28, 1).astype("float32")
        x_test_in = x_test_in.reshape(-1, 28, 28, 1).astype("float32")
        target_shape = (28, 28, 1)

    if data == "cifar10":
        (x_train_total, y_train_total), (x_test_in, y_test_in) = cifar10.load_data()
        num_train = 40000
        y_train_total = y_train_total.reshape([y_train_total.shape[0]])
        y_test_in = y_test_in.reshape([y_test_in.shape[0]])
        target_shape = (32, 32, 3)

    return x_train_total, y_train_total, x_test_in, y_test_in, num_train, target_shape


def cal_metric_outlier(tp_idx, fp_idx, in_num, out_num):
    TP = tp_idx.shape[0]
    FP = fp_idx.shape[0]
    FN = out_num - TP
    TN = in_num - FP
    TPR = TP * 1.0 / (TP + FN)
    FPR = FP * 1.0 / (TN + FP)
    TNR = TN * 1.0 / (FP + TN)
    F1 = 2.0 * TP / (2 * TP + FN + FP)
    print("TP:{}, FP:{}, TN:{}, FN:{}, TPR:{:.6f}, FPR:{:.6f}, TNR:{:.6f}, F1:{:.6f}".format(TP, FP, TN, FN, TPR, FPR,
                                                                                             TNR, F1))

    y_true = np.zeros((in_num + out_num))
    y_true[in_num:] += 1
    y_pred = np.zeros((in_num + out_num))
    y_pred[fp_idx] = 1
    y_pred[(tp_idx + in_num)] = 1
    auc = roc_auc_score(y_true, y_pred)
    print("AUC score being: ", auc)
    return auc, F1


def calc_dist(x, trains):
    # Calculate distances
    distances = np.empty(shape=(x.shape[0],))
    index = []
    for i in tqdm(range(x.shape[0])):
        # distance = np.sum(np.sqrt(np.sum(np.asarray(x[i] - trains) ** 2, axis=1)))
        dises = np.sqrt(np.sum(np.asarray(x[i] - trains) ** 2, axis=1))
        distance = np.sort(dises)[0]
        index.append(np.argsort(dises)[0])
        distances.put(i, distance)

    return distances, index


def search_th(distances_val, distances_val_tr):
    max_f = 0
    best_threshold = 0
    for th in range(70, 100):
        threshold_in = np.percentile(distances_val, th)
        fp_idx_in = np.where(distances_val > threshold_in)[0]
        tp_idx_tr = np.where(distances_val_tr > threshold_in)[0]
        auc, F1 = cal_metric_outlier(tp_idx_tr, fp_idx_in, distances_val_tr.shape[0], distances_val.shape[0])
        if F1 >= max_f:
            max_f = F1
            best_threshold = th
    return best_threshold, max_f


def threshold_val(data, model, trans):
    print("***********search***************")

    path_name = data + "/" + model
    fileName = "./tmp/sia/" + path_name + "/" + trans + "/"
    # fileName = "./tmp/sia_adv/" + path_name + "/" + trans + "/"

    distances_val = np.load(fileName + "latent_min_val.npy")
    distances_val_tr = np.load(fileName + "latent_min_val_tr.npy")

    print(
        "val: min: {}, mean: {}. max: {}".format(np.min(distances_val), np.mean(distances_val), np.max(distances_val)))
    print("val_tr: min: {}, mean: {}. max: {}".format(np.min(distances_val_tr), np.mean(distances_val_tr),
                                                      np.max(distances_val_tr)))

    best_threshold, max_f = search_th(distances_val, distances_val_tr)
    print("best_threshold: ", best_threshold, "max_F: ", max_f)
    return best_threshold, np.percentile(distances_val, best_threshold)


def collect_acc(data, model, trans, threshold):
    print("***********test***************")

    path_name = data + "/" + model
    fileName_quad = "./tmp/quad/" + path_name + "/" + trans + "/"
    # fileName_quad = "./tmp/quad_noaug/" + path_name + "/" + trans + "/"
    # fileName_quad = "./tmp/quad_adv/" + path_name + "/" + trans + "/"
    fileName_sia = "./tmp/sia/" + path_name + "/" + trans + "/"
    # fileName_sia = "./tmp/sia_adv/" + path_name + "/" + trans + "/"

    x_train_total, y_train_total, x_test_in, y_test_in, num_train, target_shape = load_dataset(data)

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
    print("max_degree: {}".format(max_degree))
    print("mean_degree: {}".format(mean_degree))

    dispatcher = {'blur': tr.image_blur, "contrast": tr.image_contrast, "bright": tr.image_brightness, "zoom": tr.image_zoom, "translation": tr.image_translation_cropped, "shear": tr.image_shear_cropped}

    x_test_tr = []
    for idx in range(x_test_in.shape[0]):
        if trans != "contrast" and degrees_test[idx] != -1 and degrees_test[idx] != 0:
            image = dispatcher[trans](x_test_in[idx], degrees_test[idx])
            x_test_tr.append(image.astype(np.float))
        elif trans == "contrast" and degrees_test[idx] != -1 and degrees_test[idx] != 1:
            image = dispatcher[trans](x_test_in[idx], degrees_test[idx])
            x_test_tr.append(image.astype(np.float))
        else:
            # image = dispatcher[trans](x_test_in[idx], max_degree)
            image = dispatcher[trans](x_test_in[idx], mean_degree)
            x_test_tr.append(image.astype(np.float))
    x_test_tr = np.array(x_test_tr)
    if len(x_test_tr.shape) == 3:
        x_test_tr = np.expand_dims(x_test_tr, axis=-1)

    x_train_total = x_train_total / 255.0
    x_test_in = x_test_in / 255.0
    x_test_tr = x_test_tr / 255.0

    x_train = x_train_total[:num_train]
    x_valid = x_train_total[num_train:]
    y_train = y_train_total[:num_train]
    y_valid = y_train_total[num_train:]

    orig_model = load_model("./origin/" + data + "_" + model + ".h5")
    preds_train = np.argmax(orig_model.predict(x_train), axis=1)
    preds_val = np.argmax(orig_model.predict(x_valid), axis=1)
    preds_in = np.argmax(orig_model.predict(x_test_in), axis=1)
    preds_tr = np.argmax(orig_model.predict(x_test_tr), axis=1)
    print("Acc of train: {}".format(np.mean(preds_train == y_train)))
    print("Acc of valid: {}".format(np.mean(preds_val == y_valid)))
    print("Acc of test: {}".format(np.mean(preds_in == y_test_in)))
    print("Acc of tr: {}".format(np.mean(preds_tr == y_test_in)))

    print("load distance")
    distances_val = np.load(fileName_sia + "latent_min_val.npy")
    distances_in = np.load(fileName_sia + "latent_min_in.npy")
    distances_tr = np.load(fileName_sia + "latent_min_tr_" + trans + ".npy")

    print(
        "val: min: {}, mean: {}. max: {}".format(np.min(distances_val), np.mean(distances_val), np.max(distances_val)))
    print("in: min: {}, mean: {}. max: {}".format(np.min(distances_in), np.mean(distances_in), np.max(distances_in)))
    print("tr: min: {}, mean: {}. max: {}".format(np.min(distances_tr), np.mean(distances_tr), np.max(distances_tr)))

    idx_val = np.load(fileName_quad + "latent_idx_val.npy")
    idx_in = np.load(fileName_quad + "latent_idx_in.npy")
    idx_tr = np.load(fileName_quad + "latent_idx_tr_" + trans + ".npy")

    fp_idx_in = np.where(distances_in > threshold)[0]
    tp_idx_tr = np.where(distances_tr > threshold)[0]
    auc, F1 = cal_metric_outlier(tp_idx_tr, fp_idx_in, distances_tr.shape[0], distances_in.shape[0])

    fn_idx_tr = np.where(distances_tr <= threshold)[0]
    tn_idx_in = np.where(distances_in <= threshold)[0]
    tp_right = np.sum(y_train[idx_tr][tp_idx_tr] == y_test_in[tp_idx_tr])
    fp_right = np.sum(y_train[idx_in][fp_idx_in] == y_test_in[fp_idx_in])
    fn_right = np.sum(preds_tr[fn_idx_tr] == y_test_in[fn_idx_tr])
    tn_right = np.sum(preds_in[tn_idx_in] == y_test_in[tn_idx_in])
    revised_count = tp_right + fp_right + fn_right + tn_right
    revised_acc = revised_count / (2 * y_test_in.shape[0]) * 100.0
    print("acc: {}, tp_right: {}, fp_right: {}, fn_right: {}, tn_right: {}".format(revised_acc, tp_right, fp_right,
                                                                                   fn_right, tn_right))
    orig_count = np.sum(preds_in == y_test_in) + np.sum(preds_tr == y_test_in)
    origin_acc = orig_count / (2 * y_test_in.shape[0]) * 100.0
    print("orig acc: {}, orig in count: {}, orig tr count: {}".format(origin_acc, np.sum(preds_in == y_test_in),
                                                                      np.sum(preds_tr == y_test_in)))
    perfect_count = np.sum(preds_in == y_test_in) + np.sum(y_train[idx_tr] == y_test_in)
    perf_acc = perfect_count / (2 * y_test_in.shape[0]) * 100.0
    print("perf acc: {}".format(perf_acc))
    aux_val = np.sum(y_train[idx_val] == y_valid)
    aux_in = np.sum(y_train[idx_in] == y_test_in)
    aux_tr = np.sum(y_train[idx_tr] == y_test_in)
    revised_in = fp_right + tn_right
    revised_tr = tp_right + fn_right

    return origin_acc, revised_acc, perf_acc, revised_in, revised_tr, aux_val, aux_in, aux_tr, np.sum(
        preds_val == y_valid), np.sum(preds_in == y_test_in), np.sum(preds_tr == y_test_in)


def run_settings():
    dataset = ["cifar10", "fmnist", "mnist"]
    models = ["conv", "vgg16", "resnet"]
    tr = ["blur", "bright", "contrast", "zoom", "shear", "translation"]

    expno = 0
    with open('./tmp/quad/results_quad.csv', 'a') as f:
        f.write(
            "ExpNo, Time, Model, Data, Transform, Th_perc, Th, Origin_acc, Revised_acc, Perf_acc, Revised_in, Revised_tr, Aux_val, Aux_in, Aux_tr, Orig_val, Orig_in, Orig_tr")
        f.write('\n')

    for m in models:
        for d in dataset:
            threshold_perc, threshold = threshold_val(d, m, "blur")
            for t in tr:
                print("model: ", m, "dataset: ", d, "trains: ", t, "threshold_perc: ", threshold_perc, "threshold: ",
                      threshold)
                origin_acc, revised_acc, perf_acc, revised_in, revised_tr, aux_val, aux_in, aux_tr, orig_val, orig_in, orig_tr = collect_acc(
                    d, m, t, threshold)

                now = datetime.now()
                current_time = now.strftime("%Y-%m-%d %H:%M:%S")

                log_str = '{expno}, {time}, {model}, {data}, {transform}, {th_perc}, {th}, {origin_acc}, {revised_acc}, {perf_acc}, {revised_in}, {revised_tr}, {aux_val}, {aux_in}, {aux_tr}, {orig_val}, {orig_in}, {orig_tr}'.format(
                    expno=expno,
                    time=current_time,
                    model=m,
                    data=d,
                    transform=t,
                    th_perc=threshold_perc,
                    th=threshold,
                    origin_acc=origin_acc,
                    revised_acc=revised_acc,
                    perf_acc=perf_acc,
                    revised_in=revised_in,
                    revised_tr=revised_tr,
                    aux_val=aux_val,
                    aux_in=aux_in,
                    aux_tr=aux_tr,
                    orig_val=orig_val,
                    orig_in=orig_in,
                    orig_tr=orig_tr
                )
                with open('./tmp/quad/results_quad.csv', 'a') as f:
                    f.write(log_str)
                    f.write('\n')

                expno += 1
    f.close()


if __name__ == "__main__":
    run_settings()
