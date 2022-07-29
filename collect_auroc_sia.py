import sys
import os
import numpy as np
from datetime import datetime

from sklearn.metrics import roc_auc_score, roc_curve


def collect_auroc(data, model, trans, out):
    path_name = data + "/" + model
    fileName = "./tmp/sia/" + path_name + "/" + trans + "/"
    # fileName = "./tmp/sia_adv/" + path_name + "/" + trans + "/"

    print("load distance")
    distances_val = np.load(fileName + "latent_min_val.npy")
    distances_in = np.load(fileName + "latent_min_in.npy")
    distances_tr = np.load(fileName + "latent_min_tr_" + trans + ".npy")
    distances_out = np.load(fileName + "latent_min_out_" + out + ".npy")
    distances_unrepair = np.load(fileName + "latent_min_unrepair_" + trans + ".npy")

    print(
        "val: min: {}, mean: {}. max: {}".format(np.min(distances_val), np.mean(distances_val), np.max(distances_val)))
    print("in: min: {}, mean: {}. max: {}".format(np.min(distances_in), np.mean(distances_in), np.max(distances_in)))
    print("tr: min: {}, mean: {}. max: {}".format(np.min(distances_tr), np.mean(distances_tr), np.max(distances_tr)))
    print(
        "out: min: {}, mean: {}. max: {}".format(np.min(distances_out), np.mean(distances_out), np.max(distances_out)))
    print("unrepair: min: {}, mean: {}. max: {}".format(np.min(distances_unrepair), np.mean(distances_unrepair),
                                                        np.max(distances_unrepair)))

    print("\r\n*****************detect unrepair*******************")
    y_true = np.zeros((distances_in.shape[0] + distances_tr.shape[0] + distances_out.shape[0]))
    y_true[(distances_in.shape[0] + distances_tr.shape[0]):] += 1

    y_pred = np.concatenate((distances_in, distances_tr, distances_unrepair), axis=0)
    AUC_unrepair = roc_auc_score(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    TNR_unrepair = 1 - fpr[np.argmax(tpr >= .95)]
    print("unrepair: AUC real: {}, TNR@TPR95: {}".format(AUC_unrepair, TNR_unrepair))

    print("\r\n*****************detect out*******************")
    y_pred = np.concatenate((distances_in, distances_tr, distances_out), axis=0)
    AUC_out = roc_auc_score(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    TNR_out = 1 - fpr[np.argmax(tpr >= .95)]
    print("out: AUC real: {}, TNR@TPR95: {}".format(AUC_out, TNR_out))

    print("\r\n*****************detect both*******************")
    y_true = np.zeros(
        (distances_in.shape[0] + distances_tr.shape[0] + distances_unrepair.shape[0] + distances_out.shape[0]))
    y_true[(distances_in.shape[0] + distances_tr.shape[0]):] += 1

    y_pred = np.concatenate((distances_in, distances_tr, distances_unrepair, distances_out), axis=0)
    AUC_both = roc_auc_score(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    TNR_both = 1 - fpr[np.argmax(tpr >= .95)]
    print("both: AUC real: {}, TNR@TPR95: {}".format(AUC_both, TNR_both))

    print("\r\n*****************detect tr*******************")
    y_true = np.zeros((distances_in.shape[0] + distances_tr.shape[0]))
    y_true[distances_in.shape[0]:] += 1

    y_pred = np.concatenate((distances_in, distances_tr), axis=0)
    AUC_tr = roc_auc_score(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    TNR_tr = 1 - fpr[np.argmax(tpr >= .95)]
    print("both: AUC real: {}, TNR@TPR95: {}".format(AUC_tr, TNR_tr))

    return AUC_unrepair, TNR_unrepair, AUC_out, TNR_out, AUC_both, TNR_both, AUC_tr, TNR_tr


def run_settings():
    dataset=["cifar10", "fmnist", "mnist"]
    models = ["conv", "vgg16", "resnet"]
    tr = ["blur", "bright", "contrast", "zoom", "shear", "translation"]
    out=["svhn", "mnist", "fmnist"]

    expno = 0
    with open('./tmp/sia/results_sia.csv', 'a') as f:
        f.write(
            "ExpNo, Time, Model, Data, Transform, Out, AUC_unrepair, TNR_unrepair, AUC_out, TNR_out, AUC_both, TNR_both, AUC_tr, TNR_tr")
        f.write('\n')

    for m in models:
        for d in dataset:
            for t in tr:
                AUC_unrepair, TNR_unrepair, AUC_out, TNR_out, AUC_both, TNR_both, AUC_tr, TNR_tr = collect_auroc(d, m,
                                                                                                                 t, out[
                                                                                                                     dataset.index(
                                                                                                                         d)])

                now = datetime.now()
                current_time = now.strftime("%Y-%m-%d %H:%M:%S")

                log_str = '{expno}, {time}, {model}, {data}, {transform}, {out}, {AUC_unrepair}, {TNR_unrepair}, {AUC_out}, {TNR_out}, {AUC_both}, {TNR_both}, {AUC_tr}, {TNR_tr}'.format(
                    expno=expno,
                    time=current_time,
                    model=m,
                    data=d,
                    transform=t,
                    out=out[dataset.index(d)],
                    AUC_unrepair=AUC_unrepair,
                    TNR_unrepair=TNR_unrepair,
                    AUC_out=AUC_out,
                    TNR_out=TNR_out,
                    AUC_both=AUC_both,
                    TNR_both=TNR_both,
                    AUC_tr=AUC_tr,
                    TNR_tr=TNR_tr
                )
                with open('./tmp/sia/results_sia.csv', 'a') as f:
                    f.write(log_str)
                    f.write('\n')

                expno += 1
    f.close()


if __name__ == "__main__":
    run_settings()
