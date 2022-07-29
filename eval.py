import sys
import os
import numpy as np

import tensorflow as tf

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
import argparse

# add args
parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-data', '--data',
                    type=str,
                    default='mnist',
                    help='Dataset: mnist/fmnist/cifar10')
parser.add_argument('-out', '--out',
                    type=str,
                    default='svhn',
                    help='Dataset: fmnist/mnist/svhn')
parser.add_argument('-model', '--model',
                    type=str,
                    default='conv',
                    help='Dataset: conv/vgg16/resnet')
parser.add_argument('-trans', '--trans',
                    type=str,
                    default='zoom',
                    help='Augmentation method. blur/contrast/bright/zoom/shear/translation')
parser.add_argument('-stage', '--stage', type=str, default='sia', help='stages: sia/quad')
parser.add_argument('-data_aug', help='use data augmentation', dest='data_aug', type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=True)
parser.add_argument('-data_aug_adv', help='use data augmentation with adversarial examples', dest='data_aug_adv',
                    type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False)
parser.add_argument('-is_diff_data', help='use totally different dataset', dest='is_diff_data', type=lambda x: (str(x).lower() in ['true','1', 'yes']),
                        default=True)
parser.add_argument("--t_in", "-t_in", help="threshold of in-distribution", type=int, default=95)
parser.add_argument("--t_out", "-t_out", help="threshold of out-of-distribution", type=int, default=98)
parser.add_argument("-gpu", type=str, default='0')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # set GPU Limits
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

print(args)


def load_dataset(args):
    x_train_total = y_train_total = x_test_in = y_test_in = x_test_out = y_test_out = num_train = x_train_out = None
    # load original data
    if args.data == "mnist" or args.out == "mnist":
        if args.data == "mnist":
            (x_train_total, y_train_total), (x_test_in, y_test_in) = mnist.load_data()
            num_train = 50000
            x_train_total = x_train_total.reshape(-1, 28, 28, 1).astype("float32")
            x_test_in = x_test_in.reshape(-1, 28, 28, 1).astype("float32")
        else:
            (x_train_out, _), (x_test_out, y_test_out) = mnist.load_data()
            x_test_out = x_test_out.reshape(-1, 28, 28, 1).astype("float32")
            x_train_out = x_train_out.reshape(-1, 28, 28, 1).astype("float32")

    if args.data == "fmnist" or args.out == "fmnist":
        if args.data == "fmnist":
            (x_train_total, y_train_total), (x_test_in, y_test_in) = fashion_mnist.load_data()
            num_train = 50000
            x_train_total = x_train_total.reshape(-1, 28, 28, 1).astype("float32")
            x_test_in = x_test_in.reshape(-1, 28, 28, 1).astype("float32")
        else:
            (x_train_out, _), (x_test_out, y_test_out) = fashion_mnist.load_data()
            x_test_out = x_test_out.reshape(-1, 28, 28, 1).astype("float32")
            x_train_out = x_train_out.reshape(-1, 28, 28, 1).astype("float32")

    if args.data == "cifar10" or args.out == "cifar10":
        if args.data == "cifar10":
            (x_train_total, y_train_total), (x_test_in, y_test_in) = cifar10.load_data()
            num_train = 40000
            y_train_total = y_train_total.reshape([y_train_total.shape[0]])
            y_test_in = y_test_in.reshape([y_test_in.shape[0]])
        else:
            (x_train_out, _), (x_test_out, y_test_out) = cifar10.load_data()
            y_test_out = y_test_out.reshape([y_test_out.shape[0]])

    if args.data == "svhn" or args.out == "svhn":
        train_images = sio.loadmat('data/svhn/train_32x32.mat')
        test_images = sio.loadmat('data/svhn/test_32x32.mat')
        if args.data == "svhn":
            x_train_total, y_train_total = train_images["X"], train_images["y"]
            x_test_in, y_test_in = test_images["X"], test_images["y"]
            x_train_total = np.transpose(x_train_total, (3, 0, 1, 2))
            x_test_in = np.transpose(x_test_in, (3, 0, 1, 2))
            # replace label "10" with label "0"
            y_train_total[y_train_total == 10] = 0
            y_test_in[y_test_in == 10] = 0
            y_train_total = y_train_total.reshape([y_train_total.shape[0]])
            y_test_in = y_test_in.reshape([y_test_in.shape[0]])
            num_train = 63257
        else:
            x_test_out_all, y_test_out_all = test_images["X"], test_images["y"]
            x_test_out_all = np.transpose(x_test_out_all, (3, 0, 1, 2))
            x_test_out = x_test_out_all[:10000]
            y_test_out_all[y_test_out_all == 10] = 0
            y_test_out_all = y_test_out_all.reshape([y_test_out_all.shape[0]])
            y_test_out = y_test_out_all[:10000]
            x_train_out, _ = train_images["X"], train_images["y"]
            x_train_out = np.transpose(x_train_out, (3, 0, 1, 2))

    return x_train_total, y_train_total, x_test_in, y_test_in, x_test_out, y_test_out, num_train, x_train_out


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

    y_true = np.zeros((in_num+out_num))
    y_true[in_num:] += 1
    y_pred = np.zeros((in_num+out_num))
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
        dises = np.sqrt(np.sum(np.asarray(x[i] - trains) ** 2, axis=1))
        distance = np.sort(dises)[0]
        index.append(np.argsort(dises)[0])
        distances.put(i, distance)

    return distances, index


def eval(args):
    path_name = args.data + "/" + args.model
    fileName = "./tmp/" + args.stage + "/" + path_name + "/" + args.trans + "/"
    if not args.data_aug:
        fileName = "./tmp/" + args.stage + "_noaug/" + path_name + "/" + args.trans + "/"
    if args.data_aug_adv:
        fileName = "./tmp/" + args.stage + "_adv/" + path_name + "/" + args.trans + "/"
    dir = os.path.dirname(fileName)
    if not os.path.exists(dir):
        os.makedirs(dir)

    save_dir = os.path.join(os.getcwd(), args.stage + '_models')
    if not args.data_aug:
        save_dir = os.path.join(os.getcwd(), args.stage + '_noaug_models')
    if args.data_aug_adv:
        save_dir = os.path.join(os.getcwd(), args.stage + '_adv_models')
    model_name = args.data + '_' + args.model + '.h5'
    filepath = os.path.join(save_dir, model_name)
    model = load_model(filepath, compile=False)

    x_train_total, y_train_total, x_test_in, y_test_in, x_test_out, y_test_out, num_train, x_train_out = load_dataset(args)

    degrees_train_tot = np.load("./data/" + path_name + "/degree/" + args.trans + "_degrees_train.npy")
    degrees_test = np.load("./data/" + path_name + "/degree/" + args.trans + "_degrees_test.npy")
    if args.trans == "contrast":
        max_degree = np.percentile(degrees_train_tot[np.where(degrees_train_tot != -1)[0]], 1)
        mean_degree = np.mean(degrees_train_tot[np.where((degrees_train_tot != -1) & (degrees_train_tot != 1))[0]])
    else:
        max_degree = np.percentile(degrees_train_tot, 99)
        mean_degree = np.mean(degrees_train_tot[np.where((degrees_train_tot != -1) & (degrees_train_tot != 0))[0]])
    print("max_degree: {}".format(max_degree))
    print("mean_degree: {}".format(mean_degree))

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

    x_test_tr = []
    for idx in range(x_test_in.shape[0]):
        if args.trans != "contrast" and degrees_test[idx] != -1 and degrees_test[idx] != 0:
            image = dispatcher[args.trans](x_test_in[idx], degrees_test[idx])
            x_test_tr.append(image.astype(np.float))
        elif args.trans == "contrast" and degrees_test[idx] != -1 and degrees_test[idx] != 1:
            image = dispatcher[args.trans](x_test_in[idx], degrees_test[idx])
            x_test_tr.append(image.astype(np.float))
        else:
            # image = dispatcher[args.trans](x_test_in[idx], max_degree)
            image = dispatcher[args.trans](x_test_in[idx], mean_degree)
            x_test_tr.append(image.astype(np.float))
    x_test_tr = np.array(x_test_tr)
    if len(x_test_tr.shape) == 3:
        x_test_tr = np.expand_dims(x_test_tr, axis=-1)
    np.save("./data/" + path_name + "/degree/" + args.trans + "_test.npy", x_test_tr)

    if args.is_diff_data:
        if x_test_out.shape[1] != x_test_in.shape[1]:
            x_test_out = np.resize(x_test_out, x_test_in.shape)
    else:
        x_test_out = []
        for idx in range(x_test_in.shape[0]):
            if args.trans != "contrast":
                if degrees_test[idx] != -1 and degrees_test[idx] != 0:
                    image = dispatcher[args.trans](x_test_in[idx], degrees_test[idx] + max_degree)
                    x_test_out.append(image.astype(np.float))
                else:
                    image = dispatcher[args.trans](x_test_in[idx], max_degree * 2)
                    x_test_out.append(image.astype(np.float))
            if args.trans == "contrast":
                if degrees_test[idx] != -1 and degrees_test[idx] != 1:
                    image = dispatcher[args.trans](x_test_in[idx], degrees_test[idx] / 10.0)
                    x_test_out.append(image.astype(np.float))
                else:
                    image = dispatcher[args.trans](x_test_in[idx], max_degree / 10.0)
                    x_test_out.append(image.astype(np.float))
        x_test_out = np.array(x_test_out)
        if len(x_test_out.shape) == 3:
            x_test_out = np.expand_dims(x_test_out, axis=-1)
        np.save("./data/" + path_name + "/degree/" + args.trans + "_out_test.npy", x_test_out)


    x_train_total = x_train_total / 255.0
    x_train_tr = x_train_tr / 255.0
    x_train_out = x_train_out / 255.0
    x_test_in = x_test_in / 255.0
    x_test_tr = x_test_tr / 255.0
    x_test_out = x_test_out / 255.0

    x_train = x_train_total[:num_train]
    x_valid = x_train_total[num_train:]
    y_train = y_train_total[:num_train]
    y_valid = y_train_total[num_train:]

    anchor_images = x_test_in
    positive_images = x_test_tr
    negative_images = x_test_out

    test_embeddings = model.predict([anchor_images, positive_images, negative_images])
    embedding_dim = int(test_embeddings.shape[1] / 3)
    anchor_embed = test_embeddings.T[:embedding_dim].T
    positive_embed = test_embeddings.T[embedding_dim:2*embedding_dim].T
    negative_embed = test_embeddings.T[2*embedding_dim:].T

    test_in_embed = anchor_embed
    test_tr_embed = positive_embed
    test_out_embed = negative_embed

    train_embeddings = model.predict([x_train_total, x_train_tr, x_train_out])
    train_embed = train_embeddings.T[:embedding_dim].T[:num_train]
    valid_embed = train_embeddings.T[:embedding_dim].T[num_train:]
    train_tr_embed = train_embeddings.T[embedding_dim:2*embedding_dim].T[:num_train]
    valid_tr_embed = train_embeddings.T[embedding_dim:2*embedding_dim].T[num_train:]


    orig_model = load_model("./origin/" + args.data + "_" + args.model + ".h5")
    preds_train = np.argmax(orig_model.predict(x_train), axis=1)
    preds_val = np.argmax(orig_model.predict(x_valid), axis=1)
    preds_in = np.argmax(orig_model.predict(x_test_in), axis=1)
    preds_tr = np.argmax(orig_model.predict(x_test_tr), axis=1)
    preds_out = np.argmax(orig_model.predict(x_test_out), axis=1)
    print("Acc of train: {}".format(np.mean(preds_train == y_train)))
    print("Acc of valid: {}".format(np.mean(preds_val == y_valid)))
    print("Acc of test: {}".format(np.mean(preds_in == y_test_in)))
    print("Acc of tr: {}".format(np.mean(preds_tr == y_test_in)))
    print("Acc of out: {}".format(np.mean(preds_out == y_test_in)))

    print("use min latent features")
    if os.path.exists(fileName + "latent_min_val.npy"):
        print("load distance")
        distances_val = np.load(fileName + "latent_min_val.npy")
        distances_in = np.load(fileName + "latent_min_in.npy")
        distances_tr = np.load(fileName + "latent_min_tr_" + args.trans + ".npy")
        distances_out = np.load(fileName + "latent_min_out_" + args.out + ".npy")
        # distances_unrepair = np.load(fileName + "latent_min_unrepair_" + args.trans + ".npy")
        distances_val_tr = np.load(fileName + "latent_min_val_tr.npy")

        idx_val = np.load(fileName + "latent_idx_val.npy")
        idx_in = np.load(fileName + "latent_idx_in.npy")
        idx_tr = np.load(fileName + "latent_idx_tr_" + args.trans + ".npy")
        idx_out = np.load(fileName + "latent_idx_out_" + args.out + ".npy")
        # idx_unrepair = np.load(fileName + "latent_idx_unrepair_" + args.trans + ".npy")
        idx_val_tr = np.load(fileName + "latent_idx_val_tr.npy")
    else:
        distances_val, idx_val = calc_dist(valid_embed, train_embed)
        np.save(fileName + "latent_min_val.npy", distances_val)
        np.save(fileName + "latent_idx_val.npy", idx_val)

        distances_val_tr, idx_val_tr = calc_dist(valid_tr_embed, train_embed)
        np.save(fileName + "latent_min_val_tr.npy", distances_val_tr)
        np.save(fileName + "latent_idx_val_tr.npy", idx_val_tr)

        distances_in, idx_in = calc_dist(test_in_embed, train_embed)
        np.save(fileName + "latent_min_in.npy", distances_in)
        np.save(fileName + "latent_idx_in.npy", idx_in)

        distances_tr, idx_tr = calc_dist(test_tr_embed, train_embed)
        np.save(fileName + "latent_min_tr_" + args.trans + ".npy", distances_tr)
        np.save(fileName + "latent_idx_tr_" + args.trans + ".npy", idx_tr)

        distances_out, idx_out = calc_dist(test_out_embed, train_embed)
        np.save(fileName + "latent_min_out_" + args.out + ".npy", distances_out)
        np.save(fileName + "latent_idx_out_" + args.out + ".npy", idx_out)

    if not args.is_diff_data:
        distances_unrepair, idx_unrepair = calc_dist(test_out_embed, train_embed)
        np.save(fileName + "latent_min_unrepair_" + args.trans + ".npy", distances_unrepair)
        np.save(fileName + "latent_idx_unrepair_" + args.trans + ".npy", idx_unrepair)

    distances_train_tr = np.sqrt(np.sum(np.asarray(train_tr_embed - train_embed) ** 2, axis=1))
    print("distances_train_tr shape: ", distances_train_tr.shape)
    print("train_tr: min: {}, mean: {}. max: {}".format(np.min(distances_train_tr), np.mean(distances_train_tr), np.max(distances_train_tr)))
    print("val: min: {}, mean: {}. max: {}".format(np.min(distances_val), np.mean(distances_val), np.max(distances_val)))
    print("val_tr: min: {}, mean: {}. max: {}".format(np.min(distances_val_tr), np.mean(distances_val_tr),
                                                      np.max(distances_val_tr)))
    print("in: min: {}, mean: {}. max: {}".format(np.min(distances_in), np.mean(distances_in), np.max(distances_in)))
    print("tr: min: {}, mean: {}. max: {}".format(np.min(distances_tr), np.mean(distances_tr), np.max(distances_tr)))
    print("out: min: {}, mean: {}. max: {}".format(np.min(distances_out), np.mean(distances_out), np.max(distances_out)))
    if not args.is_diff_data:
        print("unrepair: min: {}, mean: {}. max: {}".format(np.min(distances_unrepair), np.mean(distances_unrepair), np.max(distances_unrepair)))
        distances_out, idx_out = distances_unrepair, idx_unrepair

    y_true = np.zeros((distances_in.shape[0]+distances_out.shape[0]))
    y_true[distances_in.shape[0]:] += 1
    # outlier
    threshold_out = np.percentile(distances_val, args.t_out)
    print("threshold_out: {}, {}".format(args.t_out, threshold_out))
    print("out-dataset ****************************")
    tp_idx_out = np.where(distances_out > threshold_out)[0]
    print("in-test")
    fp_idx_in = np.where(distances_in > threshold_out)[0]
    cal_metric_outlier(tp_idx_out, fp_idx_in, distances_in.shape[0], distances_out.shape[0])
    y_pred = np.concatenate((distances_in, distances_out), axis=0)
    auc = roc_auc_score(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    TNR95 = 1 - fpr[np.argmax(tpr>=.95)]
    print("AUC real: {}, TNR@TPR95: {}".format(auc, TNR95))
    print("num of wrong in fp_in: {}".format(np.where(preds_in[fp_idx_in] != y_test_in[fp_idx_in])[0].shape[0]))

    print("in-test with {}".format("blur"))
    fp_idx_tr = np.where(distances_tr > threshold_out)[0]
    cal_metric_outlier(tp_idx_out, fp_idx_tr, distances_tr.shape[0], distances_out.shape[0])
    y_pred = np.concatenate((distances_tr, distances_out), axis=0)
    auc = roc_auc_score(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    TNR95 = 1 - fpr[np.argmax(tpr>=.95)]
    print("AUC real: {}, TNR@TPR95: {}".format(auc, TNR95))
    print("num of wrong in fp_tr: {}".format(np.where(preds_tr[fp_idx_tr] != y_test_in[fp_idx_tr])[0].shape[0]))


    threshold_in = np.percentile(distances_val, args.t_in)
    print("\nthreshold_in: {}, {}".format(args.t_in, threshold_in))
    print("in-test with {} **********************************".format("blur"))
    fp_idx_in = np.where(distances_in > threshold_in)[0]
    tp_idx_tr = np.where(distances_tr > threshold_in)[0]
    cal_metric_outlier(tp_idx_tr, fp_idx_in, distances_tr.shape[0], distances_out.shape[0])
    y_pred = np.concatenate((distances_in, distances_tr), axis=0)
    auc = roc_auc_score(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    TNR95 = 1 - fpr[np.argmax(tpr>=.95)]
    print("AUC real: {}, TNR@TPR95: {}".format(auc, TNR95))
    print("num of wrong in fp_idx_in: {}".format(np.where(preds_in[fp_idx_in] != y_test_in[fp_idx_in])[0].shape[0]))

    fn_idx_tr = np.where(distances_tr <= threshold_in)[0]
    tn_idx_in = np.where(distances_in <= threshold_in)[0]
    tp_right = np.sum(y_train[idx_tr][tp_idx_tr] == y_test_in[tp_idx_tr])
    fp_right = np.sum(y_train[idx_in][fp_idx_in] == y_test_in[fp_idx_in])
    fn_right = np.sum(preds_tr[fn_idx_tr] == y_test_in[fn_idx_tr])
    tn_right = np.sum(preds_in[tn_idx_in] == y_test_in[tn_idx_in])
    revised_count = tp_right + fp_right + fn_right + tn_right
    print("acc: {}, tp_right: {}, fp_right: {}, fn_right: {}, tn_right: {}".format(revised_count / (2 * y_test_in.shape[0]), tp_right, fp_right, fn_right, tn_right))
    orig_count = np.sum(preds_in == y_test_in) + np.sum(preds_tr == y_test_in)
    print("orig acc: {}, orig in count: {}, orig tr count: {}".format(orig_count / (2 * y_test_in.shape[0]), np.sum(preds_in == y_test_in), np.sum(preds_tr == y_test_in)))
    perfect_count = np.sum(preds_in == y_test_in) + np.sum(y_train[idx_tr] == y_test_in)
    print("perf acc: {}, revised tr count: {}".format(perfect_count / (2 * y_test_in.shape[0]), np.sum(y_train[idx_tr] == y_test_in)))
    print("revised val count: {}".format(np.sum(y_train[idx_val] == y_valid)))
    print("revised in count: {}".format(np.sum(y_train[idx_in] == y_test_in)))
    print("revised out count: {}".format(np.sum(y_train[idx_out] == y_test_in)))
    revised_in = fp_right + tn_right
    revised_tr = tp_right + fn_right
    print("revised_in: ", revised_in, "revised_tr: ", revised_tr)


if __name__ == '__main__':
    
    eval(args)

