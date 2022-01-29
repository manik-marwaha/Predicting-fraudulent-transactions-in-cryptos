#!/usr/bin/env python
# coding: utf-8

/************************************************************************************
SUBJECT:            APPLIED MACHINE LEARNING
LEVEL  :            POSTGRADUATE
NAME   :            MANIK MARWAHA
UNI ID :            a1797063
PROJECT:            Predicting fraudulent transactions incryptocurrency trading
**************************************************************************************/

from sklearn.preprocessing import MinMaxScaler
import ctgan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import glob
import os
import sys
from time import time


# global plot_path
original = sys.stdout


def main(root, i, dopca=False, minmax=False):

    print("[INFO] Make nescessary directories")
    os.makedirs(os.path.join(root[:-4], "plotsaxes"), exist_ok=True)
    os.makedirs(os.path.join(root[:-4], "gendataaxes"), exist_ok=True)
    os.makedirs(os.path.join(root[:-4], "LossDetailsaxes"), exist_ok=True)

    model_path = root
    dest_path_gen = root[:-4] + "/gendataaxes"
    plot_path = root[:-4] + "/plotsaxes"
    name = f"{dest_path_gen}/gen_sample_model{i}.csv"
    print(model_path, dest_path_gen, plot_path, i, name, sep="\n", end="\n*****\n")
    print("[INFO] Preprocessing the data")
    data = pd.read_csv(model_path, engine='python')
    data = data.drop("Unnamed: 0", axis=1)
    column_names = data.columns

    if minmax:
        scaler = MinMaxScaler()
        data = pd.DataFrame(scaler.fit_transform(data),
                            columns=list(column_names))

    if dopca:
        pca = PCA(0.9)
        data = pca.fit_transform(data)
        data = pd.DataFrame(data)

    print("[INFO] Training the model")
    model = ctgan.CTGANSynthesizer(
        batch_size=10, gen_dim=(128, 128), dis_dim=(128, 128))
    sys.stdout = open(f"{root[:-4]}/LossDetailsaxes/model{i}.txt", "w+")
    start = time()
    model.fit(data, epochs=1000)

    print("[INFO] Generate Samples")
    gen_sample = model.sample(20000)

    if dopca:
        gen_sample = pca.inverse_transform(gen_sample)

    sys.stdout = original
    print("Time required = ", time()-start)
    size = gen_sample.shape[0]
    gen_sample = pd.DataFrame(gen_sample, columns=list(column_names))
    gen_sample.to_csv(name)


if __name__ == "__main__":

    # input data
    allfiles = sorted(glob.glob("./label1_trans.csv"))
    print(allfiles)
    for root in allfiles:
        print(f"---- {root} ----")
        main(root, root.split("/")[-1][5])
        print("-"*(len(root) + 10))

