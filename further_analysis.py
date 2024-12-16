import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ann_result = pd.read_csv("results/ann_results.csv", index_col=0)
cnn_result = pd.read_csv("results/cnn_results.csv", index_col=0)
rnn_result = pd.read_csv("results/rnn_results.csv", index_col=0)
ann_boot = pd.read_csv("results/ann_boot.csv", index_col=0)
cnn_boot = pd.read_csv("results/cnn_boot.csv", index_col=0)
rnn_boot = pd.read_csv("results/rnn_boot.csv", index_col=0)

def pvalue_boot(sample1, sample2, func):
    n1 = sample1.shape[0]
    n2 = sample2.shape[0]
    observed_diff = func(sample1) - func(sample2)
    # print(observed_diff)
    uni_sample = np.concatenate((sample1, sample2))
    count = 0
    for i in range(10000):
        resample1 = np.random.choice(uni_sample, n1, replace=True)
        resample2 = np.random.choice(uni_sample, n2, replace=True)
        diff = np.abs(func(resample1) - func(resample2))
        if diff >= np.abs(observed_diff):
            count += 1
    return count / 1e4

for i in range(-1,1):
    print(pvalue_boot(ann_result["loss"], ann_boot["loss"][ann_boot["cycle"] > i], np.max))
    print(pvalue_boot(cnn_result["loss"], cnn_boot["loss"][cnn_boot["cycle"] > i], np.max))
    print(pvalue_boot(rnn_result["loss"], rnn_boot["loss"][rnn_boot["cycle"] > i], np.max))
    print()