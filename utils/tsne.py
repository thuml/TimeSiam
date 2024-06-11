import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris,load_digits

def scatter(data, labels, file_name):
    f = plt.figure(figsize=(226 / 15, 212 / 15))
    ax = plt.subplot(aspect='equal')
    color_pen = ['b', 'g', 'r', 'c', 'm', 'y', 'b']

    class_num = np.unique(labels)

    # draw
    for i in range(class_num.shape[0]):
        ax.scatter(data[labels == class_num[i], 0], data[labels == class_num[i], 1], lw=0, s=70, c=color_pen[i], label=f"token_{i}")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('tight')
    ax.legend(loc='upper right')

    f.savefig(file_name + ".png", bbox_inches='tight')
    print(file_name + ' save finished')


def plot_TSNE(data, labels, path, title):

    tsne_features = TSNE(n_components=2, random_state=20190129).fit_transform(data)
    scatter(tsne_features, labels, os.path.join(path, title))


def visualization(X, labels, token_nums, path, name):
    if not os.path.exists(path):
        os.makedirs(path)


    X = X.detach().cpu().numpy()
    y = labels.flatten().detach().cpu().numpy()

    X_sampled, _, y_sampled, _ = train_test_split(X, y, random_state=42, stratify=y)

    result = TSNE(n_components=2, random_state=42, perplexity=10).fit_transform(X_sampled)

    # 归一化处理
    result = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit_transform(result)


    # 绘制t-SNE可视化图
    plt.figure(figsize=(8, 6))
    labels = [f"token{i}" for i in range(token_nums)]
    sns.scatterplot(x=result[:, 0], y=result[:, 1], hue=np.array(labels)[y_sampled.astype(int)], hue_order=labels, palette='Set1', s=50, alpha=0.8)
    plt.axis('off')

    file_name = os.path.join(path, name)
    plt.savefig(file_name, bbox_inches='tight', dpi=120)
    print(file_name + ' save finished')

def visualization_PCA(X, labels, token_nums, path, name):
    if not os.path.exists(path):
        os.makedirs(path)

    X = X.detach().cpu().numpy()
    y = labels.flatten().detach().cpu().numpy()

    X_sampled, _, y_sampled, _ = train_test_split(X, y, stratify=y)

    result = PCA(n_components=2).fit_transform(X_sampled)

    # 归一化处理
    result = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit_transform(result)

    # 绘制PCA可视化图
    plt.figure(figsize=(8, 6))
    labels = [f"token{i}" for i in range(token_nums)]
    sns.scatterplot(x=result[:, 0], y=result[:, 1], hue=np.array(labels)[y_sampled.astype(int)], hue_order=labels, palette='Set1', s=50, alpha=0.8)
    plt.axis('off')

    file_name = os.path.join(path, name)
    plt.savefig(file_name, bbox_inches='tight', dpi=120)
    print(file_name + ' save finished')

def visualization_token_PCA(tokens, path):
    if not os.path.exists(path):
        os.makedirs(path)

    X = tokens[:, 0, 0, :].detach().cpu().numpy()
    labels = np.arange(tokens.shape[0])

    pca = PCA(n_components=2, random_state=42)
    result = pca.fit_transform(X)

    # 归一化处理
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    result = scaler.fit_transform(result)

    # 可视化展示
    df = pd.DataFrame()
    df["y"] = labels
    df["comp-1"] = result[:, 0]
    df["comp-2"] = result[:, 1]

    plt.figure(figsize=(226 / 15, 212 / 15))
    ax = sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(), palette=sns.color_palette("hls", np.unique(labels).shape[0]), s=100, data=df).set(title="PCA projection")

    f = ax[0].get_figure()

    file_name = os.path.join(path, 'case_token.png')
    f.savefig(file_name, bbox_inches='tight')
    print(file_name + ' save finished')