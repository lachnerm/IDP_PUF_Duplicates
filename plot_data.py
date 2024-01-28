import os

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from PIL import Image

from matplotlib import rcParams

def plot_corr_vs_fhd(df_corr, df_fhd, file_clean):
    fhds = []
    corrs = []
    for x in range(100):
        corrs.extend(df_corr[x].to_numpy().tolist())
        fhds.extend(df_fhd[x].to_numpy().tolist())
    corr_fhd = sns.regplot(x=fhds, y=corrs)
    corr_fhd.set(xlim=(0, 0.7), ylim=(0, 1))
    corr_fhd.get_figure().savefig(
        f'results/{folder}/plots/other/{folder}_pc_vs_fhd_{file_clean}_order1.png'
    )
    plt.clf()
    corr_fhd = sns.regplot(x=fhds, y=corrs, order=2)
    corr_fhd.set(xlim=(0, 0.7), ylim=(0, 1))
    corr_fhd.get_figure().savefig(
        f'results/{folder}/plots/other/{folder}_pc_vs_fhd_{file_clean}_order2.png'
    )
    plt.clf()
    corr_fhd = sns.regplot(x=fhds, y=corrs, order=3)
    corr_fhd.set(xlim=(0, 0.7), ylim=(0, 1))
    corr_fhd.get_figure().savefig(
        f'results/{folder}/plots/other/{folder}_pc_vs_fhd_{file_clean}_order3.png'
    )
    plt.clf()


def plot_rankings(df_corr, df_fhd, file_clean):
    val_sorted = list(df_corr.mean().sort_values(ascending=False).index)
    fhd_sorted = list(df_fhd.mean().sort_values().index)
    mse = (np.array(list(range(100))) - np.array(
        [fhd_sorted.index(item) for item in val_sorted])) ** 2

    metric_ranking = sns.heatmap(mse[None, :])
    metric_ranking.set(title=file_clean)
    metric_ranking.get_figure().savefig(
        f'results/{folder}/plots/other/{folder}_ranking_{file_clean}_metric_ranking.png')
    plt.clf()


def plot_like_dist_small(df, file_clean):
    to_remove = list(range(30)) + [39, 49, 59, 69, 79, 89, 99]
    df = df.drop(to_remove).drop(to_remove, axis=1)

    plot_like_dist(df, file_clean, is_small=True)


def plot_like_dist(df, file_clean, is_small=False):
    val_sorted = list(df.mean().sort_values(ascending=not is_corr).index)
    hist = sns.displot(data=pd.melt(df[val_sorted[:9]]), x='value',
                       bins=50, col='variable', col_wrap=3, kde=True,
                       facet_kws={'sharex': False})
    hist.fig.suptitle(file_clean, y=1.05)
    xlim = 1.0 if is_corr else 0.6
    hist.set(xlim=(0, xlim))
    hist.savefig(
        f'results/{folder}/plots/hists/{folder}_{file_clean}_like_dist{"_small" if is_small else ""}.png')
    plt.clf()


def plot_unlike_dist(df, file_clean):
    hist = sns.displot(x=df.to_numpy().flatten(), bins=50, kde=True)
    hist.fig.suptitle(file_clean, y=1.05)
    hist.set(xlim=(0, 0.6))
    hist.savefig(
        f'results/{folder}/plots/hists/{folder}_{file_clean}_unlike_dist.png')
    plt.clf()


def create_plot_folders(folder):
    if not os.path.exists(f'results/{folder}/plots'):
        os.mkdir(f'results/{folder}/plots')
        os.mkdir(f'results/{folder}/plots/heatmaps')
        os.mkdir(f'results/{folder}/plots/hists')
        os.mkdir(f'results/{folder}/plots/other')


if __name__ == '__main__':
    img1 = Image.open("data/base/struc84.png")
    img2 = Image.open("data/base/struc91.png")

    fig, axs = plt.subplots(ncols=2)
    img1 = np.array(img1, dtype=float)
    img2 = img1.copy()
    img2[0][0] = -50
    axs[0].imshow(img1)
    axs[1].imshow(img2)
    fig.savefig("speckle_example.png")
    img1 = np.array(img1)
    plt.imshow(img1)
    plt.show()
    exit()

    fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
    sns.heatmap(np.array(img1) + 3, ax=axs[0], cbar=False, xticklabels=False, yticklabels=False, vmin=0, vmax=201, cmap="hot")
    sns.heatmap(np.array(img2) + 3, ax=axs[1], cbar=False, xticklabels=False, yticklabels=False, vmin=0, vmax=201, cmap="hot")
    axs[0].set_title("Copy 1", fontsize=22, fontweight="bold", y=1.05)
    axs[1].set_title("Copy 2", fontsize=22, fontweight="bold", y=1.05)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1)
    plt.savefig("speckle_patterns.png", dpi=500)

    exit()
    data_folders = ["base", 'base_rot10']
    for folder in data_folders:
        _, _, files = next(os.walk(f'results/{folder}'))
        corr_files = list(sorted(filter(lambda f: 'corr' in f, files)))
        fhd_files = list(sorted(filter(lambda f: 'fhd' in f, files)))

        create_plot_folders(folder)

        for file in files:
            file_clean = file.replace('.csv', '')
            is_corr = file.startswith("corr")
            df = pd.read_csv(f"results/{folder}/{file}", header=None)

            heatmap = sns.heatmap(df)
            heatmap.set(title=file_clean)
            heatmap.get_figure().savefig(
                f'results/{folder}/plots/heatmaps/{folder}_{file_clean}_heatmap.png')
            plt.clf()

            plot_like_dist(df, file_clean)
            plot_like_dist_small(df, file_clean)

        for corr_file, fhd_file in zip(corr_files, fhd_files):
            file_clean = corr_file.replace('corr', '').replace('.csv', '')
            df_corr = pd.read_csv(f"results/{folder}/{corr_file}", header=None)
            df_fhd = pd.read_csv(f"results/{folder}/{fhd_file}", header=None)

            plot_corr_vs_fhd(df_corr, df_fhd, file_clean)
            plot_rankings(df_corr, df_fhd, file_clean)

    _, dirs, _ = next(os.walk(f'results'))
    unlike_dirs = list(filter(lambda d: 'vs' in d, dirs))
    for folder in unlike_dirs:
        _, _, files = next(os.walk(f'results/{folder}'))
        create_plot_folders(folder)

        for file in files:
            file_clean = file.replace('.csv', '')
            df = pd.read_csv(f"results/{folder}/{file}", header=None)
            plot_unlike_dist(df, file_clean)
