#!/usr/bin/env/ python3

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-colorblind')

from tqdm import tqdm as tqdm
import pandas as pd
from matplotlib.colors import LogNorm
import seaborn as sns
import sys

import random

if __name__=="__main__":

    N_DAYS = 27
    df = pd.read_csv('misclass_'+str(N_DAYS)+'.csv')
    df['KIC'] = df['KIC'].astype(int)
    df = df[df['evo'] != 6]
    #df['classification'] = 1.0 - df['classification']
    df2 = df.groupby(['KIC']).mean()
    df2['evo'] = df2['evo'].replace({0.: "Low-luminosity RGB", 1.0: "High-luminosity RGB", 2.0: "Confusion region RGB", 3.0: "HeCB", 4.0: "Noise"})
    print(df2.head())
    labels = [ "Low-luminosity RGB", "High-luminosity RGB", "Confusion region RGB", "HeCB", "Noise"]

    g = sns.violinplot(x=df2["classification"], y=df2["evo"], cut=0, scale="width", order=labels, alpha=0.8)#, palette="Set3")
    import textwrap
#    labels = ["High-luminosity RGB", "Confusion region RGB", "HeCB", "Noise", "Low-luminosity RGB"]
    labels = ["\n".join(textwrap.wrap(i, 10)) for i in labels]
    g.set_yticklabels(labels, wrap=True)
    plt.ylabel(r'Evolutionary State Label', fontsize=18)
    plt.xlim(0, 1)
    plt.xlabel(r'Misclassification Rate', fontsize=18)
    plt.tight_layout()
    plt.show()

    plt.plot(df2['evo'], df2['classification'], 'o')
    plt.show()

    #print(df.head())
    #print(np.shape(df))
    sys.exit()



    df180 = pd.read_csv('New_output_data_noise_180.txt')
    df80 = pd.read_csv('New_output_data_noise_80.txt')
    df27 = pd.read_csv('New_output_data_noise_27.txt')
    param = 'zc'
    df27 = df27.replace([np.inf, -np.inf], np.nan)
    print(df27.isnull().sum(), len(df27))
    print(df.columns)
    df = df[(df['var'] > 1) & (df['mc'] < 1e6)]
    labels = ['Low-luminosity RGB', 'High-luminosity RGB', 'Confusion region RGB', 'HeCB', 'Noise', 'Unknown']
    df['classification'] = np.abs(1 - df['classification'])

    df_new = df.groupby(['KIC']).mean()
    print(len(df_new['classification'][df_new['classification'] == 1])/len(df_new))
    plt.scatter(df_new['numax'], df_new['classification'], c=df_new['evo'], marker='.')
    plt.colorbar()
    plt.show()


    counts, xedges, yedges, image = plt.hist2d(np.log10(df['var'][df['classification'] == 0]), df[param][df['classification'] == 0], bins=100)
    counts1, xedges1, yedges1, image1 = plt.hist2d(np.log10(df['var'][df['classification'] == 1]), df[param][df['classification'] == 1], bins=[xedges, yedges])
    plt.close()


    plt.figure(1)
    for i in reversed(range(len(np.unique(df['evo'])))):
        plt.scatter(np.log10(df['var'])[df['evo'] == i], df[param][df['evo'] == i], s=0.1, marker='.',color='C'+str(i), label=labels[i], alpha=0.5)
    plt.xlabel(r'$\log_{10}\sigma^{2}$', fontsize=18)
    #plt.ylabel(r'Normalised zero crossings', fontsize=18)
    plt.ylabel(r'Higher order crossings', fontsize=18)
    plt.legend(loc='best')

    plt.figure(2)
    print(np.max(counts), np.min(counts), np.max(counts1), np.min(counts))
    cc = counts/(counts+counts1)
    print(np.min(cc), np.max(cc))
    print(np.nanmin(cc), np.nanmax(cc))
    cc[np.isnan(cc)] = 0
    cc[~np.isfinite(cc)] = 0
    #cc /= np.nanmax(cc)
    print(cc)
    xcentres = (xedges[:-1] + xedges[1:]) / 2
    ycentres = (yedges[:-1] + yedges[1:]) / 2
    cc[np.isnan(cc)] = 0
    cc[~np.isfinite(cc)] = 0
    #plt.contourf(cc.T, 50, extent=[xcentres.min(), xcentres.max(), ycentres.min(), ycentres.max()], cmap='bone', zorder=0)
    import matplotlib
    cmap = matplotlib.cm.magma
    cmap.set_bad('white', 0)
    plt.imshow(cc.T, origin='lower', extent=[xcentres[0], xcentres[-1], ycentres[0], ycentres[-1]], aspect='auto', cmap=cmap)
    plt.colorbar(label=r'Classification Rate')
    for i in reversed(range(len(np.unique(df['evo'])))):
        #counts, ybins, xbins, image = plt.hist2d(np.log10(df180['var'][df180['evo'] == i]), df180['zc'][df180['evo']==i], bins=100)
        counts, xbins, ybins = np.histogram2d(np.log10(df['var'][df['evo'] == i]), df[param][df['evo']==i], bins=100)
        xcentres = (xbins[:-1] + xbins[1:]) / 2
        ycentres = (ybins[:-1] + ybins[1:]) / 2
        plt.contour(counts.T, 10, extent=[xcentres[0], xcentres[-1], ycentres[0], ycentres[-1]], linewidth=2, colors='C'+str(i), label=labels[i], zorder=1)
    #plt.scatter(np.log10(df['var']), df[param], c=df['classification'], s=5, cmap='jet', zorder=1)
    plt.xlabel(r'$\log_{10}\sigma^{2}$', fontsize=18)
    #plt.ylabel(r'Normalised zero crossings', fontsize=18)
    plt.ylabel(r'Higher order crossingss', fontsize=18)
    #plt.ylabel(r'Variance of first derivative', fontsize=18)
    plt.legend(loc='best')
    plt.show()
