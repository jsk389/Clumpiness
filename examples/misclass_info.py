#!/usr/bin/env/ python3

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-colorblind')
from tqdm import tqdm as tqdm
import pandas as pd
from matplotlib.colors import LogNorm

import sys

import random

if __name__=="__main__":

    N_DAYS = 27
    df = pd.read_csv('misclass_'+str(N_DAYS)+'.csv')
    df = df[df['evo'] != 5]
    #df180 = pd.read_csv('Updated_output_data_noise_KIC_180.txt')
    #df80 = pd.read_csv('Updated_output_data_noise_KIC_80.txt')
    df27 = pd.read_csv('Colours_New_Gaps_output_data_noise_KIC_27_APOKASC.csv')
    print(df27.head(), df.head())
    df27 = df27[['KIC', 'var', 'zc', 'mc', 'evo']]
    df = df[['KIC', 'var', 'zc', 'evo', 'mc', 'classification']]
    df27.dropna(axis=0, inplace=True)
    df.dropna(axis=0, inplace=True)

    param = 'zc'
    #df27 = df27.replace([np.inf, -np.inf], np.nan)
    #print(df27.isnull().sum(), len(df27))
    #print(df.columns)
    df = df[(df['var'] > 10) & (df['mc'] < np.sqrt(1e6))]
    labels = ["LLRGB", "HLRGB", "Confusion RGB", "HeCB", "Noise"]
    #df['classification'] = np.abs(1 - df['classification'])

    df_new = df.groupby(['KIC']).mean()
    #print(len(df_new['classification'][df_new['classification'] == 1])/len(df_new))
    #plt.scatter(df_new['numax'], df_new['classification'], c=df_new['evo'], marker='.')
    #plt.colorbar()
    #plt.show()

    print(np.max(np.log10(df['var'])), np.min(np.log10(df['var'])))
    counts, xedges, yedges, image = plt.hist2d(np.log10(df['var'][df['classification'] == 0]), df[param][df['classification'] == 0], bins=100)
    counts1, xedges1, yedges1, image1 = plt.hist2d(np.log10(df['var'][df['classification'] == 1]), df[param][df['classification'] == 1], bins=[xedges, yedges])
    plt.close()


    plt.figure(1)
    for i in reversed(range(len(np.unique(df['evo'])))):
        plt.scatter(np.log10(df['var'])[df['evo'] == i], df[param][df['evo'] == i], s=0.1, marker='.',color='C'+str(i), label=labels[i], alpha=0.5)
    plt.xlabel(r'$\log_{10}\sigma^{2}$', fontsize=18)
    plt.ylabel(r'Normalised zero crossings', fontsize=18)
    #plt.ylabel(r'Higher order crossings', fontsize=18)
    plt.legend(loc='best')

    plt.figure(2)
    print(np.max(counts), np.min(counts), np.max(counts1), np.min(counts))
    cc = counts/(counts+counts1)
    print("MIN MAX: ", np.min(cc), np.max(cc))
    print("NAN MIN etc: ", np.nanmin(cc), np.nanmax(cc))
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
    cmap = matplotlib.cm.magma_r
    cmap.set_bad('white', 0)
    nbins = 100
    from scipy.stats import kde
    print(np.isfinite(cc.ravel()).sum(), len(cc.ravel()))
    k = kde.gaussian_kde(cc.T)

    # Regular grid to evaluate kde upon

    x,y = np.meshgrid(xedges, yedges)
    grid_coords = np.append(x.reshape(-1,1),y.reshape(-1,1),axis=1)

    zi = k(grid_coords)
    plt.pcolormesh(xedges, yedges, zi.reshape(xedges.shape))
    #plt.imshow(cc.T, origin='lower', extent=[xcentres[0], xcentres[-1], ycentres[0], ycentres[-1]], aspect='auto', cmap=cmap, interpolation='bicubic')
#    plt.colorbar(label=r'Classification Rate')
    import scipy.ndimage
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']#'r', '0', 'orange']
#    for i in range(len(np.unique(df['evo']))):
    #    #counts, ybins, xbins, image = plt.hist2d(np.log10(df180['var'][df180['evo'] == i]), df180['zc'][df180['evo']==i], bins=100)
#        counts, xbins, ybins = np.histogram2d(np.log10(df27['var'][df27['evo'] == i]), df27[param][df27['evo']==i], bins=500)
#        xcentres = (xbins[:-1] + xbins[1:]) / 2
#        ycentres = (ybins[:-1] + ybins[1:]) / 2
        #k = kde.gaussian_kde(cc.T)
        #zi = k(np.vstack([xbins, ybins]))
        #plt.pcolormesh(xbins, ybins, zi.reshape(xbins.shape))
  
        #print(labels[i], colors[i])
        #plt.contour(counts.T, extent=[xcentres[0], xcentres[-1], ycentres[0], ycentres[-1]], linewidth=2, label=labels[i], zorder=1, colors=colors[i])
    #plt.scatter(np.log10(df['var']), df[param], c=df['classification'], s=5, cmap='jet', zorder=1)
    plt.xlabel(r'$\log_{10}\sigma^{2}$', fontsize=18)
    plt.ylabel(r'Normalised zero crossings', fontsize=18)
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='C0', lw=2),
                    Line2D([0], [0], color='C1', lw=2),
                    Line2D([0], [0], color='C2', lw=2),
                    Line2D([0], [0], color='C3', lw=2),
                    Line2D([0], [0], color='C4', lw=2)]
    #plt.ylabel(r'Higher order crossingss', fontsize=18)
    #plt.ylabel(r'Variance of first derivative', fontsize=18)
    import textwrap
    #    labels = ["High-luminosity RGB", "Confusion region RGB", "HeCB", "Noise", "Low-luminosity RGB"]
    labels = ["\n".join(textwrap.wrap(i, 10)) for i in labels]
    plt.legend(custom_lines, labels, loc='upper right', ncol=1)
    plt.show()
