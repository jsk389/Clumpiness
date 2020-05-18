import numpy as np
import numpy as np
import glob
import matplotlib.pyplot as plt
#plt.style.use('seaborn-colorblind')
from sklearn import mixture
import os

from pylab import *
from matplotlib.colors import colorConverter
import pandas as pd
from tqdm import tqdm
import pyfits
from astropy.table import Table
import gatspy.periodic as gp
import sys
import scipy.interpolate as interp

import dropbox

dbx = dropbox.Dropbox('vz3W7ss7jxYAAAAAAAFJcDMCEY9AGA_5MA31rtgY8TQpyibVbCC4A5Pk7Yyww5eU')

class PointBrowser(object):
    """
    Click on a point to select and highlight it -- the data that
    generated the point will be shown in the lower axes.  Use the 'n'
    and 'p' keys to browse through the next and previous points
    """

    def __init__(self, ax, df, cols_to_plot=['numax', 'denv']):
        self.ax = ax
        self.df = df
        self.xs = self.df[cols_to_plot[0]].values
        self.ys = self.df[cols_to_plot[1]].values

        self.lastind = np.random.randint(len(self.df))

        self.selected, = self.ax[0].plot([self.xs[self.lastind]], [self.ys[self.lastind]], 'o', ms=12, alpha=0.4,
                                 color='yellow', visible=True)

        # Plot first randomly selected star
        kic = self.df['KIC'].iloc[self.lastind]
        data = self.read_data(kic)

        # Retrieve time series
        data = self.read_data(kic)
        self.plot_ts()
        self.plot_psd()

        self.ax[0].set_title(r'KIC {} selected'.format(self.df['KIC'].iloc[self.lastind]))

    def med_filt(self, x, y, dt=4.):
        """
        De-trend a light curve using a windowed median.
        """
        x, y = np.atleast_1d(x), np.atleast_1d(y)
        assert len(x) == len(y)
        r = np.empty(len(y))
        r1 = np.empty(len(y))
        r2 = np.empty(len(y))
        for i, t in enumerate(x):
            inds = (x >= t - 0.5 * dt) * (x <= t + 0.5 * dt)
            r[i] = np.nanmean(y[inds])
        for i, t in enumerate(x):
            inds = (x >= t - 0.5 * dt * 1.4303) * (x <= t + 0.5 * dt * 1.4303)
            r1[i] = np.nanmean(r[inds])
        for i, t in enumerate(x):
            inds = (x >= t - 0.5 * dt * 1.4303 * 1.4303) * (x <= t + 0.5 * dt * 1.4303 * 1.4303)
            r2[i] = np.nanmean(r1[inds])
        return r2

    def onpress(self, event):
        if self.lastind is None:
            return
        if event.key not in ('n', 'p'):
            return
        if event.key == 'n':
            inc = 1
        else:
            inc = -1

        self.lastind += inc
        self.lastind = np.clip(self.lastind, 0, len(self.xs) - 1)
        self.update()

    def onpick(self, event):

        if event.artist != line:
            return True

        N = len(event.ind)
        if not N:
            return True

        # the click locations
        x = event.mouseevent.xdata
        y = event.mouseevent.ydata

        distances = np.hypot(x - self.xs[event.ind], y - self.ys[event.ind])
        indmin = distances.argmin()
        dataind = event.ind[indmin]

        self.lastind = dataind
        self.update()

    def read_data(self, kic, psd=True):
        #fname = glob.glob('/home/kuszlewicz/Dropbox/Python/K2-Clumpiness/C2/hlsp_everest_k2_llc_'+str(kic)+'*.fits')[0]
        fname = glob.glob('/mnt/seismo/scratch/seismo/bell/Kepler_ALL/stitchedlcs/*'+str(kic)+'*.lc')[0]

        self.time, self.flux = np.loadtxt(fname, unpack=True)

        #self.time = df['TIME'].values#[df['QUALITY'] == 0].values
        #self.flux = df['FLUX'].values#[df['QUALITY'] == 0].values


        #med = self.med_filt(self.time, self.flux, dt=2.0)
        #plt.plot(self.time, self.flux)
        #plt.plot(self.time, med)
        #plt.show()
        # Fill single point gaps everything three days
        self.time, self.flux = self.fill_gaps(self.time, self.flux)
        self.f, self.psd = self.compute_ps(self.time, self.flux)
        # Detect large gaps and merge if necessary
        idx = np.where(np.diff(self.time) > 20)[0]
#        print(idx)
        if len(idx) > 0:
#            plt.plot(self.time, self.flux)
#            for i in range(len(idx)):
#                plt.axvline(self.time[idx[i]], color='r', linestyle='--')
#                plt.axvline(self.time[idx[i]+1], color='r', linestyle='--')
#
#            plt.show()
#            print(len(self.time))
            for i in range(len(idx)):
                self.time[idx[i]+1:] -= (self.time[idx[i]+1] - self.time[idx[i]])
            self.flux = self.flux[np.isfinite(self.time)]
            self.time = self.time[np.isfinite(self.time)]
            self.time = self.time[np.isfinite(self.flux)]
            self.flux = self.flux[np.isfinite(self.flux)]
    def fill_gaps(self, time, flux):
        mask = np.invert(np.isnan(flux))
        tm = time[mask]
        dt = np.diff(tm)
        good = np.where(dt < (((29.4*60.0)/86400.0) * 2.5))
        new_flux = interp.interp1d(tm, flux[mask], bounds_error=False)(time)
        return time[good], new_flux[good]

    def compute_ps(self, time,flux):
        time-=time[0]
        if time[1]<1:
             time=time*86400

        c=[]
        for i in range(len(time)-1):
             c.append(time[i+1]-time[i])
        c=np.median(c)
        #print(c)
        nyq=1/(2*(time[1]-time[0]))
        nyq=1/(2*c)
        #print(nyq*1e6)
        df=1/time[-1]

        f,p=gp.lomb_scargle_fast.lomb_scargle_fast(time,flux,f0=df,df=df,Nf=1*(nyq/df))
        lhs=(1/len(time))*np.sum(flux**2)
        rhs= np.sum(p)
        ratio=lhs/rhs
        p*=ratio/(df*1e6)#ppm^2/uHz
        f*=1e6
        return f,p


    def plot_psd(self):
        self.ax[2].cla()
        self.ax[2].plot(self.f, self.psd, 'k')
        self.ax[2].set_xlim(1, 283)
        self.ax[2].set_xscale('log')
        self.ax[2].set_yscale('log')
        self.ax[2].set_xlabel(r'Frequency ($\mu$Hz)', fontsize=12)
        self.ax[2].set_ylabel(r'PSD (ppm$^{2}\mu$Hz$^{-1}$)', fontsize=12)

    def plot_ts(self):
        self.ax[1].cla()
        self.ax[1].plot(self.time, self.flux, 'k')
        self.ax[1].set_xlabel(r'Time (BJD)', fontsize=12)
        self.ax[1].set_ylabel(r'Flux (ppm)', fontsize=12)

    def update(self):
        if self.lastind is None:
            return

        dataind = self.lastind

        kic = self.df['KIC'].iloc[dataind]
        data = self.read_data(kic)

        # Plot power spectrum
        self.read_data(kic)
        self.plot_ts()
        self.plot_psd()

        #ax2.text(0.05, 0.9, 'mu=%1.3f\nsigma=%1.3f' % (self.xs[dataind], self.ys[dataind]),
        #         transform=ax2.transAxes, va='top')
        #ax2.set_ylim(-0.5, 1.5)
        self.selected.set_visible(True)
        self.selected.set_data(self.xs[dataind], self.ys[dataind])

        self.ax[0].set_title(r'KIC {} selected'.format(self.df['KIC'].iloc[dataind]))
        fig.canvas.draw()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dfstar = pd.read_csv('New_Gaps_output_data_noise_KIC_-1_fullKep.txt')#-1.txt')#80.txt')
    print(len(dfstar))
    dfstar = dfstar[dfstar['fill'] > 0.5]
    dfstar = dfstar[dfstar['ndata'] > 1000]
    #print(np.min(dfstar['ndata']))
    #sys.exit()
    dr25 = pd.read_csv('dr25catalogue.txt', delimiter=r'\s+')
    dfstar = pd.merge(dfstar, dr25, how='inner', on='KIC')
    gaia = pd.read_csv('DR2PapTable1.txt', delimiter='&')
    dfstar = pd.merge(dfstar, gaia, how='inner', on='KIC')

#[['KIC', 'Teff']], how='inner', on='KIC')
    hot = dfstar[dfstar['Teff'] > 6500]
    hot['det'] = 0
    hot = hot[['KIC', 'det']]
    rg = pd.read_csv('yu_2016.csv', delimiter='|', na_values='--')
    rg['det'] = 1
    rg.rename(columns={'KICID': 'KIC'}, inplace=True)
    rg = rg[['KIC', 'det']]
    nondet = pd.read_csv('Non-Det.txt')
    nondet['det'] = 0
    new_osc = pd.read_csv('New_osc.csv')
    #print(new_osc)
    new_osc['det'] = 2
    #new_osc[['KIC']].to_csv('New_osc_kics.csv')
    #sys.exit()
    print(len(rg), len(nondet), len(hot), len(new_osc))
    rg = rg.append(nondet, ignore_index=True)
    rg = rg.append(new_osc[['KIC', 'det']], ignore_index=True)
    rg = rg.append(hot, ignore_index=True)

    plt.scatter(gaia['teff'], gaia['rad'], marker='.', s=5, color='C0', alpha=0.2, label=r'Berger et al. (2018)')
    dfstar = pd.merge(dfstar, rg, how='inner', on='KIC')
    dets = dfstar[dfstar['det']  ==  2]
    print(len(dets[dets['rad'] < 2])/len(dets))
  
    dwarfs = dets[dets['rad'] < 2]['KIC']
    plt.close('all')
    contam = pd.read_csv('new_dets_contam.txt')
    contam['min'] = np.min(contam[['Contamination0', 'Contamination1', 'Contamination2', 'Contamination3']], axis=1)
    contam['max'] = np.max(contam[['Contamination0', 'Contamination1', 'Contamination2', 'Contamination3']], axis=1)
    contam['minmax'] = contam['max'] - contam['min']
    
    print(np.shape(contam))
    contam = contam.dropna(axis='rows')
    print(np.shape(contam))
    print(contam.head())

    contams = pd.merge(dets, contam, how='inner', on='KIC')
    print(len(contams[contams['minmax'] < 0.1]))
    contams['minmax'][contams['minmax'] < 0.1] = 0
    contams['minmax'][contams['minmax'] >= 0.1] = 1
    plt.scatter(contams['teff'][contams['minmax'] == 0], contams['rad'][contams['minmax'] == 0], color='k', marker='.')
    plt.scatter(contams['teff'][contams['minmax'] == 1], contams['rad'][contams['minmax'] == 1], color='r', marker='.')
    plt.ylim(0.1, 200) 
    plt.xlim(6000, 3000)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$T_{\mathrm{eff}}$ ($K$)', fontsize=18)
    plt.ylabel(r'Stellar Radius ($R_{\odot}$)', fontsize=18)
    plt.show()   

    
    plt.scatter(dfstar[dfstar['det']  ==  1]['teff'], dfstar[dfstar['det']  ==  1]['rad'], marker='x', color='k', alpha=0.8, label=r'Yu et al. (2018)')
    plt.scatter(dfstar[dfstar['det']  ==  2]['teff'], dfstar[dfstar['det']  ==  2]['rad'], marker='x', color='r', alpha=0.4, label=r'New detections')
    plt.gca().invert_xaxis()
    plt.xlim(10000, 3000)
    plt.ylim(0.1, 200)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$T_{\mathrm{eff}}$ ($K$)', fontsize=18)
    plt.ylabel(r'Stellar Radius ($R_{\odot}$)', fontsize=18)
    plt.legend(loc='best')
    plt.show()   

    print(np.shape(dfstar))
    dfstar = dfstar.dropna(axis='rows')
    print(np.shape(dfstar))
    #sys.exit()    
    print(dfstar)
    param = 'zc'



    #plt.scatter(dfstar['zc'][dfstar['numax'] > 0], dfstar['hoc'][dfstar['numax'] > 0], c=dfstar['numax'][dfstar['numax'] > 0], marker='.', cmap='viridis')
    #plt.colorbar()
    #plt.show()

    plt.hexbin(dfstar['zc'], dfstar['hoc'], gridsize=200, bins='log', cmap='Reds_r')#c=dfstar['fill'], marker='.', cmap='viridis')
    plt.xlim(0.0, 0.6)
    plt.ylim(0,0.8)
    plt.xlabel(r'Normalised zero-crossings', fontsize=18)
    plt.ylabel(r'$\psi^{2}$', fontsize=18)
    plt.colorbar(label=r'Logarithmic number density')
    #plt.scatter(df['zc'], df['hoc'], c='C2', marker='.')
    plt.show()

    print(dfstar.head())
    fig, ax = plt.subplots(3, 1)
    #plt.subplots_adjust(hspace=0.001)
    #x = ["C0" if i == 0 else "C1" for i in df['CONS_EVSTATES'].values]
    #print(np.max(df['beta']/df['numax']), np.min(df['beta']/df['numax']))
    line = ax[0].scatter(dfstar['teff'], dfstar['rad'], c=dfstar['det'], marker='.', cmap='viridis', picker=5)  # 5 points tolerance
    #ax[0].set_yscale('log')
    #ax[0].set_xscale('log')
#    plt.gca().invert_xaxis()
    ax[0].set_xlim(10000, 3000)
    ax[0].set_ylim(0.1, 200)
    plt.xscale('log')
    plt.yscale('log')
    #ax[0].set_xlim(0, 0.6)
    #ax[0].set_ylim(0, 1.0)
    fig.colorbar(line, ax=ax[0])

    #ax[0].set_xscale('log')
    #ax[0].set_yscale('log')
    #ax[0].set_xlabel(r'$\log_{10}\sigma^{2}$', fontsize=18)
    ax[0].set_xlabel(r'Normalised zero crossings')#, fontsize=18)
    ax[0].set_ylabel(r'$\psi^{2}$')#, fontsize=18)
    #plt.legend(loc='best')
    #ax[0].set_xlim(2, 10)
    #ax[0].set_ylim(0, 0.6)

    browser = PointBrowser(ax, dfstar, cols_to_plot=['teff', 'rad'])

    fig.canvas.mpl_connect('pick_event', browser.onpick)
    fig.canvas.mpl_connect('key_press_event', browser.onpress)

    plt.show()
