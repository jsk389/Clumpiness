import numpy as np
import numpy as np
import glob
import matplotlib.pyplot as plt
plt.style.use('seaborn-colorblind')
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

import dropbox

dbx = dropbox.Dropbox('vz3W7ss7jxYAAAAAAAFJcDMCEY9AGA_5MA31rtgY8TQpyibVbCC4A5Pk7Yyww5eU')

def discrete_cmap(N, base_cmap=None):
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, 10)[:N])
    cmap_name = base.name + str(N)
    try:
        return base.from_list(cmap_name, color_list, N)
    except:
        return plt.cm.get_cmap(base_cmap, N)

class PointBrowser(object):
    """
    Click on a point to select and highlight it -- the data that
    generated the point will be shown in the lower axes.  Use the 'n'
    and 'p' keys to browse through the next and previous points
    """

    def __init__(self, ax, df, campaign='1', lc_type='K2SFF', cols_to_plot=['numax', 'denv']):
        self.ax = ax
        self.df = df
        self.campaign = campaign
        self.lc_type = lc_type
        self.xs = self.df[cols_to_plot[0]].values
        self.ys = self.df[cols_to_plot[1]].values

        self.lastind = np.random.randint(len(self.df))

        self.selected, = self.ax[0].plot([self.xs[self.lastind]], [self.ys[self.lastind]], 'o', ms=12, alpha=0.4,
                                 color='yellow', visible=True)

        # Plot first randomly selected star
        kic = self.df['EPIC'].iloc[self.lastind]
        data = self.read_data(kic)

        # Retrieve time series
        data = self.read_data(kic)
        self.plot_ts()
        self.plot_unts()
        self.plot_psd()

        self.ax[0].set_title(r'EPIC {} selected'.format(self.df['EPIC'].iloc[self.lastind]))

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
        kic = int(kic)
        if self.lc_type == 'EVEREST':
            fname = '/python/K2-Clumpiness/C'+str(self.campaign)+'/everest/hlsp_everest_k2_llc_'+str(kic)+'-c'+str(self.campaign).zfill(2)+'_kepler_v2.0_lc.fits'
            print(fname)
            f = dbx.files_download_to_file('hlsp_everest_k2_llc_'+str(kic)+'-c'+str(self.campaign).zfill(2)+'_kepler_v2.0_lc.fits', fname)
            dat = Table.read('hlsp_everest_k2_llc_'+str(kic)+'-c'+str(self.campaign).zfill(2)+'_kepler_v2.0_lc.fits', format='fits')
            os.remove('hlsp_everest_k2_llc_'+str(kic)+'-c'+str(self.campaign).zfill(2)+'_kepler_v2.0_lc.fits')
            #self.time = df['TIME'][df['QUALITY'] == 0].values
            #self.flux = df['FCOR'][df['QUALITY'] == 0].values
            #print(df['QUALITY'])
            df = dat.to_pandas()
            self.time = df['TIME'][df['QUALITY'] == 0].values
            self.flux = df['FCOR'][df['QUALITY'] == 0].values
        elif self.lc_type == 'V&J':
            fname = '/python/K2-Clumpiness/C'+str(self.campaign)+'/K2SFF/hlsp_k2sff_k2_lightcurve_'+str(kic)+'-c'+str(self.campaign).zfill(2)+'_v1_llc.fits'
            #print(fname)
            f = dbx.files_download_to_file('hlsp_k2sff_k2_lightcurve_'+str(kic)+'-c'+str(self.campaign).zfill(2)+'_v1_llc.fits', fname)
            dat = Table.read('hlsp_k2sff_k2_lightcurve_'+str(kic)+'-c'+str(self.campaign).zfill(2)+'_v1_llc.fits', format='fits')
            os.remove('hlsp_k2sff_k2_lightcurve_'+str(kic)+'-c'+str(self.campaign).zfill(2)+'_v1_llc.fits')
            df = dat.to_pandas()
            self.time = df['T'].values
            self.flux = df['FCOR'].values
        elif self.lc_type == 'NASA':
            fname = '/python/K2-Clumpiness/C'+str(self.campaign)+'/NASA/ktwo'+str(kic)+'-c'+str(self.campaign).zfill(2)+'_llc.fits'
            print(fname)
            f = dbx.files_download_to_file('ktwo'+str(kic)+'-c'+str(self.campaign).zfill(2)+'_llc.fits', fname)
            dat = Table.read('ktwo'+str(kic)+'-c'+str(self.campaign).zfill(2)+'_llc.fits', format='fits')
            os.remove('ktwo'+str(kic)+'-c'+str(self.campaign).zfill(2)+'_llc.fits')
            df = dat.to_pandas()
            self.time = df['TIME'][df['SAP_QUALITY'] == 0].values
            self.flux = df['PDCSAP_FLUX'][df['SAP_QUALITY'] == 0].values
        else:
            sys.exit('WRONG lc_type!')



        self.time = self.time[np.isfinite(self.flux)]
        self.flux = self.flux[np.isfinite(self.flux)]
        #med = self.med_filt(self.time, self.flux, dt=2.0)
        self.unflux = self.flux.copy()
        self.untime = self.time.copy()
        self.med = self.med_filt(self.time, self.flux, dt=2.0)
        #plt.plot(self.time, self.flux)
        #plt.plot(self.time, med)
        #plt.show()
        self.flux = 1e6 * ((self.flux / self.med) - 1.0)
        clip = 5.0
        self.time = self.time[np.abs(self.flux) < clip * np.std(self.flux)]
        self.flux = self.flux[np.abs(self.flux) < clip * np.std(self.flux)]
        self.time = self.time [np.abs(self.flux) < clip * np.std(self.flux)]
        self.flux = self.flux[np.abs(self.flux) < clip * np.std(self.flux)]

        self.f, self.psd = self.compute_ps(self.time, self.flux)

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
        self.ax[3].cla()
        self.ax[3].plot(self.f, self.psd, 'k')
        self.ax[3].set_xlim(1, 283)
        self.ax[3].set_xscale('log')
        self.ax[3].set_yscale('log')
        self.ax[3].set_xlabel(r'Frequency ($\mu$Hz)', fontsize=12)
        self.ax[3].set_ylabel(r'PSD (ppm$^{2}\mu$Hz$^{-1}$)', fontsize=12)

    def plot_unts(self):
        self.ax[1].cla()
        self.ax[1].plot(self.untime, self.unflux, 'k')
        self.ax[1].plot(self.untime, self.med, 'C1')
        self.ax[1].set_xlabel(r'Time (BJD)', fontsize=12)
        self.ax[1].set_ylabel(r'Flux (ppm)', fontsize=12)

    def plot_ts(self):
        self.ax[2].cla()
        self.ax[2].plot(self.time, self.flux, 'k')
        self.ax[2].set_xlabel(r'Time (BJD)', fontsize=12)
        self.ax[2].set_ylabel(r'Flux (ppm)', fontsize=12)

    def update(self):
        if self.lastind is None:
            return

        dataind = self.lastind

        kic = self.df['EPIC'].iloc[dataind]
        data = self.read_data(kic)

        # Plot power spectrum
        self.read_data(kic)
        self.plot_unts()
        self.plot_ts()
        self.plot_psd()

        #ax2.text(0.05, 0.9, 'mu=%1.3f\nsigma=%1.3f' % (self.xs[dataind], self.ys[dataind]),
        #         transform=ax2.transAxes, va='top')
        #ax2.set_ylim(-0.5, 1.5)
        self.selected.set_visible(True)
        self.selected.set_data(self.xs[dataind], self.ys[dataind])

        self.ax[0].set_title(r'EPIC {} selected'.format(self.df['EPIC'].iloc[dataind]))
        fig.canvas.draw()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import sys

    campaign = str(sys.argv[1])
    lc_type = str(sys.argv[2])
    print("Plotting data for Campaign {} using {} data".format(campaign, lc_type))
    print(campaign)
    df = pd.read_csv('New_Gaps_output_data_noise_KIC_-1_white_noise_test.txt')#pd.read_csv('New_Gaps_output_data_noise_EPIC_C1_-1_EVEREST_white_noise_test.txt')
    #df = pd.read_csv('Gaps_output_data_noise_KIC_-1.txt')
    #df = df[df['fill'] > 0.5]
    #print(len(df))
    #print(len(df[df['hoc'] > 2]))
    #print(df.sort_values(by=['hoc']))
    dfstar = pd.read_csv('New_Gaps_output_data_noise_EPIC_C'+str(campaign)+'_-1_'+str(lc_type)+'.txt')#80.txt')
    #dfstar = pd.read_csv('New_Gaps_output_data_noise_KIC_-1_APOKASC.txt')#80.txt')
    #dfstar2 = pd.read_csv('New_Gaps_output_data_noise_KIC_180_APOKASC.txt')#80.txt')
    dfstar3 = pd.read_csv('New_Gaps_output_data_noise_KIC_-1_APOKASC.txt')#80.txt')
    #dfstar3 = pd.read_csv('New_Gaps_output_data_noise_Star_TCA1_-1.txt')#80.txt')
    print(dfstar3)
    #dfstar4 = pd.read_csv('New_Gaps_output_data_noise_KIC_27_APOKASC.txt')#80.txt')
    dfstar3 = dfstar3[dfstar3['evo'] < 6]
    dfstar3['colour'] = 'C0'
    colorblind = ["#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC",
                  "#CA9161", "#FBAFE4", "#949494", "#ECE133", "#56B4E9"]
    #dfstar3['colour'] = np.array([colorblind[int(dfstar3['evo'].iloc[i])] for i in range(len(dfstar3))])
    dfstar3['colour'] = np.array(['C'+str(int(dfstar3['evo'].iloc[i])) for i in range(len(dfstar3))])
    print(dfstar3['colour'])
    plt.scatter(dfstar3['zc'], dfstar3['var'], marker='.', c=dfstar3['colour'], s=10, alpha=0.75)#, cmap=cmap, norm=norm)
    #plt.xlabel(r'$\nu_{\mathrm{max}}$ ($\mu$Hz)', fontsize=18)
    plt.ylabel(r'Variance (ppm$^{4}$)', fontsize=18)
    plt.xlabel(r'Normalised zero-crossings', fontsize=18)
    from matplotlib.lines import Line2D
    label = ['Low-luminosity RGB', 'High-luminosity RGB', 'Confusion region RGB', 'HeCB', 'Noise (long-cadence dwarfs)']
    print(len(np.unique(dfstar3['evo'])))
    legend_elements = [Line2D([0], [0], color='C'+str(int(i)), lw=2, label=label[i]) for i in range(len(np.unique(dfstar3['evo'])))]
    plt.legend(handles=legend_elements, loc='upper right')
    #plt.colorbar()
    plt.xlim(7e-3, 0.4)
    plt.xscale('log')
    plt.yscale('log')
    plt.clim(-0.5, len(np.unique(dfstar3['evo']))-0.5)
    plt.show()
    #dfstar = dfstar.dropna(axis='rows')
    #dfstar2 = dfstar2.dropna(axis='rows')

    fig, ax = plt.subplots()
    plt.scatter(dfstar['zc'], dfstar['hoc'], marker='.', zorder=2, alpha=0.1)
    #plt.scatter(dfstar2['zc'], dfstar2['hoc'], marker='.', alpha=0.1)
    plt.scatter(dfstar3['zc'], dfstar3['hoc'], marker='.', zorder=1)
    #plt.scatter(dfstar4['zc'], dfstar4['hoc'], marker='.', alpha=0.1)
    plt.show()
    #dfstar = pd.read_csv('New_Gaps_output_data_noise_KIC_-1.txt')
    #print(dfstar)
    #dfstar = dfstar[dfstar['fill'] > 0.5]
    #print(np.shape(dfstar))
    #print(np.shape(dfstar))
    param = 'zc'
    #dfstar['EPIC'] = dfstar['EPIC'].astype(int)
    # Read in dwarfs - Stello, Zinn, Elsworth 2017 (table) 4
    #dwarfs = pd.read_csv('../../K2-Clumpiness/K2-GAP-C1-dwarfs.txt', delimiter=r'\s+', comment='#')
    #dwarfs['class'] = np.argmax(dwarfs[['P_1','P_2','P_3','P_4']].values, axis=1)
    #dwarfs['EPIC'] = dwarfs['EPIC'].astype(int)

    #print(np.shape(dfstar), np.shape(dwarfs))
    #dfstar = pd.merge(dfstar, dwarfs, on='EPIC', how='left')
    print(dfstar.head())
    print(np.unique(dfstar['evo'], return_counts=True))
#    plt.scatter(dfstar['zc'], dfstar['hoc'], c=dfstar['fill'], marker='.')#c=dfstar['fill'], marker='.', cmap='viridis')
    plt.hexbin(dfstar['zc'], dfstar['hoc'], gridsize=200, bins='log', cmap='inferno_r')#c=dfstar['fill'], marker='.', cmap='viridis')
    plt.xlim(0, 0.44)
    plt.ylim(0, 0.8)
    plt.colorbar()
    #plt.scatter(df['zc'], df['hoc'], c='C2', marker='.')
    plt.show()

    #plt.scatter(dfstar['zc'], dfstar['hoc'], c=dfstar['faps'], marker='.', cmap='viridis')
    #plt.colorbar()
    #plt.show()

    #print(np.shape(dfstar))
    #detections = pd.read_csv('../../K2-Clumpiness/K2-GAP-C1-dnu-numax.txt', delimiter=r'\s+')
    #detections['EPIC'] = detections['EPIC'].astype(int)

    #dfstar = pd.merge(dfstar, detections, on='EPIC', how='left')
    #dfstar = dfstar[dfstar['class'] == 3]
    #print(dfstar.head())
    #dprint(len(df))
    fig, ax = plt.subplots(4, 1)
    #plt.subplots_adjust(hspace=0.001)
    #x = ["C0" if i == 0 else "C1" for i in df['CONS_EVSTATES'].values]
    #print(np.max(df['beta']/df['numax']), np.min(df['beta']/df['numax']))
    line = ax[0].scatter(dfstar['zc'], dfstar['hoc'], c=np.log10(dfstar['faps']), marker='.', cmap='viridis', picker=5)  # 5 points tolerance
    ax[0].set_yscale('log')
    ax[0].set_xscale('log')
    ax[0].set_xlim(2e-3, 0.6)
    ax[0].set_ylim(1e-2, 1e2)
    fig.colorbar(line, ax=ax[0])

    ax[0].set_xscale('log')
    #ax[0].set_yscale('log')
    #ax[0].set_xlabel(r'$\log_{10}\sigma^{2}$', fontsize=18)
    ax[0].set_xlabel(r'Normalised zero crossings')#, fontsize=18)
    ax[0].set_ylabel(r'$\psi^{2}$')#, fontsize=18)
    #plt.legend(loc='best')
    #ax[0].set_xlim(2, 10)
    #ax[0].set_ylim(0, 0.6)

    browser = PointBrowser(ax, dfstar, campaign=campaign, lc_type=lc_type, cols_to_plot=['zc', 'hoc'])

    fig.canvas.mpl_connect('pick_event', browser.onpick)
    fig.canvas.mpl_connect('key_press_event', browser.onpress)
    plt.show()
