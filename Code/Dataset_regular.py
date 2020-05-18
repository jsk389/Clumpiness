#!/usr/bin/env/ python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from astropy.table import Table
from astropy.io import fits
import gatspy.periodic as gp

import scipy.interpolate as interp

from operator import itemgetter
from itertools import groupby

import lightkurve as lk

# https://stackoverflow.com/questions/7460836/how-to-lengenerator
class MyGenerator(object):
    """
    This class is designed to add the length attribute to a generator such that
    the generator can be identified when used in the code i.e. when more than
    one chunk is used
    """
    def __init__(self, _generator, _n):
        self.generator = _generator
        self.n = n


class Dataset(object):
    """
    This is the Dataset class that will handle the reading in and preparation
    of the data for the rest of the analysis.
    """

    def __init__(self, _kic, _fname):

        # KIC number of star
        self.kic = _kic
        # Path to files
        self.fname = _fname

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

    def k2_data_prep(self):
        """
        If K2 data is being used then do a little extra detrending and processing
        of the data.
        """

        lc = lk.LightCurve(time=self.time, flux=self.flux).remove_nans()
        lc = lc.flatten(window_length=(48*5)+1).remove_outliers(sigma=4)

        self.flux = (lc.flux - 1) * 1e6
        self.time = lc.time

 

    def read_timeseries(self, noise=0.0, ndays=-1, chunk=False, tar=None):
        """
        Read in the timeseries from the given file path. This function can take
        data that is in either ascii or .fits format

        Parameters
        ------------------
        noise: float
            Add noise (in ppm) to the time series according to the value given. By default
            this is set to zero.
        n_days: int
            The duration of the time series in days. By default set to -1 which
            results in the entire time series being taken.
        n_sections: int
            The number of time series to return. By default set to 1, so only returns
            1 time series of length n_days. If set to -1 then will return as many
            as possible for length of data
        """
        self.ndays = ndays
        # Assess which file format the data is in
        if (self.fname.endswith(".fits")) and ("hlsp_everest_k2" in self.fname):
            # If dealing with everest lightcurves then need to do a little more preprocessing
            dat = Table.read(self.fname, hdu=1, format='fits')
            df = dat.to_pandas()
            #print(df.head())
            self.time = df['TIME'].values
            self.flux = df['FCOR'].values
            # Convert to relative flux for lightkurve
            self.flux /= np.nanmedian(self.flux)
            #plt.plot(self.time, self.flux)

            self.k2_data_prep()
            #plt.plot(self.time, self.flux)
            #plt.show()
        # For running full K2 stuff
        #elif (self.fname.endswith(".fits")) and ("hlsp_everest_k2" in self.fname):
            # If dealing with everest lightcurves then need to do a little more preprocessing
        #    dat = Table.read(tar.extractfile(self.fname), hdu=1, format='fits')
        #    tar.close()
        #    df = dat.to_pandas()
        #    self.time = df['TIME'].values
        #    self.flux = df['FCOR'].values
        #    # Convert to relative flux for lightkurve
        #    self.flux /= np.nanmedian(self.flux)
        #    #plt.plot(self.time, self.flux)
        #    #plt.show()
        #    self.k2_data_prep()
        elif 'COR_filt_inp' in self.fname:
            hdu = fits.open(self.fname)
            self.time = hdu[0].data[:,0]
            self.flux = hdu[0].data[:,1]
            self.time = self.time[self.flux != 0]
            self.flux = self.flux[self.flux != 0]

            
            #print(np.min(self.flux), np.max(self.flux))
            lc = lk.LightCurve(time=self.time, flux=1+(self.flux/1e6)).remove_nans()
            # Rafa's data only filtered with 80 days, so filter with 20 days for consistency
            lc = lc.flatten(window_length=(48*20)+1).remove_outliers(sigma=4) 
            #print(lc.flux)
            #print(np.mean(lc.flux), np.min(lc.flux), np.max(lc.flux))
            #self.flux = (lc.flux -1) * 1e6
            
            self.flux = (lc.flux - 1) * 1e6
            self.time = lc.time            


        elif self.fname.endswith(".fits"):
            #print("FNAME: ", self.fname)
            dat = Table.read(self.fname, format='fits')
            #print(dat)
            df = dat.to_pandas()
            if 'ktwo' in self.fname:
                self.time = df['TIME'][df['SAP_QUALITY'] == 0].values
                self.flux = df['PDCSAP_FLUX'][df['SAP_QUALITY'] == 0].values
                self.k2_data_prep()
            elif 'hlsp_k2sff' in self.fname:
                self.time = df['T'].values
                self.flux = df['FCOR'].values
                self.k2_data_prep()
            else:
                self.time = df['TIME'].values
                self.flux = df['FLUX'].values
                #self.k2_data_prep()
        # If doesn't end with .fits assume that it is in ascii format, since
        # ascii can have many different endings, e.g. .pow, .txt etc.
        else:
            # Assume that time and flux are the first columns
            dat = np.loadtxt(self.fname)
            self.time = dat[:,0]
            self.flux = dat[:,1]
            if np.max(self.flux) < 2:
                self.flux = (self.flux - 1) * 1e6

        # Check the units of the time array through the cadence
        # Convert time array from days into seconds
        if self.time[1] - self.time[0] < 1000.0:
            self.time -= self.time[0]
            self.time *= 86400.0
        else:
            self.time -= self.time[0]

        # Fill single point gaps everything three days
        if 'COR_filt_inp' in self.fname:
            pass
        else:
            self.time, self.flux = self.fill_gaps(self.time, self.flux)

        # Detect large gaps and merge if necessary
        idx = np.where(np.diff(self.time) > 20*86400.0)[0]
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

#            print(len(self.time))
#            plt.plot(self.time, self.flux)
            #for i in range(len(idx)):
        #        plt.axvline(self.time[idx[i]], color='r', linestyle='--')
        #        plt.axvline(self.time[idx[i]+1], color='r', linestyle='--')
#
#            plt.show()


        # TODO: REMOVE THIS BIT!
        #self.flux[self.flux != 0] = np.random.normal(0, 1, len(self.flux[self.flux != 0]))

        if self.ndays != -1:
            #print(np.nanmax(self.time)/ (self.ndays*86400.0))
            self.n_sections = int(np.ceil((np.nanmax(self.time) / 86400.0) / self.ndays))
            if self.n_sections < 1:
                self.n_sections = 1
        else:
            self.n_sections = 1

        if chunk == True:
            self.new_time = np.array_split(self.time, self.n_sections)
            self.new_flux = np.array_split(self.flux, self.n_sections)
            # If last section is too small then disregard
            # Take threshold as 3/4 * ideal length, that way it is close enough
            # to the ideal length
            if len(self.new_time[-1]) < (0.1 * self.ndays * 86400.0) / (29.4 * 60.0):
                self.new_time = self.new_time[:-1]
                self.new_flux = self.new_flux[:-1]
            # Check to see if arrays of all zeros and remove them!
            idx = []
            for i in range(len(self.new_flux)):
                if (not self.new_flux[i].any()) or (len(self.new_flux[i][self.new_flux[i] != 0])/len(self.new_flux[i]) < 0.1):
                    idx.append(int(i))
            #print("S: ", len(self.new_flux))

            if len(idx) > 0:
                for i in sorted(idx, reverse=True):
                    del self.new_time[i]
                    del self.new_flux[i]

        #print(np.shape(self.time), np.shape(self.flux))
        #plt.plot(self.time, self.flux)
        #plt.show()

        # If only want one section then proceed as normal
        #if ndays != -1:
        #if n_sections != 1:
            # Convert number of days into seconds
        #    ntime = ndays * n_sections * 86400.0
        #    self.flux = self.flux[self.time <= ntime]
        #    self.time = self.time[self.time <= ntime]

        if noise != 0.0:
            self.flux[self.flux != 0] += np.random.normal(0, noise)

    def fill_gaps(self, time, flux):
        mask = np.invert(np.isnan(flux))
        tm = time[mask]
        dt = np.diff(tm)
        good = np.where(dt < (29.4*60.0 * 2.5))
        new_flux = interp.interp1d(tm, flux[mask], bounds_error=False)(time)
        return time[good], new_flux[good]


    def read_power_spectrum(self):
        # Assess which file format the data is in
        if self.fname.endswith(".fits"):
            dat = Table.read(self.fname, format='fits')
            df = dat.to_pandas()
            self.freq = df['frequency'].values
            self.power = df['psd'].values
        # If doesn't end with .fits assume that it is in ascii format, since
        # ascii can have many different endings, e.g. .pow, .txt etc.
        else:
            # Assume that time and flux are the first columns
            dat = np.loadtxt(self.fname)
            self.freq = dat[:,0]
            self.power = dat[:,1]

    def power_spectrum(self, time, flux, df=None):

        t = time[np.isfinite(flux)]# self.time[np.isfinite(self.flux)]
        f = flux[np.isfinite(flux)]#self.flux[np.isfinite(self.flux)]

        t -= t[0]
        if t[1] < 1:
            t = t*86400

        dt = 29.4 * 60.0
        nyq=1/(2*dt)

        if not df:
            df=1/t[-1]
        else:
            pass

        freq, power = gp.lomb_scargle_fast.lomb_scargle_fast(t,
                                                             f,
                                                             f0=df,df=df,
                                                             Nf=1*(nyq/df))
        # Calibrate
        lhs = (1/len(t))*np.sum(f**2)
        rhs = np.sum(power)
        ratio = lhs/rhs
        power *= ratio/(df*1e6)#ppm^2/uHz
        freq *= 1e6
        return freq, power

    def plot_timeseries(self, save=False):
        """
        Plot the time series
        """

        plt.figure()
        plt.plot(self.time, self.flux, 'k')
        plt.xlabel(r'Time (s)', fontsize=18)
        plt.ylabel(r'Flux (ppm)', fontsize=18)
        plt.title(r'KIC '+str(self.kic))
        if save:
            plt.savefig('ts_'+str(self.kic)+'.png')

    def to_regular_sampling(self, regular=False):
        """
        Put Kepler data onto regularly sampled grid
        keyword regular: if True then puts onto regularly sampled grid
        """
        # Cadence in seconds!
        dt = (29.4 * 60.0)# / 86400.0
        # Interpolation function
        #print("LENGTH BEFORE: ", len(self.time))
        mask = np.isfinite(self.time)
        if regular:
            f = interp.interp1d(self.time[mask], self.flux[mask], kind='linear', bounds_error=False)
            # New time array
            #print(np.sum(np.isfinite(self.time)), len(self.time), np.nanmin(self.time), np.nanmax(self.time), dt)
            # Removed max time as nanmax and min time as nanmin and will go from 0 to 4 years to ensure proper limits
            # NOPE the above comment is wrong - only want to put onto regular grid between where there is and isn't data
            # Otherwise will artificially decrease fill massively!
            #if self.ndays == -1:
            self.new_time = np.arange(np.nanmin(self.time),
                                      np.nanmax(self.time),
                                      dt)
            #else:
                # Data stars from zero so will hardcode this for the time being!
            #    self.new_time = np.arange(0,
            #                              self.ndays * 86400.0,
            #                              dt)

            # Hard code for time being
            #if len(self.new_time) < 1323:
            #    self.new_time = np.arange(np.nanmin(self.time),
            #                              np.nanmin(self.time) + 1323*dt,
            #                              dt)
            # New flux array
            self.new_flux = f(self.new_time)
            # Zero centre first!
            #self.new_flux[~np.isfinite(self.new_flux)] -= np.mean(self.new_flux[~np.isfinite(self.new_flux)])
            self.new_flux[np.isfinite(self.new_flux)] -= np.mean(self.new_flux[np.isfinite(self.new_flux)])
            self.new_flux[~np.isfinite(self.new_flux)] = 0
        else:
            #pass
            self.new_flux = self.flux.copy()
            self.new_time = self.time.copy()
            #plt.plot(self.new_time, self.new_flux, color='C0')

            self.new_flux[np.isfinite(self.flux)] -= np.mean(self.new_flux[np.isfinite(self.flux)])
            self.new_flux[~np.isfinite(self.flux)] = 0

            #plt.plot(self.new_time, self.new_flux, color='C2')
            #plt.show()

        #plt.plot(self.time, self.flux, 'k')
        # Allow for slight irregular sampling and work out where gap begins
        times = np.where(np.diff(self.time[mask]) > 1800)
        for i in range(len(times[0])):
            start = self.time[mask][times[0][i]]
            finish = self.time[mask][times[0][i]]+np.diff(self.time[mask])[times[0][i]]
            self.new_flux[(self.new_time > start) & (self.new_time < finish)] = 0


        # If want it in chunks split it up now!
        # Need to think about this more carefully! As features won't end up
        # using these data!
        if self.n_sections != 1:
            n_points = int(self.ndays/ (29.4 / (60.0 * 24)))
            #print(n_points)
            #print(np.shape(self.time))
            self.new_time = []
            self.new_flux = []
            dtau = (self.ndays * 86400.0)
            #print(dtau)
            #self.new_time = np.append(self.new_time, self.time[)])
            for i in range(self.n_sections):
                self.new_time.append(self.time[(self.time >= (i*dtau)) * (self.time <= i*dtau + dtau)])
                self.new_flux.append(self.flux[(self.time >= (i*dtau)) * (self.time <= i*dtau + dtau)])
            #if self.n_sections == 1:
            #    print(self.new_time)
            self.new_time = np.array(self.new_time)
            self.new_flux = np.array(self.new_flux)
            #for i in range(self.n_sections):
            #    print(len(self.new_time[i]), len(self.new_flux[i]))
            #print(self.new_time)
            #sys.exit()

            #sys.exit()
            #self.new_time = np.array_split(self.new_time, self.n_sections)
            #self.new_flux = np.array_split(self.new_flux, self.n_sections)
            #self.new_time = np.array(self.new_time)
            #self.new_flux = np.array(self.new_flux)

            # If last section is too small then disregard
            # Take threshold as 3/4 * ideal length, that way it is close enough
            # to the ideal length
            if len(self.new_time[-1]) < (0.1 * self.ndays * 86400.0) / (29.4 * 60.0):
                self.new_time = self.new_time[:-1]
                self.new_flux = self.new_flux[:-1]
            # Check to see if arrays of all zeros and remove them!
            idx = []
            for i in range(len(self.new_flux)):
                if (not self.new_flux[i].any()) or (len(self.new_flux[i][self.new_flux[i] != 0])/len(self.new_flux[i]) < 0.1):
                    idx.append(int(i))
            #print("S: ", len(self.new_flux))

            if len(idx) > 0:
                #print(len(idx))
                #plt.figure(1)
                #plt.plot(self.time, self.flux)


                #idx = idx.astype(int)
                #print(idx)
                #print(len(self.new_time))
                #print("T: ", len(self.new_time))
                for i in sorted(idx, reverse=True):
                    del self.new_time[i]
                    del self.new_flux[i]
                #self.new_time = np.delete(self.new_time, idx)
                #self.new_flux = np.delete(self.new_flux, idx)
                #plt.figure(2)
                #for i in range(len(self.new_time)):
                #    plt.plot(self.new_time[i], self.new_flux[i])
                #print("X: ", len(self.new_time))
                #plt.show()
            #TODO: K2 detrending
#            for i in range(len(self.new_flux)):
#                self.new_time[i] = self.new_time[i][np.isfinite(self.new_flux[i])]
#                self.new_flux[i] = self.new_flux[i][np.isfinite(self.new_flux[i])]
#                #self.time = self.time[:len(self.time)//2]
#                #self.flux = self.flux[:len(self.flux)//2]
#                self.new_flux[i] = (self.new_flux[i] / 1e6) + 1
#
#                med = self.med_filt(self.new_time[i], self.new_flux[i], dt=2.0*86400.0)
#                #plt.plot(self.time, self.flux)
#                #plt.plot(self.time, med)
#                #plt.show()
#
#                self.new_flux[i] = 1e6 * ((self.new_flux[i] / med) - 1.0)
#
#                clip = 5.0
#                self.new_time[i] = self.new_time[i][np.abs(self.new_flux[i]) < clip * np.std(self.new_flux[i])]
#                self.new_flux[i] = self.new_flux[i][np.abs(self.new_flux[i]) < clip * np.std(self.new_flux[i])]
#                self.new_time[i] = self.new_time[i][np.abs(self.new_flux[i]) < clip * np.std(self.new_flux[i])]
#                self.new_flux[i] = self.new_flux[i][np.abs(self.new_flux[i]) < clip * np.std(self.new_flux[i])]

            if self.ndays != -1:
                # Remove linear trend from chunks
                # In case only one section remains
                #if len(self.new_flux) > 100:
                #    self.new_flux = [self.new_flux]
                """
                Add in new part where we properly detrend the data according to length of time observed!
                """
                if len(self.new_flux) == 1:
                    self.new_flux = self.new_flux[0]
                    self.new_time = self.new_time[0]

                    lc = lk.LightCurve(time=self.new_time, flux=self.new_flux)
                    if self.ndays == 180:
                        lc = lc.flatten(window_length=(48*10)+1)
                    elif self.ndays == 80:
                        lc = lc.flatten(window_length=(48*5)+1)
                    elif self.ndays == 27:
                        # Currently hardcoded according to Kepler long-cadence!!!
                        lc = lc.flatten(window_length=(48*2)+1)
#                    else:
#                        lc = lc.flatten(window_length=(48*20)+1)
                    self.new_flux = lc.flux
                    self.new_time = lc.time
                    trend = np.poly1d(np.polyfit(self.new_time[self.new_flux != 0], self.new_flux[self.new_flux != 0], 1))
                    self.new_flux[self.new_flux != 0] -= trend(self.new_time[self.new_flux != 0])
                else:
                    for i in range(len(self.new_flux)):
                        # Remove linear trend from data
                        #trend = self.compute_trend(self.new_time[i][self.new_flux[i] != 0], self.new_flux[i][self.new_flux[i] != 0])
                        #self.new_flux[i][self.new_flux[i] != 0] -= trend
                        lc = lk.LightCurve(time=self.new_time[i], flux=(self.new_flux[i]/1e6)+1)
                        if self.ndays == 180:
                            lc = lc.flatten(window_length=(48*10)+1)
                        elif self.ndays == 80:
                            lc = lc.flatten(window_length=(48*5)+1)
                        elif self.ndays == 27:
                            # Currently hardcoded according to Kepler long-cadence!!!
                            lc = lc.flatten(window_length=(48*3)+1)
                        #plt.plot(self.new_time[i], self.new_flux[i])
                        self.new_flux[i] = (lc.flux-1)*1e6
                        self.new_time[i] = lc.time
                        #plt.plot(lc.time, self.new_flux[i])
                        #plt.show()
                        trend = np.poly1d(np.polyfit(self.new_time[i][self.new_flux[i] != 0], self.new_flux[i][self.new_flux[i] != 0], 1))
                        self.new_flux[i][self.new_flux[i] != 0] -= trend(self.new_time[i][self.new_flux[i] != 0])

        else:
            if self.ndays == 27:
                # Remove linear trend from data
                trend = self.compute_trend(self.new_time[self.new_flux != 0], self.new_flux[self.new_flux != 0])
                #trend = np.poly1d(np.polyfit(self.new_time[i][self.new_flux[i] != 0], self.new_flux[i][self.new_flux[i] != 0], 1))
                self.new_flux[self.new_flux != 0] -= trend
            else:
                pass

    def compute_trend(self, x, y):
        """
        Compute linear trend for given data
        """
        A = np.vstack((np.ones_like(x), x)).T
        C = np.diag(np.ones_like(x))
        cov = np.linalg.inv(np.dot(A.T, np.linalg.solve(C, A)))
        b, m = np.dot(cov, np.dot(A.T, np.linalg.solve(C, y)))
        return m*x + b
