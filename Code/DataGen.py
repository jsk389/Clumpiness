#!/usr/bin/env/ python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from astropy.table import Table
import gatspy.periodic as gp

import scipy.interpolate as interp
import kplr
from operator import itemgetter
from itertools import groupby

from bisect import insort, bisect_left
from itertools import islice
from collections import deque
from tqdm import tqdm as tqdm

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


class DataGen(object):
    """
    This is the DataGen class that will handle the generation of the data from
    individual quarters - let's use pyke + kplr?
    """

    def __init__(self, _kic, _ndays, _KOI_flag=False):

        # KIC number of star
        self.kic = _kic
        # Path to files
        self.client = kplr.API()
        self.star = self.client.star(int(self.kic))
        self.ndays = _ndays


    def fetch_lightcurves(self):
        """
        Fetch the light curves of star with given KIC number using kplr
        """
        self.time = []
        self.flux = []
        self.ferr = []
        fetch = self.star.get_light_curves(short_cadence=False)
        print(fetch)
        for idx, lc in tqdm(enumerate(fetch), total=len(fetch)):
            print("DATA READ ", idx)
            data = lc.read()
            print("DONE!")
            x0 = data["TIME"]
            y0 = data["PDCSAP_FLUX"]
            m = (data["SAP_QUALITY"] == 0) & np.isfinite(x0) & np.isfinite(y0)
            mu = np.median(y0[m])
            y0 = (y0[m] / mu - 1.0) * 1e6
            #dt = np.diff(x0)
            self.flux = np.append(self.flux, y0)
            self.time = np.append(self.time, x0[m])
            self.ferr = np.append(self.ferr, 1e6 * data["PDCSAP_FLUX_ERR"][m] / mu)
            #self.time = np.concatenate(self.time)
            #self.flux = np.concatenate(self.flux)
            #self.ferr = np.concatenate(self.ferr)
            #print(len(x0), np.sum(np.isfinite(x0)))

        #plt.errorbar(self.time, self.flux, yerr=self.ferr, fmt='.')
        print("SIGMA CLIPPING")
        self.time, self.flux, self.ferr = self.sigma_clipping(self.time,
                                                              self.flux,
                                                              self.ferr,
                                                              n_rounds=3,
                                                              sigma=4.0)
        print("Moving median")
        mm = self.med_filt(self.time, self.flux, dt=30.)
        print("Done")
        idx = np.isfinite(self.flux)
        self.flux[idx] -= mm[idx]
        #plt.errorbar((self.time-self.time[0])*86400.0, self.flux, yerr=self.ferr, fmt='.', zorder=-1)
        #plt.plot(self.time, mm, zorder=10)
        #plt.show()


    def med_filt(self, x, y, dt=4.):
        """
        De-trend a light curve using a windowed median.
        """
        x, y = np.atleast_1d(x), np.atleast_1d(y)
        assert len(x) == len(y)
        r = np.empty(len(y))
        for i, t in enumerate(x):
            inds = (x >= t - 0.5 * dt) * (x <= t + 0.5 * dt)
            r[i] = np.median(y[inds])
        return r

    def std_from_mad(self, x):
        """
        Compute standard deviation from median absolute deviation from median (MAD)
        """
        return 1.4826 * (np.median(np.abs(x - np.median(x))))

    def sigma_clipping(self, t, f, ferr, n_rounds=1, sigma=4.0):

        #print(len(t), len(f), len(ferr))
        for i in range(n_rounds):
            idx = np.isfinite(f)
            #print((sigma * self.std_from_mad(f[idx] - 1)))
            #print(len(t), len(f), len(ferr))
            #print(len((abs(f - 1) < (sigma * self.std_from_mad(f[idx] - 1)))))
            t = t[(abs(f - 1) < (sigma * self.std_from_mad(f[idx] - 1)))]
            ferr = ferr[(abs(f - 1) < (sigma * self.std_from_mad(f[idx] - 1)))]
            f = f[(abs(f - 1) < (sigma * self.std_from_mad(f[idx] - 1)))]
        return t, f, ferr


    def read_timeseries(self, noise=0.0, ndays=-1, chunk=False, perturb=False):
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
        # Check the units of the time array through the cadence
        # Convert time array from days into seconds
        if self.time[1] - self.time[0] < 1.0:
            self.time -= self.time[0]
            self.time *= 86400.0
        else:
            self.time -= self.time[0]

        # Fill single point gaps everything three days
        self.time, self.flux, self.ferr = self.fill_gaps(self.time, self.flux, self.ferr)

        if self.ndays != -1:
            self.n_sections = int((np.nanmax(self.time) / 86400.0) / self.ndays)
            if self.n_sections < 1:
                self.n_sections = 1
        else:
            self.n_sections = 1
        print("SECTIONS: ", self.ndays)

        if chunk == True:
            self.new_time = np.array_split(self.time, self.n_sections)
            self.new_flux = np.array_split(self.flux, self.n_sections)
            self.new_ferr = np.array_split(self.ferr, self.n_sections)
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
                    del self.new_ferr[i]

            if perturb == True:
                # If perturb keyword set then perturb timeseries according to
                # uncertainties
                for i in range(self.new_time):
                    self.new_flux[i] += np.random.normal(0, self.new_ferr[i])

        # Perturb if not chunked
        if chunk == False and perturb == True:
            self.new_flux = self.flux.copy()# + np.random.normal(0, self.ferr)
        # If only want one section then proceed as normal
        #if ndays != -1:
        #if n_sections != 1:
            # Convert number of days into seconds
        #    ntime = ndays * n_sections * 86400.0
        #    self.flux = self.flux[self.time <= ntime]
        #    self.time = self.time[self.time <= ntime]
    #    if chunk == True:
    #        print(len(self.new_time), len(self.new_flux), len(self.new_ferr))
    #        plt.plot(self.new_time, self.new_flux, '.')
    #        plt.show()
        #else:
            #plt.plot(self.time, self.flux, '.')
            #plt.show()



        if noise != 0.0:
            self.flux[self.flux != 0] += np.random.normal(0, noise)

    def perturb_ts(self, flux, ferr):
        return flux + np.random.normal(0, ferr)

    def fill_gaps(self, time, flux, ferr):
        mask = np.invert(np.isnan(flux))
        tm = time[mask]
        dt = np.diff(tm)
        good = np.where(dt < (29.4*60.0 * 2.5))
        new_flux = interp.interp1d(tm, flux[mask], bounds_error=False)(time)

        #new_ferr[good] = np.mean(new_ferr)

        return time[good], new_flux[good], ferr[good]


    def to_regular_sampling(self, time=None, flux=None, ferr=None):
        """
        Put Kepler data onto regularly sampled grid
        """
        if not time is None:
            self.time = time
            self.flux = flux
            self.ferr = ferr
        # Cadence in seconds!
        dt = (29.4 * 60.0)# / 86400.0
        # Interpolation function
        #print("LENGTH BEFORE: ", len(self.time))
        mask = np.isfinite(self.time)
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
        # New flux array
        self.new_flux = f(self.new_time)
        # Zero centre first!
        self.new_flux[~np.isfinite(self.new_flux)] -= np.mean(self.new_flux[~np.isfinite(self.new_flux)])
        self.new_flux[~np.isfinite(self.new_flux)] = 0

        #plt.plot(self.time, self.flux, 'k')
        # Allow for slight irregular sampling and work out where gap begins
        times = np.where(np.diff(self.time[mask]) > 1800)
        for i in range(len(times[0])):
            start = self.time[mask][times[0][i]]
            finish = self.time[mask][times[0][i]]+np.diff(self.time[mask])[times[0][i]]
            self.new_flux[(self.new_time > start) & (self.new_time < finish)] = 0

        # If want it in chun1ks split it up now!
        # Need to think about this more carefully! As features won't end up
        # using these data!

        #print("SECTIONS: ", self.n_sections)
        if self.n_sections != 1:
            #print("SAFE")
            self.new_time = np.array_split(self.new_time, self.n_sections)
            self.new_flux = np.array_split(self.new_flux, self.n_sections)
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

            if self.ndays != -1:
                # Remove linear trend from chunks
                # In case only one section remains
                if len(self.new_flux) > 100:
                    self.new_flux = [self.new_flux]
                for i in range(len(self.new_flux)):
                    # Remove linear trend from data
                    #trend = self.compute_trend(self.new_time[i][self.new_flux[i] != 0], self.new_flux[i][self.new_flux[i] != 0])
                    #self.new_flux[i][self.new_flux[i] != 0] -= trend
                    trend = np.poly1d(np.polyfit(self.new_time[i][self.new_flux[i] != 0], self.new_flux[i][self.new_flux[i] != 0], 1))
                    self.new_flux[i][self.new_flux[i] != 0] -= trend(self.new_time[i][self.new_flux[i] != 0])
        else:
            if self.ndays == 27:
                # Remove linear trend from data
                trend = self.compute_trend(self.new_time[self.new_flux != 0], self.new_flux[self.new_flux != 0])
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

if __name__ == "__main__":

    kic = "3835996"
    NDAYS = 27
    d = DataGen(kic, NDAYS)
    d.fetch_lightcurves()
    N = 1000
    Nreals = 10
    perturb=False
    d.read_timeseries(ndays=NDAYS, perturb=perturb)
    plt.errorbar(d.time[:N], d.flux[:N], yerr=d.ferr[:N], fmt='None', ecolor='0')
    plt.plot(d.time[:N], d.flux[:N], color='0')
    d.to_regular_sampling(d.time, d.flux, d.ferr)
    plt.plot(d.new_time[0][:N], d.new_flux[0][:N], color='C1', marker='.')
    old_flux = d.flux
    old_ferr = d.ferr
    data = np.zeros([N, Nreals])
    time = d.time[:N]
    for i in tqdm(range(Nreals), total=Nreals):
        new_flux = d.perturb_ts(old_flux, old_ferr)
        d.to_regular_sampling(d.time, new_flux, d.ferr)
        data[:,i] = d.new_flux[0][:N]
        plt.plot(d.new_time[0][:N], d.new_flux[0][:N], '.')
        #plt.plot(d.time[:N], d.flux[:N], color='0')
    plt.violinplot(data.T, d.new_time[0][:N], points=100, widths=1e3, showmeans=False, showextrema=False, showmedians=True)
    plt.show()
        #d.to_regular_sampling()
