#!/usr/bin/env/ python3

import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate as interp

from astropy.table import Table
from astropy.io import fits

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

 

    def read_timeseries(self, noise=0.0, dt=None, ndays=-1, chunk=False, tar=None):
        """
        Read in the timeseries from the given file path. This function can take
        data that is in either ascii or .fits format

        Parameters
        ------------------
        noise: float
            Add noise (in ppm) to the time series according to the value given. By default
            this is set to zero.
        cadence: float
            Cadence of the observations (in seconds).
        n_days: int
            The duration of the time series in days. By default set to -1 which
            results in the entire time series being taken.
        n_sections: int
            The number of time series to return. By default set to 1, so only returns
            1 time series of length n_days. If set to -1 then will return as many
            as possible for length of data
        """
        self.ndays = ndays
        # 28/12/2020 Add cadence
        if dt is None:
            sys.exit('Cadence is not defined, please define it!')
        else:
            self.dt = dt

        # 11/01/2021 Hacky solution for some TESS CVZ targets
        if (self.fname.endswith("inp.fits")) and ("TIC" in self.fname):
            data = fits.open(self.fname)[0].data
            self.time = data[:,0]
            self.flux = data[:,1]
            # Remove zeros (as gaps zero-padded)
            self.time = self.time[self.flux != 0.0]
            self.flux = self.flux[self.flux != 0.0]
            #

        # Assess which file format the data is in
        elif (self.fname.endswith(".fits")) and ("hlsp_everest_k2" in self.fname):
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

            
            self.flux = (lc.flux - 1) * 1e6
            self.time = lc.time            


        elif self.fname.endswith(".fits"):

            dat = Table.read(self.fname, format='fits')

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

            for i in range(len(idx)):
                self.time[idx[i]+1:] -= (self.time[idx[i]+1] - self.time[idx[i]])
            self.flux = self.flux[np.isfinite(self.time)]
            self.time = self.time[np.isfinite(self.time)]
            self.time = self.time[np.isfinite(self.flux)]
            self.flux = self.flux[np.isfinite(self.flux)]

        if self.ndays != -1:
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
            if len(self.new_time[-1]) < (0.1 * self.ndays * 86400.0) / self.dt: #(29.4 * 60.0):
                self.new_time = self.new_time[:-1]
                self.new_flux = self.new_flux[:-1]
            # Check to see if arrays of all zeros and remove them!
            idx = []
            for i in range(len(self.new_flux)):
                if (not self.new_flux[i].any()) or (len(self.new_flux[i][self.new_flux[i] != 0])/len(self.new_flux[i]) < 0.1):
                    idx.append(int(i))
           
            if len(idx) > 0:
                for i in sorted(idx, reverse=True):
                    del self.new_time[i]
                    del self.new_flux[i]

        if noise != 0.0:
            self.flux[self.flux != 0] += np.random.normal(0, noise)

    def fill_gaps(self, time, flux):
        """
        Fill gaps 
        """
        mask = np.invert(np.isnan(flux))
        tm = time[mask]
        dt = np.diff(tm)
        good = np.where(dt < (self.dt * 2.5))#(29.4*60.0 * 2.5))
        new_flux = interp.interp1d(tm, flux[mask], bounds_error=False)(time)
        return time[good], new_flux[good]



    def processing(self, regular=False):
        """
        Does time series preparation and chunking. If regular is set to True then will
        also interpolate onto regular frid in time, but defaults to False.
        """
        # Cadence in seconds!
        # Interpolation function
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

            # New flux array
            self.new_flux = f(self.new_time)
            # Zero centre first!
            self.new_flux[np.isfinite(self.new_flux)] -= np.mean(self.new_flux[np.isfinite(self.new_flux)])
            self.new_flux[~np.isfinite(self.new_flux)] = 0
        else:
            self.new_flux = self.flux.copy()
            self.new_time = self.time.copy()

            self.new_flux[np.isfinite(self.flux)] -= np.mean(self.new_flux[np.isfinite(self.flux)])
            self.new_flux[~np.isfinite(self.flux)] = 0

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
            n_points = int(self.ndays/ (self.dt / 86400.0)) #(29.4 / (60.0 * 24)))

            self.new_time = []
            self.new_flux = []
            dtau = (self.ndays * 86400.0)

            for i in range(self.n_sections):
                self.new_time.append(self.time[(self.time >= (i*dtau)) * (self.time <= i*dtau + dtau)])
                self.new_flux.append(self.flux[(self.time >= (i*dtau)) * (self.time <= i*dtau + dtau)])

            self.new_time = np.array(self.new_time)
            self.new_flux = np.array(self.new_flux)

            # If last section is too small then disregard
            # Take threshold as 3/4 * ideal length, that way it is close enough
            # to the ideal length
            if len(self.new_time[-1]) < (0.1 * self.ndays * 86400.0) / self.dt: #(29.4 * 60.0):
                self.new_time = self.new_time[:-1]
                self.new_flux = self.new_flux[:-1]
            # Check to see if arrays of all zeros and remove them!
            idx = []
            for i in range(len(self.new_flux)):
                if (not self.new_flux[i].any()) or (len(self.new_flux[i][self.new_flux[i] != 0])/len(self.new_flux[i]) < 0.1):
                    idx.append(int(i))

            if len(idx) > 0:

                for i in sorted(idx, reverse=True):
                    del self.new_time[i]
                    del self.new_flux[i]


            if self.ndays != -1:
                # Remove linear trend from chunks
                # In case only one section remains
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

                        self.new_flux[i] = (lc.flux-1)*1e6
                        self.new_time[i] = lc.time

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
