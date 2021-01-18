#!/usr/bin/env/ python3

import matplotlib.pyplot as plt
import numpy as np

import scipy.ndimage as nd
import scipy.signal
import scipy.stats
from gatspy.periodic import LombScargleFast
from astropy.stats import mad_std
from astropy.stats import LombScargle
import astropy.units as units
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic
from .Template import Template

#from dustmaps.bayestar import BayestarQuery
#from dustmaps.bayestar import BayestarWebQuery

import astropy.units as u
from astropy.coordinates import SkyCoord
import mwdust

import warnings
warnings.filterwarnings("ignore")

from astropy.units import cds
cds.enable()

#bayes = mwdust.Green15(filter='2MASS Ks')
bayes = mwdust.Combined19(filter="2MASS Ks")

class Features(Template):
    """
    A class to compute features from a given timeseries

    :param ds: The dataset of interest
    :type ds: class

    """
    def __init__(self, ds):
        super().__init__(ds)
        # Create mask where data is finite
        self.mask = np.isfinite(self.ds.flux)
        # Check to see if there is more than one section
        if self.ds.n_sections > 1:
            if len(self.ds.new_flux) > 1000:
                self.chunks = 1
            else:
                self.chunks = len(self.ds.new_flux)
        else:
            self.chunks = self.ds.n_sections
        # Squeeze just in case extra axis is preserved
        # TODO: Why does this happen?
        if (self.chunks == 1) and (np.shape(self.ds.new_flux)[0] == 1):
            self.ds.new_flux = np.squeeze(self.ds.new_flux)

        # Coefficients for HOC and zc normalisation
        self.sigmoid_coeffs = np.array([3.625418060200669,
                                        5.250836120401338,
                                        7.117056856187291,
                                        8.862876254180602,
                                        10.608695652173914,
                                        12.234113712374581])
        self.coeffs = np.array([ 0.49912 ,  0.666367,  0.732079,  0.769869,  0.795083,  0.81284 ,
                                 0.827098,  0.838291,  0.847576,  0.855565,  0.862341,  0.868062,
                                 0.873031,  0.877556,  0.881678])

    def create_chunks(self):
        """
        Create chunks of data (according to time stamps)
        """
        raise NotImplementedError

    def calculate_fill(self, x, y):
        """
        Compute the fill of the current dataset

        :param y: array of which to calculate the fill.
        :type y: array
        """
        # 11/01/2021 Make sure to round up to avoid odd case where fill can be greater than 1
        full_npts = np.ceil((x[-1] - x[0]) / self.ds.dt) #(29.4*60.0)
        return len(y[np.isfinite(y)]) / full_npts


    def compute_fill(self, return_full_length=False):
        """
        Compute the fill of the current dataset

        :param return_full_length: Not implemented!
        :type return_full_length: bool
        """
        # Compute fill and check if multiple chunks of data to loop over
        if self.chunks == 1:
            full_npts = np.ceil((self.ds.new_time[-1] - self.ds.new_time[0]) / self.ds.dt) #(29.4 * 60.0)
            return len(self.ds.new_time[np.isfinite(self.ds.new_flux)]) / full_npts
            #return len(self.ds.new_flux[self.ds.new_flux != 0]) / len(self.ds.new_flux)
        else:
            fill = np.zeros(self.chunks)
            for i in range(self.chunks):
                full_npts = np.ceil((self.ds.new_time[i][-1] - self.ds.new_time[i][0]) / self.ds.dt) #(29.4*60.0)
                fill[i] = len(self.ds.new_time[i][np.isfinite(self.ds.new_flux[i])]) / full_npts
                #fill[i] = len(self.ds.new_flux[i][self.ds.new_flux[i] != 0]) / len(self.ds.new_flux[i])
            return fill

    def compute_zero_crossings(self):
        """
        Compute the number of zero crossings in the dataset following
        Bae et al. (1996)

        :math:`D_{1} = \sum^{N}_{i=2}(X_{i}-X_{i-1})^{2}\;\textrm{where } 0 \leq D_{1} \leq N-1`

        """
        # Normalise by number of NON-ZERO points, try to alleviate issue of fill!
        # Can always test to make sure this works properly
        if self.chunks == 1:
            fill = self.compute_fill()
            return self.compute_k_crossings(self.ds.new_flux, 0) / (len(self.ds.new_flux)) / self.correction_factor(fill, self.sigmoid_coeffs[0], 0)# / (len(self.ds.new_flux))
        else:
            zcs = np.zeros(self.chunks)
            fill = self.compute_fill()
            for i in range(self.chunks):
                zcs[i] = self.compute_k_crossings(self.ds.new_flux[i], 0) / (len(self.ds.new_flux[i])) / self.correction_factor(fill[i], self.sigmoid_coeffs[0], 0)
            return zcs

    def compute_sig_peak(self):
        """
        Compute periodogram and return power of largest peak in range 2 days - 100 days
        """
        if self.chunks == 1:
            ls = LombScargle(self.ds.new_time * cds.s, self.ds.new_flux*cds.ppm, normalization='standard')
            freq, power = ls.autopower(method='fast', nyquist_factor=1,
                                       minimum_frequency=1.0/(100*86400.0*cds.s),
                                       maximum_frequency=1.0 / (2.0 * 86400.0*cds.s), samples_per_peak=10)
            return power.max()
        else:
            fap = np.zeros(self.chunks)
            for i in range(self.chunks):
                ls = LombScargle(self.ds.new_time[i] * cds.s, self.ds.new_flux[i]*cds.ppm, normalization='standard')
                freq, power = ls.autopower(method='fast', nyquist_factor=1,
                                           minimum_frequency=1.0/(100*86400.0*cds.s),
                                           maximum_frequency=1.0 / (2.0 * 86400.0*cds.s), samples_per_peak=10)
                fap[i] = power.max()
            return fap


    def sigmoid_(self, x, t):
        return (1 / (1 + np.exp(-t*x))) - 0.5

    def sigmoid(self, x, t):
        return self.sigmoid_(x, t) / self.sigmoid_(1, t)

    def sig_single(self, x, t):
        return self.sigmoid_(x, t) / self.sigmoid_(1,t)

    def correction_factor(self, x, t, k):
        sig = self.sig_single(x, t)
        return sig# * self.coeffs[k]

    def compute_higher_order_crossings(self, k):

        if self.chunks == 1:
            psi, zc = self.compute_hocs(self.ds.new_time, self.ds.new_flux, k)
            return psi, zc
        else:
            psi = np.zeros(self.chunks)
            zc = np.zeros([self.chunks, k])
            for i in range(self.chunks):
                psi[i], zc[i,:] = self.compute_hocs(self.ds.new_time[i], self.ds.new_flux[i], k)
            return psi, zc

    def compute_hocs(self, x, y, k):
        """
        Compute higher order crossings (HOC)

        Parameters
        -----------
        k (int) : number of k crossings to compute (inclusive)
        """
        zc_gauss = np.array([ 0.49912 ,  0.666367,  0.732079,  0.769869,  0.795083,  0.81284 ,
        0.827098,  0.838291,  0.847576,  0.855565,  0.862341,  0.868062,
        0.873031,  0.877556,  0.881678])
        zc = np.zeros(k)
        fill = self.calculate_fill(x, y)#compute_fill()#x)
        #print("FILL: ", fill)
        for i in range(k):
            zc[i] = self.compute_k_crossings(y, i) / (len(y)-i)/self.correction_factor(fill, self.sigmoid_coeffs[i], i)
        delta_k = zc[0]
        delta_k_gauss = zc_gauss[0]
        for i in range(k-1):
            delta_k = np.append(delta_k, zc[i+1] - zc[i])
            delta_k_gauss = np.append(delta_k_gauss, zc_gauss[i+1] - zc_gauss[i])
        #psi = np.sum((delta_k - delta_k_gauss[:len(delta_k)])**2 / zc_gauss[:len(delta_k)])
        psi = np.sum((delta_k - delta_k_gauss[:len(delta_k)])**2 / delta_k_gauss[:len(delta_k)])
        return psi, zc

    def compute_k_crossings(self, y, k):
        y = np.diff(y, k)
        window = y.copy()
        window[window >= 0] = 1
        window[window < 0] = 0
        return np.sum(np.diff(window)**2)

    def compute_var(self, robust=True):
        """
        Compute variance of time series

        Args:
            robust (bool): If "True" the computes the variance from the
                           median absolute deviation from the median (MAD).
                           Otherwise computes in the standard manner.
        """
        if self.chunks == 1:
            if robust == True:
                mad = np.median(np.abs(self.ds.new_flux[self.ds.new_flux != 0] -
                                       np.median(self.ds.new_flux[self.ds.new_flux != 0])))
                #var = (1.4826 * mad)**2
                return mad #var
            return np.var(self.ds.flux[np.isfinite(self.ds.flux)])
        else:
            var = np.zeros(self.chunks)
            for i in range(self.chunks):
                mad = np.median(np.abs(
                                self.ds.new_flux[i][self.ds.new_flux[i] != 0] -
                      np.median(self.ds.new_flux[i][self.ds.new_flux[i] != 0])))
                var[i] = mad #(1.4826 * mad)**2
            return var

    def compute_mad(self, x):
        return np.median(np.abs(x - np.median(x)))

    def compute_variance_change(self):
        """
        Compute MAD of the first derivative
        """
        if self.chunks == 1:
            return self.compute_mad(np.diff(self.ds.new_flux))
#            return (1.4826*self.compute_mad(np.diff(self.ds.new_flux)))**2
        else:
            mc = np.zeros(self.chunks)
            for i in range(self.chunks):
                mc[i] = self.compute_mad(np.diff(self.ds.new_flux[i]))
	        # (1.4826*self.compute_mad(np.diff(self.ds.new_flux[i])))**2
            return mc

    def compute_distance(self, parallax):
        """
        Parallax (mas) to distance(pc)
        """
        return 1000/parallax

    def compute_Ak(self, ra, dec, distance, distance_error):
        """
        Compute extinction
        """
        if self.chunks == 1:
            c_icrs = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, distance=distance*u.pc, frame='icrs')
            l = c_icrs.galactic.l.value
            b = c_icrs.galactic.b.value

            #print(l, b)

            #ebv = bayes(l, b, distance)
            # Distance needs to be in kpc NOT pc!
            Ak = bayes(l, b, distance/1000)

                #ebv = bayestar(c_icrs.galactic, mode='median')
                # Table 1  Green, Schlafly, Finkbeiner et al. (2018)
                #Av = 0.161*ebv
            #Ak = 0.355*ebv
            #except:
            #    Ak = [0]
            return np.asscalar(Ak), distance
        else:
            Ak = np.zeros(self.chunks)
            new_distance = np.zeros(self.chunks)
            for i in range(self.chunks):
                distance_mod = np.random.normal(distance, distance_error)
                #distance_mod = distance
                if distance_mod < 0:
                    distance_mod = np.abs(distance_mod)
                c_icrs = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, distance=distance_mod*u.pc, frame='icrs')
                l = c_icrs.galactic.l.value
                b = c_icrs.galactic.b.value

                #ebv = bayes(l, b, distance)
                # Distance in kpc!
                Ak[i] = bayes(l, b, distance)
                #ebv = bayestar(c_icrs.galactic, mode='median')
                # Table 1  Green, Schlafly, Finkbeiner et al. (2018)
                #Av = 0.161*ebv
                #                Ak[i] = 0.355*ebv[0]
                new_distance[i] = distance_mod
            return Ak, new_distance

    def compute_dist_mod(self, distance):
        """
        Compute distance modulus
        """
        return 5*np.log10(distance)-5

    def compute_abs_K_mag(self, kmag, distance, distance_error, ra, dec, excess):
        """
        Compute absolute K-band magnitude
        """
#        print("Distance going in: ", distance)
        Ak, distance = self.compute_Ak(ra, dec, distance, distance_error)
        mu_0 = self.compute_dist_mod(distance)
#        print("Distance returned: ", distance)
        #print(excess, self.chunks)
        if excess == False and self.chunks > 1:
            return kmag - mu_0 - Ak, mu_0, Ak, np.zeros(self.chunks), distance
        elif excess == False and self.chunks == 1:
            return kmag - mu_0 - Ak, mu_0, Ak, 0, distance
        if self.chunks > 1:
            return kmag - mu_0 - Ak, mu_0, Ak, np.ones(self.chunks), distance
        else:
            return kmag - mu_0 - Ak, mu_0, Ak, 1, distance


    def rolling_window(self, a, window):
        """
        http://www.rigtorp.se/2011/01/01/rolling-statistics-numpy.html
        """
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def variance_for_binning(self, a):
        return np.nanstd(a)

    def compute_running_variance(self):
        """
        Compute running variance of dataset
        """
        if self.chunks == 1:
            # Odd behaviour if N_DAYS < -1 but only have one chunk of data, don't want it actually running through this
            #if not hasattr(self.ds.new_flux, 'len'):
        #        return np.nan
            # 90 day running variance
            cadence = np.nanmedian(np.diff(self.ds.new_time))
            window_length = int(90*86400. / cadence)
            n_bins = len(self.ds.new_flux) // window_length
            if n_bins < 1:
                n_bins = 1

            binned_val, _, _ = binned_statistic(self.ds.new_time,
                                       self.ds.new_flux,
                                       self.variance_for_binning,
                                       bins=n_bins)
            #self.rolling_window(self.ds.new_flux, window_length)
            #
            #if len(binned_val) == 1:
        #        return
        #    diffs = np.diff(binned_val)
        #    if
            if len(binned_val) == 1:
                return 0
            else:
                return np.nanmean(np.diff(binned_val))
        else:
            running_var = np.zeros(self.chunks)
            for i in range(self.chunks):
                # 90 day running variance
                cadence = np.nanmedian(np.diff(self.ds.new_time[i]))
                window_length = int(90*86400. / cadence)
                n_bins = len(self.ds.new_flux[i]) // window_length
                if n_bins < 1:
                    n_bins = 1
                binned_val, _, _ = binned_statistic(self.ds.new_time[i],
                                           self.ds.new_flux[i],
                                           self.variance_for_binning,
                                           bins=n_bins)
                #self.rolling_window(self.ds.new_flux, window_length)
                #
                #if len(binned_val) == 1:
            #        return
            #    diffs = np.diff(binned_val)
            #    if
                if len(binned_val) != 1:
                    running_var[i] = np.nanmean(np.diff(binned_val))
            return running_var
