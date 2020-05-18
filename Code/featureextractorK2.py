#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import warnings
import yaml

from astropy.table import Table
from Dataset_regular import Dataset
from features import Features
from config import readConfigFile

from tqdm import tqdm
import sys
import time

class FeatureExtractor():
    """
    Feature extractor class
    """
    def __init__(self, prior_df, NDAYS=-1):
        """
        Doc string needed!
        """
        self.prior_df = prior_df
        self.NDAYS = NDAYS

    def _extract_additional_info(self):
        """
        Doc string!
        """
        pass

    def extract(self, kicid, starid, data_dir):
        """
        Doc string!
        """
        data_file = glob.glob(data_dir+'*'+str(kicid)+'*')
        print("SAFE")
        print("DATA: ", data_file)
        star = Dataset(kicid, data_file[0])
        # Extract relevant information for star in question
        prior_info = self.prior_df.loc[self.prior_df[starid] == kicid,]

        # Extract additional information
        distance = np.asscalar(prior_info['r_est'].values)
        distance_error = np.asscalar(np.nanmean(np.c_[prior_info['r_est'] - prior_info['r_lo'], 
                                    prior_info['r_hi'] - prior_info['r_est']], axis=1))
        chi2 = np.asscalar(prior_info['astrometric_chi2_al'].values)
        nu = np.asscalar(prior_info['astrometric_n_good_obs_al'].values) - 5
        excess = np.asscalar(prior_info['astrometric_excess_noise'].values)
        kmag = np.asscalar(prior_info['kmag'].values)
        colour = np.asscalar(prior_info['bp_rp'].values)

        evol_state = np.asscalar(prior_info['evol'].values)
        numax = np.asscalar(prior_info['numax'].values)
        gmag = np.asscalar(prior_info['phot_g_mean_mag'].values)
        metric = np.asscalar(np.sqrt(chi2 / nu) < 1.2 * np.maximum(1, np.exp(-0.2 * (gmag - 19.5))))
        ra = np.asscalar(prior_info['ra'].values)
        dec = np.asscalar(prior_info['dec'].values)
        kics = np.asscalar(prior_info['kepid'].values.astype(int))

        # Full datasets
        star.read_timeseries(ndays=self.NDAYS)

        # Put into regular sampling
        star.to_regular_sampling()

        # Compute features
        feat = Features(star)

        # Compute running variance
        if feat.chunks == 1:
            running_var = feat.compute_running_variance()
        else:
            running_var = np.array([np.nan]*feat.chunks)

        # Compute features
        zs = feat.compute_zero_crossings()
        var = feat.compute_var()
        hoc, _ = feat.compute_higher_order_crossings(5)
        mc = feat.compute_variance_change()
        fill = feat.compute_fill()
        abs_mag, mu0, Ak, good_flag, distance = feat.compute_abs_K_mag(kmag, distance, distance_error, ra, dec, excess)
        # If multiple chunks then return arrays
        if feat.chunks > 1:
            ndata = np.array([len(star.new_flux[i]) for i in range(len(star.new_flux))])
            return kicid, numax, evol_state, var, zs , hoc, mc, abs_mag, mu0, np.ones(feat.chunks)*kmag, Ak, np.ones(feat.chunks)*good_flag, distance, fill, ndata, running_var, np.array([colour]*feat.chunks)
        # Else if not splitting data up then return values
        return kicid, numax, evol_state, var, zs , hoc, mc, abs_mag, mu0, kmag, Ak, good_flag, distance, fill, len(star.new_flux), running_var, colour

class Preprocessing():
    """
    Doc string!
    """
    def __init__(self, _pipeline, _df):
        """
        Doc string
        """
        self.pipeline = _pipeline
        self.df = _df

    def preprocess(self, gaia='../Tables/k2_dr2_4arcsec.fits'):
        """
        Doc string
        """
 
	    kics = np.array(list(stars[str(self.pipeline['star_id'])]))
	    evol_state = np.zeros(len(kics))
	    numax = np.zeros(len(kics))
	    # Create dataframe
	    kics = pd.DataFrame(data=np.c_[kics, evol_state, numax], columns=['kepid', 'evol', 'numax'])

        return kics
