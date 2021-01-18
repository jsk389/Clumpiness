#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tarfile
import warnings
import yaml

from astropy.io import fits
from astropy.table import Table


from .Dataset_regular import Dataset
from .features import Features
from .config import readConfigFile


class FeatureExtractor:
    """
    Feature extractor class
    """
    def __init__(self, prior_df, pipeline=None, NDAYS=-1):
        """
        Parameters
        ----------
        prior_df: pd.Dataframe

        pipeline: Optional[str]

        NDAYS: int
            Number of days to chunk the data into. Defaults to -1 which means that the
            data is not separated up into chunks. (For example when training with full 4 years
            of Kepler data but want to train on Kepler-as-K2 then this can be set to 80 to create
            80 day chunks and compute features for those).

        Returns
        -------
        None
        """

        self.prior_df = prior_df
        self.NDAYS = NDAYS
        if pipeline:
            self.pipeline = pipeline

    def _extract_additional_info(self):
        """
        Doc string!
        """
        pass

    def extract(self, kicid, data_dir):
        """
        Extract the features from the input data. The features are hard-coded to ensure
        reproducibility and no errors further down the line.

        Parameters
        ----------
        kicid: int
            Star identifier, doesn't have to be KIC (just an example).

        data_dir: str
            Data directory

        Returns
        -------
        A whole bunch of information!

        """
        # This is set up because K2 data was in .tar format so specific to that.
        # Ideally this should be completely separate so the extraction of the data
        # and the extraction of the features can be completely decoupled.

        if (self.pipeline['star_id'] == 'EPIC') and (self.pipeline['campaign'] != 'APOK2'):

            tarname = data_dir+str(kicid)[:4].ljust(9, '0')

            tar = tarfile.open(tarname+'.tar')
            intar = tar.getnames()

            fname = './'+str(kicid)[:4].ljust(9, '0')+'/'+str(kicid)[4:]+'/'+'hlsp_everest_k2_llc_'+str(kicid)+'-c'+str(self.pipeline['campaign']).zfill(2)+'_kepler_v2.0_lc.fits'
            data = fits.open(tar.extractfile(fname))

            star = Dataset(kicid, fname)

            # Full datasets
            star.read_timeseries(ndays=self.NDAYS, tar=tar)

            # Put into regular sampling
            star.processing()


        else:
            # If it doesn't find the data then just return all nans
            try:
                data_file = glob.glob(data_dir+'*'+str(kicid)+'*')
                star = Dataset(kicid, data_file[0])
            except:
                return kicid, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

            # Full datasets
            star.read_timeseries(ndays=self.NDAYS)

            # Put into regular sampling
            star.processing()

        
        prior_info = self.prior_df.loc[self.prior_df[self.pipeline['star_id']] == kicid,]

        # Extract additional information
        # Distances and uncertainties etc. from Gaia and Bailer-Jones paper
        distance = np.asscalar(prior_info['r_est'].values)
        distance_error = np.asscalar(np.nanmean(np.c_[prior_info['r_est'] - prior_info['r_lo'], 
                                    prior_info['r_hi'] - prior_info['r_est']], axis=1))
        chi2 = np.asscalar(prior_info['astrometric_chi2_al'].values)
        nu = np.asscalar(prior_info['astrometric_n_good_obs_al'].values) - 5
        excess = np.asscalar(prior_info['astrometric_excess_noise'].values)
        kmag = np.asscalar(prior_info['kmag'].values)
        colour = np.asscalar(prior_info['bp_rp'].values)

        #
        evol_state = np.asscalar(prior_info['evol'].values)
        numax = np.asscalar(prior_info['numax'].values)
        gmag = np.asscalar(prior_info['phot_g_mean_mag'].values)
        metric = np.asscalar(np.sqrt(chi2 / nu) < 1.2 * np.maximum(1, np.exp(-0.2 * (gmag - 19.5))))
        ra = np.asscalar(prior_info['ra'].values)
        dec = np.asscalar(prior_info['dec'].values)
        kics = np.asscalar(prior_info[self.pipeline['star_id']].values.astype(int))

 
        #=======================Compute Features======================================
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
            return kicid, numax, evol_state, var, zs , hoc, mc, abs_mag, mu0, np.ones(feat.chunks)*kmag, Ak, np.ones(feat.chunks)*metric, distance, fill, ndata, running_var, np.array([colour]*feat.chunks)
        # Else if not splitting data up then return values
        return kicid, numax, evol_state, var, zs , hoc, mc, abs_mag, mu0, kmag, Ak, metric, distance, fill, len(star.new_flux), running_var, colour

class Preprocessing:
    """
    Metadata pre-processing class
    """
    def __init__(self, _pipeline, _df):

        self.pipeline = _pipeline
        self.df = _df

    def preprocess(self, gaia='../Tables/kepler_dr2_4arcsec.csv',
                         evo_table='../Tables/APOKASC_cat_v3.1.7.txt', 
                         dwarfs='../Tables/Dwarf_kics.txt',
                         kois='../Tables/koi_kics_cut.csv'):
        """
        Pre-process the metadata. This consists of crossmatching against available Gaia data. For the case of the Kepler test set
        this also consists of cross-matching with APOKASC and cutting stars we don't want from the sample, for example cutting known
        red giants from the Chaplin KOI sample (since we only want main-sequence stars or subgiants in that sample).

        Parameters
        ----------
        gaia: Optional[str]
            Path and filename containing Gaia cross-match data.

        evo_table: Optional[str]
            Path and filename containing evolutionary states for training.

        dwarfs: Optional[str]
            Path and filename containing dwarf kics for KOI sample.

        kois: Optional[str]
            Path and filename containing stars to drop from KOI sample.

        Returns
        -------
        Something!

        """
        # Specific preprocessing for APOKASC stars for training
        if (self.pipeline['star_id'] == 'KIC') and (self.pipeline['data_type'] == 'APOKASC'):

            # Cross-match with Gaia
            gaia = pd.read_csv(gaia)
            gaia[self.pipeline['star_id']] = gaia[self.pipeline['star_id']].astype(int)
            
            # Only take brightest star if duplicates
            gaia = gaia.sort_values('phot_g_mean_mag', ascending=True).drop_duplicates(self.pipeline['star_id']).sort_index()
            #print("GAIA: ", gaia.duplicated(subset='kepid').sum()/len(gaia))
            
            # Read in evolutionary states from APOKASC
            evols = pd.read_csv(evo_table, delimiter=r'\s+')
            
            # Just keep KIC, evolutionary states and numax (from Benoit) for reference
            evols = evols[['KEPLER_ID', 'CONS_EVSTATES', 'COR_NU_MAX']]
            
            # Rename columns
            evols = evols.rename(columns = {'KEPLER_ID': 'KIC', 'CONS_EVSTATES': 'evol_overall'})
            evols['KIC'] = evols['KIC'].astype(int)
            #print("EVOLS: ", evols.duplicated(subset='KIC').sum()/len(evols))
            
            # Merge with stars datasets
            df_final = self.df.merge(evols, how='inner', on=self.pipeline['star_id'])
            #print("df_final: ", df_final.duplicated(subset='KIC').sum()/len(df_final))
            
            # Extract kic numbers
            kics = list(df_final[self.pipeline['star_id']])
            # Map evolutionary states to integers
            df_final['evol_overall'] = df_final['evol_overall'].astype(str)
            df_final['evol_overall'] = df_final['evol_overall'].map({'RGB': 0,
                                                                          'RGB/AGB': 0,
                                                                          'RC': 1,
                                                                          '2CL': 1,
                                                                          'RC/2CL': 1,
                                                                          '-9999': 2}).astype(int)            
            
            # If no numax detected then set to unknown evolutionary state
            df_final['evol_overall'][df_final['COR_NU_MAX'] < 0] = 2
            
            # Prepare new class labels
            # Unknown stars
            df_final['evol_overall'][df_final['evol_overall'] == 2] = 6
            # HeCB
            df_final['evol_overall'][df_final['evol_overall'] == 1] = 3
            # High luminosity RGB
            df_final['evol_overall'][(df_final['evol_overall'] == 0) &
                                     (df_final['COR_NU_MAX'] < 15.0)] = 1
            # Confusion region RGB
            df_final['evol_overall'][(df_final['evol_overall'] == 0) &
                                     (df_final['COR_NU_MAX'] > 15.0) &
                                     (df_final['COR_NU_MAX'] < 130.0)] = 2
            
            # Extract evolutionary state
            evol_state = df_final['evol_overall'].values
            #print("RG length: ", len(evol_state))
            
            # Add in dwarf sample
            dwarf_kics = np.loadtxt(dwarfs)
            dwarf_kics = dwarf_kics.astype(int)
            #print("Dwarf length: ", len(dwarf_kics))
            
            # Also add in KOIs
            kois = pd.read_csv(kois, comment='#')
            
            # Drop duplicates
            kois = kois.sort_values(self.pipeline['star_id'], ascending=True).drop_duplicates(self.pipeline['star_id']).sort_index()
            #print("KOIS: ", kois.duplicated(subset='kepid').sum()/len(kois))

            koi_kics = kois[self.pipeline['star_id']].values.astype(int)

            #TODO: Need to find a better way to do this - random sampling?
            koi_kics = koi_kics[:1500]
            dwarf_kics = np.append(dwarf_kics, koi_kics)
            #print("Total dwarf length: ", len(dwarf_kics))
            #u, c = np.unique(dwarf_kics, return_counts=True)
            #print("DWARF KICS: ", len(u), len(dwarf_kics), len(u[c > 1])/len(dwarf_kics))
            
            # Set evolutionary state to 4 for noise class
            dwarf_evol = np.ones_like(dwarf_kics).astype(int) * 4

            # Red-giant numax values
            numax = df_final['COR_NU_MAX']
            
            # Add dwarf kics
            kics = np.append(kics, dwarf_kics)
 
            # Add dwarf evolutionary states
            evol_state = np.append(evol_state, dwarf_evol)
           
            # Add dwarf numax values
            numax = np.append(numax, [np.nan]*len(dwarf_evol))

            # Create dataframe
            kics = pd.DataFrame(data=np.c_[kics, evol_state, numax], columns=[self.pipeline['star_id'], 'evol', 'numax'])
            
            # Merge with Gaia
            kics[self.pipeline['star_id']] = kics[self.pipeline['star_id']].astype(int)
            kics = kics.drop_duplicates(self.pipeline['star_id'])
            #print(kics.duplicated(subset='kepid').sum()/len(kics))
            kics = pd.merge(kics, gaia, on=[self.pipeline['star_id']], how='inner')
            #print(kics.duplicated(subset='kepid').sum()/len(kics))
            #sys.exit()
        else:
            kics = self.df[str(self.pipeline['star_id'])].values

            # Cross-match with Gaiai
            gaia = pd.read_csv(gaia)
            if str(self.pipeline['star_id']) == 'EPIC':
                gaia.rename(columns={'epic_number': 'EPIC'}, inplace=True)
            gaia[str(self.pipeline['star_id'])] = gaia[str(self.pipeline['star_id'])].astype(int)
            
            # Only take brightest star if duplicates
            gaia = gaia.sort_values('phot_g_mean_mag', ascending=True).drop_duplicates(str(self.pipeline['star_id'])).sort_index()
            evol_state = np.zeros(len(kics))
            numax = np.zeros(len(kics))

            # Create dataframe
            kics = pd.DataFrame(data=np.c_[kics, evol_state, numax], columns=[str(self.pipeline['star_id']), 'evol', 'numax'])
            kics = pd.merge(kics, gaia, on=[str(self.pipeline['star_id'])], how='inner')
            kics[str(self.pipeline['star_id'])] = kics[str(self.pipeline['star_id'])].astype(int)

        return kics

