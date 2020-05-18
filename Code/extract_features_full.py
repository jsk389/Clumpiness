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
from joblib import Parallel, delayed
from tqdm import tqdm
import sys
import time


from featureextractor import FeatureExtractor, Preprocessing


import warnings
warnings.filterwarnings("ignore")

if __name__=="__main__":
 
    # Load in settings from configuration file
    print(str(sys.argv[1]))
    fname = str(sys.argv[1])
    settings = readConfigFile(fname)
    NDAYS = -1
    if settings['ignore_warnings']:
        warnings.filterwarnings('ignore')

    run = settings['run']
    pipeline = settings[run]
    work_dir = settings['work_dir']
    data_dir = pipeline['data_dir']
    output_dir = pipeline['output_dir']

    stars = pd.read_csv(output_dir + pipeline['csv_file']) 
    # Check for duplicate kic numbers
    stars = stars.drop_duplicates(str(pipeline['star_id']))
    # Preprocess
    prepro = Preprocessing(pipeline, stars)
    kics = prepro.preprocess()
    extraction = FeatureExtractor(kics, NDAYS=NDAYS, pipeline=pipeline).extract

    print(np.shape(kics))
    df = pd.DataFrame(columns=[str(pipeline['star_id']), 'numax', 'evo', 'var', 'zc', 'hoc', 'mc', 'abs_k_mag', 'mu0', 'mk', 'Ak', 'para_flag', 'distance', 'fill', 'ndata', 'running_var', 'colour'])
    print("DATA DIR: ", data_dir)
    print(kics.head())
    # Select duplicate rows except first occurrence based on all columns
    print(kics.duplicated(subset=str(pipeline['star_id']), keep=False).sum()/len(kics))
        
    kics = kics[str(pipeline['star_id'])].values.astype(int)

    #print(kics)
    #kics = kics[15938:]
    #kics = kics[15954:]
    #sys.exit()
    #print(kics['kepid'])
    with Parallel(n_jobs=4) as parallel:
    #aprun = ParallelExecutor(use_bar='tqdm', n_jobs=4)
        #results = parallel(delayed(main)(kic, idx, evol_state, data_dir) for idx, kic in enumerate(kics))
        results = parallel(delayed(extraction)(kic, data_dir) for kic in tqdm(kics, total=len(kics)))


    if NDAYS != -1:
        new_results = []
        for i in range(len(results)):
            #print(results[i][8])
            if hasattr(results[i][3], '__len__'):
                for j in range(len(results[i][3])):
                    new_results.append((*results[i][:3],
                                         results[i][3][j],
                                         results[i][4][j],
                                         results[i][5][j],
                                         results[i][6][j],
                                         results[i][7][j],
                                         results[i][8][j],
                                         results[i][9][j],
                                         results[i][10][j],
                                         results[i][11][j],
                                         results[i][12][j],
                                         results[i][13][j],
                                         results[i][14][j],
                                         results[i][15][j],
					                     results[i][16][j]))
            else:
                new_results.append(results[i])
                #print(results[i])
        df2 = pd.DataFrame.from_records(new_results, columns=[str(pipeline['star_id']), 'numax', 'evo', 'var', 'zc', 'hoc', 'mc', 'abs_k_mag',
                                                              'mu0', 'mk', 'Ak', 'para_flag', 'distance', 'fill', 'ndata', 'running_var', 'colour'])
    else:
        df2 = pd.DataFrame.from_records(results, columns=[str(pipeline['star_id']), 'numax', 'evo', 'var', 'zc', 'hoc', 'mc', 'abs_k_mag',
                                                          'mu0', 'mk', 'Ak', 'para_flag', 'distance', 'fill', 'ndata', 'running_var', 'colour'])

    if pipeline['campaign'] == 'None':
        output_name = 'Colours_New_Gaps_output_data_noise_'+str(pipeline['star_id'])+'_'+str(NDAYS)
    else:
        output_name = 'Colours_New_Gaps_output_data_noise_'+str(pipeline['star_id'])+'_'+str(pipeline['campaign'])+'_'+str(NDAYS)
    if pipeline['data_type'] != 'None':
        output_name += '_'+str(pipeline['data_type'])
    #output_name += '_K2_detrending'
    #output_name += '_white_noise_test'
    df2.to_csv(output_name+'.csv', index=False)
