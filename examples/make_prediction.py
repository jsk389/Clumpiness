import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import xgboost

import Dataset_regular as Dataset
from featureextractor import Features


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Make a prediction using Clumpiness.")
    parser.add_argument("--input", dest="input", required=True,
                        help="Filename of the timeseries")
    parser.add_argument("--model", dest="model", required=True,
                        help="Path to trained models (not to the models themselves!).")
    parser.add_argument("--ndays", dest=ndays, type=int,
                        help="Length of dataset (optional, one of -1/180/80/27 days).")
    parser.add_argument("--metadata", dest="metadata", required=True,
                        help="Filename containing the metadata.")

    argv = parser.parse_args()

    metadata = pd.read_csv(argv.metadata)

    # Read in data
    star = Dataset.Dataset(metadata['ID'], argv.input)
    # ndays set to -1 if you want to use whole timeseries and not
    # cut down at all.
    star.read_timeseries(ndays = -1)
    star.to_regular_sampling()

    # Compute features
    feat = Features(star)
    zs = feat.compute_zero_crossings()
    var = feat.compute_var()
    hoc, _ = feat.compute_higher_order_crossings(5)
    mc = feat.compute_variance_change()
    fill = feat.compute_fill()   
    abs_mag, mu0, Ak, good_flag, distance = feat.compute_abs_K_mag(metadata['kmag'], 
                                                                   metadata['distance'], 
                                                                   0.0, # Set distance error to zero - not needed 
                                                                   metadata['ra'], 
                                                                   metadata['dec'], 
                                                                   metadata['excess'])
    # Prepare data for prediction
    predict = xgboost.DMatrix(np.array([var, zs, hoc, abs_mag, mc]).reshape(1, -1))

    # Fetch model parameters
    params = retrieve_model_parameters(argv.ndays)

    # Load in model
    bst = xgboost.Booster(params)
    bst.load_model(argv.model+str(argv.ndays)+'.model')
    # Load up best iteration
    best_iter = np.loadtxt(argv.model+'best_iter_'+str(argv.ndays)+'.txt')

    # Make prediction
    preds = bst.predict(predict, ntree_limit=best_iter).sequeeze()
    # Gives output in format with split RGB probs
    # Combined RGB probabilities
    RGB_prob = preds[0] + preds[1] + preds[2]
    RC_prob = preds[3]
    KOI_prob = preds[4]
    print(f"Probability of RGB: {RGB_prob}")
    print(f"Probability of RC: {RC_prob}")
    print(f"Probability of KOI/not RGB or RC: {KOI_prob}")

    # Do what you want with this information ... 