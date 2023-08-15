# Clumpiness
Predict the evolutionary state of a red giant (Hydrogen shell or core Helium burning) using time domain photometry.


What data is needed to make a prediction?
=======================================
The ingredients needed to make a prediction are as follows:

1.  `rest` column for the target from the table of distances computed by [Bailer-Jones et al. (2018)](https://arxiv.org/abs/1804.10121).
2. The Gaia data on this star (this is used to compute the quality of the astrometric solution), specifically these columns:
    - `astrometric_chi2_al`
    - `astrometric_n_good_obs_al`
    - `astrometric_excess_noise`
    - `bp_rp`
    - `phot_g_mean_mag`
    - `ra` and `dec`
3. The K-band magnitude (`Kmag`) from the TIC.
4. Time-series data of the format time (in days), flux (in ppm). 

The data contained in points 1 and 2 make up the "meta-data" for each star, used to make sure the Gaia parallax is good enough to compute the K-band absolute magnitude feature.

The data input (reading) part of `Clumpiness` is not implemented as well as it could (or perhaps should) be and the way in which the data is handled depends on the name of the file (generally due to different groups formatting the data in different ways) and no standardised way has yet been implemented. So it is worth checking the computed features for a few stars to make sure they make sense. For example, the variances are computed assuming the data is in units of ppm.

Once this data has been retrieved the application of `Clumpiness` is surprisingly simple.

How to install ?
================
1. clone the repository
2. `cd` into the folder and run `python setup.py install` to install the package.