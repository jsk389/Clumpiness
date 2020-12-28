#!/usr/bin/env/ python3

import os
import yaml

class ConfigurationNotFound(Exception):
    """
    Exception to be raised if the config.yml file could not be located
    """

def readConfigFile(fname):
    """
    Read in the yaml configuration file
    """

    conf_dir = '.'

    configfname = fname #conf_dir + '/config.yml'

    try:
        with open(configfname, 'r') as f:
            settings = yaml.safe_load(f)
    except FileNotFoundError as exception:
        raise ConfigurationNotFound("Could not open {}".format(configfname)) from exception

    return settings


def retrieve_model_parameters(dataset_length=-1, ncores=4):
    """
    Retrieve the correct parameters for the trained Clumpiness model
    """
    if dataset_length == -1:
        return  {'objective': 'multi:softprob',
                'eval_metric': 'mlogloss',
                'eta': 0.1,
                    'gamma': 7.5,
                    'max_depth': 3,
                    'min_child_weight': 4,
                    'subsample': 0.7,
                    'colsample': 0.7,
                    'silent': 1,
                    'nthread': ncores,
                    'num_class': 5,
                    'seed': 0}
    elif dataset_length == 180:
        return  {'objective': 'multi:softprob',
                    'eval_metric': 'mlogloss',
                    'eta': 0.375,
                        'gamma': 7.5,
                        'max_depth': 6,
                        'min_child_weight': 1.0,
                        'subsample': 0.5,
                        'colsample': 0.5,
                        'silent': 1,
                        'nthread': ncores,
                        'num_class': 5,
                        'seed': 0}
    elif dataset_length == 80:
        return {'objective': 'multi:softprob',
                'eval_metric': 'mlogloss',
                'eta': 0.275,
                    'gamma': 7.5,
                    'max_depth': 10,
                    'min_child_weight': 2.0,
                    'subsample': 0.85,
                    'colsample': 0.85,
                    'silent': 1,
                    'nthread': ncores,
                    'num_class': 5,
                    'seed': 0}
    elif dataset_length == 27:
        return {'objective': 'multi:softprob',
                'eval_metric': 'mlogloss',
                'eta': 0.025,
                    'gamma': 7.5,
                    'max_depth': 8,
                    'min_child_weight': 3.0,
                    'subsample': 0.5,
                    'colsample': 0.85,
                    'silent': 1,
                    'nthread': ncores,
                    'num_class': 5,
                    'seed': 0}
    else:
        sys.exit("Please provide a compatible dataset length!")
