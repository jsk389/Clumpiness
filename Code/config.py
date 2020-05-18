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
