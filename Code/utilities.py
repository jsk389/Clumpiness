#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys

from config import readConfigFile
from joblib import Parallel, delayed
from tqdm import tqdm


def text_progessbar(seq, total=None):
    step = 1
    tick = time.time()
    while True:
        time_diff = time.time()-tick
        avg_speed = time_diff/step
        total_str = 'of %n' % total if total else ''
        print('step', step, '%.2f' % time_diff, 'avg: %.2f iter/sec' % avg_speed, total_str)
        step += 1
        yield next(seq)

all_bar_funcs = {
                'tqdm': lambda args: lambda x: tqdm(x, **args),
}

def ParallelExecutor(use_bar='tqdm', **joblib_args):
    def aprun(bar=use_bar, **tq_args):
        def tmp(op_iter):
            if str(bar) in all_bar_funcs.keys():
                bar_func = all_bar_funcs[str(bar)](tq_args)
            else:
                raise ValueError("Value %s not supposrted as bar type"%bar)
            return Parallel(**joblib_args)(bar_func(op_iter))
        return tmp
    return aprun

def calc_numax(logg, teff):
    return ((10**logg)/27542.29) * (teff/5777)**0.5 * 3090