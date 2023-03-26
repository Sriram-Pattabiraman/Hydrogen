# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 12:08:52 2023

@author: Sri
"""

from tqdm import tqdm

import math
import random
import numpy as np

import itertools

#import sturm_liouville as sl
from interleaver import Interleave

from scipy import constants

import pyqtgraph as pg
import matplotlib.pyplot as plt
'''
import IPython
#if this is an ipy, start the qt5 event loop
%gui qt5
'''
app =  pg.Qt.mkQApp()
import pyqtgraph.opengl as gl

import matplotlib.cm as colormaps

import joblib
VERBOSITY = 0






            
def Make_Local_Energy(hamiltonian, trial_wave_function):
    pass

def expectation_energy(hamiltonian, trial_wave_function, track_local_energy_variance=False):
    pass

def minimize_expectation_energy(parameterized_trial_wave_function):
    pass

def minimize_local_energy_variance(paramaterized_trial_wave_function):
    pass