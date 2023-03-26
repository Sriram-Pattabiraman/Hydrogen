# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 13:57:19 2023

@author: srira
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


