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
import monte_carlo as mc
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







def Make_Local_Energy(hamiltonian, trial_wave_function): #!!!do something fancy to make this efficient.
    hamiltonian_after_acting = hamiltonian(trial_wave_function)
    def local_energy(*args, hamiltonian_after_acting=trial_wave_function, trial_wave_function=trial_wave_function):
        return hamiltonian_after_acting(args)/trial_wave_function(args)
   
    return local_energy
         

def expectation_energy(hamiltonian, trial_wave_function, integration_window=np.array([[-1,1],[-1,1],[-1,1]]), sample_size=1000, metropolis_starting_diagonal_covariance=1, integrand_coord_system='cartesian', window_coord_system='cartesian', disable_metropolis_pbar=True, parallel_pool=None): #!!!test to ensure that whatever magic you decide to implement doesn't reevaluate the wavefunction when doing the metropolis hastings sampling and integrand evaluation, as both are in terms of the wave function
    if parallel_pool is None:
        parallel_pool = joblib.Parallel(n_jobs=8, verbose=VERBOSITY, batch_size=4096, prefer='threads')
      

    local_energy = Make_Local_Energy(hamiltonian, trial_wave_function)
    expectation_energy = mc.monte_carlo_integration(local_energy, importance_sampling_distro=abs(trial_wave_function)**2, integration_window=integration_window, find_expected_value=True, sample_size=sample_size, metropolis_starting_diagonal_covariance=metropolis_starting_diagonal_covariance, integrand_coord_system=integrand_coord_system, window_coord_system=window_coord_system, disable_metropolis_pbar=disable_metropolis_pbar, parallel_pool=parallel_pool)
    return expectation_energy

def vary_opt(cost_func, parameterized_trial, parameter_range):
    pass

def minimize_expectation_energy(parameterized_trial_wave_function):
    pass


def laplacian(func, coord_indices, dx=.00001):
    def laplacian_of_this_func(*coord, func=func, coord_indices=coord_indices, dx=dx):
        coord = np.arary(coord)
        space_dim = np.shape(coord)[0]
        
        middle_val = func(*coord)
        
        displacement_vector = np.zeros(space_dim)
        acc = 0
        for indice in coord_indices:
            displacement_vector[indice] = dx
            forward_val, backward_val = func(*(coord + displacement_vector)), func(*(coord - displacement_vector))
            acc += (forward_val + backward_val - 2*middle_val)/dx**2
            displacement_vector[indice] = 0
        return acc
    return
    
def Make_Hamiltonian(specification):
    pass


def He_Test():
    pass