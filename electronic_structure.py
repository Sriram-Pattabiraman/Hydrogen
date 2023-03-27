# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 13:57:19 2023

@author: Sri
"""

from tqdm import tqdm

import math
import random
import numpy as np
import scipy as sp

import itertools

#import sturm_liouville as sl
import utility_funcs as util
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





def create_core_hamiltonian_and_overlap_matrix(AO_basis, nuclear_charges, nuclear_positions, integration_window=None, integration_sample_size=1000, laplace_dx=.0001, parallel_pool=None):
    if parallel_pool == None:
        parallel_pool = joblib.Parallel(n_jobs=8, verbose=VERBOSITY, batch_size=4096, prefer='threads')
      
    space_dim = integration_window.shape[0]
    AO_basis_size = AO_basis.shape[0]
    
    def core_hamiltonian_and_overlap_element_integrand_given_AOs(*r, AO_i=None, AO_j=None, nuclear_charges=nuclear_charges, nuclear_positions=nuclear_positions, laplace_dx=.0001):
        AO_i_val = AO_i(*r)
        displacements_to_sample_for_laplacian = np.diag(np.repeat(laplace_dx, space_dim))
        AO_j_val = AO_j(*r)
        AO_j_central_differences = [(AO_j(*r+delta_vector)+AO_j(*r-delta_vector)-2*AO_j_val)/laplace_dx**2 for delta_vector in displacements_to_sample_for_laplacian]
        laplacian_AO_j_val = sum(AO_j_central_differences)
        kinetic_integrand_val = (-1/2) * laplacian_AO_j_val * AO_i_val 
        overlap_integrand_val = AO_i_val * AO_j_val
        potential_integrand_val = - np.sum(nuclear_charges/(np.repeat([r], nuclear_positions.shape[0], axis=0) - nuclear_positions) ) * overlap_integrand_val
        return np.array([kinetic_integrand_val+potential_integrand_val, potential_integrand_val])
        
    def get_an_element_of_core_hamiltonian_and_overlap_matrices(i,j, AO_basis=AO_basis, nuclear_charges=nuclear_charges, nuclear_positions=nuclear_positions, integration_window=integration_window, integration_sample_size=integration_sample_size, laplace_dx=.0001, parallel_pool=parallel_pool):
        AO_i, AO_j = AO_basis[i], AO_basis[j]
        this_core_hamiltonian_and_overlap_element_integrand_given_AOs = lambda *r: core_hamiltonian_and_overlap_element_integrand_given_AOs(*r, AO_i=AO_i, AO_j=AO_j, nuclear_charges=nuclear_charges, nuclear_positions=nuclear_positions, laplace_dx=.0001)
        return mc.monte_carlo_integration(this_core_hamiltonian_and_overlap_element_integrand_given_AOs, integration_window, sample_size=integration_sample_size, parallel_pool=parallel_pool)
    
    #!!!think about doing this part parallel too/instead of the integration part (later on with the real variational method after hartree fock with stuff like linear combinations of slater determnants and jastrow factors, i think we'll definitely need to parallelize integration first and foremost due to the many-d space (which should also make the monte carlo integration truly shine). here though, i am less sure that the integration is the proper place to parallelize).
    core_hamiltonian_matrix = util.symarray(np.nans(AO_basis_size, AO_basis_size), permutable_axes_groups=[[0,1]])
    AO_overlap_matrix = util.symarray(np.nans(AO_basis_size, AO_basis_size), permutable_axes_groups=[[0,1]])
    for i in range(len(AO_basis_size)):
        for j in filter(lambda j: bool(j<=i), range(AO_basis_size)):
            core_hamiltonian_matrix[i,j], AO_overlap_matrix[i,j] = get_an_element_of_core_hamiltonian_and_overlap_matrices(i,j)
    return core_hamiltonian_matrix, AO_overlap_matrix
        
def create_density_matrices(spin_0_MO_coefficient_matrix, spin_1_MO_coefficient_matrix):
    spin_0_density_matrix = np.einsum('ik,jk', spin_0_MO_coefficient_matrix, spin_0_MO_coefficient_matrix)
    spin_1_density_matrix = np.einsum('ik,jk', spin_1_MO_coefficient_matrix, spin_1_MO_coefficient_matrix)
    
    return spin_0_density_matrix, spin_1_density_matrix

def create_two_electron_repulsion_integral_tensor(AO_basis, nuclear_positions, integration_window, integration_sample_size=1000, parallel_pool=None):
    integration_window = np.repeat(integration_window, 2, axes=0)
    two_coord_integration_window = np.tile(integration_window, [2,1])
    def integrand_ijkl_as_func_of_combined_coords_vars(*r_1_and_2, i=None, j=None, k=None, l=None):
        two_coord_dim = r_1_and_2.shape[0]
        space_dim = two_coord_dim/2
        r_1 = r_1_and_2[:space_dim/2]
        r_2 = r_1_and_2[space_dim/2:]
        return AO_basis[i](r_1)*AO_basis[j](r_1)*AO_basis[k](r_2)*AO_basis[l](r_2) * (1/np.linalg.norm(r_2-r_1))
    
    basis_size = AO_basis.shape[0]
    two_electron_repulsion_integral_tensor = util.symarray(np.nans([basis_size,basis_size,basis_size,basis_size]), permutable_axes_groups=[[0,1], [2,3]])
    for i in range(basis_size):
        for j in filter(lambda j: bool(j<=i), range(basis_size)):
            for k in range(basis_size):
                for l in filter(lambda k: bool(l<=k), range(basis_size)):
                    two_electron_repulsion_integral_tensor[i,j,k,l] = mc.monte_carlo_integration(integrand_ijkl_as_func_of_combined_coords_vars(i=i, j=j, k=k, l=l), two_coord_integration_window, sample_size=integration_sample_size, parallel_pool=parallel_pool)

    return two_electron_repulsion_integral_tensor
        
def create_fock_matrices(spin_0_density_matrix, spin_1_density_matrix, core_hamiltonian_matrix, two_electron_repulsion_integral_tensor, AO_basis):
    #F^s_ij = h_ij + J^s_ij + J^s'_ij - K^s_ij where s is spin and s' is the other spin and h is the core hamiltonian matrix and J^s is the coulumb matrix for the s-spin electrons and K^s is the s spin exchange. 
    #E = h_ij  D^\alpha_ij + h_ij D^\beta_ij  + 1/2 (D^\alpha_ij F^\alpha_ij) + 1/2 (D^\beta_ij F^\beta_ij)
    #h_ij = T_ij + V_ij = -1/2 \int \chi_i(r) \nabla^2_r \chi_j (r) dr - \int \chi_i(r) (\sum_A Z_A/||r-r_A||) \chi_j(r) dr = pre_computed_core_hamiltonian
    #where A, Z_A, r_A are the indice, charge, and position of nucleus A.
    #J^s_ij = D^s_kl (ij|kl)
    #K^s_ij = D^s_kl (il|kj)
    #D^s_ij = C^s_ik C^s_jk = density_matrix
    #\phi^s_k = C_ik \chi_i
    #so D^s_ij = <\phi_k|\chi_i> <\phi_k|chi_j>
    #(ij|kl)=two_electron_repulsion_integral(chi_i,chi_j,chi_k,chi_l) is the two electron repulsion integral \int \int \chi^*_i(r_1) \chi_j(r_1) \chi^*_k(r_2) \chi_l(r_2) / ||r_1 - r_2||  dr_1 dr_2
    #the overlap is S^s_ij = \int \chi_i (r) \chi_j (r) dr
    #clearly, F,E,h,J,K,D,S are all symmetric
    
    spin_0_coulomb_matrix = np.einsum('kl,ijkl->ij', spin_0_density_matrix,  two_electron_repulsion_integral_tensor)
    spin_1_coulomb_matrix = np.einsum('kl,ijkl->ij', spin_1_density_matrix,  two_electron_repulsion_integral_tensor)
    
    spin_0_exchange_matrix = np.einsum('kl,ilkj->ij', spin_0_density_matrix,  two_electron_repulsion_integral_tensor)   
    spin_1_exchange_matrix = np.einsum('kl,ilkj->ij', spin_1_density_matrix,  two_electron_repulsion_integral_tensor)   

    totalled_coulomb_matrix = spin_0_coulomb_matrix + spin_1_coulomb_matrix
    spin_0_fock_matrix = core_hamiltonian_matrix + totalled_coulomb_matrix + spin_0_exchange_matrix
    spin_1_fock_matrix = core_hamiltonian_matrix + totalled_coulomb_matrix + spin_1_exchange_matrix
    
    return spin_0_fock_matrix, spin_1_fock_matrix

def solve_roothan(spin_fock_matrix_0, spin_fock_matrix_1, AO_overlap_matrix):
    spin_0_diag_eigs, new_spin_0_MO_coefficient_matrix = sp.linalp.eigh(a=spin_fock_matrix_0, b=AO_overlap_matrix)
    spin_1_diag_eigs, new_spin_1_MO_coefficient_matrix = sp.linalp.eigh(a=spin_fock_matrix_1, b=AO_overlap_matrix)

    return spin_0_diag_eigs, spin_1_diag_eigs, new_spin_0_MO_coefficient_matrix, new_spin_1_MO_coefficient_matrix

def create_and_solve_roothaan(guess_spin_0_MO_coefficient_matrix, guess_spin_1_MO_coefficient_matrix, core_hamiltonian_matrix, two_electron_repulsion_integral_tensor, AO_overlap_matrix, AO_basis):
    #F^sC^s = SC^s\epsilon^s, where all are matrices. F is fock matrix, C is matrix of eigenvectors, and \epsilon is diagonal matrix with the eigenvalues. Note that S, the overlap, doesn't depend on spin,
    spin_0_density_matrix, spin_1_density_matrix = create_density_matrices(guess_spin_0_MO_coefficient_matrix, guess_spin_1_MO_coefficient_matrix)
    spin_fock_matrix_0, spin_fock_matrix_1 = create_fock_matrices(spin_0_density_matrix, spin_1_density_matrix, core_hamiltonian_matrix, two_electron_repulsion_integral_tensor, AO_basis)

    return solve_roothan(spin_fock_matrix_0, spin_fock_matrix_1, AO_overlap_matrix)

def hartree_fock_method(AO_basis, nuclear_charges, nuclear_positions, integration_window, fixed_point_iterations=100, integration_sample_size=1000, laplace_dx=.0001, parallel_pool=None, disable_hartree_fock_fixed_point_iterations_pbar=False):
    if parallel_pool == None:
        parallel_pool = joblib.Parallel(n_jobs=8, verbose=VERBOSITY, batch_size=4096, prefer='threads')
    
    core_hamiltonian_matrix, AO_overlap_matrix = create_core_hamiltonian_and_overlap_matrix(AO_basis, nuclear_charges, nuclear_positions, integration_window=integration_window, integration_sample_size=integration_sample_size, laplace_dx=.0001, parallel_pool=parallel_pool)
    two_electron_repulsion_integral_tensor = create_two_electron_repulsion_integral_tensor(AO_basis, integration_window, integration_sample_size=integration_sample_size, parallel_pool=None)
    
    initial_fock_matrix = core_hamiltonian_matrix
    this_spin_0_diag_eigs, this_spin_1_diag_eigs, this_spin_0_MO_coefficient_matrix, this_spin_1_MO_coefficient_matrix = solve_roothan(initial_fock_matrix, initial_fock_matrix, AO_overlap_matrix)
    this_spin_0_density_matrix, this_spin_1_density_matrix = create_density_matrices(this_spin_0_MO_coefficient_matrix, this_spin_1_MO_coefficient_matrix)
    for i in tqdm(range(fixed_point_iterations), desc="Calculating Hartree-Fock Orbital...", disable=disable_hartree_fock_fixed_point_iterations_pbar):
        
        spin_fock_matrix_0, spin_fock_matrix_1 = create_fock_matrices(this_spin_0_density_matrix, this_spin_1_density_matrix, core_hamiltonian_matrix, two_electron_repulsion_integral_tensor, AO_basis)
        this_spin_0_diag_eigs, this_spin_1_diag_eigs, this_spin_0_MO_coefficient_matrix, this_spin_1_MO_coefficient_matrix = solve_roothan()
        