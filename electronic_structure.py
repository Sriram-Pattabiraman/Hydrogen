# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 110:57:19 2023

@author: Sri
"""

from tqdm import tqdm

import math
import numpy as np
import scipy as sp
import pyshtools as shtools

import itertools

#import sturm_liouville as sl
import utility_funcs as util
import monte_carlo as mc

import matplotlib.pyplot as plt

import joblib
VERBOSITY = 0



def this_real_spherical_harmonic(l, m, theta, phi):
    if m >= 0:
        return shtools.legendre.legendre_lm(l, m, math.cos(theta), normalization='ortho')*math.cos(m*phi)
    else:
        return shtools.legendre.legendre_lm(l, abs(m), math.cos(theta), normalization='ortho')*math.sin(abs(m)*phi)
    
def sto_orbital_creator(effective_nuclear_charge, nuclear_position, n, l, m):
    #breakpoint()
    '''
    radial_normalizing_constant = ( (2 * effective_nuclear_charge)**n )*(( 2*n/math.factorial(2*n) )**.5)
    def this_sto_orbital(x,y,z, effective_nuclear_charge=effective_nuclear_charge, nuclear_position=nuclear_position, n=n, radial_normalizing_constant=radial_normalizing_constant, l=l, m=m):
        shifted_cart_coords = np.array([x,y,z]) - nuclear_position
        r, theta, phi = util.convert_coord(shifted_cart_coords, from_system='cartesian', out_system='spherical')
        r_norm = np.linalg.norm(r)
        radial_component = radial_normalizing_constant * (r_norm**(n-1)) * math.exp(-effective_nuclear_charge*r_norm)
        angular_component = this_real_spherical_harmonic(l, m, theta, phi)
        return radial_component * angular_component
    return this_sto_orbital
    '''
    def this_gaussian_orbital(x,y,z, effective_nuclear_charge=effective_nuclear_charge, nuclear_position=nuclear_position):
        if effective_nuclear_charge == 1:
            norm_cons = .3696
            exp_cons = -.4166
        elif effective_nuclear_charge == 2:
            norm_cons = .5881
            exp_cons =  -.7739
        
        return norm_cons * math.exp(exp_cons * (np.linalg.norm(np.array([x,y,z])-nuclear_position)**2))
            
    return this_gaussian_orbital
            

def make_indiv_AO_basis(nuclear_charge, nuclear_position, principal_quantum_numbers_to_consider=1):
    this_indiv_AO_basis = []
    for principal_quantum_number in range(1, principal_quantum_numbers_to_consider+1):
        for l in range(0,principal_quantum_number):
            for abs_m in range(0,l+1):
                this_indiv_AO_basis.append(sto_orbital_creator(nuclear_charge, nuclear_position, principal_quantum_number, l, abs_m))

    this_indiv_AO_basis = np.array(this_indiv_AO_basis)
    return this_indiv_AO_basis

def assemble_AO_basis_from_nuclear_data(nuclear_charges, nuclear_positions, principal_quantum_numbers_to_consider=1):
    AO_basis = np.array([])
    for nucleus_center_indice in range(len(nuclear_positions)):
        this_nucleus_position = nuclear_positions[nucleus_center_indice]
        this_nuclear_charge = nuclear_charges[nucleus_center_indice]
        AO_basis = np.concatenate([AO_basis, make_indiv_AO_basis(this_nuclear_charge, this_nucleus_position, principal_quantum_numbers_to_consider=principal_quantum_numbers_to_consider)])

    return AO_basis

    

def create_core_hamiltonian_and_overlap_matrix(AO_basis, nuclear_charges, nuclear_positions, integration_window=None, integration_sample_size=4e4, stages=5, bin_num_vector=100, laplace_dx=.00001, parallel_pool=None):
    if parallel_pool == None:
        parallel_pool = joblib.Parallel(n_jobs=8, verbose=VERBOSITY, batch_size=4096, prefer='threads')
      
    space_dim = len(integration_window)
    AO_basis_size = len(AO_basis)
    
    def kin_element_integrand_given_AOs(*r, AO_i=None, AO_j=None, laplace_dx=.00001):
        AO_i_val = AO_i(*r)
        displacements_to_sample_for_laplacian = np.diag(np.repeat(laplace_dx, space_dim))
        AO_j_val = AO_j(*r)
        AO_j_central_differences = [(AO_j(*r+delta_vector)+AO_j(*r-delta_vector)-2*AO_j_val)/laplace_dx**2 for delta_vector in displacements_to_sample_for_laplacian]
        laplacian_AO_j_val = sum(AO_j_central_differences)
        kinetic_integrand_val = (-1/2) * laplacian_AO_j_val * AO_i_val 
        return kinetic_integrand_val
    
    def kin_pot_and_overlap_element_integrand_given_AOs(*r, AO_i=None, AO_j=None, nuclear_charges=nuclear_charges, nuclear_positions=nuclear_positions):
        AO_i_val = AO_i(*r)
        #displacements_to_sample_for_laplacian = np.diag(np.repeat(laplace_dx, space_dim))
        AO_j_val = AO_j(*r)
        #AO_j_central_differences = [(AO_j(*r+delta_vector)+AO_j(*r-delta_vector)-2*AO_j_val)/laplace_dx**2 for delta_vector in displacements_to_sample_for_laplacian]
        #laplacian_AO_j_val = sum(AO_j_central_differences)
        #kinetic_integrand_val = (-1/2) * laplacian_AO_j_val * AO_i_val 
        overlap_integrand_val = AO_i_val * AO_j_val
        potential_integrand_val = 0
        for nucleus_index in range(len(nuclear_positions)):
            this_nuclear_position = nuclear_positions[nucleus_index]
            this_nuclear_charge = nuclear_charges[nucleus_index]
            r_diff = r - this_nuclear_position
            r_diff_norm = 0
            for part in r_diff:
                r_diff_norm += part**2
            r_diff_norm = r_diff_norm**.5
            potential_integrand_val -= this_nuclear_charge/r_diff_norm
            
        potential_integrand_val *= overlap_integrand_val
        #vectorized_norm = np.vectorize(np.linalg.norm)
        #potential_integrand_val = - np.sum(np.divide( nuclear_charges.reshape((len(nuclear_charges),1)), vectorized_norm(np.repeat([r], len(nuclear_positions), axis=0) - nuclear_positions) )) * overlap_integrand_val
        #return np.array([kinetic_integrand_val, potential_integrand_val, overlap_integrand_val])
        return np.array([potential_integrand_val, overlap_integrand_val])
    
        
    def get_an_element_of_kin_pot_and_overlap_matrices(i,j, AO_basis=AO_basis, nuclear_charges=nuclear_charges, nuclear_positions=nuclear_positions, integration_window=integration_window, integration_sample_size=integration_sample_size, stages=stages, bin_num_vector=bin_num_vector, laplace_dx=.00001, parallel_pool=parallel_pool):
        AO_i, AO_j = AO_basis[i], AO_basis[j]
        this_kin_element_integrand_given_AOs = lambda *r: kin_element_integrand_given_AOs(*r, AO_i=AO_i, AO_j=AO_j, laplace_dx=laplace_dx)
        this_pot_and_overlap_element_integrand_given_AOs = lambda *r: kin_pot_and_overlap_element_integrand_given_AOs(*r, AO_i=AO_i, AO_j=AO_j, nuclear_charges=nuclear_charges, nuclear_positions=nuclear_positions)
        '''
        breakpoint()
        xs = np.linspace(-1,1,100)
        actuals = [this_kin_element_integrand_given_AOs(x,0,0) for x in xs]
        import matplotlib.pyplot as plt
        plt.plot(xs, actuals)
        '''
        kin_element = mc.VEGAS_integration(this_kin_element_integrand_given_AOs, integration_window, sample_size=integration_sample_size, stages=stages, bin_num_vector=bin_num_vector, parallel_pool=parallel_pool) 
        pot_and_overlap_element = mc.VEGAS_integration(this_pot_and_overlap_element_integrand_given_AOs, integration_window, sample_size=integration_sample_size, stages=stages, bin_num_vector=bin_num_vector, parallel_pool=parallel_pool) 
        return np.array([kin_element, *pot_and_overlap_element])
    
    
    kin = util.symarray(np.zeros((AO_basis_size, AO_basis_size)), permutation_generators=[[[0,1]]],)
    pot = util.symarray(np.zeros((AO_basis_size, AO_basis_size)), permutation_generators=[[[0,1]]], )
    core_hamiltonian_matrix = util.symarray(np.zeros((AO_basis_size, AO_basis_size)), permutation_generators=[[[0,1]]],)
    AO_overlap_matrix = util.symarray(np.zeros((AO_basis_size, AO_basis_size)), permutation_generators=[[[0,1]]],)
    
    #breakpoint()
    #'''
    for i in range(AO_basis_size):
        for j in filter(lambda j: bool(j<=i), range(AO_basis_size)):
            kin[i,j], pot[i,j], AO_overlap_matrix[i,j] = get_an_element_of_kin_pot_and_overlap_matrices(i,j)
            core_hamiltonian_matrix[i,j] = kin[i,j] + pot[i,j]
            breakpoint()
    #'''
    
    '''
    indexes_to_compute_list = []
    for i in range(AO_basis_size):
        for j in filter(lambda j: bool(j<=i), range(AO_basis_size)):
            indexes_to_compute_list.append([i,j])
        
    flat_work_generator = (joblib.delayed(get_an_element_of_core_hamiltonian_and_overlap_matrices)(i,j) for i,j in indexes_to_compute_list)
    flat_matrix_outs = parallel_pool(flat_work_generator)
    for indexes_indice in range(len(indexes_to_compute_list)):
        i,j = indexes_to_compute_list[indexes_indice]
        core_hamiltonian_matrix[i,j], AO_overlap_matrix[i,j] = flat_matrix_outs[indexes_indice]
    '''
        
    #core_hamiltonian_matrix, AO_overlap_matrix = (util.symarray([[ 0.41222943, -1.32350684],
    #          [-1.32350684, -2.46838966]]), util.symarray([[0.0708652 , 0.14331325],
    #          [0.14331325, 0.29684872]]))
    
    return core_hamiltonian_matrix, AO_overlap_matrix
        
def create_density_matrices(spin_0_MO_coefficient_matrix, spin_1_MO_coefficient_matrix):
    spin_0_density_matrix = np.einsum('ik,jk', spin_0_MO_coefficient_matrix, spin_0_MO_coefficient_matrix)
    spin_1_density_matrix = np.einsum('ik,jk', spin_1_MO_coefficient_matrix, spin_1_MO_coefficient_matrix)
    
    return spin_0_density_matrix, spin_1_density_matrix

def create_two_electron_repulsion_integral_tensor(AO_basis, nuclear_positions, integration_window, integration_sample_size=4e4, stages=5, bin_num_vector=100, disable_monte_carlo_repulsion_integral_pbar=False, parallel_pool=None):
    two_coord_integration_window = np.repeat(integration_window, 2, axis=0)
    #breakpoint()
    def integrand_ijkl_as_func_of_combined_coords_vars(*r_1_and_2, i=None, j=None, k=None, l=None):
        two_coord_dim = len(r_1_and_2)
        space_dim = two_coord_dim/2
        assert space_dim.is_integer()
        space_dim = int(space_dim)
        r_1 = r_1_and_2[:space_dim]
        r_2 = r_1_and_2[space_dim:]
        r_1, r_2 = np.array(r_1), np.array(r_2)
        r_diff = r_2-r_1
        r_diff_norm = 0
        for part in r_diff:
            r_diff_norm += part**2
        r_diff_norm = r_diff_norm**.5
        return AO_basis[i](*r_1)*AO_basis[j](*r_1)*AO_basis[k](*r_2)*AO_basis[l](*r_2) * (1/r_diff_norm)
    
    basis_size = len(AO_basis)
    two_electron_repulsion_integral_tensor = util.symarray(np.zeros([basis_size,basis_size,basis_size,basis_size]), permutation_generators=[ [[0,1]], [[2,3]], [[0,2],[1,3]] ])
    
    #there's symarray takes care of propogating the symmetries, but we need a way to iterate through them while picking exactly one element from each equivalence class
    #remember that the symmetries are generated by: (ij|kl)=(ji|kl)=(ij|lk)=(kl|ij)
    #so, we pick an i. we then want to pick a j so that exactly one of ( (i,j) , (j,i) ) end up being picked. we can do this by requiring j>=i.
    #likewise we require l>=k.
    #the trickier bit is the (ij|kl)=(kl|ij) symmetry. we want exactly one of ( (i,j,k,l), (k,l,i,j) ) to pass our criterion.
    #the most obvious way to do that is to require that (i,j) precedes (or is equivalent to) (k,l) w.r.t the dictionary order
    
    for i in range(basis_size):
        for j in filter(lambda j: bool(j>=i), range(basis_size)):
            for k in range(basis_size):
                for l in filter(lambda l: bool(l>=k), range(basis_size)):
                    if not util.lex_less_equals((i,j), (k,l)):
                        continue
                    #breakpoint()
                    two_electron_repulsion_integral_tensor[i,j,k,l] = mc.VEGAS_integration(lambda *r_1_and_2: integrand_ijkl_as_func_of_combined_coords_vars(*r_1_and_2, i=i, j=j, k=k, l=l), two_coord_integration_window, sample_size=integration_sample_size, stages=stages, bin_num_vector=bin_num_vector, disable_first_metropolis_pbar=disable_monte_carlo_repulsion_integral_pbar, disable_vegas_stage_pbar=disable_monte_carlo_repulsion_integral_pbar, parallel_pool=parallel_pool)
                    #two_electron_repulsion_integral_tensor[i,j,k,l] = mc.monte_carlo_integration(lambda *r_1_and_2: integrand_ijkl_as_func_of_combined_coords_vars(*r_1_and_2, i=i, j=j, k=k, l=l), two_coord_integration_window, sample_size=integration_sample_size, disable_metropolis_pbar=disable_monte_carlo_repulsion_integral_pbar, parallel_pool=parallel_pool)
    #breakpoint()
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
    #clearly, F,E,h,J,K,D,S are all symmetric (note that C is not symmetric)
    
    spin_0_coulomb_matrix = np.einsum('kl,ijkl', spin_0_density_matrix,  two_electron_repulsion_integral_tensor)
    spin_1_coulomb_matrix = np.einsum('kl,ijkl', spin_1_density_matrix,  two_electron_repulsion_integral_tensor)
    
    spin_0_exchange_matrix = np.einsum('kl,ilkj', spin_0_density_matrix,  two_electron_repulsion_integral_tensor)   
    spin_1_exchange_matrix = np.einsum('kl,ilkj', spin_1_density_matrix,  two_electron_repulsion_integral_tensor)   

    totalled_coulomb_matrix = spin_0_coulomb_matrix + spin_1_coulomb_matrix
    
    spin_0_fock_matrix = core_hamiltonian_matrix + totalled_coulomb_matrix - spin_0_exchange_matrix
    spin_1_fock_matrix = core_hamiltonian_matrix + totalled_coulomb_matrix - spin_1_exchange_matrix
    
    return spin_0_fock_matrix, spin_1_fock_matrix

def solve_roothan(spin_fock_matrix_0, spin_fock_matrix_1, AO_overlap_matrix):
    spin_0_diag_eigs, new_spin_0_MO_coefficient_matrix = sp.linalg.eigh(a=spin_fock_matrix_0, b=AO_overlap_matrix)
    spin_1_diag_eigs, new_spin_1_MO_coefficient_matrix = sp.linalg.eigh(a=spin_fock_matrix_1, b=AO_overlap_matrix)

    return spin_0_diag_eigs, spin_1_diag_eigs, new_spin_0_MO_coefficient_matrix, new_spin_1_MO_coefficient_matrix

def create_and_solve_roothaan(guess_spin_0_MO_coefficient_matrix, guess_spin_1_MO_coefficient_matrix, core_hamiltonian_matrix, two_electron_repulsion_integral_tensor, AO_overlap_matrix, AO_basis):
    #F^sC^s = SC^s\epsilon^s, where all are matrices. F is fock matrix, C is matrix of eigenvectors, and \epsilon is diagonal matrix with the eigenvalues. Note that S, the overlap, doesn't depend on spin,
    spin_0_density_matrix, spin_1_density_matrix = create_density_matrices(guess_spin_0_MO_coefficient_matrix, guess_spin_1_MO_coefficient_matrix)
    spin_fock_matrix_0, spin_fock_matrix_1 = create_fock_matrices(spin_0_density_matrix, spin_1_density_matrix, core_hamiltonian_matrix, two_electron_repulsion_integral_tensor, AO_basis)

    return solve_roothan(spin_fock_matrix_0, spin_fock_matrix_1, AO_overlap_matrix)

def hartree_fock_method(AO_basis, nuclear_charges, nuclear_positions, integration_window, fixed_point_iterations=100, integration_sample_size=4e4, stages=5, bin_num_vector=100, laplace_dx=.00001, parallel_pool=None, disable_hartree_fock_fixed_point_iterations_pbar=False):
    if parallel_pool == None:
        parallel_pool = joblib.Parallel(n_jobs=8, verbose=VERBOSITY, batch_size=4096, prefer='threads')
    
    core_hamiltonian_matrix, AO_overlap_matrix = create_core_hamiltonian_and_overlap_matrix(AO_basis, nuclear_charges, nuclear_positions, integration_window=integration_window, integration_sample_size=integration_sample_size, stages=stages, bin_num_vector=bin_num_vector, laplace_dx=.0001, parallel_pool=parallel_pool)
    #breakpoint()
    #print(core_hamiltonian_matrix)
    #print(AO_overlap_matrix)
    
    two_electron_repulsion_integral_tensor = create_two_electron_repulsion_integral_tensor(AO_basis, nuclear_positions, integration_window, integration_sample_size=integration_sample_size, stages=stages, bin_num_vector=bin_num_vector, parallel_pool=None)
    
    #print(two_electron_repulsion_integral_tensor)
    #breakpoint()
    initial_fock_matrix = core_hamiltonian_matrix
    history = []
    breakpoint()
    this_spin_0_diag_eigs, this_spin_1_diag_eigs, this_spin_0_MO_coefficient_matrix, this_spin_1_MO_coefficient_matrix = solve_roothan(initial_fock_matrix, initial_fock_matrix, AO_overlap_matrix)
    for i in tqdm(range(fixed_point_iterations), desc="Iterating Hartree-Fock Molecular Orbitals...", disable=disable_hartree_fock_fixed_point_iterations_pbar):
        history.append([this_spin_0_diag_eigs, this_spin_1_diag_eigs, this_spin_0_MO_coefficient_matrix, this_spin_1_MO_coefficient_matrix])
        this_spin_0_density_matrix, this_spin_1_density_matrix = create_density_matrices(this_spin_0_MO_coefficient_matrix, this_spin_1_MO_coefficient_matrix)
        spin_fock_matrix_0, spin_fock_matrix_1 = create_fock_matrices(this_spin_0_density_matrix, this_spin_1_density_matrix, core_hamiltonian_matrix, two_electron_repulsion_integral_tensor, AO_basis)
        this_spin_0_diag_eigs, this_spin_1_diag_eigs, this_spin_0_MO_coefficient_matrix, this_spin_1_MO_coefficient_matrix = solve_roothan(spin_fock_matrix_0, spin_fock_matrix_1, AO_overlap_matrix)
        #print(this_spin_0_diag_eigs, this_spin_1_diag_eigs,)
        #print(this_spin_0_MO_coefficient_matrix, this_spin_1_MO_coefficient_matrix)

    return this_spin_0_diag_eigs, this_spin_1_diag_eigs, this_spin_0_MO_coefficient_matrix, this_spin_1_MO_coefficient_matrix, history

def H2_Test():
    # 1.40104295
    bond_length = 1.40104295
    nuclear_positions = np.array([[-bond_length/2,0,0], [bond_length/2,0,0]])
    nuclear_charges = np.array([1, 1])
    
    integration_window = np.array([[-2-bond_length/2, 2+bond_length/2], [-2-bond_length/2, 2+bond_length/2], [-2-bond_length/2, 2+bond_length/2]])
    
    AO_basis = assemble_AO_basis_from_nuclear_data(nuclear_charges, nuclear_positions, principal_quantum_numbers_to_consider=1)
    fixed_point_iterations=5
    integration_sample_size=1e5
    stages=5
    bin_num_vector=1000
    
    return hartree_fock_method(AO_basis, nuclear_charges, nuclear_positions, integration_window, fixed_point_iterations=fixed_point_iterations, integration_sample_size=integration_sample_size, stages=stages, bin_num_vector=bin_num_vector, laplace_dx=.0001, disable_hartree_fock_fixed_point_iterations_pbar=False)

def H_HE_Test():
    # 1.5117
    bond_length = 1.5117
    nuclear_positions = np.array([[-bond_length/2,0,0], [bond_length/2,0,0]])
    nuclear_charges = np.array([1, 2])
    
    integration_window = np.array([[-2-bond_length/2, 2+bond_length/2], [-2-bond_length/2, 2+bond_length/2], [-2-bond_length/2, 2+bond_length/2]])
    AO_basis = assemble_AO_basis_from_nuclear_data(nuclear_charges, nuclear_positions, principal_quantum_numbers_to_consider=1)
    
    fixed_point_iterations=5
    integration_sample_size=1e4
    stages=5
    bin_num_vector=1e3
    
    #breakpoint()
    return hartree_fock_method(AO_basis, nuclear_charges, nuclear_positions, integration_window, fixed_point_iterations=fixed_point_iterations, integration_sample_size=integration_sample_size, stages=stages, bin_num_vector=bin_num_vector, laplace_dx=.0001, disable_hartree_fock_fixed_point_iterations_pbar=False)


#breakpoint()
#out1 = H2_Test()
out2=H_HE_Test()