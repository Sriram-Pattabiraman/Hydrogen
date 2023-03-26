# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 09:34:19 2023

@author: Sri
"""


from tqdm import tqdm

import math
import numpy as np

import itertools


import sturm_liouville as sl
import utility_funcs as util
from utility_funcs import interleave
import monte_carlo


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


def Solve_For_Eigens(problem_funcs, x_start=None, x_end=None, baked_x_mesh_override=None, mesh_dx=.05, which_basis_init_vector=0, stop_at_candidate_roots_num_thresh=5, get_eigens_up_to_n=3, potentially_ad_hoc_start_eigen_index=0, liouville_n=2, dx=.05, bisect_tol=.1, parallel_pool=None, force_monotone=False, force_monotone_start_val=-.126, stepping_from_anchor_dx=.001, candidate_root_start_val_guess=-.6, displacement_mag_quit_thresh=30, disable_candidate_roots_pbar=False):
    basis_init_vectors = [0,1], [1,0]

    if parallel_pool is None:
        parallel_pool = joblib.Parallel(n_jobs=8, verbose=VERBOSITY, batch_size=4096, prefer='threads')
    if baked_x_mesh_override is None:
        if (not (x_start is None)) and (not (x_end is None)):
            baked_x_mesh = np.arange(x_start, x_end, mesh_dx)
            baked_x_mesh = sl.Make_And_Bake_Potential_Of_X_Double_Coordinate_Mesh_Given_Original_Problem(*problem_funcs, baked_x_mesh, liouville_n=liouville_n, dx=dx)
        else:
            raise TypeError("Solve_For_Eigens needs either a x_start and x_end, or a baked_x_mesh_override!")
    else:
        baked_x_mesh = baked_x_mesh_override

    baked_mesh, mis_eigen = sl.CPM_Method_Liouville_Mismatch(*problem_funcs, baked_x_mesh, basis_init_vectors[which_basis_init_vector], basis_init_vectors[which_basis_init_vector], mesh_was_already_baked=True, liouville_n=liouville_n, dx=dx, parallel_pool=parallel_pool, force_monotone=force_monotone, force_monotone_start_val=force_monotone_start_val, stepping_from_anchor_dx=stepping_from_anchor_dx)
    #breakpoint()
    if stop_at_candidate_roots_num_thresh==1: #only doing one root means we just use the non-stability-checking root finding method
        get_tailored_prufer = lambda index: lambda lambda_: mis_eigen(lambda_)[1] - index
        guess = 0
        stable_root = lambda index: sl.Secant_Method(get_tailored_prufer(index), guess, guess+.01, bisect_tol, f_index_to_rootfind=False)
        #stable_root = lambda index: sl.Monotone_Root_Find(get_tailored_prufer(index), 0, bisect_tol, initial_step=dx)[0]
    else:
        stable_root = lambda index: next(sl.find_stable_roots_in_mis_and_cpm_prufer(mis_eigen, index, stop_at_candidate_roots_num_thresh=stop_at_candidate_roots_num_thresh, candidate_root_start_val_guess=candidate_root_start_val_guess, disable_candidate_roots_pbar=disable_candidate_roots_pbar, displacement_mag_quit_thresh=displacement_mag_quit_thresh))[0]
    #breakpoint()
    eigens = (stable_root(i) for i in tqdm(range(potentially_ad_hoc_start_eigen_index, get_eigens_up_to_n), desc='Finding Eigens...'))
    for eigen in eigens:
        if eigen != None and not np.isnan(eigen):
            yield eigen

def Make_Eigen_Func_Given_Eigen(problem_funcs, x_start, x_end, dx=.0001, which_basis_init_vector=0, asymptotic_clamping=False, disable_simple_ivp_pbar=True, parallel_pool=None):

    if parallel_pool is None:
        parallel_pool = joblib.Parallel(n_jobs=8, verbose=VERBOSITY, batch_size=4096, prefer='threads')

    def Eigen_Func_Given_Eigen(lambda_, which_basis_init_vector=which_basis_init_vector, dx=dx, asymptotic_clamping=asymptotic_clamping, disable_simple_ivp_pbar=disable_simple_ivp_pbar, parallel_pool=parallel_pool):
        basis_init_vectors = [0,1], [1,0]
        with parallel_pool as pool:
            return sl.Solve_LSLP_IVP(lambda_, *problem_funcs, x_start, *basis_init_vectors[which_basis_init_vector], x_end, dx, asymptotic_clamping=asymptotic_clamping, parallel_pool=pool, disable_pbar=disable_simple_ivp_pbar, store_solution=True)[3]
    return Eigen_Func_Given_Eigen

def unravel_eigens(eigens_for_each_coord, parallel_pool=None): #this function looks very simple. it's actually a little sophisticated. hopefully, it's easy to understand even if it was tricky to write. the basic idea is to run through the states of the past coords, and put into a new state list  (*past state, new_possible_coord) for every new_possible_coord in prob.pop()(past state)
    if parallel_pool is None:
        parallel_pool = joblib.Parallel(n_jobs=8, verbose=VERBOSITY, batch_size=4096, prefer='threads')
    states = [()]
    pbar = tqdm(total=len(eigens_for_each_coord), desc='Unraveling...')
    while len(eigens_for_each_coord) > 0 :
        new_states = []
        this_func = eigens_for_each_coord.pop(0)
        for state in states:
            with parallel_pool as pool:
                for next_coord in this_func(*state, parallel_pool=pool):
                    if abs(next_coord) == np.inf or np.isnan(next_coord) or next_coord is None:
                        continue
                    out_val = (*state, next_coord)
                    new_states.append(out_val)
                    if len(eigens_for_each_coord) == 0:
                        yield out_val

        pbar.update(1)
        states = new_states
    
'''
def find_surrounding_indice_in_mono_arr(val, arr, use_ends_of_arr_if_not_in_arr=True):
    for i in range(len(arr)):
        arr_val = arr[i]
        if i == 0:
            if val <= arr_val:
                if use_ends_of_arr_if_not_in_arr:
                    return [0, 0]
                else:
                    raise ValueError("Val not in arr!")
        else:
            if val <= arr_val:
                return (i-1, i)
    else:
        if use_ends_of_arr_if_not_in_arr:
            return [len(arr)-1, len(arr)-1]
        else:
            raise ValueError("Val not in arr!")
'''


def lerp(time, point_0, point_1):
    try:
        right_len = bool(len(point_0) == len(point_1))
    except TypeError:
        out = time * point_0 + (1-time) * point_1
        return out

    if right_len:
        out = []
        for i in range(len(point_0)):
            out[i] = time * point_0[i] + (1-time) * point_1[i]
        return out
    else:
        raise(ValueError("point_0 and point_1 have different lengths!"))

def interp(dim, points, time_vector, points_indice=0, base_interp=lerp):
    if dim > 1:
        int_0 = interp(dim-1, points, time_vector, points_indice=points_indice, base_interp=base_interp)
        int_1 = interp(dim-1, points, time_vector, points_indice=points_indice + 2**(dim-1), base_interp=base_interp)
        return base_interp(time_vector[dim-1], int_0, int_1)
    elif dim == 1:
        return base_interp(time_vector[0], points[points_indice], points[points_indice+1])

def interp_in_coord_out_arr(coord_out_arr):
    if type(coord_out_arr) != np.ndarray:
        coord_out_arr = np.array(coord_out_arr)
    
    def out_interp(coord):
        surrounding_indices_func = util.find_indices_given_coords([coord], [coord_out_arr[:,0]])[0]
        left_coord, left_val = coord_out_arr[surrounding_indices_func[0]]
        right_coord, right_val = coord_out_arr[surrounding_indices_func[1]]
        if right_coord - left_coord == 0:
            return left_val
        else:
            time = (coord - left_coord) / (right_coord - left_coord)
        return lerp(time, left_val, right_val)
    return out_interp

class LerpedFunc:
    def __init__(self, func, distance_thresh=.01, known_coords=None, known_func_vals=None):
        self.base_func = func
        self.distance_thresh = distance_thresh
        self.known_coords = known_coords
        self.known_func_vals = known_func_vals
        
    
    def __call__(self, *args):
        coord = args
        total_dim = len(coord)
        if self.known_coords == None:
            self.known_coords = np.array()
          
        
        this_dim = 0
        for coord_component in coord:
            #this_coord_array = self.known_coords[:, this_dim]
            this_allowed_range = np.apply_along_axis(lambda one_dim_coord: np.searchsorted(one_dim_coord, ), )
            
            this_dim += 1
        
        
    

def Make_Make_Total_Eigen_Func_Given_Eigens_Given_Component_Eigen_Funcs(eigen_funcs_for_each_coord, total_out_given_components=lambda comps: np.prod(comps)):
    eigen_funcs_for_each_coord = np.array(eigen_funcs_for_each_coord)
    def Make_Total_Eigen_Func_Given_Eigens(*eigens, eigen_funcs_for_each_coord=eigen_funcs_for_each_coord):
        all_outs = []
        all_coords = []
        for i in range(len(eigen_funcs_for_each_coord)):
            this_func = np.array(eigen_funcs_for_each_coord[i](*eigens[:i])(eigens[i]), dtype='float64')
            all_coords.append(this_func[:,0])
            all_outs.append(this_func[:,1])
        def Total_Eigen_Func(*position_vect, eigens=eigens, eigen_funcs_for_each_coord=eigen_funcs_for_each_coord, all_coords=all_coords, all_outs=all_outs):
            if len(eigens) < len(eigen_funcs_for_each_coord):
                raise(TypeError(f"Vector_Eigen missing {len(eigen_funcs_for_each_coord) - len(eigens)} required positional arguments"))
            elif len(eigens) > len(eigen_funcs_for_each_coord):
                raise(TypeError(f"Vector_Eigen takes {len(eigen_funcs_for_each_coord)} positional arguments but {len(eigens)} were given"))
                
            #breakpoint()    

            all_surrounding_indices = []
            for i in range(len(eigens)):
                this_surrounding_indice = util.find_surrounding_indice_in_mono_arr(position_vect[i], all_coords[i])
                all_surrounding_indices.append(this_surrounding_indice)

            surrounding_indices_vects = itertools.product(*all_surrounding_indices)
            #surrounding_coords_vects = [all_coords[surrounding_indice_vect] for surrounding_indice_vect in surrounding_indices_vects]
            surrounding_outs_vects = [] #not actually a vect in this case. thankfully, it's better to be accidentally  too general than accidentally too specific.
            for surrounding_indice_vect in surrounding_indices_vects:
                components = []
                for i in range(len(surrounding_indice_vect)):
                    components.append(all_outs[i][surrounding_indice_vect[i]])
                surrounding_outs_vects.append(total_out_given_components(components))
            #surrounding_outs_vects = [total_out_given_components([all_outs[i] for i in surrounding_indice_vect]) for surrounding_indice_vect in surrounding_indices_vects]
            time_vect = []
            for i in range(len(all_surrounding_indices)):
                time_vect.append( (position_vect[i] - all_coords[i][all_surrounding_indices[i][0]]) / (all_coords[i][all_surrounding_indices[i][1]] - all_coords[i][all_surrounding_indices[i][0]]) )
            return interp(len(all_surrounding_indices), surrounding_outs_vects, time_vect) #!!!figure out the issue with this still being slow??
        return Total_Eigen_Func
    return Make_Total_Eigen_Func_Given_Eigens



def scalar_3D_plot(coord_ranges, scalar_func, color_func=lambda normed_val: np.array(colormaps.ScalarMappable(cmap='Blues').to_rgba(normed_val, alpha=1, norm=False))*255, coord_system_of_ranges='cartesian', coord_system_of_func='spherical', func_args_as_direct_vect_arr=False, make_opengl_graphic=False, make_color_data=False, make_mag_data=False):
    len1, len2, len3 = len(coord_ranges[0]), len(coord_ranges[1]), len(coord_ranges[2])
    total_length = len1*len2*len3

    coord_range_vects = itertools.product(*coord_ranges)
    coord_indices = []
    coord_range_vects_with_valid_domain = []
    function_outs = []
    coord_and_func_outs = []
    pbar = tqdm(total=total_length, desc="3D Func Calculating...") #!!!parallelize this?
    try:
        for (i,j,k) in interleave(list(itertools.product(list(range(len1)), list(range(len2)), list(range(len3))))):
            if ( (i+1)*len2*len3 + (j+1)*len3 + (k+1) ) % 10000 == 0:
                print(f"Num of Good (not None or NaN): {len(function_outs)}")
            in_coord = [coord_ranges[0][i], coord_ranges[1][j], coord_ranges[2][k]]
            func_in_coord = util.convert_coord(in_coord, from_system=coord_system_of_ranges, out_system=coord_system_of_func)
            if func_in_coord is None or np.isnan(func_in_coord).any():
                pbar.update(1)
                continue
            else:
                if not func_args_as_direct_vect_arr:
                    function_out = scalar_func(*func_in_coord)
                else:
                    function_out = scalar_func(func_in_coord)

                if function_out is None or np.isnan(function_out).any():
                    pbar.update(1)
                    continue
                else:
                    function_outs.append(function_out)
                    coord_range_vects_with_valid_domain.append(in_coord)
                    coord_indices.append([i,j,k])

                    coord_and_func_outs.append([*in_coord, function_out])
                    pbar.update(1)
    except KeyboardInterrupt:
        #to be safe, discard the untrustworthy last values
        function_outs.pop()
        coord_range_vects_with_valid_domain.pop()
        coord_indices.pop()

    '''
    for comp_1 in coord_ranges[0]:
        j = 0
        for comp_2 in coord_ranges[1]:
            k = 0
            for comp_3 in coord_ranges[2]:
                in_coord = [comp_1, comp_2, comp_3]
                func_in_coord = convert_coord(in_coord, from_system=coord_system_of_ranges, out_system=coord_system_of_func)
                if func_in_coord is None or np.isnan(func_in_coord).any():
                    k += 1
                    pbar.update(1)
                    continue
                else:
                    if not func_args_as_direct_arr:
                        function_out = scalar_func(*func_in_coord)
                    else:
                        function_out = scalar_func(func_in_coord)

                    if function_out is None or np.isnan(function_out).any():
                        k += 1
                        pbar.update(1)
                        continue
                    else:
                        function_outs.append(function_out)
                        coord_range_vects_with_valid_domain.append(in_coord)
                        coord_indices.append([i,j,k])
                        pbar.update(1)
                        k += 1
            j += 1
        i += 1
        '''

    max_func_out = np.max(function_outs)
    min_func_out = np.min(function_outs)
    norm_val = lambda func_out: func_out#lambda func_out: (func_out - min_func_out)/(max_func_out - min_func_out)
    if make_color_data:
        color_data = np.zeros((len1, len2, len3, 4), dtype=np.uint8)
    if make_mag_data:
        mag_data = np.zeros((len1, len2, len3), dtype=np.float64)
    for vect_n in range(len(coord_indices)):
        i,j,k = coord_indices[vect_n]

        normed_val = norm_val(function_outs[vect_n])
        if make_color_data:
            color_data[i,j,k] = color_func(normed_val)
        if make_mag_data:
            mag_data[i,j,k] = normed_val

    #breakpoint()
    if make_opengl_graphic:
        glvw = gl.GLViewWidget()
        vol = gl.GLVolumeItem(color_data, sliceDensity=1)
        glvw.addItem(vol) #!!!todo - fixy fix
    
    out_list = []
    if make_opengl_graphic:
        out_list.extend([glvw, vol, color_data])
    elif make_color_data:
        out_list.append(color_data)
    elif make_mag_data:
        out_list.append(mag_data)
    out_list.extend([coord_range_vects_with_valid_domain, function_outs, coord_and_func_outs])
    return out_list



def plot3D_point_list(points, fig=None, ax=None, alpha=.5, plotting_window=[ [-5, 5], [-5, 5], [-5, 5] ]):
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    if type(points) != np.ndarray:
        points = np.array(points)
    
    ax.scatter(points[:,0], points[:,1], points[:,2], alpha=alpha, s=.5)
    ax.axes.set_xlim3d(*plotting_window[0])
    ax.axes.set_ylim3d(*plotting_window[1]) 
    ax.axes.set_zlim3d(*plotting_window[2]) 
    return fig, ax
    
def point_cloud_plot(un_normed_probability_density, metropolis_starting_window=[[-1,1], [-1, 1], [-1, 1]], func_coord_system='spherical', window_coord_system='cartesian', num_points=100, alpha=.01, parallel_pool=None, fig=None, ax=None, plot_title=None, filename=None):
    if parallel_pool is None:
        parallel_pool = joblib.Parallel(n_jobs=8, verbose=VERBOSITY, batch_size=4096, prefer='threads')

    metropolis_starting_window = np.array(metropolis_starting_window)
    if window_coord_system == 'cartesian':
        cartesian_un_normed_prob_density = lambda x,y,z: un_normed_probability_density(*util.convert_coord([x,y,z], from_system='cartesian', out_system=func_coord_system))
        
        num_of_short_runs = parallel_pool.n_jobs
        run_time_of_a_short_run = math.ceil(num_points/num_of_short_runs)
        
        short_runs_generator = ( joblib.delayed(lambda run_indice: list(monte_carlo.metropolis_hastings(cartesian_un_normed_prob_density, starting_window=metropolis_starting_window, run_time=run_time_of_a_short_run, pbar_desc_id=run_indice, only_desc_0th_id=True, disable_metropolis_pbar=False)))(run_indice) for run_indice in range(num_of_short_runs) )
        points_from_runs = parallel_pool(short_runs_generator)
        points = np.concatenate(points_from_runs, )
        fig, ax = plot3D_point_list(points, fig=fig, ax=ax, alpha=alpha, plotting_window=metropolis_starting_window)
        ax.set_title(f"{plot_title}")
        if filename is not None:
            fig.savefig(f"Images/Point_Clouds/{filename}.png")
            fig.savefig(f"Images/Point_Clouds/{filename}.svg")
        return fig,ax
            



def vector_eigen_for_choice_of_basis_init_wronsks(which_basis_init_wronsks=[0,0,0], do_point_cloud_plot=True, do_radial_distro_plot=False, parallel_pool=None): #!!!generalize!
    if parallel_pool is None:
        parallel_pool = joblib.Parallel(n_jobs=8, verbose=VERBOSITY, batch_size=4096, prefer='threads')
      
    #azimuthal (magnetic quantum numbers) equation
    azi_problem = [lambda x: 1, lambda x: 0, lambda x: 1]
    azi_dx = .001
    bisect_tol = .001
    azi_mesh = np.arange(0, 2*math.pi, azi_dx)
    baked_azi_mesh = sl.Make_And_Bake_Potential_Of_X_Double_Coordinate_Mesh_Given_Original_Problem(*azi_problem, azi_mesh, dx=azi_dx)
    which_basis_init_vector_0 = which_basis_init_wronsks[0]
    #breakpoint()
    azi_eigens = lambda *prev_coord_eigens, parallel_pool=parallel_pool: Solve_For_Eigens(azi_problem, baked_x_mesh_override=baked_azi_mesh, which_basis_init_vector=which_basis_init_vector_0, stop_at_candidate_roots_num_thresh=1, potentially_ad_hoc_start_eigen_index=1, dx=azi_dx, bisect_tol=bisect_tol, parallel_pool=parallel_pool, force_monotone=False, force_monotone_start_val=-.1, stepping_from_anchor_dx=.001)
    azi_eigen_func_given_eigen = lambda *prev_coord_eigens: Make_Eigen_Func_Given_Eigen(azi_problem, azi_mesh[0], azi_mesh[-1], which_basis_init_vector=which_basis_init_vector_0, dx=azi_dx*(10**-2), parallel_pool=parallel_pool)


    #theta (azimuthal quantum numbers) equation
    theta_problem_given_azi_eig = lambda *prev_coord_eigens: [lambda x: math.sin(x), lambda x: prev_coord_eigens[0]/math.sin(x), lambda x: math.sin(x)]

    boundary_epsilon_theta = .001
    mesh_dtheta_left = .001
    mesh_theta_mid_left = .2
    mesh_dtheta_mid = .01
    mesh_theta_mid_right = .8
    mesh_dtheta_end = .001
    theta_liouville_sigma_dx = .01
    theta_liouville_n = 2
    bisect_tol = .001
    theta_mesh = np.concatenate(np.array([np.arange(boundary_epsilon_theta, mesh_theta_mid_left, mesh_dtheta_left), np.arange(mesh_theta_mid_left, mesh_theta_mid_right, mesh_dtheta_mid), np.arange(mesh_theta_mid_right, math.pi-boundary_epsilon_theta, mesh_dtheta_end)], dtype=object))
    baked_theta_mesh_given_azi_eig = lambda *prev_coord_eigens: sl.Make_And_Bake_Potential_Of_X_Double_Coordinate_Mesh_Given_Original_Problem(*theta_problem_given_azi_eig(*prev_coord_eigens), theta_mesh, liouville_n=theta_liouville_n, dx=theta_liouville_sigma_dx)
    which_basis_init_vector_1 = which_basis_init_wronsks[1]
    theta_eigens_given_azi_eig = lambda *prev_coord_eigens, parallel_pool=parallel_pool: Solve_For_Eigens(theta_problem_given_azi_eig(*prev_coord_eigens), baked_x_mesh_override=baked_theta_mesh_given_azi_eig(*prev_coord_eigens), which_basis_init_vector=which_basis_init_vector_1, stop_at_candidate_roots_num_thresh=1, potentially_ad_hoc_start_eigen_index=1, liouville_n=theta_liouville_n, dx=theta_liouville_sigma_dx, bisect_tol=bisect_tol, get_eigens_up_to_n=3, parallel_pool=parallel_pool, force_monotone=False, force_monotone_start_val=-.126, stepping_from_anchor_dx=.001)
    theta_eigen_func_given_azi_eig = lambda *prev_coord_eigens: Make_Eigen_Func_Given_Eigen(theta_problem_given_azi_eig(*prev_coord_eigens), theta_mesh[0], theta_mesh[-1], dx=theta_liouville_sigma_dx*(10**-2), which_basis_init_vector=which_basis_init_vector_1, parallel_pool=parallel_pool)


    #radial
    boundary_epsilon_radial = .01
    boundary_inf_approx = 50 #!!!should be at least twice the desired plotting range - note that 50 suffices for eigenfinding but you need the twice the range thing for actually plotting eigenfunctions accurately. perhaps find a way to set two different values for this
    #p_of_r,q_of_r,w_of_r = lambda x: x**2, lambda x: l*(l+1) - ( ((2*reduced_mass*(x**2))/(hbar**2)) * ((electron_charge**2)/(4*math.pi*vaccuum_permittivity*x))  ), lambda x: ((2*reduced_mass*(x**2))/(hbar**2))
    #r_problem_given_theta_eig = lambda *prev_coord_eigens: [lambda x: 1, lambda x: prev_coord_eigens[1]/(x**2) - (1/x), lambda x: 1]
    r_problem_given_theta_eig = lambda *prev_coord_eigens: [lambda x: x**2, lambda x: prev_coord_eigens[1]-2*x, lambda x: 2*(x**2)]
    mesh_dr = .01
    r_mesh = np.arange(boundary_epsilon_radial, boundary_inf_approx, mesh_dr)
    radial_Num_D_sigma_dx = .0001
    r_liouville_n = 2
    baked_radial_mesh_given_theta_eig = lambda *prev_coord_eigens: sl.Make_And_Bake_Potential_Of_X_Double_Coordinate_Mesh_Given_Original_Problem(*r_problem_given_theta_eig(*prev_coord_eigens), r_mesh, liouville_n=r_liouville_n, dx=radial_Num_D_sigma_dx)
    which_basis_init_vector_2 = which_basis_init_wronsks[2]
    bisect_tol = .0001
    force_monotone = True
    rad_stop_at_candidate_roots_num_thresh = 5
    rad_get_eigens_up_to_n = 4
    rad_displacement_mag_quit_thresh = .55

    radial_eigens_given_theta_eig = lambda *prev_coord_eigens, parallel_pool=parallel_pool: Solve_For_Eigens(r_problem_given_theta_eig(*prev_coord_eigens), baked_x_mesh_override=baked_radial_mesh_given_theta_eig(*prev_coord_eigens), stop_at_candidate_roots_num_thresh=rad_stop_at_candidate_roots_num_thresh, get_eigens_up_to_n=rad_get_eigens_up_to_n, dx=radial_Num_D_sigma_dx, which_basis_init_vector=which_basis_init_vector_2, bisect_tol=bisect_tol, parallel_pool=parallel_pool, force_monotone=force_monotone, force_monotone_start_val=-.126, stepping_from_anchor_dx=.001, displacement_mag_quit_thresh=rad_displacement_mag_quit_thresh)
    #breakpoint()
    #radial_eigen_func_given_theta_eig = lambda *prev_coord_eigens: Make_Eigen_Func_Given_Eigen(r_problem_given_theta_eig(*prev_coord_eigens), r_mesh[0], r_mesh[-1], dx=mesh_dr_start, which_basis_init_vector=which_basis_init_vector_2, asymptotic_clamping=True)
    radial_eigen_func_given_theta_eig = lambda *prev_coord_eigens: Make_Eigen_Func_Given_Eigen(r_problem_given_theta_eig(*prev_coord_eigens), r_mesh[0], r_mesh[-1], dx=mesh_dr, which_basis_init_vector=which_basis_init_vector_2, asymptotic_clamping=True, parallel_pool=parallel_pool)
    
    #print("test")
    #breakpoint()
    #breakpoint()
    '''
    m_list = azi_eigens()
    for m in tqdm(m_list):
    '''
    '''
    for m in tqdm(range(-3,3)):
        l_list = theta_eigens_given_azi_eig(m**2)
        for l in tqdm(l_list):
            '''
            
    for l in tqdm(range(0, 3)):
        for m in tqdm(range(-l, l+1)):
            '''
            data_for_r_sph = []
            x_min, x_max, y_min, y_max = 0, 2*math.pi, 0, math.pi
            for i in np.arange(0, 2*math.pi, .1):
                sub=[]
                for j in np.arange(0, 1*math.pi, .1):
                    sub.append(hopefully_real_sph_harm(i,j))
                data_for_r_sph.append(sub)
            data_for_r_sph = np.array(data_for_r_sph)
            data_for_r_sph = data_for_r_sph.transpose()
            data_for_r_sph = np.flip(data_for_r_sph, axis=0)
            data_for_r_sph.clip(-15, 15, out=data_for_r_sph)
            fig, ax = plt.subplots()
            axImage = ax.imshow(data_for_r_sph, extent=[x_min, x_max, y_min, y_max], cmap='viridis')
            fig.colorbar(axImage, ax=ax)
            fig.savefig(f"Images/Hopefully_Correct_Real_Spherical_Harmonics_{which_basis_init_wronsks[0]}_{which_basis_init_wronsks[1]}/l={l}_m={m}")
            plt.close()
            '''
        
            energy_list = radial_eigens_given_theta_eig(m**2, l*(l+1))
            n = l+1
            #breakpoint()
            print(n,l,m)
            for energy in energy_list:
                azi_func = interp_in_coord_out_arr(azi_eigen_func_given_eigen()(m**2))
                #breakpoint()
                theta_func = interp_in_coord_out_arr(theta_eigen_func_given_azi_eig(m**2)(l*(l+1)))
                #breakpoint()
                hopefully_real_sph_harm = lambda phi, theta: azi_func(phi) * theta_func(theta)
                radial_func = interp_in_coord_out_arr(np.array(radial_eigen_func_given_theta_eig(m**2,l*(l+1))(energy)))
                hopefully_hydrogen_wave_function = lambda r, phi, theta: hopefully_real_sph_harm(phi, theta) * radial_func(r)
                #breakpoint()
                end_of_radius = min((n**2)+l+1, (40))
                window = [[-end_of_radius, end_of_radius],[-end_of_radius, end_of_radius],[-end_of_radius, end_of_radius]]
                un_normed_probability_density = lambda r, phi, theta: hopefully_hydrogen_wave_function(r, phi, theta)**2
                
                if do_point_cloud_plot:
                    plot_title = f"Orbital_{n}_{l}_{m}, energy={round(energy, 10)} hartrees"
                    point_cloud_plot(un_normed_probability_density, metropolis_starting_window=window, func_coord_system='spherical', window_coord_system='cartesian', num_points=100000, alpha=.01, plot_title=plot_title, filename=f'Orbital_{n}_{l}_{m}', parallel_pool=parallel_pool)
                    print('check result!')
                elif do_radial_distro_plot:
                    fig, ax = plt.subplots()
                    ax.set_title(f"n,l,m={n},{l},{m}; energy={round(energy, 10)} hartrees")
                    ax.plot(np.arange(0,end_of_radius,.0001), [(radial_func(r)**2) * (r**2) for r in np.arange(0,end_of_radius,.0001)], '.')
                    fig.savefig(f"Images/Radial_Probability_Densities/Rad_Prob_{n}_{l}_{m}.png")
                    fig.savefig(f"Images/Radial_Probability_Densities/Rad_Prob_{n}_{l}_{m}.svg")
                    
                else:
                    #testing1 = [(radial_func(r)**2) * (r**2) for r in np.arange(0,end_of_radius,.0001)]
                    #testing2 = point_cloud_plot(un_normed_probability_density, metropolis_starting_window=window, func_coord_system='spherical', window_coord_system='cartesian', num_points=10000, alpha=.1, filename=None)
                    pass

                
                #scalar_3D_plot([np.arange(-1, 1, .1), np.arange(-1, 1, .1), np.arange(-1, 1, .1)], hopefully_hydrogen_wave_function, )
                n+=1

    
    

    



parallel_pool = joblib.Parallel(n_jobs=8, verbose=VERBOSITY, batch_size=4096, prefer='threads')
#o1=vector_eigen_for_choice_of_basis_init_wronsks(which_basis_init_wronsks=[0,0,0], parallel_pool=parallel_pool) 
#o2=vector_eigen_for_choice_of_basis_init_wronsks(which_basis_init_wronsks=[0,1,0], parallel_pool=parallel_pool) 
#o3=vector_eigen_for_choice_of_basis_init_wronsks(which_basis_init_wronsks=[1,0,0], parallel_pool=parallel_pool) 
o4=vector_eigen_for_choice_of_basis_init_wronsks(which_basis_init_wronsks=[1,1,0], parallel_pool=parallel_pool) 
#breakpoint()
#out_func = vector_eigen_for_choice_of_basis_init_wronsks()[1](0,0,-.0625)
#out_val = out_func(1,1,1)


#out_list = scalar_3D_plot([np.arange(-2,2,.1),np.arange(-2,2,.1),np.arange(-2,2,.1)], out_func)
#out_list = scalar_3D_plot([np.arange(0,2*math.pi,1),np.arange(0,math.pi,1),np.arange(0,1,.1)], out_func, coord_system_of_ranges='spherical')

print("done!")
#w.show()
#app.exec()

'''
#breakpoint()
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X,Y,Z = np.mgrid[:11,:11,:11]
ax.scatter(X, Y, Z, c=mag_data.ravel(), cmap=plt.get_cmap("Greys"), depthshade=True)
fig.add_axes(ax)
'''


