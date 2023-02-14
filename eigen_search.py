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
from interleaver import Interleave

from scipy import constants

import pyqtgraph as pg
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


def Solve_For_Eigens(problem_funcs, x_start=None, x_end=None, baked_x_mesh_override=None, mesh_dx=.05, which_basis_init_vector=0, stop_at_candidate_roots_num_thresh=3, potentially_ad_hoc_start_eigen_index=0, get_eigens_up_to_n=3, dx=.05, bisect_tol=.1, parallel_pool=None):
    basis_init_vectors = [0,1], [1,0]

    if parallel_pool is None:
        parallel_pool = joblib.Parallel(n_jobs=6, verbose=VERBOSITY, batch_size=4096)
    if baked_x_mesh_override is None:
        if (not (x_start is None)) and (not (x_end is None)):
            baked_x_mesh = np.arange(x_start, x_end, mesh_dx)
            baked_x_mesh = sl.Make_And_Bake_Potential_Of_X_Double_Coordinate_Mesh_Given_Original_Problem(*problem_funcs, baked_x_mesh, dx=dx)
        else:
            raise TypeError("Solve_For_Eigens needs either a x_start and x_end, or a baked_x_mesh_override!")
    else:
        baked_x_mesh = baked_x_mesh_override

    baked_mesh, mis_eigen = sl.CPM_Method_Liouville_Mismatch(*problem_funcs, baked_x_mesh, basis_init_vectors[which_basis_init_vector], basis_init_vectors[which_basis_init_vector], mesh_was_already_baked=True, dx=dx, parallel_pool=parallel_pool)
    if stop_at_candidate_roots_num_thresh==1: #only doing one root means we just use the non-stability-checking root finding method
        get_tailored_prufer = lambda index: lambda lambda_: mis_eigen(lambda_)[1] - index
        guess = 0
        stable_root = lambda index: sl.Secant_Method(get_tailored_prufer(index), guess, guess+.01, bisect_tol, f_index_to_rootfind=False)
        #stable_root = lambda index: sl.Monotone_Root_Find(get_tailored_prufer(index), 0, bisect_tol, initial_step=dx)[0]
    else:
        stable_root = lambda index: sl.find_stable_roots_in_mis_and_cpm_prufer(mis_eigen, index, stop_at_candidate_roots_num_thresh=stop_at_candidate_roots_num_thresh)[0][0]
    #breakpoint()
    eigens = [stable_root(i) for i in tqdm(range(potentially_ad_hoc_start_eigen_index, get_eigens_up_to_n))]
    eigens = list(filter( lambda x: (x!=None) and (not np.isnan(x)) , eigens))
    return eigens

def Make_Eigen_Func_Given_Eigen(problem_funcs, x_start, x_end, dx=.0001, which_basis_init_vector=0, parallel_pool=None):
    basis_init_vectors = [0,1], [1,0]

    if parallel_pool is None:
        parallel_pool = joblib.Parallel(n_jobs=6, verbose=VERBOSITY, batch_size=4096)

    def Eigen_Func_Given_Eigen(lambda_, which_basis_init_vector=which_basis_init_vector, parallel_pool=parallel_pool):
        with parallel_pool as pool:
            return sl.Solve_LSLP_IVP(lambda_, *problem_funcs, x_start, *basis_init_vectors[which_basis_init_vector], x_end, dx, parallel_pool=pool, store_solution=True)[3]

    return Eigen_Func_Given_Eigen

def unravel_eigens(eigens_for_each_coord, parallel_pool=None): #this function looks very simple. it's actually a little sophisticated. hopefully, it's easy to understand even if it was tricky to write. the basic idea is to run through the states of the past coords, and put into a new state list  (*past state, new_possible_coord) for every new_possible_coord in prob.pop()(past state)
    if parallel_pool is None:
        parallel_pool = joblib.Parallel(n_jobs=6, verbose=VERBOSITY, batch_size=4096)
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
                    yield out_val

        pbar.update(1)
        states = new_states

def find_surrounding_indice_in_mono_arr(val, arr, use_ends_of_arr_if_not_in_arr=True):
    for i in range(len(arr)):
        arr_val = arr[i]
        if i == 0:
            if val < arr_val:
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

def find_indices_given_coords(coords, arrs):
    return [find_surrounding_indice_in_mono_arr(coords[i], arrs[i]) for i in range(len(coords))]

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

def Make_Make_Total_Eigen_Func_Given_Eigens_Given_Component_Eigen_Funcs(eigen_funcs_for_each_coord, total_out_given_components=lambda comps: np.prod(comps)):
    eigen_funcs_for_each_coord = np.array(eigen_funcs_for_each_coord)
    def Make_Total_Eigen_Func_Given_Eigens(*eigens, eigen_funcs_for_each_coord=eigen_funcs_for_each_coord):
        def Total_Eigen_Func(*position_vect, eigens=eigens, eigen_funcs_for_each_coord=eigen_funcs_for_each_coord):
            if len(eigens) < len(eigen_funcs_for_each_coord):
                raise(TypeError(f"Vector_Eigen missing {len(eigen_funcs_for_each_coord) - len(eigens)} required positional arguments"))
            elif len(eigens) > len(eigen_funcs_for_each_coord):
                raise(TypeError(f"Vector_Eigen takes {len(eigen_funcs_for_each_coord)} positional arguments but {len(eigens)} were given"))


            all_outs = []
            all_coords = []
            for i in range(len(eigen_funcs_for_each_coord)):
                this_func = np.array(eigen_funcs_for_each_coord[i](*eigens[:i])(eigens[i]), dtype='float64')
                all_coords.append(this_func[:,0])
                all_outs.append(this_func[:,1])

            all_surrounding_indices = []
            for i in range(len(eigens)):
                this_surrounding_indice = find_surrounding_indice_in_mono_arr(position_vect[i], all_coords[i])
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


def convert_coord(in_coord, from_system='cartesian', out_system='spherical'):
    if from_system=='cartesian' and out_system=='spherical': #using the ISO convention for physics: azimuth is \phi and inclination is \theta. the azimuth is constricted to [0, 2\pi] and the inclination to [0, \pi]
        x,y,z = in_coord
        if x==0 and y==0:
            return None
        hxy = (x**2 + y**2)**.5
        r = (hxy**2 + z**2)**.5
        if z > 0:
            theta = math.atan(hxy/z)
        elif z < 0:
            theta = math.pi + math.atan(hxy/z)
        elif z==0:
            theta = math.pi/2

        phi = math.atan2(y, x)
        if phi < 0:
            phi += 2*math.pi

        if any([np.isnan(r), np.isnan(theta), np.isnan(phi)]):
            return None
        return [r, theta, phi]

def scalar_3D_plot(coord_ranges, scalar_func, color_func=lambda normed_val: np.array(colormaps.ScalarMappable(cmap='Blues').to_rgba(normed_val, alpha=1, norm=False))*255, coord_system_of_ranges='cartesian', coord_system_of_func='spherical', func_args_as_direct_vect_arr=False):
    len1, len2, len3 = len(coord_ranges[0]), len(coord_ranges[1]), len(coord_ranges[2])
    total_length = len1*len2*len3

    coord_range_vects = itertools.product(*coord_ranges)
    coord_indices = []
    coord_range_vects_with_valid_domain = []
    function_outs = []
    coord_and_func_outs = []
    pbar = tqdm(total=total_length, desc="3D Func Calculating...") #!!!parallelize this?
    try:
        for (i,j,k) in Interleave(list(itertools.product(list(range(len1)), list(range(len2)), list(range(len3))))):
            print(f"Num of Good (not None or NaN): {len(function_outs)}")
            in_coord = [coord_ranges[0][i], coord_ranges[1][j], coord_ranges[2][k]]
            func_in_coord = convert_coord(in_coord, from_system=coord_system_of_ranges, out_system=coord_system_of_func)
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
    norm_val = lambda func_out: (func_out - min_func_out)/(max_func_out - min_func_out)
    color_data = np.zeros((len1, len2, len3, 4), dtype=np.uint8)
    mag_data = np.zeros((len1, len2, len3), dtype=np.float64)
    for vect_n in range(len(coord_indices)):
        i,j,k = coord_indices[vect_n]

        normed_val = norm_val(function_outs[vect_n])
        color_data[i,j,k] = color_func(normed_val)
        mag_data[i,j,k] = normed_val

    #breakpoint()
    #glvw = gl.GLViewWidget()
    #vol = gl.GLVolumeItem(color_data, sliceDensity=10)
    #glvw.addItem(vol)
    glvw, vol = None, None #!!!todo - fixy fix
    return glvw, vol, color_data, mag_data, coord_range_vects_with_valid_domain, function_outs, coord_and_func_outs


def vector_eigen_for_choice_of_basis_init_wronsks(which_basis_init_wronsks=[0,0,0]): #!!!generalize!
    #azimuthal (magnetic quantum numbers) equation
    azi_problem = [lambda x: 1, lambda x: 0, lambda x: 1]
    lazy_small_number = .0001
    bisect_tol = .05
    azi_mesh = np.arange(0, 2*math.pi, lazy_small_number)
    baked_azi_mesh = sl.Make_And_Bake_Potential_Of_X_Double_Coordinate_Mesh_Given_Original_Problem(*azi_problem, azi_mesh, dx=lazy_small_number)
    which_basis_init_vector = which_basis_init_wronsks[0]
    #breakpoint()
    azi_eigens = lambda *prev_coord_eigens, parallel_pool=None: Solve_For_Eigens(azi_problem, baked_x_mesh_override=baked_azi_mesh, which_basis_init_vector=which_basis_init_vector, stop_at_candidate_roots_num_thresh=1, potentially_ad_hoc_start_eigen_index=1, dx=lazy_small_number, bisect_tol=bisect_tol, parallel_pool=parallel_pool)
    azi_eigen_func_given_eigen = lambda *prev_coord_eigens: Make_Eigen_Func_Given_Eigen(azi_problem, azi_mesh[0], azi_mesh[-1], which_basis_init_vector=which_basis_init_vector, dx=lazy_small_number*(10**-2))


    #theta (azimuthal quantum numbers) equation
    theta_problem_given_azi_eig = lambda *prev_coord_eigens: [lambda x: math.sin(x), lambda x: prev_coord_eigens[0]/math.sin(x), lambda x: math.sin(x)]

    boundary_epsilon_theta = .001
    mesh_dtheta_left = .001
    mesh_theta_mid_left = .1
    mesh_dtheta_mid = .01
    mesh_theta_mid_right = .9
    mesh_dtheta_end = .001
    lazy_small_number = .001
    bisect_tol = .01
    theta_mesh = np.concatenate(np.array([np.arange(boundary_epsilon_theta, mesh_theta_mid_left, mesh_dtheta_left), np.arange(mesh_theta_mid_left, mesh_theta_mid_right, mesh_dtheta_mid), np.arange(mesh_theta_mid_right, math.pi-boundary_epsilon_theta, mesh_dtheta_end)], dtype=object))
    baked_theta_mesh_given_azi_eig = lambda *prev_coord_eigens: sl.Make_And_Bake_Potential_Of_X_Double_Coordinate_Mesh_Given_Original_Problem(*theta_problem_given_azi_eig(*prev_coord_eigens), theta_mesh, dx=lazy_small_number)
    which_basis_init_vector = which_basis_init_wronsks[1]
    theta_eigens_given_azi_eig = lambda *prev_coord_eigens, parallel_pool=None: Solve_For_Eigens(theta_problem_given_azi_eig(*prev_coord_eigens), baked_x_mesh_override=baked_theta_mesh_given_azi_eig(*prev_coord_eigens), mesh_dx=lazy_small_number, dx=lazy_small_number, which_basis_init_vector=which_basis_init_vector, bisect_tol=bisect_tol,  get_eigens_up_to_n=3, parallel_pool=parallel_pool)
    theta_eigen_func_given_azi_eig = lambda *prev_coord_eigens: Make_Eigen_Func_Given_Eigen(theta_problem_given_azi_eig(*prev_coord_eigens), theta_mesh[0], theta_mesh[-1], dx=lazy_small_number*(10**-2), which_basis_init_vector=which_basis_init_vector)


    #radial
    electron_mass, proton_mass = constants.electron_mass, constants.proton_mass
    reduced_mass = electron_mass*proton_mass/(electron_mass+proton_mass)
    electron_charge = constants.elementary_charge
    vaccuum_permittivity = constants.epsilon_0
    hbar = constants.hbar

    boundary_epsilon_radial = .001
    mid_r_start = 1
    mid_r_end = 10
    boundary_inf_approx = 100
    #p_of_r,q_of_r,w_of_r = lambda x: x**2, lambda x: l*(l+1) - ( ((2*reduced_mass*(x**2))/(hbar**2)) * ((electron_charge**2)/(4*math.pi*vaccuum_permittivity*x))  ), lambda x: ((2*reduced_mass*(x**2))/(hbar**2))
    r_problem_given_theta_eig = lambda *prev_coord_eigens: [lambda x: 1, lambda x: prev_coord_eigens[1]/(x**2) - (1/x), lambda x: 1]
    mesh_dr_start = .001
    mesh_dr_mid = 1
    mesh_dr_end = .001
    r_mesh_start = np.arange(boundary_epsilon_radial, mid_r_start, mesh_dr_start)
    r_mesh_mid = np.arange(mid_r_start, mid_r_end, mesh_dr_mid)
    r_mesh_end = np.arange(mid_r_end, boundary_inf_approx, mesh_dr_end)
    r_mesh = np.concatenate([r_mesh_start, r_mesh_mid, r_mesh_end])
    Num_D_Liouville_dx = .01
    lazy_small_number = .01
    baked_radial_mesh_given_theta_eig = lambda *prev_coord_eigens: sl.Make_And_Bake_Potential_Of_X_Double_Coordinate_Mesh_Given_Original_Problem(*r_problem_given_theta_eig(*prev_coord_eigens), r_mesh, dx=lazy_small_number)
    which_basis_init_vector = which_basis_init_wronsks[2]
    bisect_tol = .0001
    radial_eigens_given_theta_eig = lambda *prev_coord_eigens, parallel_pool=None: Solve_For_Eigens(r_problem_given_theta_eig(*prev_coord_eigens), baked_x_mesh_override=baked_radial_mesh_given_theta_eig(*prev_coord_eigens), dx=Num_D_Liouville_dx, which_basis_init_vector=which_basis_init_vector, bisect_tol=bisect_tol, parallel_pool=parallel_pool)
    radial_eigen_func_given_theta_eig = lambda *prev_coord_eigens: Make_Eigen_Func_Given_Eigen(r_problem_given_theta_eig(*prev_coord_eigens), r_mesh[0], r_mesh[-1], dx=mesh_dr_start, which_basis_init_vector=which_basis_init_vector)

    eigen_funcs_for_each_coord = [azi_eigen_func_given_eigen, theta_eigen_func_given_azi_eig, radial_eigen_func_given_theta_eig]
    vector_eigen_func = Make_Make_Total_Eigen_Func_Given_Eigens_Given_Component_Eigen_Funcs(eigen_funcs_for_each_coord)
    #eigen_func_for_each_coord = lambda azi_eig, theta_eig, radial_eig: [azi_eigen_func_given_eigen_b1(azi_eig), theta_eigen_func_given_azi_eig_b1(azi_eig)(theta_eig), radial_eigen_func_given_theta_eig_b1(azi_eig, theta_eig)(radial_eig)]
    #prod_func = lambda azi_eig, theta_eig, radial_eig: [eigen_func_for_each_coord(azi_eig, theta_eig, radial_eig)]
    eigens_for_each_coord = [azi_eigens, theta_eigens_given_azi_eig, radial_eigens_given_theta_eig]
    coord_eigens = unravel_eigens(eigens_for_each_coord)
    #breakpoint()
    return coord_eigens, vector_eigen_func

def listify_meshgrids_and_remove_zeros(X, Y, Z, C): #!!! here and earlier, find way to distinguish between default_initialized zero and actual function zero [try initialzing with nans via a function that takes in shape and makes an empty*nan]
    newx, newy, newz = np.zeros(X.shape[0]), np.zeros(Y.shape[1]), np.zeros(Z.shape[2])
    newc = np.zeros(X.shape[0]*Y.shape[1]*Z.shape[2])
    point_indice = 0
    for i,j,k in itertools.product(range(len(X)), range(len(Y)), range(len(Z))):
        if C[i,j,k] != 0:
            newx[point_indice], newy[point_indice], newz[point_indice], newc[point_indice] = X[i,j,k], Y[i,j,k], Z[i,j,k], C[i,j,k]
            point_indice += 1
    return newx, newy, newz, newc

#breakpoint()
out_func = vector_eigen_for_choice_of_basis_init_wronsks()[1](0,2,-.0625)
#out_val = out_func(1,1,1)


w, vol, color_data, mag_data, coord_range_vects_with_valid_domain, function_outs, coord_and_func_outs = scalar_3D_plot([np.arange(-.5,.5,.1),np.arange(-.5,.5,.1),np.arange(-.5,.5,.1)], out_func)
print("done!")
#w.show()
#app.exec()

#breakpoint()
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X,Y,Z = np.mgrid[:10,:10,:10]
ax.scatter(X, Y, Z, c=mag_data.ravel(), cmap=plt.get_cmap("hot"), depthshade=False)
fig.add_axes(ax)
