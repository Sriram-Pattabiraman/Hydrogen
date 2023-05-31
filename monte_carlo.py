# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 11:33:35 2023

@author: Sri
"""


import random
import numpy as np
import scipy as sp
import math
from uncertainties import ufloat
ufloat = np.vectorize(ufloat)

import utility_funcs as util 

import joblib
VERBOSITY=0

from tqdm import tqdm

import matplotlib.pyplot as plt


#metropolis_hastings
def metropolis_hastings(un_normed_probability_density, starting_window=np.array([[-1,1], [-1,1], [-1,1]]), starting_diagonal_covariance=1, run_time=1000, restrict_to_window=True, adjust_covariance=True, yield_prob=False, yield_proposal_point_and_acceptance_prob=False, desc_text="Traversing Markov Chain", pbar_desc_id=None, only_desc_0th_id=False, disable_metropolis_pbar=True):
    if desc_text is None:
        desc_text = "Traversing Markov Chain"
    
    dimension = np.shape(starting_window)[0]
    if type(starting_diagonal_covariance) != np.ndarray:
        if hasattr(starting_diagonal_covariance, '__iter__'):
            starting_diagonal_covariance = np.array(starting_diagonal_covariance)
        else:
            starting_diagonal_covariance = np.repeat(starting_diagonal_covariance, dimension)
    starting_diagonal_covariance = starting_diagonal_covariance.astype('float64', copy=False)

    starting_window = np.array(starting_window)
    volume = np.product(np.diff(starting_window, axis=1))

    
    starting_window_lows, starting_window_highs = starting_window[:,0], starting_window[:,1]
    if restrict_to_window:
        un_normed_probability_density = util.restrict(un_normed_probability_density, restriction_window_lows=starting_window_lows, restriction_window_highs=starting_window_highs)
    else:
        un_normed_probability_density = un_normed_probability_density, restriction_window_lows=starting_window_lows, restriction_window_highs=starting_window_highs
        
        
    starting_state = np.random.default_rng().uniform(low=starting_window_lows, high=starting_window_highs) 
    while np.any(un_normed_probability_density(*starting_state)==0):
        starting_state = np.random.default_rng().uniform(low=starting_window_lows, high=starting_window_highs)

    def make_displacement_proposal_sampler(diagonal_covariance):
        displacement_proposal_sampler = lambda dimension=dimension, diagonal_covariance=diagonal_covariance: np.random.default_rng().multivariate_normal(np.zeros((dimension,)), np.diag(diagonal_covariance))
        return displacement_proposal_sampler
    
    state_list = []
    this_state = starting_state
    prev_state = this_state
    max_maximum_diagonal_covariance = starting_diagonal_covariance.max()*2
    min_minimum_diagonal_covariance = starting_diagonal_covariance.min()*.5
    current_diagonal_covariance = starting_diagonal_covariance
    total_iters = 0
    accept_list_size = 100 #!!!
    accept_list = []
    acceptance_rate = None

    total_num_of_pbar_updates = 20
    pbar_update_amount_per_iter = run_time//total_num_of_pbar_updates
    if pbar_desc_id is None:
        desc = f"{desc_text}..."
    elif only_desc_0th_id:
        if pbar_desc_id != 0:
            disable_metropolis_pbar=True
            desc=''
        else:
            desc=f"{desc_text} 0 (rest hidden)..."
    else:
        desc = f"{desc_text} {pbar_desc_id}..."
        
    pbar = tqdm(total=run_time, desc=desc, disable=disable_metropolis_pbar)
    
    proposal, this_weight, proposal_weight, acceptance_probability = None, None, None, None #not necessary for the code, but my IDE yells at me otherwise
    for i in range(run_time):
        if yield_prob:
            if i == 0:
                yield [this_state, 1/volume]
            else:
                yield [this_state, this_weight if np.all(this_state==prev_state) else proposal_weight]
        elif yield_proposal_point_and_acceptance_prob:
            if i != 0:
                yield [prev_state, proposal, acceptance_probability]
        else:
            yield this_state
            
        if total_iters%pbar_update_amount_per_iter==0 and total_iters!=0:
            pbar.update(pbar_update_amount_per_iter)
            
        total_iters += 1
        state_list.append(this_state)
        
        if adjust_covariance:
            if total_iters > 100 and True:
                if acceptance_rate < .3:
                    if current_diagonal_covariance.min() > min_minimum_diagonal_covariance:
                        current_diagonal_covariance *= .9
                elif acceptance_rate > .4:
                    if current_diagonal_covariance.max() < max_maximum_diagonal_covariance:
                        current_diagonal_covariance *= 1.1
            
        current_displacement_proposal_sampler = make_displacement_proposal_sampler(current_diagonal_covariance) 
        '''
        if restrict_to_window:
            while True:
                proposal = this_state + current_displacement_proposal_sampler()
                if np.all(np.logical_and(np.less_equal(starting_window_lows, proposal), np.less_equal(proposal, starting_window_highs))):
                    break
        else:
            proposal = this_state + current_displacement_proposal_sampler()
        '''
        proposal = this_state + current_displacement_proposal_sampler()
    
        this_weight = un_normed_probability_density(*this_state)
        proposal_weight = un_normed_probability_density(*proposal)
        acceptance_probability = min(1,proposal_weight/this_weight)
            
        if acceptance_probability == 1:
            prev_state = this_state
            this_state = proposal
            if len(accept_list) >= accept_list_size:
                accept_list.pop(0)
            accept_list.append(True)
            acceptance_rate = sum(accept_list)/len(accept_list)
        else:
            random_roll = np.random.default_rng().uniform(low=0,high=1)
            if random_roll <= acceptance_probability:
                prev_state = this_state
                this_state = proposal
                if len(accept_list) >= accept_list_size:
                    accept_list.pop(0)
                accept_list.append(True)
                acceptance_rate = sum(accept_list)/len(accept_list)
            else:
                prev_state=this_state
                if len(accept_list) >= accept_list_size:
                    accept_list.pop(0)
                accept_list.append(False)
                acceptance_rate = sum(accept_list)/len(accept_list)
                continue
    
    return# state_list


#monte_carlo_integration
def monte_carlo_integration(integrand, integration_window, importance_sampling_distro=None, sample_generator=None, find_expected_value=False, use_unnormalized_importance_sampler_correction=False, sample_size=1000, metropolis_starting_diagonal_covariance=1, store_indep_var_history=False, store_integrand_history=False, estimate_error=True, return_error_estimate=False, integrand_coord_system='cartesian', window_coord_system='cartesian', override_desc_text=None, disable_metropolis_pbar=False, parallel_pool=None):
    if return_error_estimate:
        assert(estimate_error)
    
    #if find_expected_value==False, then importance_sampling_distro needs to be normalized or use_unnormalized_importance_sampler_correction must be true
    if parallel_pool is None:
        parallel_pool = joblib.Parallel(n_jobs=8, verbose=VERBOSITY, batch_size=4096, prefer='threads')
    elif parallel_pool == False:
        parallel_pool = joblib.Parallel(n_jobs=1, verbose=VERBOSITY, batch_size=4096, prefer='threads')
        
    if integrand_coord_system != window_coord_system:
        raise NotImplementedError
    else:
        #effectively, convert the problem to a cartesian one by doing a transform of the integrand and having the window values be the same value but this time in cartesian coordinates.
        if integrand_coord_system == 'cartesian':
            pass
        elif integrand_coord_system == 'spherical': 
            integrand = lambda r, phi, theta: integrand(r, phi, theta) * (r**2) * math.sin(theta) 
        else:
            raise NotImplementedError
    
    volume = np.product(np.diff(integration_window, axis=1))
    if importance_sampling_distro == None:
        if find_expected_value:
            importance_sampling_distro = lambda *args: 1
        else:
            importance_sampling_distro = lambda *args: 1/volume
    
    #integration_window_lows, integration_window_highs = integration_window[:,0], integration_window[:,1]
    #dimension = integration_window.shape[0]
    

    num_of_short_runs = parallel_pool.n_jobs
    per_run_sample_size = math.ceil(sample_size/num_of_short_runs)
    #breakpoint()
    if sample_generator==None:
        make_a_sample_run = lambda run_indice: metropolis_hastings(importance_sampling_distro, starting_window=integration_window, starting_diagonal_covariance=metropolis_starting_diagonal_covariance, run_time=per_run_sample_size, yield_prob=not(bool(find_expected_value)), yield_proposal_point_and_acceptance_prob=bool(find_expected_value), desc_text="Importance Sampling Integrand" if override_desc_text is None else override_desc_text, pbar_desc_id=run_indice, only_desc_0th_id=True, disable_metropolis_pbar=disable_metropolis_pbar)
    else:
        def make_a_sample_run(run_indice, sample_generator=sample_generator, run_time=per_run_sample_size, desc_text="Importance Sampling Integrand" if override_desc_text is None else override_desc_text, only_desc_0th_id=True, disable_metropolis_pbar=disable_metropolis_pbar):
            pbar_desc_id=run_indice
            if desc_text is None:
                desc_text = "Starting Sample Run"
            total_num_of_pbar_updates = 20
            pbar_update_amount_per_iter = run_time//total_num_of_pbar_updates
            if pbar_desc_id is None:
                desc = f"{desc_text}..."
            elif only_desc_0th_id:
                if pbar_desc_id != 0:
                    disable_metropolis_pbar=True
                    desc=''
                else:
                    desc=f"{desc_text} 0 (rest hidden)..."
            else:
                desc = f"{desc_text} {pbar_desc_id}..."
                
            pbar = tqdm(total=run_time, desc=desc, disable=disable_metropolis_pbar)
            total_iters = 0
            for sample_and_prob in sample_generator(run_time):
                yield sample_and_prob
                if total_iters%pbar_update_amount_per_iter==0 and total_iters!=0:
                    pbar.update(pbar_update_amount_per_iter)
                    
                total_iters += 1
            return
            
        
    #breakpoint()
    def integration_sample_run(sample_generator, integrand=integrand, per_run_sample_size=per_run_sample_size):
        acc = 0
        indep_var_history = []
        integrand_history = []
        prob_history = []
        #breakpoint()
        prev_sample = None
        if find_expected_value:
            prev_proposed_sample = None
        prev_integrand_val = None
        if find_expected_value:
            prev_proposed_integrand_val = None
        first = True
        for sample in sample_generator:
            if find_expected_value:
                indep_var_sample, proposed_indep_var_sample, acceptance_prob = sample
            else:
                indep_var_sample, prob = sample
                if prob == 0:
                    continue
            if first:
                this_integrand_val = integrand(*indep_var_sample)
                if find_expected_value:
                    proposed_integrand_val = integrand(*proposed_indep_var_sample)
                prev_sample = indep_var_sample
                prev_integrand_val = this_integrand_val
                if find_expected_value:
                    prev_proposed_sample = proposed_indep_var_sample
                    prev_proposed_integrand_val = proposed_integrand_val
                first = False
            else:
                if np.all(prev_sample == indep_var_sample):
                    this_integrand_val = prev_integrand_val
                else:
                    this_integrand_val = integrand(*indep_var_sample)
                    prev_sample = indep_var_sample
                    prev_integrand_val = this_integrand_val
                
                if find_expected_value:
                    if prev_proposed_sample == proposed_indep_var_sample:
                        proposed_integrand_val = prev_proposed_integrand_val
                    else:
                        proposed_integrand_val = integrand(*proposed_indep_var_sample)
                        prev_proposed_sample = proposed_indep_var_sample
                        prev_proposed_integrand_val = proposed_integrand_val
                    
            if find_expected_value:
                acc += (1-acceptance_prob) * this_integrand_val + acceptance_prob * proposed_integrand_val #has the same mean as a sum of accepted samples/probability, but reduces variance
            else:
                acc += this_integrand_val/prob
            if store_indep_var_history or store_integrand_history or estimate_error or ((not find_expected_value) and use_unnormalized_importance_sampler_correction):
                if store_indep_var_history or estimate_error:
                    indep_var_history.append(prev_sample)
                if store_integrand_history or estimate_error:
                    integrand_history.append(prev_integrand_val)
                if (estimate_error and not find_expected_value) or use_unnormalized_importance_sampler_correction:
                    prob_history.append(prob)
        
        if not find_expected_value and use_unnormalized_importance_sampler_correction:
            out_estimate = (acc/np.sum(prob_history))*(1/volume) #from https://stats.stackexchange.com/a/484693
        else:
            out_estimate = acc / per_run_sample_size
            
        if store_indep_var_history or store_integrand_history or estimate_error:
            if estimate_error:
                if not find_expected_value:
                    return [out_estimate, indep_var_history, integrand_history, prob_history]
                else:
                    return [out_estimate, indep_var_history, integrand_history]
            elif store_indep_var_history:
                if store_integrand_history:
                    return [out_estimate, indep_var_history, integrand_history]
                else:
                    return [out_estimate, indep_var_history]
            elif store_integrand_history:
                return [out_estimate, integrand_history]
        else:
            return [out_estimate]
    
    work_generator = ( joblib.delayed(integration_sample_run)(make_a_sample_run(run_indice)) for run_indice in range(num_of_short_runs)  )
    
    work_out = np.array(parallel_pool(work_generator), dtype=object)
    integral_estimate = sum(work_out[:,0])/num_of_short_runs
    if store_indep_var_history or store_integrand_history or estimate_error:
        indep_var_history_total = np.concatenate(work_out[:,1])
        integrand_history_total = np.concatenate(work_out[:,2])
        if estimate_error and not find_expected_value:
            prob_history_total = np.concatenate(work_out[:,3])
        
        if estimate_error:
            if not find_expected_value:
                divided_array = np.copy(integrand_history_total)
                for observation_indice in range(len(divided_array)):
                    divided_array[observation_indice] /= prob_history_total[observation_indice]
                    
                variance = ( sum(divided_array**2) - (1/sample_size)*sum(divided_array)**2 ) / (sample_size*(sample_size-1))
                error_estimate = variance**.5
            else:
                std = np.std(integrand_history_total)
                error_estimate = volume * std/(sample_size**.5)
                
        if store_indep_var_history or store_integrand_history:
            if return_error_estimate:
                if not find_expected_value:
                    outs = [ufloat(integral_estimate, error_estimate), indep_var_history_total, integrand_history_total, prob_history_total]
                else:
                    outs = [ufloat(integral_estimate, error_estimate), indep_var_history_total, integrand_history_total]
            else:
                outs = [integral_estimate, indep_var_history_total, integrand_history_total]
            if not store_integrand_history:
                outs.pop(2)
            elif not store_indep_var_history:
                outs.pop(1)
        elif return_error_estimate:
            outs = ufloat(integral_estimate, error_estimate)
        else:
            outs = integral_estimate
    else:
        outs = integral_estimate
        
    return outs

def normalize(func, window, sample_size=1e4, **kwargs):
    normalization_constant = 1/monte_carlo_integration(func, window, sample_size=sample_size, **kwargs)
    return lambda *args: func(*args) * (normalization_constant)

def tint(f, low, high, n):
    grid, step = np.linspace(low,high,retstep=True)
    acc = step*(f(low)+f(high))/2
    for i in range(1,len(grid)-1):
        acc += step*f(low+i*step)
    return acc

def nint(f, window, n_per_dim):
    dim = len(window)
    if dim == 1:
        return tint(f, window[0][0], window[0][1], n_per_dim)
    else:
        integral_over_all_but_first = lambda first_coord: nint(lambda *rest_of_coords: f(first_coord, *rest_of_coords), window[1:], n_per_dim)
        return nint(integral_over_all_but_first, window[:1], n_per_dim)
        
def histogram_given_data(new_var_data, new_func_data, undivided_histo=None, window=None, bin_num_vector=10, testing_dont_use_lerp=True):
    new_var_data_shape = new_var_data.shape
    new_func_data_shape = new_func_data.shape
    if len(new_func_data_shape) == 1:
        should_return_scalar = True
        new_func_data = np.expand_dims(new_func_data, axis=1)
        new_func_data_shape = new_func_data.shape
    else:
        should_return_scalar = False
        
    assert(np.all(new_var_data_shape[0] == new_func_data_shape[0]))
    if window is not None:
        window = np.array(window)
        assert(new_var_data_shape[1] == window.shape[0])
    
    var_dim = new_var_data_shape[1]
    func_dim = new_func_data_shape[1]
    
    try:
        iter(bin_num_vector)
    except TypeError:
        bin_num_vector = np.repeat(np.array([bin_num_vector], np.int64), var_dim)

    #var_histogram_of_func_out_arrays = [[np.histogram(var_data[:,this_var_dim], bins=bin_num_vector[this_var_dim], range=(window[this_var_dim] if window is not None else None), weights=func_data[:,this_func_dim], density=True) for this_func_dim in range(func_dim)] for this_var_dim in range(var_dim)]
    #breakpoint()
    new_var_histo_of_func_out_arrays = []
    var_bin_edges_arrays = []
    
    #!!!with current simple allocation there's a closed form for bin edges
    def make_closed_form_bin_indice_location_func(this_low, step):
        #mimic the output of np.searchsorted
        def this_closed_form_bin_indice_location_func(val, this_low=this_low, step=step):
            to_ceiled = (val - this_low)/step
            out = np.empty_like(to_ceiled, dtype=np.int16)
            steps_taken = np.ceil(to_ceiled, out, casting='unsafe')
            return steps_taken
        
        return this_closed_form_bin_indice_location_func
        
    new_undivided_histo = []
    if testing_dont_use_lerp:
        histo_total_integral_accs = []
    
    for this_var_dim in range(var_dim):
        this_new_var_data = new_var_data[:, this_var_dim]
        this_bin_num = bin_num_vector[this_var_dim]
        
        #!!!think of a smarter way to allocate bin edges - maybe by looking at variance or derivative of the func outs? or maybe instead look at the large f is? think more on this. maybe use the np histogram only for allocating bin edges.
        this_var_bin_edges = np.linspace(window[this_var_dim][0], window[this_var_dim][1], this_bin_num+1)
        
        this_low = window[this_var_dim][0]
        step = (window[this_var_dim][1] - window[this_var_dim][0])/this_bin_num
        this_closed_form_bin_indice_location_func = make_closed_form_bin_indice_location_func(this_low, step)

        #breakpoint()
        this_undivided_histo = undivided_histo[this_var_dim] if not(undivided_histo is None) else np.zeros( (len(this_var_bin_edges)-1, 2) ) #!!!previously instead of -1, 2 it was -1, func_dim+1
        this_new_var_histo_of_func_out_array = np.copy(this_undivided_histo)
        '''
        for datum_indice in range(len(this_var_data)):
            this_var_datum = this_var_data[datum_indice]
            this_func_datum = func_data[datum_indice]
            bin_indice = this_closed_form_bin_indice_location_func(this_var_datum) - 1
            #bin_indice = np.searchsorted(this_var_bin_edges, this_var_datum) - 1
            this_var_histo_of_func_out_array[bin_indice] += [1, *this_func_datum]
        '''
        bin_indices = this_closed_form_bin_indice_location_func(this_new_var_data) - 1
        #bin_indice = np.searchsorted(this_var_bin_edges, this_var_datum) - 1
        #breakpoint()
        for i in range(len(new_func_data)):
            this_new_var_histo_of_func_out_array[bin_indices[i]] +=  [1, np.linalg.norm(new_func_data[i])] #before one-d histoing, it was np.concatenate([ [1], new_func_data[i]], axis=-1)
    
        this_new_var_histo_of_func_out_array += this_undivided_histo
        new_undivided_histo.append(np.copy(this_new_var_histo_of_func_out_array))

        
        divided_array = this_new_var_histo_of_func_out_array[:,1:]
        #breakpoint()
        np.divide(this_new_var_histo_of_func_out_array[:,1:], np.expand_dims(this_new_var_histo_of_func_out_array[:,0], axis=-1), out=divided_array)
        #divided_array
        
        
        '''
        #normalize (if func_out has multiple dimensions than the norm of the output is what's being normalized for):
        #first, integrate our histo approximation over each bin
        acc = 0 #before the 1d histoing it was np.zeros((func_dim))
        total_width = 0
        not_nanned_width = 0
        nanned_width = 0
        for bin_indice in range(len(this_var_bin_edges)-1):
            bin_left, bin_right = this_var_bin_edges[bin_indice], this_var_bin_edges[bin_indice+1]
            width = bin_right - bin_left
            if np.any(np.logical_not(np.isfinite(divided_array[bin_indice]))):
                nanned_width += width
            else:
                not_nanned_width += width
                acc += width * np.linalg.norm(divided_array[bin_indice]) #before the 1-d histoing same just without norm
                
            total_width += width
        '''
        
        #for the samples with a sample_size of 0, there will be a np.nan.
        #!!!think about whether the following replacement strategy is good
        #our strategy is to assume the function's average over the 0-sample (aka nanned) region is the average over the not_nanned region.
        
        #over the nearest region
        for bin_indice in range(len(this_var_bin_edges)-1):
            func_out = divided_array[bin_indice]
            if not np.all(np.isfinite(func_out)):
                #search to left and right for good indice
                left_good_indice_displacement = 0
                right_good_indice_displacement = 0
                considering_left_indice = bin_indice - left_good_indice_displacement
                considering_right_indice = bin_indice + right_good_indice_displacement
                considering_left_func_out = func_out
                considering_right_func_out = func_out
                left_search_done = np.all(np.isfinite(considering_left_func_out))
                right_search_done = np.all(np.isfinite(considering_right_func_out))
                while (not left_search_done) or (not right_search_done):
                    if not left_search_done:
                        left_good_indice_displacement += 1
                        considering_left_indice = bin_indice - left_good_indice_displacement
                    
                    if not right_search_done:
                        right_good_indice_displacement += 1
                        considering_right_indice = bin_indice + right_good_indice_displacement
                        
                    if considering_left_indice == -1:
                        considering_left_indice = None
                        considering_left_func_out = None
                        left_search_done = True
                    if considering_right_indice == len(this_var_bin_edges)-1:
                        considering_right_indice = None
                        considering_right_func_out = None
                        right_search_done = True
                
                    if not left_search_done:
                        considering_left_func_out = divided_array[considering_left_indice]
                    if not right_search_done:
                        considering_right_func_out = divided_array[considering_right_indice]

                    
                    if not left_search_done:
                        if np.all(np.isfinite(considering_left_func_out)):
                            left_search_done = True
                    if not right_search_done:
                        if np.all(np.isfinite(considering_right_func_out)):
                            right_search_done = True
                    
                
                
                left_indep_var = this_var_bin_edges[considering_left_indice] if considering_left_indice!=None else None
                right_indep_var = this_var_bin_edges[considering_right_indice] if considering_right_indice!=None else None
                if left_indep_var!=None and right_indep_var!=None:
                    slope = (considering_right_func_out - considering_left_func_out)/(right_indep_var - left_indep_var)
                    displacement = this_var_bin_edges[bin_indice] - left_indep_var
                    anchor = considering_left_func_out
                    func_out_to_replace_nan = displacement*slope + anchor
                    
                elif left_indep_var == None:
                    if considering_right_indice < len(this_var_bin_edges)-1: #note the strict inequality
                        extra_right_indice = considering_right_indice+1
                        considering_extra_right_func_out = divided_array[extra_right_indice]
                        if np.all(np.isfinite(considering_extra_right_func_out)):
                            extra_right_indep_var = this_var_bin_edges[extra_right_indice]
                            slope = (considering_extra_right_func_out - considering_right_func_out)/(extra_right_indep_var - right_indep_var)
                            displacement = this_var_bin_edges[bin_indice] - right_indep_var
                            anchor = considering_right_func_out
                            func_out_to_replace_nan = displacement*slope + anchor
                        else:
                            func_out_to_replace_nan = considering_right_func_out#that is, we don't trust lerping when there's only one on one side and none immediately after it
                    else:
                        func_out_to_replace_nan = considering_right_func_out#that is, we don't trust lerping when there's only one on one side and none immediately after it
                            
                elif right_indep_var == None:
                    if considering_left_indice > 0: #note the strict inequality
                        extra_left_indice = considering_left_indice-1
                        considering_extra_left_func_out = divided_array[extra_left_indice]
                        if np.all(np.isfinite(considering_extra_left_func_out)):
                            extra_left_indep_var = this_var_bin_edges[extra_left_indice]
                            slope = (considering_extra_left_func_out - considering_left_func_out)/(extra_left_indep_var - left_indep_var)
                            displacement = this_var_bin_edges[bin_indice] - left_indep_var
                            anchor = considering_left_func_out
                            func_out_to_replace_nan = displacement*slope + anchor
                        else:
                            func_out_to_replace_nan = considering_right_func_out#that is, we don't trust lerping when there's only one on one side and none immediately after it
        
                    else:
                        func_out_to_replace_nan = considering_right_func_out#that is, we don't trust lerping when there's only one on one side and none immediately after it
                         
                
                divided_array[bin_indice] = np.linalg.norm(func_out_to_replace_nan)

                #new_undivided_histo[this_var_dim][bin_indice][0] = 1
                #new_undivided_histo[this_var_dim][bin_indice][1:] = np.linalg.norm(func_out_to_replace_nan)
            #else:
                #func_out_to_replace_nan = func_out
                
                        
        '''    
        if nanned_width != 0:
            func_integral_over_not_nanned_region = acc
            func_average_over_not_nanned_region = acc / not_nanned_width
            
            #replace nans with func_average_over_not_nanned_region
            for bin_indice in range(len(this_var_bin_edges)-1):
                func_out = divided_array[bin_indice]
                if np.any(np.logical_not(np.isfinite(func_out))):
                    divided_array[bin_indice] = func_average_over_not_nanned_region
                    new_undivided_histo[this_var_dim][bin_indice][0] = 1
                    new_undivided_histo[this_var_dim][bin_indice][1:] = func_average_over_not_nanned_region
                    
            #correct the total_integral for the normalization
            func_integral_over_nanned_region = func_average_over_not_nanned_region * nanned_width
            func_integral_over_total_region = func_integral_over_not_nanned_region + func_integral_over_nanned_region
        else:
            func_integral_over_total_region = acc
        '''
        
        '''
        #normalize (if func_out has multiple dimensions than the norm of the output is what's being normalized for):
        #first, integrate our histo approximation over each bin
        acc = 0 #before the 1d histoing it was np.zeros((func_dim))
        total_width = 0
        for bin_indice in range(len(this_var_bin_edges)-1):
            bin_left, bin_right = this_var_bin_edges[bin_indice], this_var_bin_edges[bin_indice+1]
            width = bin_right - bin_left
            acc += width * np.linalg.norm(divided_array[bin_indice]) #before the 1-d histoing same just without norm
            total_width += width
        '''
        

            
            
        
        #histo is usually pretty noisy. therefore, we smooth it. 
        #according to https://stackoverflow.com/a/20642478, the "Savitzky–Golay filter" (https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter) is a smart way of doing this
        #breakpoint()
        #note that divided_array is of shape (n,1). in the past vectorvalued funcs would make it (n,m) but now the norms are stored as we want only the probability density (meaning that vector valued funcs are only more efficient than splitting it into many single valued funcs when the components are highly correlated)
        #therefore, divided_array[:,0] is actually just all the data
        smoothing_window_size = 50
        smoothing_poly_order = 3
        smoothing_size_ratio_thresh = 2
        #breakpoint()
        if len(divided_array) <= smoothing_size_ratio_thresh*smoothing_window_size:
            smoothing_window_size = len(divided_array)//smoothing_size_ratio_thresh
        
        divided_array[:,0] = sp.signal.savgol_filter(divided_array[:,0], smoothing_window_size, smoothing_poly_order)
        
                
        
        if testing_dont_use_lerp: #calculate necessary normalization constant
            this_histo_total_integral_acc = 0
            for bin_indice in range(len(this_var_bin_edges)-1):
                bin_left, bin_right = this_var_bin_edges[bin_indice], this_var_bin_edges[bin_indice+1]
                width = bin_right - bin_left
                this_histo_total_integral_acc +=  width * divided_array[bin_indice]
                
            histo_total_integral_accs.append(this_histo_total_integral_acc)
            #now, divide by our total integral to normalize
            for bin_indice in range(len(this_var_bin_edges)-1):
                divided_array[bin_indice] /= this_histo_total_integral_acc
                
        new_var_histo_of_func_out_arrays.append(divided_array)
        var_bin_edges_arrays.append(this_var_bin_edges)
    
    #breakpoint()
    histo_and_bin_edges_arrays = [new_var_histo_of_func_out_arrays, var_bin_edges_arrays]
            
    
    def sample_from_histo(run_time, histo_and_bin_edges_arrays=histo_and_bin_edges_arrays, return_norm=True):
        #var_histo_of_func_out_arrays, var_bin_edges_arrays = histo_and_bin_edges_arrays
        #assert(var_dim == len(var_histo_of_func_out_arrays))
        #bin_len, func_dim = var_histo_of_func_out_arrays[0].shape
        new_var_histo_of_func_out_arrays, var_bin_edges_arrays = histo_and_bin_edges_arrays[0], histo_and_bin_edges_arrays[1]
        
        sampled_point_count = 0
        while sampled_point_count < run_time:
            out_coords = []
            prob_density_of_this_choice = 1
            for this_var_dim in range(var_dim):
                this_divided_array = new_var_histo_of_func_out_arrays[this_var_dim]
                this_var_bin_edges_arr = var_bin_edges_arrays[this_var_dim]
                this_var_prob_thresh = random.random()
                running_this_prob_acc = 0
                i = 0
                while running_this_prob_acc < this_var_prob_thresh:
                    bin_left, bin_right = this_var_bin_edges[i], this_var_bin_edges[i+1]
                    width = bin_right - bin_left
                    prob_for_this_bin = this_divided_array[i] *  width
                    running_this_prob_acc += prob_for_this_bin
                    i += 1
                this_picked_var = this_var_bin_edges_arr[i]
                out_coords.append(this_picked_var)
                width = (this_var_bin_edges[i+1] - this_var_bin_edges[i]) if i < len(this_var_bin_edges) else (this_var_bin_edges[i]-this_var_bin_edges[i-1])
                prob_density_of_this_choice *= this_divided_array[i]
            yield [out_coords, prob_density_of_this_choice]
            sampled_point_count += 1
    
    '''
    interp_mode = ('nearest' if testing_dont_use_lerp else 'lerp')
    def this_histo_func(*var, histo_and_bin_edges_arrays=histo_and_bin_edges_arrays, should_return_scalar=should_return_scalar, return_norm=True, interp_mode=interp_mode):
        #should_return_scalar is meant for arrays that contain only one element
        #return_norm is meant to return the norm here for prob distro purposes
        var_histo_of_func_out_arrays, var_bin_edges_arrays = histo_and_bin_edges_arrays
        var_dim = len(var)
        assert(var_dim == len(var_histo_of_func_out_arrays))
        bin_len, func_dim = var_histo_of_func_out_arrays[0].shape
        
        acc_func_outs = np.ones(func_dim)
        for this_var_dim in range(var_dim):
            this_var = var[this_var_dim]
            this_var_histo_of_func_out_array = var_histo_of_func_out_arrays[this_var_dim]
            these_bin_edges = var_bin_edges_arrays[this_var_dim]
            
            
            this_high, this_low = these_bin_edges[-1], these_bin_edges[0]
            this_bin_num = len(these_bin_edges) - 1
            step = (this_high - this_low) / this_bin_num
            right_edge_indice = this_closed_form_bin_indice_location_func(this_var, this_low=this_low, step=step)
            if right_edge_indice==len(this_var_histo_of_func_out_array):
                right_edge_indice -= 1
                left_edge_indice = right_edge_indice
            else:
                left_edge_indice = right_edge_indice-1 if right_edge_indice !=0 else 0
            
            if interp_mode=='right': 
                these_func_outs = this_var_histo_of_func_out_array[right_edge_indice]
            elif interp_mode=='left':
                these_func_outs = this_var_histo_of_func_out_array[left_edge_indice]
            elif interp_mode=='nearest':
                left_edge = these_bin_edges[left_edge_indice]
                if (this_var-left_edge)/step < .5:
                    these_func_outs = this_var_histo_of_func_out_array[left_edge_indice]
                else:
                    these_func_outs = this_var_histo_of_func_out_array[right_edge_indice]
            elif interp_mode=='lerp': 
                left_edge = these_bin_edges[left_edge_indice]
                
                left_func_out, right_func_out = this_var_histo_of_func_out_array[left_edge_indice], this_var_histo_of_func_out_array[right_edge_indice]
                
                lerp_indep_var = (this_var-left_edge)/step
                these_func_outs = left_func_out*(1-lerp_indep_var) + right_func_out*(lerp_indep_var)
            elif interp_mode=='average':
                left_func_out, right_func_out = this_var_histo_of_func_out_array[left_edge_indice], this_var_histo_of_func_out_array[right_edge_indice]
                these_func_outs = (left_func_out+right_func_out)/2
            

            for comp_index in range(len(these_func_outs)):
                if these_func_outs[comp_index] < 0:
                    these_func_outs[comp_index] = abs(these_func_outs[comp_index])
                    
            acc_func_outs *= these_func_outs

        if should_return_scalar:
            return acc_func_outs[0]
        elif return_norm:
            return acc_func_outs #before the 1d histoing it was same but with norm
        else:
            return acc_func_outs

    if testing_dont_use_lerp:
        norm_hist_func = this_histo_func #normalize(this_histo_func, window)
    else:
        norm_hist_func = normalize(this_histo_func, window)
    
    
    #breakpoint()

    
    
    
    def landau_distro_approx(x):
        return math.exp(-(x**2))/((3.332307087093)**(1/3))
        #return (1/(2*math.pi))**.5 * math.exp(-.5*(x+math.exp(-x)))
    
    def f(*x):
        #integration_window = np.repeat([[-10,30]], dim, axis=0)
        #actual_value = 1 #approximate
        acc = 1
        i = 0
        for part in x:
            acc *= landau_distro_approx(part)
            i += 1
        
        return acc
    
    
    normed_f = f
    
    
    #start,end = -10,10
    start,end = -3,3
    line_x = np.linspace(start,end,100)
    line_y = np.linspace(start,end,100)
    line_z = np.linspace(start,end,100)
    mesh_x,mesh_y,mesh_z = np.meshgrid(line_x, line_y, line_z, indexing='ij')
    xy_grid_x, xy_grid_y = mesh_x[:,:,0], mesh_y[:,:,0]
    yz_grid_y, yz_grid_z = mesh_y[0,:,:], mesh_z[0,:,:]
    xz_grid_x, xz_grid_z = mesh_x[:,0,:], mesh_z[:,0,:]
    
    
    nf,nh = normed_f, norm_hist_func,
    lh = lambda var: this_histo_func(*var, testing_dont_use_lerp=False)
    vf,vh = np.vectorize(nf), np.vectorize(nh)
    
    x_line_fs = vf(line_x,0,0)
    x_line_hs = vh(line_x,0,0)
    
    y_line_fs = vf(0,line_y,0)
    y_line_hs = vh(0,line_y,0)
    
    z_line_fs = vf(0,0,line_z)
    z_line_hs = vh(0,0,line_z)
    

    xy_fs = vf(xy_grid_x, xy_grid_y, 0)
    xy_hs = vh(xy_grid_x, xy_grid_y, 0)
    
    yz_fs = vf(0, yz_grid_y, yz_grid_z)
    yz_hs = vh(0, yz_grid_y, yz_grid_z)
    
    xz_fs = vf(xz_grid_x, 0, xz_grid_z)
    xz_hs = vh(xz_grid_x, 0, xz_grid_z)
    
    x_line_rmse = np.sqrt(np.mean((x_line_hs - x_line_fs)**2))
    
    
    
    
    fig, axs = plt.subplots(nrows=3, ncols=3)
    
    
    axs[0][0].plot(line_x, x_line_fs, line_x, x_line_hs)
    axs[0][1].plot(line_y, y_line_fs, line_y, y_line_hs)
    axs[0][2].plot(line_z, z_line_fs, line_z, z_line_hs)
    
    axs[1][0].pcolormesh(xy_grid_x, xy_grid_y, xy_fs)
    axs[2][0].pcolormesh(xy_grid_x, xy_grid_y, xy_hs)
    
    axs[1][1].pcolormesh(yz_grid_y, yz_grid_z, yz_fs)
    axs[2][1].pcolormesh(yz_grid_y, yz_grid_z, yz_hs)
    
    axs[1][2].pcolormesh(xz_grid_x, xz_grid_z, xz_fs)
    axs[2][2].pcolormesh(xz_grid_x, xz_grid_z, xz_hs)
    
    plt.show()
    #breakpoint()
    '''
    
    
    #breakpoint()
    
    return new_undivided_histo, sample_from_histo


def uncertainty_weighted_average(ufloats):    
    noms = np.empty_like(ufloats)
    inverse_variances = np.empty_like(ufloats)
    i = 0
    for ufloat_ in ufloats:
        noms[i] = util.nominalize(ufloat_)
        inverse_variances[i] = 1/(util.std_devize(ufloat_)**2)
        i+=1
    
    '''
    total_variance = sum(inverse_variances)**-1
    estimate = total_variance * sum(noms*inverse_variances)
    '''
    
    squared_nom_div_squared_var = (noms**2)*(inverse_variances)
    sum_of_squared_nom_div_squared_var = sum(squared_nom_div_squared_var)
    estimate = sum(noms*squared_nom_div_squared_var) / sum_of_squared_nom_div_squared_var
    total_variance = (estimate**2) * (sum_of_squared_nom_div_squared_var**-1)
    
    
    return ufloat(estimate, total_variance**.5)
    
def VEGAS_integration(integrand, integration_window, find_expected_value=False, sample_size=1e4, first_importance_sampling_distro=None, first_sample_generator=None, first_stage_sample_proportion=None, stages=5, bin_num_vector=1000, discard_num_of_initial_stages=0, aitken_accelerate=False, metropolis_starting_diagonal_covariance=1, return_error_estimate=False, only_return_best_estimate=True, integrand_coord_system='cartesian', window_coord_system='cartesian', first_integral_desc_text="Integrating First Vegas Stage", vegas_stage_desc="Vegas Stages", disable_first_metropolis_pbar=False, disable_vegas_stage_pbar=False, testing_dont_use_lerp=True, parallel_pool=None):
    if stages <= discard_num_of_initial_stages:
        discard_num_of_initial_stages = 0
    
    if first_stage_sample_proportion==None:
        first_stage_sample_proportion = 1/stages
        
    first_stage_sample_size = math.ceil(sample_size*first_stage_sample_proportion)
    remaining_sample_size = sample_size - first_stage_sample_size
    samples_per_stage = math.ceil(remaining_sample_size/(stages-1))
    if not (vegas_stage_desc is None):
        vegas_stage_desc += "..."
    def do_a_stage(sample_generator, bin_num_vector, undivided_histo=None, integrand=integrand, integration_window=integration_window, find_expected_value=find_expected_value, sample_size=samples_per_stage, metropolis_starting_diagonal_covariance=metropolis_starting_diagonal_covariance, integrand_coord_system=integrand_coord_system, window_coord_system=window_coord_system, first_integral_desc_text=first_integral_desc_text, disable_metropolis_pbar=True, testing_dont_use_lerp=testing_dont_use_lerp, parallel_pool=parallel_pool):
        #breakpoint()
        if not find_expected_value:
            uncertain_integral_estimate, new_indep_var_history, new_integrand_history, _throw_away_prob_history = monte_carlo_integration(integrand, integration_window, sample_generator=sample_generator, store_indep_var_history=True, store_integrand_history=True, estimate_error=True, return_error_estimate=True,  find_expected_value=find_expected_value, sample_size=sample_size, metropolis_starting_diagonal_covariance=metropolis_starting_diagonal_covariance, integrand_coord_system=integrand_coord_system, window_coord_system=window_coord_system, override_desc_text=first_integral_desc_text, disable_metropolis_pbar=disable_metropolis_pbar, parallel_pool=parallel_pool)
        else:
            uncertain_integral_estimate, new_indep_var_history, new_integrand_history = monte_carlo_integration(integrand, integration_window, sample_generator=sample_generator, store_indep_var_history=True, store_integrand_history=True, estimate_error=True, return_error_estimate=True,  find_expected_value=find_expected_value, sample_size=sample_size, metropolis_starting_diagonal_covariance=metropolis_starting_diagonal_covariance, integrand_coord_system=integrand_coord_system, window_coord_system=window_coord_system, override_desc_text=first_integral_desc_text, disable_metropolis_pbar=disable_metropolis_pbar, parallel_pool=parallel_pool)
             
        #breakpoint()
        undivided_histo, this_histo_sample_generator = histogram_given_data(new_indep_var_history, new_integrand_history, undivided_histo=undivided_histo, window=integration_window, bin_num_vector=bin_num_vector, testing_dont_use_lerp=testing_dont_use_lerp)
        return uncertain_integral_estimate, new_indep_var_history, new_integrand_history, undivided_histo, this_histo_sample_generator
    
    sample_generator = first_sample_generator
    integrand_estimates = []
    total_indep_var_history = []
    total_integrand_history = []
    first = True
    for stage in tqdm(range(stages), desc=vegas_stage_desc, disable=disable_vegas_stage_pbar):
        #breakpoint()
        if first:
            uncertain_integral_estimate, indep_var_history, integrand_history, undivided_histo, this_histo_sample_generator = do_a_stage(sample_generator, bin_num_vector, sample_size=first_stage_sample_size, disable_metropolis_pbar=disable_first_metropolis_pbar)
        else:
            #prev_indep_var_history=total_indep_var_history, prev_integrand_history=total_integrand_history
            uncertain_integral_estimate, indep_var_history, integrand_history, undivided_histo, this_histo_sample_generator = do_a_stage(sample_generator, bin_num_vector, undivided_histo=undivided_histo)
        print("Vegas Estimate: {uncertain_integral_estimate}")
        integrand_estimates.append(uncertain_integral_estimate+0) #+0 is for convenience to make the array of ufloat become ufloat
        if first:
            total_indep_var_history.append(indep_var_history)
            total_integrand_history.append(integrand_history)
            total_indep_var_history = np.concatenate(total_indep_var_history)
            total_integrand_history = np.concatenate(total_integrand_history)
            first = False
        else:
            total_indep_var_history = np.concatenate([total_indep_var_history, indep_var_history])
            total_integrand_history = np.concatenate([total_integrand_history, integrand_history])
            
        sample_generator = this_histo_sample_generator

    
    integrand_estimates = integrand_estimates[discard_num_of_initial_stages:]
    if aitken_accelerate:
        combined_integrand_estimate = list(util.aitken(integrand_estimates, override_finish_sequence_iterable=True))[-1]
    else:
        combined_integrand_estimate = uncertainty_weighted_average(integrand_estimates)
        #combined_integrand_estimate = sum(integrand_estimates)/len(integrand_estimates)
        
    #breakpoint()
    if not return_error_estimate:
        combined_integrand_estimate = combined_integrand_estimate + 0
        combined_integrand_estimate = util.nominalize(combined_integrand_estimate)
    
    if only_return_best_estimate:
        return combined_integrand_estimate+0
    else:
        return combined_integrand_estimate+0, integrand_estimates, total_indep_var_history, total_integrand_history


#integrand = lambda x,y,z: x**2+y**2+z**2

#integration_window = [ [0,1], [0,1], [0,1] ]

#out = VEGAS_integration(integrand, integration_window)





def vegas_test(sample_size=1e5, first_stage_sample_proportion=None, stages=5, bin_num_vector=200, disable_testing_pbar=False, testing_ait=False, testing_dont_use_lerp=True):
    from time import time
    '''
    def f(*x):
        dx2 = 0
        for d in range(4):
            dx2 += (x[d] - 0.5) ** 2
        return math.exp(-dx2 * 100.) * 1013.2118364296088

    integration_window = [[-1, 1], [0, 1], [0, 1], [0, 1]]
    '''


    '''
    def f(*x): 
        #integration_window = [[0, 1], [0, 1], [0, 1]]
        #actual_value = .7391807524660602
        dists = np.zeros((7))
        for i in range(3):
            dists += (np.repeat([x[i]], 7) - [1/8,2/8,3/8,4/8,5/8,6/8,7/8]) ** 2
            
        
        acc = 1
        for dist in dists:
            acc *= math.exp(-.1*dist)
            
        return acc
    '''
    
    '''
    def f(*x):
        dists = np.zeros((7))
        for i in range(3):
            dists += (np.repeat([x[i]], 7) - [1/8,2/8,3/8,4/8,5/8,6/8,7/8]) ** 2
            
        
        acc = 1
        for dist in dists:
            acc *= math.exp(-.1*dist)
            
        return [acc,1]
    '''
    
    
    '''
    dim=5
    def osc(*x,dim=dim):
        x=np.array(x)
        c, r = np.array(list(range(1,dim+1))), 0
        return math.cos(2*math.pi*r + np.sum(c*x))
    '''
    
    def landau_distro_approx(x):
        #return math.exp(-(x**2))/((3.332307087093)**(1/3))
        return (1/.9880617451241765) * ( (1/(2*math.pi))**.5 * math.exp(-.5*(x+math.exp(-x))) )
    
    def f(*x):
        #integration_window = np.repeat([[-10,30]], dim, axis=0)
        #actual_value = 1 #approximate
        acc = 1
        i = 0
        for part in x:
            acc *= landau_distro_approx(part)
            i += 1
        
        return acc
    
    
    #integration_window = [[0, 1], [0, 1], [0, 1]]
    #actual_value = .7391807524660602
    
    #integration_window = np.repeat([[-5,5]], dim, axis=0)
    #actual_value=
    
    dim = 3
    start, end = -2, 10
    #start, end = -10, 30
    integration_window = np.repeat([[start, end]], dim, axis=0)
    actual_value = 1 #approximate
    
    
    #breakpoint()
    
    start1 = time()
    out1 = monte_carlo_integration(f, integration_window, sample_size=sample_size, return_error_estimate=True, disable_metropolis_pbar=disable_testing_pbar)
    
    if not disable_testing_pbar:
        print(out1)
    end1 = time()
    elapsed1 = end1 - start1
    
    start2 = time()
    out2 = VEGAS_integration(f, integration_window, first_stage_sample_proportion=first_stage_sample_proportion, sample_size=sample_size, stages=stages, bin_num_vector=bin_num_vector, return_error_estimate=True, only_return_best_estimate=False, disable_first_metropolis_pbar=disable_testing_pbar, disable_vegas_stage_pbar=disable_testing_pbar, testing_dont_use_lerp=testing_dont_use_lerp, aitken_accelerate=testing_ait)
    best_vegas_estimate = out2[0]
    all_vegas_estimates = out2[1]
    end2 = time()
    elapsed2 = end2 - start2

    elapsed_cost = elapsed2 - elapsed1

    estimates = out1, best_vegas_estimate
    if not disable_testing_pbar:
        print(f"Plain: {out1}, Vegas:{out2}")
    errs_actual =  (estimates[0]-actual_value), (estimates[1]-actual_value)
    
    if not disable_testing_pbar:
        print(errs_actual)
    vegas_estimate_err_improvement = abs(errs_actual[0]) - abs(errs_actual[1])
    
   
    return vegas_estimate_err_improvement, elapsed_cost, errs_actual, estimates, all_vegas_estimates, elapsed1, elapsed2




def tests(relative_error=.1):
    def error(act, calc): #purely for convenience/laziness.
        return (calc - act)/act #if not any([np.isnan(val) for val in [act, calc]]) else 0
    
    interval = np.array([[-1,1]])
    
    
    tri = lambda x: 1 if abs(x)>.5 else 2-abs(x)
    
    bump = lambda x: math.exp(-1/(1-abs(x)**2)) if abs(x) < 1 else 0
    norm_bump = lambda x: bump(x)/.4439937793801389
    
    
    calc0 = monte_carlo_integration(lambda x: abs(x), integration_window=interval, find_expected_value=False, sample_size=1e3, metropolis_starting_diagonal_covariance=.1)
    act0 = 1
    e0 = error(calc0, act0)
    
    #triangle func
    calc1 = monte_carlo_integration(tri, integration_window=interval, find_expected_value=False, sample_size=1e4, metropolis_starting_diagonal_covariance=.1)
    act1 = 2.5
    e1 = error(calc1, act1)
    
    
    #triangle func importance sampled by normalized bump func
    calc2 = monte_carlo_integration(tri, integration_window=interval, importance_sampling_distro=norm_bump, find_expected_value=False, sample_size=1e4, metropolis_starting_diagonal_covariance=.1)
    act2 = 2.5
    e2 = error(calc2, act2)
    
    #triangle func expected value wrt identity
    calc3 = monte_carlo_integration(tri, integration_window=interval, find_expected_value=True, sample_size=1e4, metropolis_starting_diagonal_covariance=.1)
    act3 = 1.25
    e3 = error(calc3, act3)
    
    #triangle func expected value wrt normalized bump func
    calc4 = monte_carlo_integration(tri, integration_window=interval, importance_sampling_distro=norm_bump, find_expected_value=True, sample_size=1e3, metropolis_starting_diagonal_covariance=.1)
    act4 = calc4 #idk actual value
    e4 = error(calc4, act4)
    
    #triangle func expected value wrt bump func
    calc5 = monte_carlo_integration(tri, integration_window=interval, importance_sampling_distro=bump, find_expected_value=True, sample_size=1e3, metropolis_starting_diagonal_covariance=.1)
    act5 = calc5 #idk actual value
    e5 = error(calc5, act5)
    
    
    #f(x)=x^2 on [0,1] sampled by p(x)=2*x 
    calc6 = monte_carlo_integration(lambda x: x**2, integration_window=np.array([[0,1]]), importance_sampling_distro=(lambda x: 2*x), find_expected_value=False, sample_size=1e3, metropolis_starting_diagonal_covariance=.1)
    act6 = 1/3
    e6 = error(calc6, act6)
    
    #vectors!
    v_func = lambda r: np.array([r, r**2, r**3])
    calc7 = monte_carlo_integration(v_func, np.array([[0,1]]), sample_size=1e3)
    act7 = np.array([1/2, 1/3, 1/4])
    e7 = calc7 - act7
    e7 = (e7[0]**2 + e7[1]**2 + e7[2]**2)**.5
    
    
    if not all(np.less(np.abs([e0,e1,e2,e3,e4,e5,e6,e7]), np.repeat([relative_error],8))):
        return False
    else:
        return True
    
if __name__ == '__main__':
    testing_dont_use_lerp = True
    out1=vegas_test(sample_size=3e4, stages=5, bin_num_vector=1e3)
    
    #yes_ait=[vegas_test(testing_ait=True) for i in range(50)]
    #no_ait=[vegas_test(testing_ait=False) for i in range(50)]
    #print(out1)
    #out2=tests()
    
    '''
    i,j,k,l = 0,0,0,0
    already_did_tuple = (0,0,9,8)
    for sample_size in tqdm(np.linspace(1e4,1e6,5), desc="Sample Sizes..."):
        j=0
        for bin_num_vector in tqdm(np.arange(10, 210, 10), desc="Bin nums..."):
            k=0
            for stages in tqdm(range(1,11), desc="Stages..."):
                l=0
                for test_datum_num in tqdm(range(10), desc="Collecting Test Data..."):
                    if util.lex_less((i,j,k,l), already_did_tuple):
                        continue
                    
                    if test_datum_num == 0:
                        with open("Parameter_Testing/vegas_testing.txt", 'a') as f:
                            f.write(f"\n{sample_size}, {stages}, {bin_num_vector}::\n\n")
                            
                    out = vegas_test(sample_size=sample_size, stages=stages, bin_num_vector=bin_num_vector, disable_testing_pbar=True)[:2]
                    
                    with open("Parameter_Testing/vegas_testing.txt", 'a') as f:
                        f.write(f"{out[0]}, {out[1]}\n")
                        
                    l+=1                
                k+=1
            j+=1
        i+=1
    '''
    pass