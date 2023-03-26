# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 11:33:35 2023

@author: Sri
"""


import numpy as np
import math

import utility_funcs as util 

import joblib
VERBOSITY=0

from tqdm import tqdm


#metropolis_hastings
def metropolis_hastings(un_normed_probability_density, starting_window=np.array([[-1,1], [-1,1], [-1,1]]), starting_diagonal_covariance=1, run_time=1000, restrict_to_window=True, yield_prob=False, yield_proposal_point_and_acceptance_prob=False, desc_text="Traversing Markov Chain", pbar_desc_id=None, only_desc_0th_id=False, disable_metropolis_pbar=True):
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
        
        
    starting_state = np.random.default_rng().uniform(low=starting_window_lows, high=starting_window_highs) #!!!generalize to nonuniform starting distribution

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
                yield [this_state, this_weight if this_state==prev_state else proposal_weight]
        elif yield_proposal_point_and_acceptance_prob:
            if i != 0:
                yield [prev_state, proposal, acceptance_probability]
        else:
            yield this_state
            
        if total_iters%pbar_update_amount_per_iter==0 and total_iters!=0:
            pbar.update(pbar_update_amount_per_iter)
            
        total_iters += 1
        state_list.append(this_state)
        
        if total_iters > 100 and True: #!!!
            if acceptance_rate < .27:
                if current_diagonal_covariance.min() > min_minimum_diagonal_covariance:
                    current_diagonal_covariance *= .9
            elif acceptance_rate > .33:
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
def monte_carlo_integration(integrand, integration_window=np.array([[-1,1], [-1,1], [-1,1]]), importance_sampling_distro=None, find_expected_value=False, sample_size=1000, metropolis_starting_diagonal_covariance=1, integrand_coord_system='cartesian', window_coord_system='cartesian', disable_metropolis_pbar=True, parallel_pool=None):
    #if find_expected_value==True, then importance_sampling_distro needs to be normalized
    if parallel_pool is None:
        parallel_pool = joblib.Parallel(n_jobs=8, verbose=VERBOSITY, batch_size=4096, prefer='threads')
      
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
    
    if importance_sampling_distro == None:
        if find_expected_value:
            importance_sampling_distro = lambda *args: 1
        else:
            volume = np.product(np.diff(integration_window, axis=1))
            importance_sampling_distro = lambda *args: 1/volume
    
    #integration_window_lows, integration_window_highs = integration_window[:,0], integration_window[:,1]
    #dimension = integration_window.shape[0]
    

    num_of_short_runs = parallel_pool.n_jobs
    per_run_sample_size = math.ceil(sample_size/num_of_short_runs)
    #breakpoint()
    make_a_sample_run = lambda run_indice: metropolis_hastings(importance_sampling_distro, starting_window=integration_window, starting_diagonal_covariance=metropolis_starting_diagonal_covariance, run_time=per_run_sample_size, yield_prob=not(bool(find_expected_value)), yield_proposal_point_and_acceptance_prob=bool(find_expected_value), desc_text="Importance Sampling Integrand", pbar_desc_id=run_indice, only_desc_0th_id=True, disable_metropolis_pbar=disable_metropolis_pbar)
      
    #breakpoint()
    def integration_sample_run(sample_generator, integrand=integrand, importance_sampling_distro=importance_sampling_distro, per_run_sample_size=per_run_sample_size):
        acc = 0
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
                if prev_sample == indep_var_sample:
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
            
        return acc/per_run_sample_size
    
    work_generator = ( joblib.delayed(integration_sample_run)(make_a_sample_run(run_indice)) for run_indice in range(num_of_short_runs)  )
    
    return sum(parallel_pool(work_generator))/num_of_short_runs

def monte_carlo_integrate_3D(func, window=[[0,1], [0, 2*math.pi], [0, math.pi]], func_coord_system='spherical', window_coord_system='spherical', monte_carlo_n=10000): #!!!generalize
    if func_coord_system == 'spherical':
        integrand = lambda r, phi, theta: func(r, phi, theta) * (r**2) * math.sin(theta) 
    elif func_coord_system == 'cartesian':
        integrand = lambda x, y, z: func(x,y,z)
    else:
        raise NotImplementedError
        
    if not (monte_carlo_n is False):
        if window_coord_system == func_coord_system:
            if window_coord_system == 'spherical':
                volume = (window[0][1] - window[0][0]) * (window[1][1] - window[1][0]) * (window[2][1] - window[2][0])
            elif window_coord_system == 'cartesian':
                volume = (window[0][1] - window[0][0]) * (window[1][1] - window[1][0]) * (window[2][1] - window[2][0])
        else:
            raise NotImplementedError
            
        running_total = 0
        for i in range(monte_carlo_n):
            starting_window_lows, starting_window_highs = window[:,0], window[:,1]
            random_coords = np.random.default_rng().uniform(low=starting_window_lows, high=starting_window_highs)
            running_total += integrand(*random_coords)
        
        return volume * running_total/(monte_carlo_n)
    
    
    
def tests(relative_error=.1):
    #breakpoint()
    def error(act, calc): #purely for convenience/laziness.
        return (calc - act)/act if not any([np.isnan(val) for val in [act, calc]]) else 0
    
    interval = np.array([[-1,1]])
    
    
    tri = lambda x: 1 if abs(x)>.5 else 2-abs(x)
    
    bump = lambda x: math.exp(-1/(1-abs(x)**2)) if abs(x) < 1 else 0
    norm_bump = lambda x: bump(x)/.4439937793801389
    
    
    calc0 = monte_carlo_integration(lambda x: abs(x), integration_window=interval, find_expected_value=False, sample_size=1000, metropolis_starting_diagonal_covariance=.1)
    act0 = 1
    e0 = error(calc0, act0)
    
    #triangle func
    calc1 = monte_carlo_integration(tri, integration_window=interval, find_expected_value=False, sample_size=1000, metropolis_starting_diagonal_covariance=.1)
    act1 = 2.5
    e1 = error(calc1, act1)
    
    
    #triangle func importance sampled by normalized bump func
    calc2 = monte_carlo_integration(tri, integration_window=interval, importance_sampling_distro=norm_bump, find_expected_value=False, sample_size=1000, metropolis_starting_diagonal_covariance=.1)
    act2 = 2.5
    e2 = error(calc2, act2)
    
    #triangle func expected value wrt identity
    calc3 = monte_carlo_integration(tri, integration_window=interval, find_expected_value=True, sample_size=1000, metropolis_starting_diagonal_covariance=.1)
    act3 = 1.25
    e3 = error(calc3, act3)
    
    #triangle func expected value wrt normalized bump func
    calc4 = monte_carlo_integration(tri, integration_window=interval, importance_sampling_distro=norm_bump, find_expected_value=True, sample_size=1000, metropolis_starting_diagonal_covariance=.1)
    act4 = np.nan
    e4 = error(calc4, act4)
    
    #triangle func expected value wrt bump func
    calc5 = monte_carlo_integration(tri, integration_window=interval, importance_sampling_distro=bump, find_expected_value=True, sample_size=1000, metropolis_starting_diagonal_covariance=.1)
    act5 = np.nan
    e5 = error(calc5, act5)
    
    
    #f(x)=x^2 on [0,1] sampled by p(x)=2*x 
    calc6 = monte_carlo_integration(lambda x: x**2, integration_window=np.array([[0,1]]), importance_sampling_distro=(lambda x: 2*x), find_expected_value=False, sample_size=1000, metropolis_starting_diagonal_covariance=.1)
    act6 = 1/3
    e6 = error(calc6, act6)
    
    if not all(np.less_than(np.abs([e0,e1,e2,e3,e4,e5,e6]), np.repeat([relative_error],7))):
        return False
    else:
        return True