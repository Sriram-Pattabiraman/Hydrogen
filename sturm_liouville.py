# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 10:40:35 2022

@author: Sri
"""



from tqdm import tqdm

from warnings import warn

import pyqtgraph as pg
import PyQt5
DashLine = PyQt5.QtCore.Qt.DashLine #doing this as a from import doesn't work for unknown reasons.
import datetime

'''
import IPython
#if this is an ipy, start the qt5 event loop
%gui qt5
app =  pg.Qt.mkQApp()
'''

import math
import numpy as np
import scipy as sp
from scipy import constants

import itertools

import joblib
VERBOSITY = 1


#references for future me:
#the one that everyone copies from: numerical solution of sturm liouville problems by J.D. pryce, isbn 13: 9780198534150.
#https://libstore.ugent.be/fulltxt/RUG01/001/050/950/RUG01-001050950_2010_0001_AC.pdf#page=28&zoom=130,39,29 - verle ledoux sturm liouville apprx
#https://www.jstor.org/stable/2007594?seq=3#metadata_info_tab_contents - magic eigenfinder using minimum subeigenfinders (maybe it's a shooter?). advantages are it's just as good for far out eigenvals as for the first one (requiring ), the finding of the first eigenvalue on the subinterval is more accurate (same for the eigenfunction) [equilibriating the eigenfunctions is also accurate [also probably more accurate if itp as opposed to newton is used]],
#https://dl.acm.org/doi/10.1145/3423597 and https://docs.rs/kurbo/0.8.1/src/kurbo/common.rs.html#194-237 - itp method - since prufer mismatch is monotonic (and 2nd diff? 2nd cont diff?) we get the error bound of epsilon [using the rust code's convention] on the zero, regardless (i think??) of the starting interval that contains the root.
#https://www.sciencedirect.com/science/article/pii/S0010465508003305#fd007 - general numerical introduction, better explanation of pruess/coefficient approximation than ledoux - fixed mesh, resistant by instability nor stiffness, resistant/actively better error when eigenvalues go up, interval truncation, parralelizable, potentionally similar but easier and better magic than before

'''
We use the convention that the Sturm-Liouville problem is a real 2nd order ODE of the form d/dx [p(x)dy(x)/dx] - q(x)y(x) = -lambda w(x) y(x), where finding the possible eigenvalue lambda is part of the problem and thus unspecified
We use the convention that the regular problem has boundary conditions of a_0 y(a) + b_0 p(a) y'(a) = 0 and a_1 y(b) + b_1 p(b) y'(b) = 0. Importantly, a and b cannot both be 0 in either of those conditions. This implies that y=0 and y'=0 is not a regular boundary condition [otherwise we'll get constant 0 as the solution].
When going from those boundary conditions to initial y and y' at a and b for the purposes of the shooting method, there's [sometimes? need to do the math on this] freedom in selecting one of the a_i or b_i. The fact that setting them all to 0 would lead to a trivial solution means we need to be careful about what we arbitrarily end up picking.

Sometimes the problem with the DE in the above form is "regular" and the conditions are either "separated" as above or "periodic".
Sometimes, and for the programmed prufer implementation, p, q or w [for us q] >0 is required. note that since the azimuthal equation has q=0, there is a problem with there not existing a 0th eigenval/func for when the boundary condition has value 0 and derivative 1. this is remediated by noticing that in this case, it's an eigenvalue of -inf. this screws everything up (even more than it would for inf, cuz the eigenvalue in sturm liouville theory is to the negative second derivative operator so it's inf eigen to the second deriv operator). so, we just throw out infinite eigenvalues.
'''


def NumD(f, dx, cmp_step=False): #returns NUMericalDerivatives using the complex step method, with finite difference method as a fallback for higher orders.
    if cmp_step:
        def f_prime(x):
            try:
                #since this is quantum mechanics, things tend to be complex valued. assuming continuity is equivalent to assuming it's holomorphic is equivalent to analytic. expanding f(x + ih) as a taylor series and then taking the imaginary part, f'(x) = Im(f(x+ih))/h + O(h^2). this also doesn't have as bad issues of floating point error when h is small, as unlike finite differences, this is well conditioned.
                return f(x+dx*1j).imag/dx #only works for first order for reasons.
            except TypeError:
                return NumD(f, dx, cmp_step=False)(x)
        return f_prime
    else:
        return lambda x: (f(x+dx)-f(x-dx))/(2*dx)

#!!!use the ITP method - perhaps use secant to find an interval first.
def Secant_Method(f, x_0, x_1, tolerance, f_index_to_rootfind=False, low_deriv_warning_thresh=.0001, give_up_iterations=100, give_up_displacement_mag_factor=200, sanity_give_up_displacement_difference=5, random_inits_if_fail=10, random_n0_range=[-10,50], random_step_range=[-.01,.01]): #secant method is finite difference version of newton's method
    #WARNING: The oscillatory nature of the mismatch function means this may fail catastrophically.
    low_deriv_warning_fired = False
    i=0
    randos = 0
    chosen_randos = []
    if f_index_to_rootfind is False: #using "is" to prevent 0==False
        if i > give_up_iterations:
            warn("More than give_up_iterations have been exceeded! We may be in a cycle, or diverging to ±infinity. ")
            raise BaseException #more than give_up_iterations #yes i know it's bad practice to not actually use a helpfully named exception class.

        x_n0, x_n1 = x_0, x_1
        prevF = f(x_0)
        thisF = f(x_n1)
        while abs(thisF) >= tolerance:
            dx = x_n1-x_n0
            df = thisF-prevF
            #print(f"secant debugging: {thisF}, {dx}, {df}")
            #print(f"secant deriv: {df/dx}")
            derivative = df/dx
            if abs(derivative) < low_deriv_warning_thresh and not low_deriv_warning_fired: #don't warn if already warned in this run of the method
                warn("Derivative is smaller than warning threshold! Catastrophe may ensue!")
                low_deriv_warning_fired = True
            x_n2 = x_n1 - thisF*(1/derivative) #f * dx/df = f / (df/dx) is why this agrees with wikipedia
            if (x_0 > 0 and abs(x_n2/x_0) > give_up_displacement_mag_factor) and (abs(x_n2-x_0)>sanity_give_up_displacement_difference):
                warn("x/x_0 is bigger than the give_up_displacement_mag_factor! It probably has no zeros, or has a zero at ±infinity")
                if randos >= random_inits_if_fail:
                    if random_inits_if_fail > 0:
                        warn(f"Failed for {randos} random inits! here are the sampled inits:\n{chosen_randos}")
                    warn("We're brazenly assuming it has a zero at ±infinity")
                    if x_n2 > 0:
                        return np.inf
                    elif x_n2 < 0:
                        return -np.inf
                elif random_inits_if_fail > 0:
                    randos += 1
                    x_0 = np.random.uniform(*random_n0_range)
                    x_n0 = x_0
                    x_n1 = x_n0+np.random.uniform(*random_step_range)

                    chosen_randos.append([x_n0, x_n1])
                    if randos > 0:
                        warn("trying random initial guesses!")
                    prevF = f(x_n0)
                    thisF = f(x_n1)
                    i = 0
                    low_deriv_warning_fired = False
                    continue


            x_n0, x_n1 = x_n1, x_n2
            i+=1
            prevF = thisF
            thisF = f(x_n1)

        return x_n1

    else:
        root = Secant_Method(lambda x: f(x)[f_index_to_rootfind], x_0, x_1, tolerance, f_index_to_rootfind=False, low_deriv_warning_thresh=low_deriv_warning_thresh, give_up_iterations=give_up_iterations, give_up_displacement_mag_factor=give_up_displacement_mag_factor, random_inits_if_fail=random_inits_if_fail, random_n0_range=random_n0_range, random_step_range=random_step_range)
        return root, f(root)
        '''
        x_n0, x_n1 = x_0, x_1
        thisF = f(x_n1)
        while abs(thisF[f_index_to_rootfind]) >= tolerance:
            dx = x_n1-x_n0
            df = f(x_n1)[f_index_to_rootfind]-f(x_n0)[f_index_to_rootfind]
            reciprocal_derivative = dx/df
            if abs(reciprocal_derivative) < low_deriv_warning_thresh and not low_deriv_warning_fired:
                warn("Reciprocal Derivative is less than warning threshold! Catastrophe may ensue!")
                low_deriv_warning_fired = True
            x_n2 = x_n1 - f(x_n1)[f_index_to_rootfind]*(reciprocal_derivative) #f * dx/df = f / (df/dx)

            x_n0, x_n1 = x_n1, x_n2
            i+=1
            thisF = f(x_n1)

        return x_n1, thisF[:f_index_to_rootfind]+thisF[f_index_to_rootfind:]
        '''


def Monotone_Find_Bracket(f, x_0, initial_step=1, adjusting_factor=2, giveup_steps=10, giveup_displacement=100, disable_pbar=False):
    starting_x = x_0
    thisF = f(x_0)

    #find initial direction
    if abs(thisF) == 0:
        return [x_0, x_0]
    elif thisF > 0:
        direction = -1
    else:
        direction = 1

    #now let's loop to find the other end of the interval
    step = initial_step
    bracket = [x_0, np.inf] if direction == 1 else [-np.inf, x_0]
    step_count = 0
    if not disable_pbar:
        start = thisF
        pbar = tqdm(total=giveup_steps, desc="Bracketing...")
    while True:
        if thisF > 0:
            new_direction = -1
        else:
            new_direction = 1

        if new_direction != direction:
            if direction == 1:
                bracket[1] = x_0
            else:
                bracket[0] = x_0

            return bracket
        else:
            if direction == 1:
                bracket[0] = x_0
            else:
                bracket[1] = x_0

        if abs(x_0 - starting_x) > giveup_displacement:
            print("gaveup displacement")
            warn(f"Moved more than {giveup_displacement} while trying to find a sign change to bracket, but still failed! Assuming the other end is that infinity")
            return bracket

        if step_count > giveup_steps:
            print("gaveup steps")
            warn(f"Moved more than {giveup_steps} while trying to find a sign change to bracket, but still failed! Assuming the other end is that infinity")
            return bracket

        step_count += 1
        step = initial_step * adjusting_factor
        x_0 += step * direction
        thisF = f(x_0)
        if not disable_pbar:
            pbar.set_description(f"Bracketing: thisF={thisF:.3f}")
            pbar.update(1)


def Bisection_Method(f, init_bracket, tolerance, giveup_steps=100, disable_pbar=False, report=False):
    if init_bracket[0] == -np.inf and init_bracket[1] == np.inf:
        warn("bad bracket! can't have both infinities!")
        raise BaseException
    elif init_bracket[0] == -np.inf:
        warn("lower bound of bracket is -np.inf! assuming root is -np.inf")
        return -np.inf, np.nan
    elif init_bracket[1] == np.inf:
        warn("upper bound of bracket is np.inf! assuming root is np.inf")
        return np.inf, np.nan

    bracket = init_bracket
    steps = 0

    if not disable_pbar:
        log_base = 1/1.1
        inverse_log_tol = math.log(tolerance, log_base)
        pbar = tqdm(total=int(inverse_log_tol), desc="Root finding...")
    while True:
        errorBound = (bracket[1] - bracket[0]) / 2
        midpoint = (bracket[0] + bracket[1])/2
        thisF = f(midpoint)
        if not disable_pbar:
            inverse_log_error = math.log(thisF, log_base)

        if report:
            print(f"Bisection report: bracket, midpoint, thisF: {bracket}, {midpoint}, {thisF}")

        if thisF > tolerance:
            bracket[1] = midpoint
        elif thisF < -tolerance:
            bracket[0] = midpoint
        else:
            return midpoint, errorBound

        if steps > giveup_steps:
            warn("more than giveup steps while bisecting! returning best so far.")
            print('bisection gave up steps')
            return midpoint, errorBound

        steps += 1
        if not disable_pbar:
            new_inverse_log_error = math.log(thisF, log_base)
            if int(new_inverse_log_error) > int(inverse_log_error):
                pbar.update(int(new_inverse_log_error) - int(inverse_log_error))




def Monotone_Root_Find(f, x_0, tolerance, initial_step=1, adjusting_factor=2): #!!!consider specialized prufer root find or dx-paramaterized root find that dynamically adjusts dx as it zeros in on the root (early on, don't actually need as much precision as later). same goes for the rest of this category of functions.
    bracket = Monotone_Find_Bracket(f, x_0, initial_step=initial_step, adjusting_factor=adjusting_factor)
    root, errorBound = Bisection_Method(f, bracket, tolerance)

    return root, errorBound


def LSLP_FD2(lambda_, p_x, q_x, w_x, dp__dx_x, y_x, dy__dx_x): #LSLP is Lambda Sturm-Liouville Problem aka lambda is given. #returns the finite 2nd order difference as a function of lambda_, y_x, and dy__dx_x; where p_x,q_x,y_x are p(x),q(x),y(x) evaluated at x (think of solving an IVP using Euler's method)
    return (-lambda_*w_x*y_x + q_x*y_x - dp__dx_x * dy__dx_x)/p_x #hopefully i did my algebra right. if i did, then this is d^2y/dx^2

def exponential_fit(point_list):
    if type(point_list) != np.ndarray:
        point_list = np.array(point_list)
    a,b = sp.optimize.curve_fit(lambda r,a,b: a*math.e**(b*r), point_list[:,0], point_list[:,1])[0]
    return lambda r: a*math.e**(b*r)

def exponential_over_r_fit(point_list):
    if type(point_list) != np.ndarray:
        point_list = np.array(point_list)
    a,b = sp.optimize.curve_fit(lambda r,a,b: a*math.e**(b*r)/r, point_list[:,0], point_list[:,1])[0]
    return lambda r: a*math.e**(b*r)/r

#!!!use parareal
def Solve_LSLP_IVP(lambda_, p, q, w, x_init, y_init, dy__dx_init, x_end, dx, asymptotic_fitting_n=10000, asymptotic_start_thresh=16, asymptotic_clamping=False, parallel_pool=None, disable_pbar=False, store_solution=False): #for goodest results, dx should divide x_end-x_init #quadratic approximation, aka updates y at each time step as y + dy/dx dx + 1/2 d^2y/dx^2 dx^2
    x, y, dy__dx = x_init, y_init, dy__dx_init    
    indep_var_displacement = 0
    
    if asymptotic_clamping is True:
        asymptotic_clamper_fitter = exponential_fit
    
    fit_has_happened_already = False
        
    dp__dx = NumD(p, dx, cmp_step=True) #cmp_step=True
    zero_count = 0
    
    if store_solution:
        point_list = []
    elif asymptotic_clamping:
        point_list = []

    if dx<0:
        continue_condition = lambda x: x>x_end
    elif dx>0:
        continue_condition = lambda x: x<x_end

    if not disable_pbar:
        pbar = tqdm(desc="Simple IVP Solving...", total=((x_end-x_init)//dx)+1)
    while continue_condition(x):
        if store_solution:
            point_list.append((x,y))
        elif asymptotic_clamping:
            point_list.append((x,y))
            if len(point_list) > asymptotic_fitting_n:
                point_list.pop(0)
                
        if asymptotic_clamping and len(point_list) >= asymptotic_fitting_n and indep_var_displacement >= asymptotic_start_thresh and not fit_has_happened_already:
            asymptotic_clamper = asymptotic_clamper_fitter(point_list[-asymptotic_fitting_n:])
            fit_has_happened_already = True
            
        d2y__dx2 = LSLP_FD2(lambda_, p(x), q(x), w(x), dp__dx(x), y, dy__dx)
        x_new, y_new, dy__dx_new = x+dx, y+dy__dx*dx+.5*d2y__dx2*(dx**2), dy__dx + d2y__dx2*dx
        indep_var_displacement = x_new - x_init
        if asymptotic_clamping and fit_has_happened_already:
            predicted_val = asymptotic_clamper(x_new)
            y_sign = 1 if y > 0 else (-1 if y < 0 else 0)
            if abs(y_new) > abs(predicted_val):
                y_new = y_sign * predicted_val
                dy__dx_new = (y_new-y) / dx
        
        if (y_new < 0 and y > 0) or (y < 0 and y_new > 0):
            zero_count += 1 #by IVT
        
        x, y, dy__dx = x_new, y_new, dy__dx_new
        
        if not disable_pbar:
            pbar.update()
        

    if store_solution:
        return y, dy__dx, zero_count, point_list
    else:
        return y, dy__dx, zero_count

#!!!make a parallel version of this so that eigen_search gets parallel speedups for eigenfunc_given_eig finding.

''' Old Shooting Algorithm (without prufer)
def LR_Shots_Mismatch_lambda(lambda_, p, q, w, x_a, y_a, dy__dx_a, x_b, y_b, dy__dx_b, dx): #calculates the mismatch function at lambda by solving the LSLP_IVP at one boundary toward the other, stopping halfway. it takes the y's and the dy/dx's at the halfway point for each shot and then calculates left_y'*right_y - right_y'*left_y
    number_of_samples = (x_b - x_a)//dx

    halfway_indice = number_of_samples//2 #for all i know flooring still makes this just (x_b - x_a)/2 or (x_b - x_a)//2, but i'm pretty sure it isn't
    x_halfway = x_a + halfway_indice*dx

    left_shot = Solve_LSLP_IVP(lambda_, p, q, w, x_a, y_a, dy__dx_a, x_halfway, dx)
    right_shot = Solve_LSLP_IVP(lambda_, p, q, w, x_b, y_b, dy__dx_b, x_halfway, -dx)

    eigen_index = left_shot[2] + right_shot[2] #sum of the left and right roots is total number of roots is the eigenvalue index due to a theorem of SLP's.

    return p(x_halfway)*left_shot[1]*right_shot[0] - p(x_halfway)*right_shot[1]*left_shot[0], eigen_index

def Make_Mismatch(p, q, w, x_a, y_a, dy__dx_a, x_b, y_b, dy__dx_b, dx):
    #ideas to refactor: do the x_0 sample of all the eigens at once, then do the x_0+1dx, and so on. this way the p,q,w,dp__dx_x are still only computed once, but now don't need to be stored beyond that step.
    def Mismatch(lambda_, dx=dx): #returns the mismatch and the eigenindex
        return LR_Shots_Mismatch_lambda(lambda_, p, q, w, x_a, y_a, dy__dx_a, x_b, y_b, dy__dx_b, dx)

    return Mismatch
'''

def Prufer_FD1(lambda_, p_x, q_x, w_x, theta_x): #d\theta/dx = 1/p cos^2(\theta) + (\lambda * w - q) * sin^2(\theta)
    return (1/p_x)*(math.cos(theta_x)**2) + (lambda_ * w_x - q_x) * (math.sin(theta_x)**2) #this time, Veerle Ledoux did the algebra for me!
    #return (1/p_x)*(cmath.cos(theta_x)**2) + (lambda_ * w_x - q_x) * (cmath.sin(theta_x)**2) #this time, Veerle Ledoux did the algebra for me!


def original_boundary_conditions_to_prufer_shooters(p, x_a, y_a, dy__dx_a, x_b, y_b, dy__dx_b): #!!!verify complexification.
    if y_a != 0 and p(x_a)*dy__dx_a == 0:
        sign = abs(y_a)/y_a * abs(p(x_a))/p(x_a)
        theta_a = math.atan(sign*np.inf)
        #theta_a = cmath.acos((p(x_a)*dy__dx_a).real / (y_a*y_a.conjugate() + (p(x_a)*dy__dx_a)*(p(x_a)*dy__dx_a).conjugate())**.5)
    elif dy__dx_a != 0:
        theta_a = math.atan(y_a/(p(x_a)*dy__dx_a)) #the convention by verle ledoux puts this within 0,pi but the part that actually matters is that they are normalized in a way that the mismatch for the kth eigenvalue is k. this convention also does that. also means no do the atan2.
        #theta_a = cmath.acos((p(x_a)*dy__dx_a).real / (y_a*y_a.conjugate() + (p(x_a)*dy__dx_a)*(p(x_a)*dy__dx_a).conjugate())**.5)
    else:
        warn("Can't have boundary condition with wronskian [well, the almost wronskian because it uses py' instead of y']  0!")
        raise(BaseException) #zero boundary wronskian

    if y_b != 0 and p(x_b) * dy__dx_b == 0:
        sign = abs(y_b)/y_b * abs(p(x_b))/p(x_b)
        theta_b = math.atan(sign*np.inf)
        #theta_b = cmath.acos((p(x_b)*dy__dx_b).real / (y_b*y_b.conjugate() + (p(x_b)*dy__dx_b)*(p(x_b)*dy__dx_b).conjugate())**.5)
    elif p(x_b)* dy__dx_b != 0:
        theta_b = math.atan(y_b/(p(x_b)*dy__dx_b))
        #theta_b = cmath.acos((p(x_b)*dy__dx_b).real / (y_b*y_b.conjugate() + (p(x_b)*dy__dx_b)*(p(x_b)*dy__dx_b).conjugate())**.5)
    else:
        warn("Can't have boundary condition with wronskian [kinda wronskian cuz py' instead of p] [in this case, y, py'] = 0!")
        raise BaseException #zero boundary wronskian

    return [x_a, theta_a], [x_b, theta_b]

def Solve_Prufer_IVP(lambda_, p, q, w, x_init, theta_init, x_end, dx, store_solution=False):
    x, theta = x_init, theta_init
    if dx<0:
        continue_condition = lambda x: x>x_end
    elif dx>0:
        continue_condition = lambda x: x<x_end

    if store_solution:
        point_list = []
    while continue_condition(x):
        if store_solution:
            point_list.append((x,theta))
        dtheta__dx = Prufer_FD1(lambda_, p(x), q(x), w(x), theta)
        x_new, theta_new = x+dx, theta+dtheta__dx*dx
        x, theta = x_new, theta_new

    if store_solution:
        return theta, point_list
    else:
        return [theta]

def LR_Prufer_Shots_Mismatch_lambda(lambda_, p, q, w, x_a, theta_a, x_b, theta_b, dx):
    number_of_samples = (x_b - x_a)//dx

    halfway_indice = number_of_samples//2 #for all i know flooring still makes this just (x_b - x_a)/2 or (x_b - x_a)//2, but i'm pretty sure it isn't
    x_halfway = x_a + halfway_indice*dx

    left_shot = Solve_Prufer_IVP(lambda_, p, q, w, x_a, theta_a, x_halfway, dx)
    right_shot = Solve_Prufer_IVP(lambda_, p, q, w, x_b, theta_b, x_halfway, -dx)

    return (left_shot[0] - right_shot[0])/(2*math.pi) #the dividing by 2*math.pi is probably correct. unsure if it misses eigenvals.

def Make_Prufer_Mismatch(p, q, w, x_a, theta_a, x_b, theta_b, dx):
    def Prufer_Mismatch(lambda_, dx=dx):
        return LR_Prufer_Shots_Mismatch_lambda(lambda_, p, q, w, x_a, theta_a, x_b, theta_b, dx)

    return Prufer_Mismatch

def Make_Prufer_Mismatch_given_original_boundary_conditions(p, q, w, x_a, y_a, dy__dx_a, x_b, y_b, dy__dx_b, dx):
    shooter1, shooter2 = original_boundary_conditions_to_prufer_shooters(p, x_a, y_a, dy__dx_a, x_b, y_b, dy__dx_b)
    return Make_Prufer_Mismatch(p, q, w, *shooter1, *shooter2, dx)








def full_shot_from_start_and_end(start_shot_vector, end_shot_vector): #for schrodinger problems
    full_shot_vector = np.stack([np.concat(start_shot_vector, [0,0]), np.concat([0,0], end_shot_vector)]).transpose()
    return full_shot_vector

def bake_potentials_into_mesh(potential, baked_mesh, indep_var_mesh_key="mesh", disable_pbar=False): #is an inplace function #takes a baked_mesh, and outputs a baked_mesh with the potentials.
    #!!!find a way to parralelize this.     
    #baked so that the potential on the interval is stored at the left endpoint indice. the rightmost endpoint simply gets the potential value there as an approximation - the rest are averages of the potential at the endpoints.
    if type(baked_mesh) == dict: #it's a baked mesh
        length = len(baked_mesh[indep_var_mesh_key])
    else:
        raise(TypeError) #mesh_or_baked_mesh wasn't of an expected type!

    approximate_potentials = []
    for left_indice in tqdm(range(length), desc="Baking Potentials", disable=disable_pbar):
        approximate_potentials.append(potential(baked_mesh, left_indice))

    baked_mesh["potential_mesh"] = approximate_potentials
    return


#the following Ixaru stuff can be relatively easily obtained by solving the differential equation with constant potential approximation on each interval. the cos and sin and cosh and sinh come from the fundamental solutions of that, and the arguments of them (that is, Z) comes from the constant in the differential equation. putting delta in Z seems to just be a way to compute Z once and use it repeatedly, but to delta it just doesn't do anything as it squares and then square roots to place it into the trig funcs.
#modified version of what verle ledoux calls \xi and \eta_0, and they are used to construct the prop_mats by giving them a value Z(\delta). Z itself depends on the potential's approximation on the interval and on the claimed eigenvalue, and of course depends on \delta (which is the amout you are stepping forward)
#these are modified to avoid the silly redundancies like dividing by sqrt(abs(delta)**2) and then multiplying by delta^2 and then dividing by delta.
def Modified_Ixaru_Xi(pot_minus_lambda, delta):
    if pot_minus_lambda <= 0:
        try:
            return math.cos(delta * (abs(pot_minus_lambda)**.5))
        except ValueError:
            print("debbuging: value error for some reason")
            print(pot_minus_lambda)
            print(delta)
            breakpoint()
            print(math.cos(delta * (abs(pot_minus_lambda)**.5)))
    else:
        return math.cosh(delta * ((pot_minus_lambda)**.5))

def Modified_Ixaru_Eta_0(pot_minus_lambda, delta): #modified eta_0
    if pot_minus_lambda < 0:
        return -math.sin(delta * (abs(pot_minus_lambda)**.5))
    elif pot_minus_lambda == 0:
        return 1
    else:
        return math.sinh(delta * (pot_minus_lambda**.5))



def Make_Reference_Funcs(pot_minus_lambda, delta, xi_out, eta_out):
    #modified version of what ledoux just calls "Z". used to put into the reference propogators that form the prop mats.
    abs_z_over_delta_times_sqrt_abs_z = abs(pot_minus_lambda)**.5 #sign taken care of in modified_ixaru_eta_0
    #!!!handle div by 0
    reference_u = xi_out
    reference_u_prime = abs_z_over_delta_times_sqrt_abs_z * eta_out
    reference_v = abs(eta_out/abs_z_over_delta_times_sqrt_abs_z) if abs_z_over_delta_times_sqrt_abs_z!=0 else delta
    reference_v_prime = xi_out
    return reference_u, reference_u_prime, reference_v, reference_v_prime

def Make_Prop_Mat_Forward(pot_minus_lambda, delta, xi_out, eta_out):
    reference_u, reference_u_prime, reference_v, reference_v_prime = Make_Reference_Funcs(pot_minus_lambda, delta, xi_out, eta_out)
    return np.array([ [reference_u, reference_v], [reference_u_prime, reference_v_prime] ])

def Make_Prop_Mat_Backward(pot_minus_lambda, delta, xi_out, eta_out):
    reference_u, reference_u_prime, reference_v, reference_v_prime = Make_Reference_Funcs(pot_minus_lambda, delta, xi_out, eta_out)
    return np.array([ [reference_v_prime, -reference_v], [-reference_u_prime, reference_u] ])

def Bake_Delta_Forward(baked_mesh, indep_var_mesh_key="mesh", disable_pbar=True):
    baked_mesh["delta_forward_mesh"] = []
    for left_indice in tqdm(range(len(baked_mesh[indep_var_mesh_key])-1), desc="Baking delta_forwards...", disable=disable_pbar):
        baked_mesh["delta_forward_mesh"].append(baked_mesh[indep_var_mesh_key][left_indice+1] - baked_mesh[indep_var_mesh_key][left_indice])
    return

def forward_mat_for_one_baking_iter(this_pots_minus_lambda, this_delta_forward):
    xi_out_left = Modified_Ixaru_Xi(this_pots_minus_lambda, this_delta_forward)
    eta_out_left = Modified_Ixaru_Eta_0(this_pots_minus_lambda, this_delta_forward)
    return Make_Prop_Mat_Forward(this_pots_minus_lambda, this_delta_forward, xi_out_left, eta_out_left)

def backward_mat_for_one_baking_iter(this_pots_minus_lambda, this_delta_backward):
    xi_out_right = Modified_Ixaru_Xi(this_pots_minus_lambda, this_delta_backward)
    eta_out_right = Modified_Ixaru_Eta_0(this_pots_minus_lambda, this_delta_backward)
    return Make_Prop_Mat_Backward(this_pots_minus_lambda, this_delta_backward, xi_out_right, eta_out_right)

def Bake_Prop_Mats(baked_mesh, lambda_, indep_var_mesh_key="mesh", matching_point_index=None, disable_pbar=False, parallel_pool=None): #assumes mesh has delta_forward_mesh baked in
    if parallel_pool is None:
        parallel_pool = joblib.Parallel(n_jobs=8, verbose=VERBOSITY, batch_size=4096)
    if matching_point_index is None:
        matching_point_index = len(baked_mesh[indep_var_mesh_key])//2

    #baked_mesh["prop_mat_forward_mesh"] = []
    #baked_mesh["prop_mat_backward_mesh"] = [] #note that this will be backwards of what you would normally order it - this way, program can propogate left and right by just asking the left and right prop mats and then going up one index. as in, it's ordered according to a simultaneous shooter
    #left_index = 0
    #right_index = len(baked_mesh[indep_var_mesh_key])-1
    if not disable_pbar:
        print("Baking Prop Mats...")

    if type(baked_mesh['potential_mesh']) != np.ndarray:
        baked_mesh['potential_mesh'] = np.array(baked_mesh['potential_mesh'])


    pots_minus_lambda = baked_mesh["potential_mesh"] - lambda_
    delta_forwards = baked_mesh["delta_forward_mesh"]

    forward_generator = (joblib.delayed(forward_mat_for_one_baking_iter)(pots_minus_lambda[i], delta_forwards[i]) for i in range(matching_point_index))
    backward_generator = (joblib.delayed(backward_mat_for_one_baking_iter)(pots_minus_lambda[i], delta_forwards[i-1]) for i in range(len(pots_minus_lambda)-1, matching_point_index, -1))

    baked_mesh['prop_mat_forward_mesh'] = parallel_pool(forward_generator)
    baked_mesh['prop_mat_backward_mesh'] = parallel_pool(backward_generator)
    #baked_mesh["prop_mat_forward_mesh"] = map(forward_mat_for_one_baking_iter, zip(pots_minus_lambda[:matching_point_index], delta_forwards[:matching_point_index]))
    #baked_mesh["prop_mat_backward_mesh"] = map(backward_mat_for_one_baking_iter, zip(pots_minus_lambda[-1:matching_point_index:-1], delta_forwards[-2:matching_point_index-1:1])) #using delta_forwards[-2:matching_point_index-1:-1] as delta_backward = baked_mesh["delta_forward_mesh"][right_index-1]


    '''
    with tqdm(total=max(matching_point_index-left_index, right_index-matching_point_index), disable=disable_pbar) as pbar:
        while left_index < matching_point_index and right_index > matching_point_index:
            if left_index < matching_point_index:
                pot_minus_lambda_left = baked_mesh["potential_mesh"][left_index] - lambda_
                delta_forward = baked_mesh["delta_forward_mesh"][left_index]
                xi_out_left = Modified_Ixaru_Xi(pot_minus_lambda_left, delta_forward)
                eta_out_left = Modified_Ixaru_Eta_0(pot_minus_lambda_left, delta_forward)
                baked_mesh["prop_mat_forward_mesh"].append(Make_Prop_Mat_Forward(pot_minus_lambda_left, delta_forward, xi_out_left, eta_out_left))

            if right_index > matching_point_index: #this is literally just the inverse. because of relationships between the derivatives of sin sinh cos cosh they have this nice form.
                pot_minus_lambda_right = baked_mesh["potential_mesh"][right_index] - lambda_
                delta_backward = baked_mesh["delta_forward_mesh"][right_index-1]
                xi_out_right = Modified_Ixaru_Xi(pot_minus_lambda_right, delta_backward)
                eta_out_right = Modified_Ixaru_Eta_0(pot_minus_lambda_right, delta_backward)
                baked_mesh["prop_mat_backward_mesh"].append(Make_Prop_Mat_Backward(pot_minus_lambda_right, delta_backward, xi_out_right, eta_out_right))

            left_index += 1
            right_index -= 1
            pbar.update(1)
    '''
    return

def Bake_Local_Scaling_Factors(baked_mesh, global_scaling_factor, lambda_, indep_var_mesh_key="mesh", matching_point_index=None, disable_pbar=False):
    if matching_point_index is None:
        matching_point_index = len(baked_mesh[indep_var_mesh_key])//2

    baked_mesh["local_scaling_factor_left_mesh"] = []
    baked_mesh["local_scaling_factor_right_mesh"] = [] #same reversed caveat as in the bake prop mats function
    left_index = 0
    right_index = len(baked_mesh[indep_var_mesh_key])-1
    if not disable_pbar:
        print("Baking Local_Scaling_Factors...")
    with tqdm(total=max(matching_point_index-left_index, right_index-matching_point_index), disable=disable_pbar) as pbar:
        while left_index < matching_point_index and right_index > matching_point_index:
            if left_index < matching_point_index:
                if lambda_ > baked_mesh["potential_mesh"][left_index]:
                    baked_mesh["local_scaling_factor_left_mesh"].append((lambda_ - baked_mesh["potential_mesh"][left_index])**.5)
                else:
                    baked_mesh["local_scaling_factor_left_mesh"].append("Barrier Case!")

            if right_index > matching_point_index:
                if lambda_ > baked_mesh["potential_mesh"][right_index]:
                    baked_mesh["local_scaling_factor_right_mesh"].append((lambda_ - baked_mesh["potential_mesh"][right_index])**.5)
                else:
                    baked_mesh["local_scaling_factor_right_mesh"].append("Barrier Case!")

            left_index += 1
            right_index -= 1
            pbar.update(1)

def Adjust_Prufer_Angle(unadjusted_angle, rescaling_ratio): #rescaling ratio is new_scale/old_scale, and is denoted in many sources (pyrce, ledoux) as "\sigma", not to be confused with the unrelated "\sigma" used in the Liouville transformation.
    return unadjusted_angle + math.atan2((rescaling_ratio-1)*math.sin(rescaling_ratio)*math.cos(rescaling_ratio), 1 + (rescaling_ratio - 1) * (math.sin(rescaling_ratio)**2))

#ad_hoc_pi_epsilon=.01
def Constricted_Looping_Monotone_Angle_To_Total_Monotone_Angle(current_total_monotone_angle, constricted_looping_monotone_angle, prev_constricted_looping_monotone_angle, ad_hoc_pi_epsilon=.001, direction="up"): #constricted_looping_monotone_angle constricted between -\pi/2, \pi/2,  monotone increasing but loops
    intra_loop_angle_delta = constricted_looping_monotone_angle - prev_constricted_looping_monotone_angle
    '''
    if min(abs(intra_loop_angle_delta), abs(prev_intra_loop_angle_delta)) > abs(first_last_intra_loop_angle_delta):
        if direction=='up':
            if (prev_intra_loop_angle_delta > 0 and intra_loop_angle_delta < 0):
                print('triple step up')
                potentially_spiked_prev_constricted_looping_monotone_angle -= math.pi
        elif direction=='down':
            if (prev_intra_loop_angle_delta < 0 and intra_loop_angle_delta > 0):
                print('triple step down')
                potentially_spiked_prev_constricted_looping_monotone_angle +=  math.pi
        intra_loop_angle_delta = constricted_looping_monotone_angle - potentially_spiked_prev_constricted_looping_monotone_angle
    '''

    if abs(intra_loop_angle_delta) < math.pi - ad_hoc_pi_epsilon:
        current_total_monotone_angle += intra_loop_angle_delta
        return current_total_monotone_angle

    if direction=="up":
        if intra_loop_angle_delta < 0:
            current_total_monotone_angle += 0#math.pi + intra_loop_angle_delta
        else:
            current_total_monotone_angle += intra_loop_angle_delta
    elif direction=="down":
        if intra_loop_angle_delta > 0:
            current_total_monotone_angle += 0#-math.pi + intra_loop_angle_delta
        else:
            current_total_monotone_angle += intra_loop_angle_delta
    '''
    if adhoc_correct_jump_threshold < abs(prev_intra_loop_angle_delta):
        if direction=='up':
            if (prev_intra_loop_angle_delta > 0 and intra_loop_angle_delta < 0):
                if triple_total_intra_loop_angle_delta > 0:
                    print('triple step up')
                    potentially_spiked_prev_constricted_looping_monotone_angle -= math.pi
                elif triple_total_intra_loop_angle_delta < 0:
                    print('triple shift up')
                    current_total_monotone_angle += math.pi
        elif direction=='down':
            if (prev_intra_loop_angle_delta < 0 and intra_loop_angle_delta > 0):
                if triple_total_intra_loop_angle_delta < 0:
                    print('triple step down')
                    potentially_spiked_prev_constricted_looping_monotone_angle +=  math.pi
                elif triple_total_intra_loop_angle_delta > 0:
                    print('triple shift down')
                    current_total_monotone_angle -= math.pi
        intra_loop_angle_delta = constricted_looping_monotone_angle - potentially_spiked_prev_constricted_looping_monotone_angle
        return current_total_monotone_angle, potentially_spiked_prev_constricted_looping_monotone_angle
    '''
    return current_total_monotone_angle


def Step_Prufer_Left_Forward(baked_mesh, current_prufer_left, prev_intra_loop_angle_left, left_index, global_scaling_factor, current_left_shot_vector, next_left_shot_vector):
    local_scaling_factor = baked_mesh["local_scaling_factor_left_mesh"][left_index]
    #breakpoint()
    if local_scaling_factor=="Barrier Case!" and False: #barrier case: E<=V
        raw_end_theta = math.atan2(global_scaling_factor*next_left_shot_vector[0], next_left_shot_vector[1])
        prufer_left_global = math.atan2(global_scaling_factor*current_left_shot_vector[0], current_left_shot_vector[1])
        if (current_left_shot_vector[0] == 0 or next_left_shot_vector[0] == 0) or (current_left_shot_vector[0] > 0 and next_left_shot_vector[0] > 0) or (current_left_shot_vector[0] < 0 and next_left_shot_vector[0] < 0):
            if prufer_left_global > 0 and raw_end_theta < 0:
                final_right_angle = raw_end_theta + math.pi
            elif prufer_left_global < 0 and raw_end_theta > 0:
                final_right_angle = raw_end_theta - math.pi
            else:
                final_right_angle = raw_end_theta
        else:
            if (prufer_left_global > 0 and raw_end_theta > 0) or (prufer_left_global < 0 and raw_end_theta < 0):
                final_right_angle = raw_end_theta + math.pi
            else:
                final_right_angle = raw_end_theta

        new_total_prufer_right = Constricted_Looping_Monotone_Angle_To_Total_Monotone_Angle(current_prufer_left, final_right_angle, prev_intra_loop_angle_left)
        return new_total_prufer_right, final_right_angle

    else: #well case: E>V
        if local_scaling_factor == "Barrier Case!":
            local_scaling_factor = 1
        
        
        #ledoux uses "atan", but it should be "atan2" - this is probably just her forgetting to clarify or assuming the reader implicitly understood that fact.
        linear_unadjusted_right_angle = local_scaling_factor * baked_mesh["delta_forward_mesh"][left_index] + math.atan2(local_scaling_factor*current_left_shot_vector[0], current_left_shot_vector[1])
        end_constant_corrector = math.atan2(local_scaling_factor*next_left_shot_vector[0], next_left_shot_vector[1])

        #for whatever reason, integer part doesn't do anything? i'm assuming that ledoux somehow doesn't normalize prufer angles during the interval and this code definitely does and requires separate somewhat-adhoc corrections via the Constricted_Looping_Monotone_Angle_To_Total_Monotone_Angle function
        predicted_linear_phase = linear_unadjusted_right_angle
        phase_correction = end_constant_corrector-predicted_linear_phase
        #print(f"phase_correction raw: {phase_correction}")
        #print(f"predicted lin left {predicted_linear_phase}, end const corr left {end_constant_corrector}, lin unadjust left for right {linear_unadjusted_right_angle}")
        if phase_correction < -math.pi/2:
            range_canonicalization_corrector = math.pi
        elif phase_correction > math.pi/2:
            range_canonicalization_corrector = -math.pi
        else:
            range_canonicalization_corrector = 0

        if abs(phase_correction + range_canonicalization_corrector) > .01:
            #input('ahh!')
            pass

        #print(f"the thing: {linear_unadjusted_right_angle/math.pi}")

        final_unadjusted_right_angle = linear_unadjusted_right_angle + phase_correction + range_canonicalization_corrector
        final_right_angle = Adjust_Prufer_Angle(final_unadjusted_right_angle, global_scaling_factor/local_scaling_factor)
        #print(f"final_right: {final_right_angle}")
        new_total_prufer_right = Constricted_Looping_Monotone_Angle_To_Total_Monotone_Angle(current_prufer_left, final_right_angle, prev_intra_loop_angle_left)

        return new_total_prufer_right, final_right_angle



def Step_Prufer_Right_Backward(baked_mesh, current_prufer_right, prev_intra_loop_angle_right, logical_index, right_index, global_scaling_factor, current_right_shot_vector, next_right_shot_vector):
    local_scaling_factor = baked_mesh["local_scaling_factor_right_mesh"][logical_index]
    if local_scaling_factor=="Barrier Case!": #barrier case: E<=V
        raw_end_theta = math.atan2(global_scaling_factor*next_right_shot_vector[0], next_right_shot_vector[1])
        prufer_right_global = math.atan2(global_scaling_factor*next_right_shot_vector[0], next_right_shot_vector[1])
        if (current_right_shot_vector[0] == 0 or next_right_shot_vector[0] == 0) or (current_right_shot_vector[0] > 0 and next_right_shot_vector[0] > 0) or (current_right_shot_vector[0] < 0 and next_right_shot_vector[0] < 0):
            if prufer_right_global > 0 and raw_end_theta < 0:
                final_left_angle = raw_end_theta + math.pi
            elif prufer_right_global < 0 and raw_end_theta > 0:
                final_left_angle = raw_end_theta - math.pi
            else:
                final_left_angle = raw_end_theta
        else:
            if (prufer_right_global > 0 and raw_end_theta > 0) or (prufer_right_global < 0 and raw_end_theta < 0):
                final_left_angle = raw_end_theta + math.pi
            else:
                final_left_angle = raw_end_theta

        #print(f"still looping right angle: {final_left_angle}", end=' ')
        new_total_prufer_left = Constricted_Looping_Monotone_Angle_To_Total_Monotone_Angle(current_prufer_right, final_left_angle, prev_intra_loop_angle_right, direction="down")
        return new_total_prufer_left, final_left_angle

    else: #well case: E>V
        linear_unadjusted_left_angle = -local_scaling_factor * baked_mesh["delta_forward_mesh"][logical_index] + math.atan2(local_scaling_factor*current_right_shot_vector[0], current_right_shot_vector[1])
        end_constant_corrector = math.atan2(local_scaling_factor*next_right_shot_vector[0], next_right_shot_vector[1])

        #for whatever reason, integer part doesn't do anything? i'm assuming that ledoux somehow doesn't normalize prufer angles during the interval and this code definitely does and requires separate somewhat-adhoc corrections via the Constricted_Looping_Monotone_Angle_To_Total_Monotone_Angle function
        predicted_linear_phase = linear_unadjusted_left_angle
        phase_correction = end_constant_corrector - predicted_linear_phase
        #print(f"phase_correction raw: {phase_correction}")

        if phase_correction < -math.pi/2:
            range_canonicalization_corrector = math.pi
        elif phase_correction > math.pi/2:
            #print(f"phase_correction > math.pi/2, {phase_correction}, {end_constant_corrector}, {predicted_linear_phase}, {local_scaling_factor}")
            range_canonicalization_corrector = -math.pi
        else:
            range_canonicalization_corrector = 0

        if abs(phase_correction + range_canonicalization_corrector) > .01:
            #input('ahh')
            pass


        #print(f"the thing: {linear_unadjusted_left_angle/math.pi}")

        final_unadjusted_left_angle = linear_unadjusted_left_angle + phase_correction + range_canonicalization_corrector
        final_left_angle = Adjust_Prufer_Angle(final_unadjusted_left_angle, global_scaling_factor/local_scaling_factor)
        #print(f"still looping right angle: {final_left_angle}", end=' ')
        new_total_prufer_left = Constricted_Looping_Monotone_Angle_To_Total_Monotone_Angle(current_prufer_right, final_left_angle, prev_intra_loop_angle_right, direction="down")

        return new_total_prufer_left, final_left_angle,


def CPM_Method_Shoot_And_Mismatch(baked_mesh, left_shot_vector, right_shot_vector, lambda_, indep_var_mesh_key="mesh", indep_var_mesh_key_for_history_override=None, matching_point_index=None, disable_delta_forward_baking_pbar=True, disable_prop_baking_pbar=True, disable_shooting_pbar=False, disable_local_scaling_factors_baking_pbar=True, store_solution=False, parallel_pool=None, adhoc_two=False): #takes in the prop_mats for a schrodinger problem and the full_shot_vector is the initial shot at both endpoints giving left then right in a single vector. baked_mesh has a bunch of information, including the mesh itself, and is a dictionary so that i don't forget what the things are.
    if indep_var_mesh_key_for_history_override is None:
        indep_var_mesh_key_for_history_override = indep_var_mesh_key
    if parallel_pool is None:
        parallel_pool = joblib.Parallel(n_jobs=8, verbose=VERBOSITY, batch_size=4096)
    if "delta_forward_mesh" not in baked_mesh.keys():
        Bake_Delta_Forward(baked_mesh, indep_var_mesh_key=indep_var_mesh_key, disable_pbar=disable_delta_forward_baking_pbar)

    if matching_point_index is None:
        matching_point_index = len(baked_mesh[indep_var_mesh_key])//2

    global_scaling_factor_determiner = lambda_ - baked_mesh["potential_mesh"][matching_point_index]
    if global_scaling_factor_determiner == np.nan:
        print(f"determiner was nan. {lambda_} minus {baked_mesh['potential_mesh'][matching_point_index]}")
    if global_scaling_factor_determiner < 1:
        global_scaling_factor = 1
    elif global_scaling_factor_determiner >= 1:
        global_scaling_factor = (global_scaling_factor_determiner)**.5
    else:
        print(f"what? not greater or less than or equal to 1. {global_scaling_factor_determiner}")
        print(f"what follow up: {lambda_}, {baked_mesh['potential_mesh'][matching_point_index]}")

    Bake_Local_Scaling_Factors(baked_mesh, global_scaling_factor, lambda_, indep_var_mesh_key=indep_var_mesh_key, matching_point_index=matching_point_index, disable_pbar=disable_local_scaling_factors_baking_pbar)


    Bake_Prop_Mats(baked_mesh, lambda_, indep_var_mesh_key=indep_var_mesh_key, matching_point_index=matching_point_index, disable_pbar=disable_prop_baking_pbar, parallel_pool=parallel_pool)
    current_left_shot_vector, current_right_shot_vector = np.copy(left_shot_vector), np.copy(right_shot_vector)
    prufer_boundary_conditions = original_boundary_conditions_to_prufer_shooters(lambda x: 1, baked_mesh[indep_var_mesh_key][0], *left_shot_vector, baked_mesh[indep_var_mesh_key][-1], *right_shot_vector)
    current_prufer_left, current_prufer_right = prufer_boundary_conditions[0][1], prufer_boundary_conditions[1][1]
    #print(current_prufer_left, current_prufer_right)
    left_index = 0
    right_index = len(baked_mesh[indep_var_mesh_key])-1
    logical_index = 0
    prev_looping_angle_left = 0
    prev_looping_angle_right = 0
    #if not disable_shooting_pbar:
    #    print("Shooting...")

    if store_solution:
        history_left, history_right = [np.array([baked_mesh[indep_var_mesh_key_for_history_override][0], *current_left_shot_vector])], [np.array([baked_mesh[indep_var_mesh_key_for_history_override][-1], *current_right_shot_vector])]
    with tqdm(desc='Shooting...', total=max(matching_point_index-left_index, right_index-matching_point_index), disable=disable_shooting_pbar) as pbar:
        while left_index < matching_point_index and right_index > matching_point_index:
            #print('prufers:',end='')
            #print(current_prufer_left, current_prufer_right) #!!!probably sign error

            if left_index < matching_point_index:
                next_left_shot_vector = np.dot(baked_mesh["prop_mat_forward_mesh"][logical_index], current_left_shot_vector)
                #print(Step_Prufer_Left_Forward(baked_mesh, current_prufer_left, prev_looping_angle_left, left_index, global_scaling_factor, current_left_shot_vector, next_left_shot_vector))
                next_prufer_left, prev_looping_angle_left = Step_Prufer_Left_Forward(baked_mesh, current_prufer_left, prev_looping_angle_left, left_index, global_scaling_factor, current_left_shot_vector, next_left_shot_vector)

                current_left_shot_vector = next_left_shot_vector
                current_prufer_left = next_prufer_left
                if store_solution:
                    history_left.append(np.array([baked_mesh[indep_var_mesh_key_for_history_override][left_index], *current_left_shot_vector]))

            if right_index > matching_point_index:
                next_right_shot_vector = np.dot(baked_mesh["prop_mat_backward_mesh"][logical_index], current_right_shot_vector)
                next_prufer_right, prev_looping_angle_right  = Step_Prufer_Right_Backward(baked_mesh, current_prufer_right, prev_looping_angle_right, logical_index, right_index, global_scaling_factor, current_right_shot_vector, next_right_shot_vector)

                current_right_shot_vector = next_right_shot_vector
                current_prufer_right = next_prufer_right
                if store_solution:
                    history_right.append(np.array([baked_mesh[indep_var_mesh_key_for_history_override][right_index], *current_right_shot_vector]))

            left_index += 1
            logical_index += 1
            right_index -= 1
            pbar.update(1)
            #input(f"{current_prufer_left}, {current_prufer_right}")


    mismatch =  current_left_shot_vector[1]*current_right_shot_vector[0] - current_right_shot_vector[1]*current_left_shot_vector[0]

    final_left_prufer = current_prufer_left
    final_right_prufer = current_prufer_right
    theta_diff = final_left_prufer - final_right_prufer
    #print(f'diffs: {final_left_prufer} and {final_right_prufer}')
    #print(f'ending prufs shots: {final_left_prufer}, {final_right_prufer}')
    eigen_index = theta_diff/(2*math.pi)
    if adhoc_two:
        eigen_index *= 2
    if store_solution:
        #breakpoint()
        history_right.reverse()
        history_left.extend(history_right)
        return mismatch, eigen_index, history_left
    else:
        return mismatch, eigen_index
#correct for: mathieu with k_0=1 e_0=5, magnetic, coffey-evans with beta=20 as long as you make very close initial guesses,

'''
potential = lambda x: 10*math.cos(2*x) #mathieu problem with k_0 = 1 and e_0 = 5 according to https://www.researchgate.net/figure/46_tbl2_344652900
mesh = {"mesh": np.arange(0,math.pi,.01)}
bake_potentials_into_mesh(potential,mesh)
left_shot_vector, right_shot_vector = np.array([0,1]), np.array([0,1])
#generic_plot(mesh["mesh"], lambda x: mesh["potential_mesh"][np.where(mesh["mesh"]==x)[0][0]])
mis = lambda lambda_: CPM_Method_Shoot_And_Mismatch(mesh, left_shot_vector, right_shot_vector, lambda_)
generic_plot(np.arange(-10,110,.1), mis)
#first ten eigens according to source: -5.7901 2.0995 9.2363 16.6482 25.5109 36.3589 49.2696 64.2047 81.7732 100.6887
'''

'''
potential = lambda x: -2*20*math.cos(2*x) + 400*(math.sin(2*x)**2) #coffey-evans problem with beta = 20
mesh = {"mesh": np.arange(-math.pi/2,math.pi/2,.01)}
bake_potentials_into_mesh(potential,mesh)
left_shot_vector, right_shot_vector = np.array([0,1]), np.array([0,1])
#generic_plot(mesh["mesh"], lambda x: mesh["potential_mesh"][np.where(mesh["mesh"]==x)[0][0]])
mis = lambda lambda_: CPM_Method_Shoot_And_Mismatch(mesh, left_shot_vector, right_shot_vector, lambda_)
generic_plot(np.arange(0,100,.1), mis)
'''


def CPM_Convert_EigenvalBound_To_EigenindiceBound(delta_theta, search_bounds): #takes in delta_theta and search_bounds, outputs indices
    return math.floor(delta_theta(search_bounds[0])/math.pi), math.floor(delta_theta(search_bounds[1])/math.pi - 1)


#all the previous cpm stuff works only for schrodinger problems, that is one's that look like y''(x) - V(x) = -E y(x). fortunately, there's a transformation known as the liouville transformation that puts sturm liouville problems in "Liouville Normal Form" that has the right form. unfortunately, it's not pretty.
#for the following: x is the new independent variable, r is the old independent variable, V is the new potential, p q and w are old coefficient functions, z is the old dependent variable, y is the new dependent variable.

def Custom_Liouville_Integration(func, r_start, r_end, n=10): #!!!code this specially? meshify both the q integrand and the integration accumulated values.
    dx = (r_end - r_start) / n
    acc = 0
    for i in range(n): #MRAM
        acc += func(r_start + (i/2)*dx) * dx
    return acc

def find_contained_interval(val, mesh_of_vals): #suppose mesh_of_vals is a list of monotone vals (e.g. one representing a mesh). this function takes in a value, and finds the mesh vals/mesh "interval" that surrounds it. note that this really works for all ordered lists.
    lower_indice, upper_indice = 0, len(mesh_of_vals) - 1
    while True:
        indice_diff = upper_indice - lower_indice
        if indice_diff == 1:
            return lower_indice

        middle_indice = lower_indice + indice_diff//2
        middle_val = mesh_of_vals[middle_indice]
        if val >= middle_val:
            lower_indice = middle_indice
        elif val < middle_val:
            upper_indice = middle_indice

def Make_Liouville_Q_Integrand(p_of_r, w_of_r):#this is the integrand used to do a liouville transformation of the independent variables. x = \int_{r_{min}}^{r} \sqrt {\frac {w(r')} {p(r')}} dr'
    #!!!store this in a baked mesh, do the custom liouville integration baked into the mesh, as these are all energy independent values.
    return lambda r: (w_of_r(r)/p_of_r(r))**.5

def Dumb_Double_Mesh_Make(Q_Integrand, r_mesh, liouville_n=10, disable_coord_pbar=False):
    double_coordinate_mesh = {"r_mesh": r_mesh}
    x_vals = [0.0]
    for left_indice in tqdm(range(len(double_coordinate_mesh["r_mesh"])-1), desc="Making double_coordinate_mesh...", disable=disable_coord_pbar):
        net_change_in_x = Custom_Liouville_Integration(Q_Integrand,r_mesh[left_indice],r_mesh[left_indice+1], n=liouville_n) #also called Q.
        x_vals.append(x_vals[-1]+net_change_in_x)
    double_coordinate_mesh["x_mesh"] = x_vals
    return double_coordinate_mesh

def Smart_Double_Mesh_Make(p_of_r, w_of_r, r_min, r_max): #!!!code this
    raise(NotImplemented)

def Make_Smart_Find_X_Given_R(Q_Integrand, double_coordinate_mesh):
    def r_to_x(requested_r):
        lower_contained_indice = find_contained_interval(requested_r, double_coordinate_mesh["r_mesh"])
        return double_coordinate_mesh["x_mesh"][lower_contained_indice] + Custom_Liouville_Integration(Q_Integrand, double_coordinate_mesh["r_mesh"][lower_contained_indice], requested_r)
    return r_to_x
def Make_Smart_Find_R_Given_X(Q_Integrand, double_coordinate_mesh, tol=.001): #!!!make the tol changeable in other funcs
    def x_to_r(requested_x):
        lower_contained_indice = find_contained_interval(requested_x, double_coordinate_mesh["x_mesh"])
        func_with_zero_at_the_right_r_val = lambda r: Custom_Liouville_Integration(Q_Integrand, double_coordinate_mesh["r_mesh"][lower_contained_indice], r) - requested_x + double_coordinate_mesh["x_mesh"][lower_contained_indice]
        out_r = Secant_Method(func_with_zero_at_the_right_r_val, double_coordinate_mesh["r_mesh"][lower_contained_indice], double_coordinate_mesh["r_mesh"][lower_contained_indice]+.01, tol)
        return out_r
    return x_to_r

def Make_Liouville_Sigma_Given_R(p_of_r, w_of_r):
    def out(r):
        try:
            return (p_of_r(r) * w_of_r(r)) ** (-.25)
        except ZeroDivisionError:
            print(f"zero problem! r, p, w: {r}, {p_of_r(r)}, {w_of_r(r)}")
            print(f"going to do some definitely wrong guesses!")
            if p_of_r(r) == w_of_r(r):
                return 1
            else:
                return (p_of_r(r) * w_of_r(r)) ** (-.25)
    return out
    #return lambda r: (p_of_r(r) * w_of_r(r)) ** (-.25)

def Bake_Sigma(sigma_of_r, double_coordinate_mesh, r_var_name="r_mesh"):
    double_coordinate_mesh["sigma_mesh"] = []
    for r_indice in range(len(double_coordinate_mesh[r_var_name])):
        double_coordinate_mesh["sigma_mesh"].append(sigma_of_r(double_coordinate_mesh[r_var_name][r_indice]))
    return

def Find_Reciprocal_Sigma_2nd_Deriv_At_A_Good_Indice(baked_mesh, q_of_r, w_of_r, indice, r_var_mesh_name="r_mesh", x_var_mesh_name="x_mesh"):
    x = baked_mesh[x_var_mesh_name][indice]
    recip_sigma_of_r = 1/baked_mesh["sigma_mesh"][indice]
    x_b1 = baked_mesh[x_var_mesh_name][indice-1]
    recip_sigma_bd1 = recip_sigma_of_r - 1/baked_mesh["sigma_mesh"][indice-1]
    recip_sigma_deriv_x_b1 = (recip_sigma_bd1)/(x-x_b1)

    x_f1 = baked_mesh[x_var_mesh_name][indice+1]
    recip_sigma_fd1 = 1/baked_mesh["sigma_mesh"][indice+1] - recip_sigma_of_r
    recip_sigma_deriv_x_f1 = (recip_sigma_fd1)/(x_f1-x)
    recip_sigma_2nd_deriv_x = (recip_sigma_deriv_x_f1 - recip_sigma_deriv_x_b1)/( (x_f1-x)/2 - (x_b1-x)/2 )
    return recip_sigma_2nd_deriv_x

def Make_Potential_At_An_Indice(q_of_r, w_of_r, backup_sigma=None, backup_x_of_r=None, backup_r_of_x=None, r_var_mesh_name="r_mesh", x_var_mesh_name="x_mesh", dx=.001): #all the sources have this, but the only one that actually makes it clear what the equation means is J.D. Pryce's book Numerical Solution of Sturm Liouville Problems, and it really looks like everyone else copied the specific equation from Pyrce while forgetting to copy the explanatory note.
    if backup_sigma is None:
        def pot_of_x(baked_mesh, indice):
            original_r = baked_mesh[r_var_mesh_name][indice]
            if indice > 0 and indice < len(baked_mesh[x_var_mesh_name]) - 1:
                recip_sigma_2nd_deriv_x = Find_Reciprocal_Sigma_2nd_Deriv_At_A_Good_Indice(baked_mesh, q_of_r, w_of_r, indice, r_var_mesh_name="r_mesh", x_var_mesh_name="x_mesh")

            elif indice == 0: #assume quadratic so use 2nd deriv of nearby point
                recip_sigma_2nd_deriv_x = Find_Reciprocal_Sigma_2nd_Deriv_At_A_Good_Indice(baked_mesh, q_of_r, w_of_r, indice+1, r_var_mesh_name="r_mesh", x_var_mesh_name="x_mesh")

            elif indice == len(baked_mesh[x_var_mesh_name]) - 1:
                recip_sigma_2nd_deriv_x = Find_Reciprocal_Sigma_2nd_Deriv_At_A_Good_Indice(baked_mesh, q_of_r, w_of_r, indice-1, r_var_mesh_name="r_mesh", x_var_mesh_name="x_mesh")

            return q_of_r(original_r)/w_of_r(original_r) + baked_mesh["sigma_mesh"][indice] * recip_sigma_2nd_deriv_x
    else:
        def pot_of_x(baked_mesh, indice):
            r = backup_r_of_x(baked_mesh[x_var_mesh_name][indice])
            #return q_of_r(r)/w_of_r(r) + backup_sigma(r) * NumD(NumD(lambda x: 1/backup_sigma(backup_r_of_x(x)), dx, cmp_step=False), dx, cmp_step=False)(backup_x_of_r(r))
            return q_of_r(r)/w_of_r(r) + backup_sigma(r) * NumD(NumD(lambda x: 1/backup_sigma(x), dx, cmp_step=False), dx, cmp_step=False)(baked_mesh[x_var_mesh_name][indice])
    return pot_of_x

def Make_And_Bake_Potential_Of_X_Double_Coordinate_Mesh_Given_Original_Problem(p_of_r, q_of_r, w_of_r, r_mesh, liouville_n=10, dx=.001, disable_pot_pbar=False, disable_coord_pbar=False):
    Q_Integrand = Make_Liouville_Q_Integrand(p_of_r, w_of_r)
    double_coordinate_mesh = Dumb_Double_Mesh_Make(Q_Integrand, r_mesh, liouville_n=liouville_n, disable_coord_pbar=disable_coord_pbar)
    sigma_of_r = Make_Liouville_Sigma_Given_R(p_of_r, w_of_r)
    Bake_Sigma(sigma_of_r, double_coordinate_mesh)
    r_of_x = Make_Smart_Find_R_Given_X(Q_Integrand, double_coordinate_mesh)
    x_of_r = Make_Smart_Find_X_Given_R(Q_Integrand, double_coordinate_mesh)
    #backup_sigma=sigma_of_r, backup_x_of_r=x_of_r, backup_r_of_x=r_of_x
    #pot = Make_Potential_At_An_Indice(q_of_r, w_of_r, backup_sigma=sigma_of_r, backup_x_of_r=x_of_r, backup_r_of_x=r_of_x, dx=dx)
    pot = Make_Potential_At_An_Indice(q_of_r, w_of_r, backup_sigma=None, backup_x_of_r=x_of_r, backup_r_of_x=r_of_x, dx=dx)
    bake_potentials_into_mesh(pot, double_coordinate_mesh, indep_var_mesh_key="x_mesh", disable_pbar=disable_pot_pbar)
    return double_coordinate_mesh

def find_surrounding_indice_in_mono_arr(val, arr, use_ends_of_arr_if_not_in_arr=True):
    if val <= arr[0]:
        if use_ends_of_arr_if_not_in_arr:
            return [0, 0]
        else:
            raise ValueError("Val not in arr!")
    elif arr[-1] < val:
        if use_ends_of_arr_if_not_in_arr:
            return [len(arr)-1, len(arr)-1]
        else:
            raise ValueError("Val not in arr!")
    else:
        i = np.searchsorted(arr, val, side="left")
        return (i-1, i)
    
def find_indices_given_coords(coords, arrs):
    return [find_surrounding_indice_in_mono_arr(coords[i], arrs[i]) for i in range(len(coords))]

class MonotonizedFunc: #this is the way to make a function with memory. yes, the break from the functional-style of the rest of this code irks me too.    
    #!!!assumes monotone up - generalize this
    #!!!assumes known lists are sorted
    def __init__(self, func, index_to_monotonize=None, start_val=-.1, known_indep_var_coords=np.array([], dtype=np.float64), known_monotonized_and_non_monotonized_vals=np.empty((0,2), dtype=np.float64), other_indices_func_outs=np.empty((0,0)), stepping_from_anchor_dx=.001, sustain_thresh=5, is_a_jump_thresh=.45, disable_scaffold_climbing_pbar=False):  
        #breakpoint()
        insertion_index = known_indep_var_coords.searchsorted(start_val)
        if len(known_indep_var_coords)==0 or (known_indep_var_coords[insertion_index] != start_val and (insertion_index==0 or known_indep_var_coords[insertion_index-1])):
            known_indep_var_coords = np.insert(known_indep_var_coords, insertion_index, start_val)
            raw_start_func_val = func(start_val)
            if index_to_monotonize is not None:
                raw_start_func_val = list(raw_start_func_val)
            start_func_val = raw_start_func_val if index_to_monotonize is None else raw_start_func_val.pop(index_to_monotonize)
            if index_to_monotonize is not None:
                #breakpoint()
                if len(other_indices_func_outs) == 0:
                    other_indices_func_outs = np.array([raw_start_func_val])
                else:  
                    other_indices_func_outs = np.insert(other_indices_func_outs, insertion_index, raw_start_func_val, axis=0)
            
            known_monotonized_and_non_monotonized_vals = np.insert(known_monotonized_and_non_monotonized_vals, insertion_index, [start_func_val, start_func_val], axis=0)
        
        self.base_func = func
        self.index_to_monotonize = index_to_monotonize
        self.start_val = start_val
        self.stepping_from_anchor_dx = stepping_from_anchor_dx
        self.known_indep_var_coords = known_indep_var_coords
        self.known_monotonized_and_non_monotonized_vals = known_monotonized_and_non_monotonized_vals
        if index_to_monotonize is not None:
            self.other_indices_func_outs = other_indices_func_outs
        self.sustain_thresh = sustain_thresh
        self.is_a_jump_thresh = is_a_jump_thresh
        self.disable_scaffold_climbing_pbar = disable_scaffold_climbing_pbar
        
    def __call__(self, indep_var):
        #breakpoint()
        insertion_index = self.known_indep_var_coords.searchsorted(indep_var)
        
        if insertion_index == len(self.known_indep_var_coords):
            upper_coord = np.inf
        else:
            upper_coord = self.known_indep_var_coords[insertion_index]
            
        if insertion_index > 0:
            lower_coord = self.known_indep_var_coords[insertion_index-1]
        else:
            lower_coord = -np.inf
        
        
        if upper_coord == indep_var:
            if self.index_to_monotonize is None:
                return self.known_monotonized_and_non_monotonized_vals[insertion_index][0]
            else:
                out_at_monotonized_index = self.known_monotonized_and_non_monotonized_vals[insertion_index][0]
                other_outs =  self.other_indices_func_outs[insertion_index]
                all_outs = np.insert(other_outs, self.index_to_monotonize, out_at_monotonized_index)
                return all_outs
        elif lower_coord == indep_var:
            if self.index_to_monotonize is None:
                return self.known_monotonized_and_non_monotonized_vals[insertion_index-1][0]
            else:
                out_at_monotonized_index = self.known_monotonized_and_non_monotonized_vals[insertion_index-1][0]
                other_outs =  self.other_indices_func_outs[insertion_index-1]
                all_outs = np.insert(other_outs, self.index_to_monotonize, out_at_monotonized_index)
                return all_outs
                
        else:
            lower_coord_diff = indep_var - lower_coord
            upper_coord_diff = indep_var - upper_coord
            if abs(lower_coord_diff) <= abs(upper_coord_diff):
                which_is_anchor, anchor_coord = 'lower', lower_coord,
            else:
                which_is_anchor, anchor_coord = 'upper', upper_coord
            
            scaffolding_coords = np.arange(anchor_coord, indep_var, self.stepping_from_anchor_dx)[1:]
            if len(scaffolding_coords) == 0 or scaffolding_coords[-1] != indep_var:
                scaffolding_coords = np.append(scaffolding_coords, indep_var)
            
            prev_coord_index = insertion_index + (-1 if which_is_anchor=='lower' else 0)
            this_insertion_index = insertion_index
            buffered_next_func_vals_for_sustain_testing = []
            buffered_next_other_indices_vals_for_sustain_testing = []
            try_the_buffer_length=0
            for coord in tqdm(scaffolding_coords, desc="Climbing Monotone Scaffolding...", disable=self.disable_scaffold_climbing_pbar):
                if try_the_buffer_length > 0:
                    this_func_val = buffered_next_func_vals_for_sustain_testing.pop(0)
                    raw_this_func_val = buffered_next_other_indices_vals_for_sustain_testing.pop(0)
                    try_the_buffer_length -= 1
                else:
                    raw_this_func_val = list(self.base_func(coord)) if self.index_to_monotonize is not None else self.base_func(coord)
                    this_func_val = raw_this_func_val.pop(self.index_to_monotonize) if self.index_to_monotonize is not None else raw_this_func_val
                prev_monotonized_val, prev_non_monotonized_val = self.known_monotonized_and_non_monotonized_vals[prev_coord_index]
                func_diff = this_func_val - prev_non_monotonized_val
                if which_is_anchor == 'lower': #!!!generalize for monotone down
                    if func_diff >= 0:
                        if func_diff < self.is_a_jump_thresh:
                            this_monotonized_val = prev_monotonized_val + abs(func_diff)
                        else:
                            if try_the_buffer_length > 0:
                                passed = False
                                try_the_buffer_length -= 1
                            else:
                                buffered_next_func_vals_for_sustain_testing = []
                                buffered_next_other_indices_vals_for_sustain_testing = []
                                passed = True
                                prev_sustain_test_func_val = this_func_val
                                for x in np.append(np.arange(coord, coord+self.sustain_thresh*self.stepping_from_anchor_dx, self.stepping_from_anchor_dx)[1:], coord+self.sustain_thresh*self.stepping_from_anchor_dx):
                                    raw_sustain_test_val = list(self.base_func(x)) if self.index_to_monotonize is not None else self.base_func(coord)
                                    this_sustain_test_val = raw_sustain_test_val.pop(self.index_to_monotonize) if self.index_to_monotonize is not None else raw_sustain_test_val
                                    buffered_next_func_vals_for_sustain_testing.append(this_sustain_test_val)
                                    buffered_next_other_indices_vals_for_sustain_testing.append(raw_sustain_test_val)
                                    sustain_test_func_diff = this_sustain_test_val - prev_sustain_test_func_val
                                    if abs(sustain_test_func_diff) > self.is_a_jump_thresh or sustain_test_func_diff < 0:
                                        passed = False
                                        break
                                    prev_sustain_test_func_val = this_sustain_test_val
                                    
                                try_the_buffer_length = len(buffered_next_func_vals_for_sustain_testing)
                            
                            if passed:
                                this_monotonized_val = prev_monotonized_val + abs(func_diff)
                            else:
                                input(f"Fake jump at {coord}")
                                this_monotonized_val = prev_monotonized_val
                    else:
                        #breakpoint()
                        if try_the_buffer_length > 0:
                            passed = False
                            try_the_buffer_length -= 1
                        else:
                            buffered_next_func_vals_for_sustain_testing = []
                            buffered_next_other_indices_vals_for_sustain_testing = []
                            passed = True
                            prev_sustain_test_func_val = this_func_val
                            for x in np.append(np.arange(coord, coord+self.sustain_thresh*self.stepping_from_anchor_dx, self.stepping_from_anchor_dx)[1:], coord+self.sustain_thresh*self.stepping_from_anchor_dx):
                                raw_sustain_test_val = list(self.base_func(x)) if self.index_to_monotonize is not None else self.base_func(coord)
                                this_sustain_test_val = raw_sustain_test_val.pop(self.index_to_monotonize) if self.index_to_monotonize is not None else raw_sustain_test_val
                                buffered_next_func_vals_for_sustain_testing.append(this_sustain_test_val)
                                buffered_next_other_indices_vals_for_sustain_testing.append(raw_sustain_test_val)
                                sustain_test_func_diff = this_sustain_test_val - prev_sustain_test_func_val
                                if abs(sustain_test_func_diff) > self.is_a_jump_thresh or sustain_test_func_diff < 0:
                                    passed = False
                                    break
                                prev_sustain_test_func_val = this_sustain_test_val
                                
                            try_the_buffer_length = len(buffered_next_func_vals_for_sustain_testing)
                        
                        if passed:
                            this_monotonized_val = prev_monotonized_val + abs(func_diff)
                        else:
                            input(f"Fake jump at {coord}")
                            this_monotonized_val = prev_monotonized_val
                elif which_is_anchor == 'upper': #!!!generalize for monotone down
                    if func_diff <= 0:
                        if func_diff > self.is_a_jump_thresh:
                            this_monotonized_val = prev_monotonized_val - abs(func_diff)
                        else:
                            if try_the_buffer_length > 0:
                                passed = False
                                try_the_buffer_length -= 1
                            else:
                                buffered_next_func_vals_for_sustain_testing = []
                                buffered_next_other_indices_vals_for_sustain_testing = []
                                passed = True
                                prev_sustain_test_func_val = this_func_val
                                for x in np.append(np.arange(coord, coord-self.sustain_thresh*self.stepping_from_anchor_dx, -self.stepping_from_anchor_dx)[1:], coord-self.sustain_thresh*self.stepping_from_anchor_dx):
                                    raw_sustain_test_val = list(self.base_func(x)) if self.index_to_monotonize is not None else self.base_func(coord)
                                    this_sustain_test_val = raw_sustain_test_val.pop(self.index_to_monotonize) if self.index_to_monotonize is not None else raw_sustain_test_val
                                    buffered_next_func_vals_for_sustain_testing.append(this_sustain_test_val)
                                    buffered_next_other_indices_vals_for_sustain_testing.append(raw_sustain_test_val)
                                    sustain_test_func_diff = this_sustain_test_val - prev_sustain_test_func_val
                                    if abs(sustain_test_func_diff) > self.is_a_jump_thresh or sustain_test_func_diff > 0:
                                        passed = False
                                        break
                                    prev_sustain_test_func_val = this_sustain_test_val
                                    
                                try_the_buffer_length = len(buffered_next_func_vals_for_sustain_testing)
                                
                            if passed:
                                this_monotonized_val = prev_monotonized_val - abs(func_diff)
                            else:
                                input(f"Fake jump at {coord}")
                                this_monotonized_val = prev_monotonized_val
                    else:
                        breakpoint()
                        if try_the_buffer_length > 0:
                            passed = False
                            try_the_buffer_length -= 1
                        else:
                            buffered_next_func_vals_for_sustain_testing = []
                            buffered_next_other_indices_vals_for_sustain_testing = []
                            passed = True
                            prev_sustain_test_func_val = this_func_val
                            for x in np.append(np.arange(coord, coord-self.sustain_thresh*self.stepping_from_anchor_dx, -self.stepping_from_anchor_dx)[1:], coord-self.sustain_thresh*self.stepping_from_anchor_dx):
                                raw_sustain_test_val = list(self.base_func(x)) if self.index_to_monotonize is not None else self.base_func(coord)
                                this_sustain_test_val = raw_sustain_test_val.pop(self.index_to_monotonize) if self.index_to_monotonize is not None else raw_sustain_test_val
                                buffered_next_func_vals_for_sustain_testing.append(this_sustain_test_val)
                                buffered_next_other_indices_vals_for_sustain_testing.append(raw_sustain_test_val)
                                sustain_test_func_diff = this_sustain_test_val - prev_sustain_test_func_val
                                if abs(sustain_test_func_diff) > .5 or sustain_test_func_diff > 0:
                                    passed = False
                                    break
                                prev_sustain_test_func_val = this_sustain_test_val
                                
                            try_the_buffer_length = len(buffered_next_func_vals_for_sustain_testing)
                            
                        if passed:
                            this_monotonized_val = prev_monotonized_val - abs(func_diff)
                        else:
                            input(f"Fake jump at {coord}")
                            this_monotonized_val = prev_monotonized_val
                self.known_indep_var_coords = np.insert(self.known_indep_var_coords, this_insertion_index, coord)
                self.known_monotonized_and_non_monotonized_vals = np.insert(self.known_monotonized_and_non_monotonized_vals, this_insertion_index, [this_monotonized_val, this_func_val], axis=0)
                if self.index_to_monotonize is not None:
                    self.other_indices_func_outs = np.insert(self.other_indices_func_outs, this_insertion_index, raw_this_func_val, axis=0)
                
                if which_is_anchor == 'lower':
                    prev_coord_index += 1
                    this_insertion_index += 1
                elif which_is_anchor == 'upper':
                    continue
                
              
            if self.index_to_monotonize is None:
                return self.known_monotonized_and_non_monotonized_vals[prev_coord_index][0]
            else:
                out_at_monotonized_index = self.known_monotonized_and_non_monotonized_vals[prev_coord_index][0]
                other_outs =  self.other_indices_func_outs[prev_coord_index]
                all_outs = np.insert(other_outs, self.index_to_monotonize, out_at_monotonized_index)
                return all_outs


def CPM_Method_Liouville_Mismatch(p_of_r, q_of_r, w_of_r, r_mesh, init_left_shot_vector, init_right_shot_vector, mesh_was_already_baked=False, liouville_n=10, dx=.001, disable_shooting_pbar=True, disable_pot_pbar=False, disable_coord_pbar=False, disable_local_scaling_factors_baking_pbar=True, store_solution=False, parallel_pool=None, adhoc_two=False, force_monotone=False, force_monotone_start_val=-.1, stepping_from_anchor_dx=.001):
    if parallel_pool is None:
        parallel_pool = joblib.Parallel(n_jobs=8, verbose=VERBOSITY, batch_size=4096)

    if not mesh_was_already_baked:
        pot_of_x_baked_double_coordinate_mesh = Make_And_Bake_Potential_Of_X_Double_Coordinate_Mesh_Given_Original_Problem(p_of_r, q_of_r, w_of_r, r_mesh, liouville_n=liouville_n, dx=dx, disable_pot_pbar=disable_pot_pbar, disable_coord_pbar=disable_coord_pbar)
    else:
        pot_of_x_baked_double_coordinate_mesh = r_mesh
    #r_mesh is now a double_coordinate_mesh baked with potentials
    def mis(lambda_):
        if lambda_ > 9.15:
            #breakpoint()
            pass
        with parallel_pool as pool:
            return CPM_Method_Shoot_And_Mismatch(pot_of_x_baked_double_coordinate_mesh, init_left_shot_vector, init_right_shot_vector, lambda_, indep_var_mesh_key="x_mesh", indep_var_mesh_key_for_history_override='r_mesh', matching_point_index=None, disable_delta_forward_baking_pbar=True, disable_prop_baking_pbar=True, disable_shooting_pbar=disable_shooting_pbar, disable_local_scaling_factors_baking_pbar=disable_local_scaling_factors_baking_pbar, store_solution=store_solution, parallel_pool=pool, adhoc_two=adhoc_two)
    return pot_of_x_baked_double_coordinate_mesh, MonotonizedFunc(mis, start_val=force_monotone_start_val, stepping_from_anchor_dx=stepping_from_anchor_dx, index_to_monotonize=1) if force_monotone else mis
#works for azimuthal (magnetic quantum numbers), polar (azimuthal quantum numbers), radial. prufer not so much - not at all monotonic sometimes.for radial problem, has dips at eigen values possibly?



def check_stable_part_of_unstable_monotone(unstable_monotone, x_val_to_test, sustained_monotone_radius=.01, uni_directional_sustained_monotone_step_sample_count=2, sustained_monotone_success_sample_thresh=.75):
    upper_test_lim, lower_test_lim = x_val_to_test + sustained_monotone_radius, x_val_to_test - sustained_monotone_radius
    upper_x_test_vals = np.arange(x_val_to_test, upper_test_lim, (upper_test_lim - x_val_to_test) / uni_directional_sustained_monotone_step_sample_count)
    lower_x_test_vals = np.arange(x_val_to_test, lower_test_lim, (lower_test_lim - x_val_to_test) / uni_directional_sustained_monotone_step_sample_count)

    func_val_at_test = unstable_monotone(x_val_to_test)
    upper_test_result = 0
    for upper_x in upper_x_test_vals:
        upper_test_result += int(unstable_monotone(upper_x) >= func_val_at_test)

    lower_test_result = 0
    for lower_x in lower_x_test_vals:
        lower_test_result += int(unstable_monotone(lower_x) <= func_val_at_test)

    return bool( (upper_test_result + lower_test_result) / (len(upper_x_test_vals) + len(lower_x_test_vals)) >= sustained_monotone_success_sample_thresh) #the bool here is really only for clarity.



def find_candidate_roots(tailored_prufer, displacement_mag_quit_thresh=30, stop_at_candidate_roots_num_thresh=3, stop_at_total_steps_total_thresh=100, stop_at_total_steps_per_root_thresh=10, subsequent_root_agreement_tol=.001, interval_tol=.001, mis_val_tol=.01, start_val=0, start_direction="right", start_step_mag=.001, step_mag_increase_factor=2, sustained_monotone_radius_factor_as_ratio_of_prev_step=.1, uni_directional_sustained_monotone_step_sample_count=2, sustained_monotone_success_sample_thresh=.75, fake_root_restart_movement_factor_as_ratio_of_start_step=1): #mis_and_cpm_prufer_mis is a func taking lambda_ and outputting [mismatch, eigen_index]
    #breakpoint()
    total_steps_total = 0
    current_interval = [-np.inf, np.inf]
    current_x = start_val
    original_step_mag = start_step_mag
    current_step_mag = start_step_mag
    prev_direction = start_direction
    total_steps_for_this_root = 0
    candidate_roots = []
    pbar = tqdm(desc="Finding Candidate Roots...", total=stop_at_total_steps_total_thresh)
    while total_steps_total < stop_at_total_steps_total_thresh and len(candidate_roots) < stop_at_candidate_roots_num_thresh:
        #breakpoint()
        #find a candidate root
        interval_found = False
        while total_steps_for_this_root < stop_at_total_steps_per_root_thresh and total_steps_total < stop_at_total_steps_total_thresh:
            if abs(current_x-start_val) >= displacement_mag_quit_thresh:
                print("displacement mag exceeded displacement_mag_quit_thresh")
                curr_pruf_val = tailored_prufer(current_x)
                if curr_pruf_val < 0:
                    return [[-np.inf, current_x - start_val, {"no_root_found"}]]
                elif curr_pruf_val > 0:
                    return [[np.inf, current_x - start_val, {"no_root_found"}]]

            if not interval_found:
                #breakpoint()
                curr_pruf_val = tailored_prufer(current_x)
                if curr_pruf_val < 0: #don't put the 0-mis_val_tol here --- tol checking is in the main stopping conditional, and this way if it's in the mis_val_tol but the interval still hasn't been found it will still look for an interval.
                    current_interval[0] = max(current_interval[0], current_x)

                    if prev_direction == "right":
                        current_step_mag *= step_mag_increase_factor
                        current_x += current_step_mag
                    elif prev_direction == "left":
                        if check_stable_part_of_unstable_monotone(tailored_prufer, current_x, sustained_monotone_radius=.01, uni_directional_sustained_monotone_step_sample_count=2, sustained_monotone_success_sample_thresh=.75):
                            prev_direction = "right"
                            current_step_mag /= step_mag_increase_factor
                            current_x += current_step_mag
                        else: #keep going in prev_direction
                            current_step_mag *= step_mag_increase_factor
                            current_x -= current_step_mag


                elif curr_pruf_val > 0:
                    current_interval[1] = min(current_interval[1], current_x)

                    if prev_direction == "left":
                        current_step_mag *= step_mag_increase_factor
                        current_x -= current_step_mag
                    elif prev_direction == "right":
                        if check_stable_part_of_unstable_monotone(tailored_prufer, current_x, sustained_monotone_radius=.01, uni_directional_sustained_monotone_step_sample_count=2, sustained_monotone_success_sample_thresh=.75):
                            prev_direction = "left"
                            current_step_mag /= step_mag_increase_factor
                            current_x -= current_step_mag
                        else: #keep going in prev_direction
                            current_step_mag *= step_mag_increase_factor
                            current_x += current_step_mag

                if abs(current_interval[0]) != np.inf and abs(current_interval[1]) != np.inf:
                    interval_found = True

            else:
                current_x = (current_interval[0] + current_interval[1])/2
                curr_pruf_val = tailored_prufer(current_x)
                if curr_pruf_val < 0:
                    current_interval[0] = current_x
                elif curr_pruf_val > 0:
                    current_interval[1] = current_x


            if (abs(curr_pruf_val) < mis_val_tol and (current_interval[1]!=np.inf and current_interval[0]!=np.inf)) or abs(current_interval[1] - current_interval[0]) < interval_tol:
                candidate_roots.append( [(current_interval[0]+current_interval[1])/2, (current_interval[1] - current_interval[0]), {"untagged"}] ) #average, error_radius
                break

            elif curr_pruf_val==0:
                candidate_roots.append( [current_x, 0, {"untagged"}] )
                break

            total_steps_for_this_root += 1
            total_steps_total += 1
            pbar.update()

        else: #else in the while means that this branch executes if the condition in the while is false but not if there was a break statement.#one of the total step threshes has been exceeded
            if total_steps_total >= stop_at_total_steps_total_thresh:
                if curr_pruf_val < 0:
                    print("total_steps_total exceeded stop_at_total_steps_total_thresh!")
                    return [[-np.inf, current_x - start_val, {"no_root_found"}]]
                elif curr_pruf_val > 0:
                    return [[np.inf, current_x - start_val, {"no_root_found"}]]
            else: # we know it's because the this root thresh is exceeded.
                print("total_steps_for_this_root exceeded stop_at_total_steps_per_root_thresh!")
                if curr_pruf_val < 0:
                    return [[-np.inf, current_x - start_val, {"no_root_found"}]]
                elif curr_pruf_val > 0:
                    return [[np.inf, current_x - start_val, {"no_root_found"}]]


        if len(candidate_roots) >= 2 and ((candidate_roots[-2][0] + subsequent_root_agreement_tol < candidate_roots[-1][0] < candidate_roots[-2][0] - subsequent_root_agreement_tol) or ((candidate_roots[-2][0] - candidate_roots[-2][1]) < (candidate_roots[-1][0] + candidate_roots[-1][1])) or ((candidate_roots[-1][0] - candidate_roots[-1][1]) < (candidate_roots[-2][0] + candidate_roots[-2][1]))):
            candidate_roots.pop()
            candidate_roots[-1][2].discard("untagged")
            candidate_roots[-1][2].add("subsequent_candidate_root_agreement")
            break
        elif len(candidate_roots) != 0:
            current_interval = [-np.inf, np.inf]
            current_x = candidate_roots[-1][0] + candidate_roots[-1][1] + original_step_mag*fake_root_restart_movement_factor_as_ratio_of_start_step
            current_step_mag = start_step_mag
            prev_direction = start_direction
            total_steps_for_this_root = 0
        else:
            print("no candidate roots after first search! breaking the while")
            break
    else:
        if len(candidate_roots) >= stop_at_candidate_roots_num_thresh:
            return candidate_roots
        elif total_steps_total >=  stop_at_total_steps_total_thresh:
            if len(candidate_roots) == 0:
                print("Exceeded total_steps_total, but no candidate roots!")
                warn("Exceeded total_steps_total, but no candidate roots!")
                if curr_pruf_val < 0:
                    return [[-np.inf, current_x - start_val, {"no_root_found"}]]
                elif curr_pruf_val > 0:
                    return [[np.inf, current_x - start_val, {"no_root_found"}]]
                else:
                    warn("this shouldn't be possible")
                    raise ArithmeticError #impossible condition, pru val is 0 but no candidate roots after exceeding total steps
            else:
                return candidate_roots

    if "subsequent_candidate_root_agreement" in candidate_roots[-1][2]:
        return candidate_roots
    else:
        warn("this shouldn't be possible")
        raise ArithmeticError #impossible condition, shouldn't have gotten here without returning already

def find_stable_roots_in_mis_and_cpm_prufer(mis_and_cpm_prufer, target_index, sanity_check_tol=.1, sanity_check_step_for_zero_radius_pruf=.005, stop_at_candidate_roots_num_thresh=3, stop_at_total_steps_total_thresh=50, stop_at_total_steps_per_root_thresh=30, subsequent_root_agreement_tol=.001, interval_tol=.0001, pruf_mis_val_tol=.01, start_val=0, start_direction="right", start_step_mag=.001, step_mag_increase_factor=2, sustained_monotone_radius_factor_as_ratio_of_prev_step=.1, uni_directional_sustained_monotone_step_sample_count=3, sustained_monotone_success_sample_thresh=.75, fake_root_restart_movement_factor_as_ratio_of_start_step=1):
    #breakpoint()
    mis = lambda lambda_: mis_and_cpm_prufer(lambda_)[0]
    tailored_prufer = lambda lambda_: mis_and_cpm_prufer(lambda_)[1] - target_index
    candidate_roots = find_candidate_roots(tailored_prufer, stop_at_candidate_roots_num_thresh=stop_at_candidate_roots_num_thresh, stop_at_total_steps_total_thresh=stop_at_total_steps_total_thresh, stop_at_total_steps_per_root_thresh=stop_at_total_steps_per_root_thresh, subsequent_root_agreement_tol=subsequent_root_agreement_tol, interval_tol=interval_tol, mis_val_tol=pruf_mis_val_tol, start_val=start_val, start_direction=start_direction, start_step_mag=start_step_mag, step_mag_increase_factor=step_mag_increase_factor, sustained_monotone_radius_factor_as_ratio_of_prev_step=sustained_monotone_radius_factor_as_ratio_of_prev_step, uni_directional_sustained_monotone_step_sample_count=uni_directional_sustained_monotone_step_sample_count, sustained_monotone_success_sample_thresh=sustained_monotone_success_sample_thresh, fake_root_restart_movement_factor_as_ratio_of_start_step=fake_root_restart_movement_factor_as_ratio_of_start_step)
    verified_roots = []
    print("Testing Candidate Roots...")
    for candidate_root in tqdm(candidate_roots):
        print(f"Testing {candidate_root}")
        #breakpoint()
        if abs(candidate_root[0]) != np.inf and ("no_root_found" not in candidate_root[2]):
            if abs(mis(candidate_root[0])) < sanity_check_tol:
                verified_roots.append(candidate_root)

            if abs(candidate_root[1]) == 0:
                left_sanity_check, right_sanity_check = candidate_root[0] - sanity_check_step_for_zero_radius_pruf, candidate_root[0] + sanity_check_step_for_zero_radius_pruf
            else:
                left_sanity_check, right_sanity_check = candidate_root[0] - candidate_root[1], candidate_root[0] + candidate_root[1]

            left_sanity_result, right_sanity_result = mis(left_sanity_check), mis(right_sanity_check)
            sanity_check_signs = np.sign([left_sanity_result, right_sanity_result])
            same_sign_bool = sanity_check_signs[0]*sanity_check_signs[1]
            if same_sign_bool == -1:
                verified_roots.append(candidate_root)

        else:
            continue

    if len(verified_roots) == 0:
        return [[None, np.inf, {'no_root_found'}]]
    else:
        return verified_roots



def find_ith_eigen_val_given_Prufer_Mismatch(Prufer_Mismatch, eigen_index, tolerance, guess_0=0, initial_step=1, adjusting_factor=2): #i is the eigen_index
    tailored_prufer_mismatch = lambda lambda_: Prufer_Mismatch(lambda_) - eigen_index
    eigen_val, error = Monotone_Root_Find(tailored_prufer_mismatch, guess_0, tolerance, initial_step=initial_step, adjusting_factor=adjusting_factor)
    return eigen_val, error

def find_first_n_eigen_val_given_Prufer_Mismatch(Prufer_Mismatch, up_to_n_eigens, tolerance, custom_eigen_list=False, initial_guess_0=0, initial_step=1, adjusting_factor=2): #custom_eigen_list overrides up_to_n_eigens #since monotonic, and since we tend to approach from the left for most problems, and from general way things work, using the left works better
    if not custom_eigen_list:
        custom_eigen_list = range(up_to_n_eigens)

    guess_0 = initial_guess_0
    out_list = []
    for eigen_index in tqdm(custom_eigen_list):
        eigen_val, error = find_ith_eigen_val_given_Prufer_Mismatch(Prufer_Mismatch, eigen_index, tolerance, guess_0=guess_0, initial_step=initial_step, adjusting_factor=adjusting_factor)
        if abs(eigen_val)==np.inf: #throw out infinite eigen_val
            continue

        out_list.append([eigen_val, error])
        guess_0 = eigen_val

    return out_list


def boundary_conditions_to_shooters(conditions, mode="periodic"): #takes in some form of boundary conditions and outputs the shooter conditions. the solutions to the shooters for a given eigenvalue are a basis for the eigenspace of that eigenvalue. i'm pretty sure that's right. hopefully they do indeed have the same eigenvalue...
    if mode=="periodic":
        #conditions = [x_a, x_b]
        #so y(a) = y(b) and y'(a) = y'(b), and we look at the interval y(a) to y(b). then the solutions are a 2D space. you can represent the solutions by their initial conditions, so [y(a), y'(a)] and then once you found the solutions to 2 lin indep init cond vectors, you then have a basis for the solution space. this is the space of solutions given an eigen value
        #this function takes in the periodic conditions and outputs two pairs of shooter conditions, each pair being solvable for eigensolutions and eigenvalues. each eigenvalue has solutions that are linear combinations of the two eigensolutions thus generated. at least, i'm pretty sure that's how that works.
        #shooters = x_a, y_a, dy__dx_a, x_b, y_b, dy__dx_b
        shooters1 = conditions[0], 1, 0, conditions[1], 1, 0 #this is one normalized initial condition vector in the basis
        shooters2 = conditions[0], 0, 1, conditions[1], 0, 1 #this is the other.
        return shooters1, shooters2




def generic_plot(x_vals, func, disable_pbar=False, title=None):
    if title is None:
        title = str(datetime.datetime.today())
    y_vals = [func(x_val) for x_val in tqdm(x_vals, desc='Calculating Plot Values...', disable=disable_pbar)]

    min_y, max_y = min(y_vals), max(y_vals)
    y_abs_limit = min([abs(min_y), abs(max_y)])
    y_range = [-y_abs_limit, y_abs_limit]

    pw = pg.plot(x_vals, y_vals, pen='r') #this is the plot widget
    origin_x_axis = pg.InfiniteLine(0, 0, pen={'color':'b', 'style':DashLine})
    pi = pw.getPlotItem()
    pi.setTitle(title)
    vb = pi.getViewBox()
    vb.setYRange(*y_range)
    vb.addItem(origin_x_axis)

def plot_mismatch(possible_eigens, mismatch):
    mismatch_vals, eigen_indices = np.vectorize(mismatch)(possible_eigens)

    min_miss, max_miss = np.min(mismatch_vals), np.max(mismatch_vals)
    y_abs_limit = np.min([abs(min_miss), abs(max_miss)])
    y_range = [-y_abs_limit, y_abs_limit]

    pw = pg.plot(possible_eigens, mismatch_vals, pen='r') #this is the plot widget
    origin_x_axis = pg.InfiniteLine(0, 0, pen={'color':'b', 'style':DashLine})
    pi = pw.getPlotItem()
    vb = pi.getViewBox()
    vb.setYRange(*y_range)
    vb.addItem(origin_x_axis)

def Make_Displacement_Newton(func, dlambda=.1):
    #for a prufer mismatch:
    #add the sought index/f' to get the displacement_newton_function [which tells you where newton's method will move to get the next iteration] for the tailored prufer mismatch for that index [since -(f-i)/(f-i)' = -(f-i)/f' = -f/f' + i/f'. what this means is that if the displacement_newton_function equals i/f' at a point, then that point is a fixed point of newton's method on the tailored prufer mismatch for the ith eigenval.
    return lambda x: -func(x)/NumD(func, dlambda, cmp_step=False)(x)

def plot_many_funcs(list_of_list_of_x_vals, funcs, hue_converge=True, val_converge=True):
    pw = pg.plot()
    pi = pw.getPlotItem()
    index = 0
    total_indices = len(list_of_list_of_x_vals)
    for x_vals in tqdm(list_of_list_of_x_vals):
        x_vals = np.array(x_vals)
        phi = (1 + 5**0.5) / 2
        if hue_converge:
            hue = (index+1)/(total_indices*2) #here index+1 is not actually necessary, could just do index, but keeping with the pattern [that is required] seen elsewhere in this function. #multiply the denominator by 2 to get hue from 0 to .5 to get from red to green instead of red to cyan.
        else:
            hue = ((index+1)*phi)%1 #if you multiply by phi mod 1 you get equidistribution by that one theorem (some source said phi was special for this, but i had thought it was true of all irrationals...). do index+1 times phi so that the start color isn't black so it's visible on the default (and morally superior) black background of pyqtgraph graphs

        if val_converge:
            val = (index+1)/(total_indices*2) + .5 #ensures val is at least .5, which makes the early entries still visible while being darker.
        else:
            val = 1

        uniqueish_pen = pg.mkPen(hsv=[hue, 1.0, val])
        pi.plot(list_of_list_of_x_vals[index], np.vectorize(funcs[index])(x_vals), pen=uniqueish_pen)

        index +=1

    origin_x_axis = pg.InfiniteLine(0, 0, pen={'color':'b', 'style':DashLine})

    vb = pi.getViewBox()
    vb.addItem(origin_x_axis)


def plot_eigen_funcs(list_of_list_of_coords, eigen_vals, color_eigen_as_zero_precision=3):
    pw = pg.plot()
    pi = pw.getPlotItem()
    lowest_eigen_val = min(abs(eigen_val) for eigen_val in eigen_vals if abs(round(eigen_val, color_eigen_as_zero_precision)) > 0) #hugely negative and positive eigens are darker.
    index = 0
    for coord_list in list_of_list_of_coords:
        #coord_list = np.array(coord_list)
        eigen_val = eigen_vals[index]
        phi = (1 + 5**0.5) / 2
        hue = ((index+1)*phi)%1 #if you multiply by phi mod 1 you get equidistribution by that one theorem (some source said phi was special for this, but i had thought it was true of all irrationals...). do index+1 times phi so that the start color isn't black so it's visible on the default (and morally superior) black background of pyqtgraph graphs
        val = 1/math.log(abs(eigen_val/lowest_eigen_val), 3) if eigen_val>lowest_eigen_val else 1.0
        val = val if 0<val<=1 else 1
        uniqueish_pen = pg.mkPen(hsv=[hue, 1.0, val])
        pi.plot(coord_list[:, 0], coord_list[:, 1], pen=uniqueish_pen)

        index +=1

    origin_x_axis = pg.InfiniteLine(0, 0, pen={'color':'b', 'style':DashLine})

    vb = pi.getViewBox()
    vb.addItem(origin_x_axis)





#prufer tests

'''
#azimuth (magnetic quantum number)
p,q,w = lambda x:1, lambda x: 0, lambda x: 1 #azimuthal equation
x_a, y_a, dy__dx_a, x_b, y_b, dy__dx_b = boundary_conditions_to_shooters([0, 2*math.pi])[1]
#d^2y/dx^2 = -\lambda y with periodic boundary conditions y(0)=y(2*\pi)=0. y'(0)=y'(2*\pi)=1 This is the equation for the azimuthal angle. The eigen_vals are the eigenvalues of -d^2/dx^2, and therefore the magnetic quantum number m=sqrt(\lambda)
'''


'''#polar (azimuthal quantum number)
m=3
boundary_epsilon = .01
p,q,w = lambda x: math.sin(x), lambda x: m**2/math.sin(x), lambda x: math.sin(x)
init_wronsk = [0,1]
x_a, y_a, dy__dx_a, x_b, y_b, dy__dx_b = 0+boundary_epsilon, *init_wronsk, math.pi-boundary_epsilon, *init_wronsk
'''


'''
#radial
electron_mass, proton_mass = constants.electron_mass, constants.proton_mass
reduced_mass = electron_mass*proton_mass/(electron_mass+proton_mass)

electron_charge = constants.elementary_charge
vaccuum_permittivity = constants.epsilon_0

hbar = constants.hbar


l = 1
boundary_epsilon = .01
boundary_inf_approx = 100
#p,q,w = lambda x: x**2, lambda x: l*(l+1) - ((2*reduced_mass)*(x**2)/(hbar**2)) * ((electron_charge**2)/(4*math.pi*vaccuum_permittivity*x)), lambda x: ((2*reduced_mass)*(x**2)/(hbar**2))
coulomb_z = 1/2
p,q,w = lambda x: 1, lambda x: l*(l+1)/(x**2) - (2*coulomb_z)/(x), lambda x: 1
deriv_at_zero, deriv_at_inf = 1,1
free_boundary_params = [deriv_at_zero, deriv_at_inf]
x_a, y_a, dy__dx_a, x_b, y_b, dy__dx_b = 0+boundary_epsilon, 0, free_boundary_params[0], boundary_inf_approx, 0, free_boundary_params[1]
'''





'''
prufer_mismatch = Make_Prufer_Mismatch_given_original_boundary_conditions(p, q, w, x_a, y_a, dy__dx_a, x_b, y_b, dy__dx_b, dx)
generic_plot(np.arange(-.1,0,.01), prufer_mismatch)
'''

'''
prufer_mismatches_param_by_dx = lambda dx: (lambda x: Make_Prufer_Mismatch_given_original_boundary_conditions(p, q, w, x_a, y_a, dy__dx_a, x_b, y_b, dy__dx_b, dx)(x) - 1)
displacement_newton_param_by_dx = lambda dx: Make_Displacement_Newton(prufer_mismatches_param_by_dx(dx))
index_addition_scaling_funcs_param_by_dx = lambda dx: lambda lambda_: 1/NumD(prufer_mismatches_param_by_dx(dx), dx, cmp_step=False)(lambda_) #on the plot, i_p*these + (the newton function for i=0) = (the newton function for i=i_p)

dx_list = np.arange(.02, .001, -.001)
displacement_funcs = np.vectorize(displacement_newton_param_by_dx)(dx_list)
index_addition_scaling_funcs = np.vectorize(index_addition_scaling_funcs_param_by_dx)(dx_list)
interleaved_list = np.reshape(np.column_stack((displacement_funcs, index_addition_scaling_funcs)), -1) #shape dimension of -1 lets numpy figure out the length of the flattened interleaving from knowing it should by flat

list_of_x_vals = np.tile(np.arange(.1,100,.1), [len(dx_list)*2,1])
plot_many_funcs(list_of_x_vals, interleaved_list)
'''

'''
displacement_newton_param_by_i = lambda i: Make_Displacement_Newton(lambda x: Make_Prufer_Mismatch_given_original_boundary_conditions(p, q, w, x_a, y_a, dy__dx_a, x_b, y_b, dy__dx_b, dx)(x) - i)
i_list = np.array(range(0,10))
list_of_x_vals = np.tile(np.arange(.1,16,.1), [len(i_list),1])
plot_many_funcs(list_of_x_vals, np.vectorize(displacement_newton_param_by_i)(i_list))
'''


'''
dx = .001
tolerance = .01
up_to_n_eigens = 3

prufer_mismatch = Make_Prufer_Mismatch_given_original_boundary_conditions(p, q, w, x_a, y_a, dy__dx_a, x_b, y_b, dy__dx_b, dx)
#generic_plot(np.arange(-10,20,.1), prufer_mismatch)
eigen_vals = np.array(list(find_first_n_eigen_val_given_Prufer_Mismatch(prufer_mismatch, up_to_n_eigens, tolerance)))[:,0].round(5)
eigen_funcs = np.array(list(find_eigen_funcs_given_eigen_vals(eigen_vals, p, q, w, x_a, y_a, dy__dx_a, x_b, dx*2)))
plot_eigen_funcs(eigen_funcs, eigen_vals)
#magnetic_quantum_numbers = np.sqrt(eigen_vals)
'''

'''
while True:
    free_boundary_params = [float(i) for i in input('derivs: ').split(',')]
    boundary_epsilon, boundary_inf_approx = [float(i) for i in input('boundaries: ').split(',')]
    func_vals = [float(i) for i in input('func_vals: ').split(',')]
    x_a, y_a, dy__dx_a, x_b, y_b, dy__dx_b = 0+boundary_epsilon, func_vals[0], free_boundary_params[0], boundary_inf_approx, func_vals[1], free_boundary_params[1]

    prufer_mismatch = Make_Prufer_Mismatch_given_original_boundary_conditions(p, q, w, x_a, y_a, dy__dx_a, x_b, y_b, dy__dx_b, dx)
    eigen_vals = np.array(list(find_first_n_eigen_val_given_Prufer_Mismatch(prufer_mismatch, up_to_n_eigens, tolerance, custom_eigen_list=range(1,up_to_n_eigens))))[:,0].round(5)
    eigen_funcs = np.array(list(find_eigen_funcs_given_eigen_vals(eigen_vals, p, q, w, x_a, y_a, dy__dx_a, x_b, boundary_inf_approx/100)))
    plot_eigen_funcs(eigen_funcs, eigen_vals)
    print("look!")
    print(eigen_vals)
    print(free_boundary_params)
    print(boundary_epsilon, boundary_inf_approx)
    print(func_vals)
    while True:
        try:
            app.processEvents()
        except KeyboardInterrupt:
            break
'''




#cpm tests

# Something happens here


#azimuthal equation (magnetic quantum numbers)

def azi(which_init=1):
    p_of_r, q_of_r, w_of_r = lambda x: 1, lambda x: 0, lambda x: 1
    mesh_dr = .001
    newton_tol = .001
    r_mesh = np.arange(0, 2*math.pi, mesh_dr)
    init_left_shot_vector, init_right_shot_vector = [[0,1], [1,0]][which_init], [[0,1], [1,0]][which_init]
    pool = joblib.Parallel(n_jobs=8, verbose=VERBOSITY, batch_size=4096)
    baked_mesh, mis_eigen = CPM_Method_Liouville_Mismatch(p_of_r, q_of_r, w_of_r, r_mesh, init_left_shot_vector, init_right_shot_vector, dx=.01, parallel_pool=pool)
    generic_plot(np.arange(-.1, 10, .2), lambda e: mis_eigen(e)[1])
    #guess = 2#float(input("guess: "))
    #print(Secant_Method(mis_eigen, guess, guess+.01, newton_tol, f_index_to_rootfind=0))




#polar (azimuthal quantum numbers) equation
def polar(which_init=1):
    m=3
    p_of_r, q_of_r, w_of_r = lambda x: math.sin(x), lambda x: m**2/math.sin(x), lambda x: math.sin(x)

    boundary_epsilon = .001

    mesh_dr_left = .001

    mesh_r_mid_left = .2
    mesh_dr_mid = .01
    mesh_r_mid_right = .8

    mesh_dr_end = .001

    newton_tol=.001
    r_mesh = np.concatenate(np.array([np.arange(boundary_epsilon, mesh_r_mid_left, mesh_dr_left), np.arange(mesh_r_mid_left, mesh_r_mid_right, mesh_dr_mid), np.arange(mesh_r_mid_right, math.pi-boundary_epsilon, mesh_dr_end)], dtype=object))
    init_left_shot_vector_0, init_right_shot_vector_0 = [[0,1], [1,0]][which_init], [[0,1], [1,0]][which_init]
    baked_mesh_0, mis_eigen_0 = CPM_Method_Liouville_Mismatch(p_of_r, q_of_r, w_of_r, r_mesh, init_left_shot_vector_0, init_right_shot_vector_0, dx=.01, adhoc_two=False)
    #breakpoint()
    #while True:
        #guess = float(input("guess: "))
        #print(Secant_Method(mis_eigen, guess, guess+.01, newton_tol, f_index_to_rootfind=0))
    #generic_plot(np.arange(0, 30.1, .1), lambda l: mis_eigen_0(l)[1])
    generic_plot(np.arange(9.16087, 9.1742, .0000005), lambda l: mis_eigen_0(l)[1])
    
#radial
def radial(which_init=0, boundary_epsilon=.01, mid_r_start=1, mid_r_end=49, boundary_inf_approx=50, mesh_dr_start=.01, mesh_dr_mid=.01, mesh_dr_end=.01, Num_D_sigma_dx=.0001, liouville_n=2):
    #electron_mass, proton_mass = constants.electron_mass, constants.proton_mass
    #reduced_mass = electron_mass*proton_mass/(electron_mass+proton_mass)
    #electron_charge = constants.elementary_charge
    #vaccuum_permittivity = constants.epsilon_0
    #hbar = constants.hbar

    l = 1
 
    #p_of_r,q_of_r,w_of_r = lambda x: x**2, lambda x: l*(l+1) - ( ((2*reduced_mass*(x**2))/(hbar**2)) * ((electron_charge**2)/(4*math.pi*vaccuum_permittivity*x))  ), lambda x: ((2*reduced_mass*(x**2))/(hbar**2))
    #p_of_r,q_of_r,w_of_r = lambda x: 1, lambda x: l*(l+1)/(x**2) - (1/x), lambda x: 1
    p_of_r,q_of_r,w_of_r = lambda x: x**2, lambda x: l*(l+1) - (2*x), lambda x: 2*(x**2)


    #mesh_dr_start = .0001
    #mesh_dr_mid = .001
    #mesh_dr_end = .1

    r_mesh_start = np.arange(boundary_epsilon, mid_r_start, mesh_dr_start)
    r_mesh_mid = np.arange(mid_r_start, mid_r_end, mesh_dr_mid)
    r_mesh_end = np.arange(mid_r_end, boundary_inf_approx, mesh_dr_end)
    r_mesh = np.concatenate([r_mesh_start, r_mesh_mid, r_mesh_end])
    init_left_shot_vector, init_right_shot_vector = [[0,1], [1,0]][which_init], [[0,1], [1,0]][which_init]


    #!!!very high mismatch value...for the correct eigenvalue.
    pool = joblib.Parallel(n_jobs=8, verbose=VERBOSITY, batch_size=4096)
    #breakpoint()
    disable_shooting_pbar = False
    disable_coord_pbar = False
    disable_pot_pbar = False
    store_solution = False
    force_monotone = True
    baked_mesh, mis_eigen = CPM_Method_Liouville_Mismatch(p_of_r,q_of_r,w_of_r, r_mesh, init_left_shot_vector, init_right_shot_vector, liouville_n=liouville_n, dx=Num_D_sigma_dx, disable_shooting_pbar=disable_shooting_pbar, disable_coord_pbar=disable_coord_pbar, disable_pot_pbar=disable_pot_pbar, store_solution=store_solution, parallel_pool=pool, force_monotone=force_monotone, force_monotone_start_val=-.1261, stepping_from_anchor_dx=.001, adhoc_two=False)
    #breakpoint()
    #print("plotting")
    #generic_plot(np.arange(-.1251, -.124, .00025), lambda e: mis_eigen(e)[1])
    generic_plot(np.arange(-.1261, -.03, .001), lambda e: mis_eigen(e)[1])
    #generic_plot(np.arange(-.126, -.01, .001), lambda e: mis_eigen(e)[1]) 
    #breakpoint()
    #eig_out = mis_eigen(-.0625)
    #plot_data = eig_out[2]
    #plot_data = np.array(plot_data)
    #mis_eigen(-.0625)[1]
#radial()

    #stable_roots = lambda index: find_stable_roots_in_mis_and_cpm_prufer(mis_eigen, index)
    #print([stable_roots(i) for i in range(1,2)])

#boundary_epsilon=.0001,
#boundary_inf_approx=1000,
#mesh_dr_start=.00001, mesh_dr_mid=.001, mesh_dr_end=.01, Num_D_sigma_dx=.0001

'''
import time
filename = 'parameter_testing.txt'
def parameter_test():
    args_pre_prod = [np.arange(-2.0,-6.0,-1.0), np.arange(2,4.0,1.0), np.arange(-2.0,-6.0,-1.0), np.arange(-1.0,-4.0,-1.0), np.arange(0.0,-4.0,-1.0), np.arange(-2.0, -5.0, -1.0)]
    args_prod = itertools.product(*args_pre_prod)
    with open('parameter_testing.txt', 'r') as f:
        text = f.read()
    left_off = int(text.split('\n')[-1])
    computed_data = text.split('\n')[:-2]
    text = ''
    for line in computed_data:
        text += line + '\n'
    with open('parameter_testing.txt', 'w') as f:
        f.write(text)

    args_prod = list(args_prod)[left_off:]
    i = left_off
    for (p1,p2,p3,p4,p5,p6) in tqdm(args_prod, total=len(args_prod), desc="Parameter Testing..."):
        these_args = [10.0**p1, 10.0**p2, 10.0**p3, 10.0**p4, 10.0**p5, 10.0**p6]
        with open('parameter_testing.txt', 'a') as f:
            f.write(str(these_args)+'; ')
            try:
                start_time = time.time()
                err=1-radial(boundary_epsilon=these_args[0], boundary_inf_approx=these_args[1], mesh_dr_start=these_args[2], mesh_dr_mid=these_args[3], mesh_dr_end=these_args[4], Num_D_sigma_dx=these_args[5])
                end_time = time.time()
                dt = end_time - start_time
                f.write(str(err))
                f.write(f'; {dt}; {math.log(err, 10**-1)/(dt)}')
            except KeyboardInterrupt:
                f.write(f'{np.nan}; {np.nan}; {np.nan}\n{i}')
                raise(KeyboardInterrupt)
            except:
                f.write(f'{np.nan}; {np.nan}; {np.nan}')
            f.write('\n')
            i += 1

def read_test_results():
    with open('parameter_testing.txt', 'r') as f:
        text=f.read()
    entries = text.split('\n')
    left_off = entries.pop()
    nanned = entries.pop()
    for i in range(len(entries)):
        entry = entries[i]
        entry = entry.split('; ')
        entry = [*eval(entry[0]), float(entry[1]), float(entry[2]), float(entry[3])]
        entries[i] = entry
    return np.array(entries)

data = read_test_results()
sort = np.flipud(data[data[:, 8].argsort()])
'''

'''
while True:
    guess = float(input("guess: "))
    print(Secant_Method(mis, guess, guess+.01, .01))
'''
