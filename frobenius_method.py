# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 10:06:28 2022

@author: Sri
"""


from warnings import warn

import numpy as np
import math

'''
for sturm liouville problem d/dx [p(x) y'(x)] - q(x)y(x) = -\lambda * w(x)y(x)
so py'' + p'y' + (\lambda * w(x) - q)y = 0
so y'' + (p'/p) y' + ( (\lambda * w(x) - q)/p ) y = 0
using https://en.wikipedia.org/wiki/Frobenius_method and converting the conventions so that wikipedia's u, z, p/z, q/z^2 -> our y, x-center, (p'/p), ( (\lambda * w(x) - q)/p ) respectively.
therefore wiki's p, q -> our (p'/p) (x-center), ( (\lambda * w(x) - q)/p )(x-center)^2
'''


def FD1(f, dx, cmp_step=False):  # returns Finite Differences of order 1
    if cmp_step:
        # since this is quantum mechanics, things tend to be complex valued. assuming continuity is equivalent to assuming it's holomorphic is equivalent to analytic. expanding f(x + ih) as a taylor series and then taking the imaginary part, f'(x) = Im(f(x+ih))/h + O(h^2). this also doesn't have as bad issues of floating point error when h is small, as unlike finite differences, this is well conditioned.
        # only works for first order for reasons.
        return lambda x: f(x+dx*1j).imag/dx
    else:
        return lambda x: (f(x+dx)-f(x-dx))/(2*dx)


def FDN(f, n, dx):  # returns Finite Differences of order n, using
    recip_dxN = 1/(dx**n)

    def nth_deriv(x):
        acc = 0
        for i in range(0, n+1):
            acc += recip_dxN * (-1)**(i) * (math.comb(n, i)) * f(x + (n/2 - i)*dx)

        return acc

    return nth_deriv


# for even slightly bad functions, doesn't give the actual limit nor recognizing the lack of a limit. just averages around the limit point for some delta.
def Limitable_Func(func, limiting_epsilon, give_up_iters=100, diverge_thresh=1000, diverge_thresh_iters_wait=20):
    def out(point_to_limit):
        def converging_x_seq(i):
            return point_to_limit + (-1)**(i) * (.75)**(i)
        def func_seq(i):
            return func(converging_x_seq(i))
        i = 3
        f_0, f_1 = func_seq(1), func_seq(2)
        while abs(f_1 - f_0) >= limiting_epsilon:
            f_0, f_1 = f_1, func_seq(i)
            i += 1
            if i > give_up_iters:
                warn("More than give_up_iters have passed without meeting cauchy condition for epsilon = limiting_epsilon! This function may not converge! The latest value was still returned.")
                return f_1

            if abs(f_1 - f_0) >= diverge_thresh and i >= diverge_thresh_iters_wait:
                warn("Forward Diff exceeded diverge_thresh after waiting diverge_thresh_iters_wait iters. It probably diverges, so this function, boldly hoping it still converges in the extended reals, either returned infinity or -infinity")
                if f_1 > 0:
                    return np.inf
                elif f_1 < 0:
                    return -np.inf


        return f_1

    return out


def Make_Analytic_Funcs(p, q, w, lambda_, singular_point, dx, limiting_delta):
    dp__dx = FD1(p, dx)

    #!!!this is the wrong part?
    analytic_for_y_func = Limitable_Func(lambda x: (x-singular_point)**2 * (lambda_*w(x-x_singular) - q(x-x_singular)) / p(x-x_singular) + bool(print(f"77: {x}")), limiting_delta)
    analytic_for_dy__dx_func = Limitable_Func(lambda x: (x-singular_point) * dp__dx(x-x_singular) / p(x-x_singular) + bool(print(f"78: {x}")), limiting_delta)

    return analytic_for_y_func, analytic_for_dy__dx_func


# using our conventions.
def Make_Indicial(analytic_for_y_coeff, analytic_for_dy__dx_coeff):
    return lambda r: r*(r-1) + analytic_for_dy__dx_coeff * r + analytic_for_y_coeff

# i needs to be greater than 0. if it's 0, then the coeff can i think be anything


# the ith coeff is the coeff to x^(i+r) in the series, past coeff is in ascending order.
'''
def ith_solution_series_coeff_given_indicial_and_analytic_coeffs_and_r_and_past_coeffs(indicial, r, analytic_for_y_func, analytic_for_dy__dx_func, singular_point, past_coeffs, i, dx):
    acc = 0
    prev_part_fact = 1
    for j in range(i-1, -1, -1):
        prev_part_fact *= i-j
        #acc += (bool(print(f"96: j, i: {j, i}")) + (j+r) * FDN(lambda x: analytic_for_y_func(x) + bool(print(f"96.1: {x}")), i-j, dx)(singular_point) + FDN(lambda x: analytic_for_dy__dx_func(x) + bool(print(f"96.2: {x}")), i-j, dx)(singular_point)) / prev_part_fact * past_coeffs[j]
        fdif = (bool(print(f"96: j, i: {j, i}")) + (j+r) * analytic_for_y_func(singular_point)**(i-j) + analytic_for_dy__dx_func(singular_point)**(i-j) / prev_part_fact * past_coeffs[j])
        acc += fdif
        input(str(fdif)+' line 100')


    return acc * -1/(indicial(i+r))
'''

#!!!for some reason, only last term in sum is used.
#for the besesl function of order 1/2
def ith_solution_series_coeff_given_indicial_and_analytic_coeffs_and_r_and_past_coeffs(indicial, r, analytic_for_y_func, analytic_for_dy__dx_func, singular_point, past_coeffs, i, dx):
    this_coeff = (-1)**(i) * (bool(print(f"108.0:  i: {i}")) + (r) * FDN(lambda x: analytic_for_y_func(x) + bool(print(f"108.1: {x}")), i, dx)(singular_point) + FDN(lambda x: analytic_for_dy__dx_func(x) + bool(print(f"108.2: {x}")), i, dx)(singular_point)) / math.factorial(i) * past_coeffs[0]

    return this_coeff




dx = .00001
limiting_epsilon = .0001

#bessel
nu = .5
p = lambda x: x
q = lambda x: nu**2/x - x
w = lambda x: 0
lambda_ = 0

coeff_i = -(2/math.pi)**.5 #
x_singular = 0


'''
#legendre
l=3
m=2
p = lambda x: 1-x**2
q = lambda x: m**2/(1-x**2)
w = lambda x: 1
lambda_ = l*(l+1)

coeff_i = 15
x_singular = 1
'''

anal_y, anal_yprime = Make_Analytic_Funcs(p, q, w, lambda_, x_singular, dx, limiting_epsilon)
anal_y_coeff, anal_yprime_coeff = anal_y(x_singular), anal_yprime(x_singular)

indicial = Make_Indicial(anal_y_coeff, anal_yprime_coeff)

r = (-(anal_yprime_coeff-1) + ((anal_yprime_coeff-1)**2 - 4*anal_y_coeff)**.5)/2


coeff_list = []
i = 0
while True:
    coeff_list.append(coeff_i)
    i += 1
    coeff_i = ith_solution_series_coeff_given_indicial_and_analytic_coeffs_and_r_and_past_coeffs(indicial, r, anal_y, anal_yprime, 1, coeff_list, i, dx)
    input(str(coeff_i) + " outer loop, line 146")

#!!!note that something is wrong:
