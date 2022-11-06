# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 10:02:24 2022

@author: Sri
"""


from icecream import ic


import functools

import numpy as np
import scipy as sp


'''
Using math:
The problem is to find $\psi$ and $\lambda$ such that $D \psi = \lambda \psi$ for a given partial differential operator $D$.
Suppose we have a nice $D$ that can be readily separated, specifically a $D$ of the form $\sum_i \sum_j f_i(x_i)\partial_{x_i}^{j}$
Let us now make the ansatz $\psi = \prod_k \phi_k(x_k)$ to perform separation of variables
After performing the separation, we get the eigenvalue equation $\sum_j f_i(x_i) \frac{\phi^{(j)}_{i}(x_i)} {\phi_i(x_i)} = \omega_i$ for every dimension i, and $\sum_i \omega_i = \lambda$

Therefore, the code here will separate $D$ and then call up the ode_solver on each of the ordinary eigenvalue equations
'''




#def separator():


def eigenfunc_evolutions(eigenfuncs, eigenvals, time): #eigens of H #this is the propogator but with the outer product replaced with just the ket of the eigenfunc, the intent being to allow for a later application to \psi(0) by taking the innerproduct with the eigenbras and multiplying element wise with the output of this function, and then adding up all the elements
    np.vectorize(lambda eigenfunc, eigenval: sp.e**((-1j*eigenval*time)/sp.hbar)*eigenfunc)#!!!finish
