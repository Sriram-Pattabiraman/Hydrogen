# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 11:28:53 2023

@author: Sri
"""


import numpy as np
import scipy as sp
import math
import sympy as symp
import sympy.combinatorics as comb


import itertools


#find in array
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




#convert coordinate systems
def convert_coord(in_coord, from_system='cartesian', out_system='spherical'):
    if from_system==out_system:
        return in_coord
    elif from_system=='cartesian' and out_system=='spherical': #using the ISO convention for physics: azimuth is \phi and inclination is \theta. the azimuth is constricted to [0, 2\pi] and the inclination to [0, \pi]
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

        if any([np.isnan(r), not (np.isfinite(theta)), not(np.isfinite(phi))]):
            return None
        return [r, theta, phi]
    
    
    
    
#interleave  
def inplace_binlist_succ(current_bin):
    i = -1
    while True:
        if current_bin[i] == 0:
            current_bin[i] = 1
            return
        else:
            current_bin[i] = 0
            i -= 1

def Bin_Order_N(n, list_mode=True):
    if n == 0:
        yield
        return
    elif n == 1:
        yield 0
        return
    else:
        n -= 1 #following code was written to be to n inclusive, this makes the behaviour like range
    already_encountered = set()
    for depth_past_0th_bit in range(0, math.ceil(math.log2(n))):
        if depth_past_0th_bit == 0:
            this_point = n//2
            if this_point not in already_encountered:
                yield this_point
                already_encountered.add(this_point)
        elif list_mode:
            current_bin = [0]*(1+depth_past_0th_bit)
            for i in range(2**(1+depth_past_0th_bit)):
                if i != 0:
                    inplace_binlist_succ(current_bin)
                    #current_bin = bin(int(current_bin, 2) + 1)[2:].zfill(1+depth_past_0th_bit)
                left_point = 0
                right_point = n
                this_point = n//2
                for bit in current_bin[1:]:
                    if bit == 0:
                        right_point = this_point
                        this_point = (this_point - left_point)//2 + left_point
                    elif bit == 1:
                        left_point = this_point
                        this_point = math.ceil((right_point - this_point)/2 )+ this_point
                if this_point not in already_encountered:
                    yield this_point
                    already_encountered.add(this_point)
                else:
                    continue
                    #input([n, current_bin, this_point])
        else:
            current_bin = '0'.zfill(1+depth_past_0th_bit)
            for i in range(2**(1+depth_past_0th_bit)+1):
                if i != 0:
                    current_bin = bin(int(current_bin, 2) + 1)[2:].zfill(1+depth_past_0th_bit)
                left_point = 0
                right_point = n
                this_point = n//2
                for bit in current_bin[1:]:
                    if bit == '0':
                        right_point = this_point
                        this_point = (this_point - left_point)//2 + left_point
                    elif bit == '1':
                        left_point = this_point
                        this_point = math.ceil((right_point - this_point)/2 )+ this_point
                if this_point not in already_encountered:
                    yield this_point
                    already_encountered.add(this_point)


    for i in range(n+1):
        if i not in already_encountered:
            yield i

def interleave(list_):
    indice_max = len(list_)
    indice_order_gen = Bin_Order_N(indice_max)
    for i in indice_order_gen:
        yield list_[i]




#restrict
def restrict(func, restriction_window_lows=None, restriction_window_highs=None):
    def restricted_func(*args, func=func, restriction_window_lows=restriction_window_lows, restriction_window_highs=restriction_window_highs):
        coord = np.array(args)  
        if np.all(np.less_equal(coord, restriction_window_highs)) and np.all(np.greater_equal(coord, restriction_window_lows)):
            return func(*args)
        else:
            return 0
    return restricted_func
    

    
'''
class symarray(np.ndarray):
    def __new__(cls, in_arr, permutable_axes_groups=np.empty((0,0))):
        obj = np.asarray(in_arr).view(cls)
        obj.permutable_axes_groups = permutable_axes_groups
        return obj

    def __array_finalize__(self, obj):
        assert self.check_symmetry(getattr(obj, "permutable_axes", np.empty((0,0))))
        if obj is None: #explicit constructor
            return
        else:
            self.permutable_axes_groups = getattr(obj, "permutable_axes", np.empty((0,0)))
            
        
    def __setitem__(self, indexes, value):
        super(symarray, self).__setitem__(indexes, value)   
        
        permutable_axes_groups = self.permutable_axes_groups
        equivalent_index_tuples = [np.copy(indexes)]
        for permutable_axes_group in permutable_axes_groups:
            up_to_this_group_applied_equivalent_indices = []
            permuting_indices = []
            for axis in permutable_axes_group:
                if axis >= len(indexes):
                    continue
                else:
                    permuting_indices.append(indexes[axis])
            
            for permutation in itertools.permutations(permuting_indices):
                for these_indexes in equivalent_index_tuples:
                    working_these_indexes = np.copy(these_indexes)
                    this_permuting_axis_num = 0
                    for axis in range(len(indexes)):
                        if axis in permutable_axes_group:
                            working_these_indexes[axis] = permutation[this_permuting_axis_num]
                            this_permuting_axis_num += 1
                            
                    up_to_this_group_applied_equivalent_indices.append(working_these_indexes)
                    
            equivalent_index_tuples = np.copy(up_to_this_group_applied_equivalent_indices)
            
        for equivalent_index_tuple in equivalent_index_tuples:
            if type(indexes) == tuple:
                type_corrected_setting_indice = tuple(equivalent_index_tuple)
            elif type(indexes) == list:
                type_corrected_setting_indice = list(equivalent_index_tuple)    
            elif issubclass(type(indexes), np.ndarray):
                index_type = type(indexes)
                type_corrected_setting_indice = np.array(equivalent_index_tuple).view(index_type)
                
            super(symarray, self).__setitem__(type_corrected_setting_indice, value)
   
    def check_symmetry(self, permutable_axes_groups):
        arr_it = np.nditer(self, flags=['multi_index', 'refs_ok']) 
        for part in arr_it:
            indexes = arr_it.multi_index
            equivalent_index_tuples = [np.copy(indexes)]
            for permutable_axes_group in permutable_axes_groups:
                up_to_this_group_applied_equivalent_indices = []
                permuting_indices = []
                for axis in permutable_axes_group:
                    if axis >= len(indexes):
                        continue
                    else:
                        permuting_indices.append(indexes[axis])
                
                for permutation in itertools.permutations(permuting_indices):
                    for these_indexes in equivalent_index_tuples:
                        working_these_indexes = np.copy(these_indexes)
                        this_permuting_axis_num = 0
                        for axis in range(len(indexes)):
                            if axis in permutable_axes_group:
                                working_these_indexes[axis] = permutation[this_permuting_axis_num]
                                this_permuting_axis_num += 1
                                
                        up_to_this_group_applied_equivalent_indices.append(working_these_indexes)
                        
                equivalent_index_tuples = np.copy(up_to_this_group_applied_equivalent_indices)
                
            val = self[indexes]
            for equivalent_index_tuple in equivalent_index_tuples:
                if type(indexes) == tuple:
                    type_corrected_setting_indice = tuple(equivalent_index_tuple)
                elif type(indexes) == list:
                    type_corrected_setting_indice = list(equivalent_index_tuple)    
                elif issubclass(type(indexes), np.ndarray):
                    index_type = type(indexes)
                    type_corrected_setting_indice = np.array(equivalent_index_tuple).view(index_type)
                    
                if not self[type_corrected_setting_indice] == val:
                    return False
            else:
                return True
'''
def lex_less(tuple1, tuple2): #lexicographic_le
    if len(tuple1) < len(tuple2):
        return True
    elif len(tuple1) > len(tuple2):
        return False
    
    for i in range(len(tuple1)):
         if tuple1[i] < tuple2[i]:
             return True
         elif tuple1[i] > tuple2[i]:
             return False
         else:
             continue
    else:
        return True
    
def lex_less_equals(tuple1, tuple2): #lexicographic_le
    if len(tuple1) < len(tuple2):
        return True
    elif len(tuple1) > len(tuple2):
        return False
    
    for i in range(len(tuple1)):
         if tuple1[i] < tuple2[i]:
             return True
         elif tuple1[i] > tuple2[i]:
             return False
         else:
             continue
    else:
        return True
    
    
def IdentityPerm(size):
    return comb.Permutation(range(size))

def IdentityPermGroup(size):
    return comb.PermutationGroup(IdentityPerm(size))

def perm_from_arr(nested_array, size=None):
    this_perm = comb.Permutation(size=size)
    for cycle in nested_array:
        this_perm = this_perm(*cycle)
        
    return this_perm
        
def permgroup_from_arr(nested_array, size=None):
    if len(nested_array) == 0:
        return IdentityPermGroup(size)
    
    perm_gens = []
    for perm_gen_as_arr in nested_array:
        perm_gens.append(perm_from_arr(perm_gen_as_arr, size=size))
    
    permgroup = comb.PermutationGroup(*perm_gens)
    return permgroup
        
class symarray(np.ndarray):
    def __new__(cls, in_arr, permutation_generators=np.empty((0,0,0))):
        obj = np.asarray(in_arr).view(cls)
        obj.permgroup = permgroup_from_arr(permutation_generators, size=len(in_arr.shape))
        return obj

    def __array_finalize__(self, obj):
        assert self.check_symmetry(getattr(obj, "permgroup", IdentityPermGroup(len(self.shape))))
        if obj is None: #explicit constructor
            return
        else:
            self.permgroup = getattr(obj, "permgroup", IdentityPermGroup(len(obj.shape)))
            
        
    def __setitem__(self, indexes, value):
        super(symarray, self).__setitem__(indexes, value)   
        
        
        for perm in self.permgroup._elements:
            equivalent_index_tuple = perm(indexes)
            
            if type(indexes) == tuple:
                type_corrected_setting_indice = tuple(equivalent_index_tuple)
            elif type(indexes) == list:
                raise NotImplementedError #!!!test this? never going to plan to use this so should be fine anyways
                type_corrected_setting_indice = list(equivalent_index_tuple)    
            elif issubclass(type(indexes), np.ndarray):
                raise NotImplementedError #!!!test this? never going to plan to use this so should be fine anyways
                index_type = type(indexes)
                type_corrected_setting_indice = np.array(equivalent_index_tuple).view(index_type)
                
            super(symarray, self).__setitem__(type_corrected_setting_indice, value)
   
    def check_symmetry(self, permgroup, isclose_rel_tol=1e-10, isclose_abs_tol=1e-50):
        arr_it = np.nditer(self, flags=['multi_index', 'refs_ok']) 
        already_good_indexes = []
        for this_val in arr_it:
            indexes = arr_it.multi_index
            if indexes in already_good_indexes:
                continue
            for perm in permgroup._elements:
                equivalent_index_tuple = perm(indexes)
                if equivalent_index_tuple in already_good_indexes:
                    already_good_indexes.append(indexes)
                    break
                should_match_val = self[tuple(equivalent_index_tuple)]
                if not math.isclose(should_match_val, this_val, rel_tol=isclose_rel_tol, abs_tol=isclose_abs_tol):
                    return False
                else:
                    already_good_indexes.append(equivalent_index_tuple)
        else:
            return True
            
           
            
def test():
    arr=symarray(np.array([ [ [[1,2,3], [2,4,5], [3,5,6]], [[11,12,13], [12,14,15], [13,15,16]] ], [ [[11,12,13], [12,14,15], [13,15,16]], [[1,2,3], [2,4,5], [3,5,6]] ] ]), permutation_generators=[ [[0,1]], [[2,3]] ])

    arr[0,1,0,1]=100
    equalities = np.equal([arr[0,1, 0,1], arr[0,1, 1,0], arr[1,0, 0,1], arr[1,0, 1,0]], np.repeat(100, 4))

    assert all(equalities)
    
    arr2 = symarray(np.zeros((4,4,4,4)), permutation_generators=[ [[0,1]], [[2,3]], [[0,2],[1,3]] ])
    arr2[0,1,2,3] = 1
    equalities2 = np.equal([arr2[1,0,2,3], arr2[0,1,3,2], arr2[1,0,3,2], arr2[2,3,0,1], arr2[3,2,0,1], arr2[2,3,1,0], arr2[3,2,1,0]],  np.repeat(1, 7))
    assert all(equalities2)
    
#test()



#sequence acceleration
def aitken(sequence_iterable, tol=1e-5, dont_divide_thresh=1e-8, up_to_n_terms=10, override_finish_sequence_iterable=True):
    #plays nice with generators (including infinite ones)
    #note: stops if override_finish_sequence_iterable OR |correction| < tol OR |second_delta| < dont_divide_thresh OR i >= up_to_n_terms OR sequence_generator has been exhausted
    i = 0
    for next_next_x in sequence_iterable:
        if i == 0:
            this_x = next_next_x
        elif i == 1:
            next_x = next_next_x
        else:
            this_delta = next_x - this_x
            next_delta = next_next_x - next_x
            second_delta = next_delta - this_delta
            if (abs(second_delta) < dont_divide_thresh or i>=up_to_n_terms):
                if (not override_finish_sequence_iterable):
                    return
                else:
                    yield this_x
            else:
                correction = this_delta**2 / second_delta
                corrected_x = this_x - correction
                yield corrected_x
                if (not override_finish_sequence_iterable) and (abs(correction) < tol):
                    return
                this_x = next_x
                next_x = next_next_x
        
        i+=1
        
def aitken_test():
    def pi_sequence():
        acc = 0
        n = 0
        while True:
            acc += 4* (-1)**n / (2*n + 1)
            n+=1
            yield acc
    
    accelerated_sequence = aitken(pi_sequence)


def std_devize(possibly_arr):
    def single_std_devize(elem):
        try:
            return (elem+0).std_dev
        except AttributeError:
            return (elem+0)
    
    try:
        if len(possibly_arr) == 1:
            return single_std_devize(possibly_arr[0]+0)
        else:
            return  np.vectorize(single_std_devize)(possibly_arr)
    except TypeError:
        return single_std_devize(possibly_arr)
       
def nominalize(possibly_arr):
    def single_nominalize(elem):
        try:
            return (elem+0).nominal_value
        except AttributeError:
            return (elem+0)
    
    try:
        if len(possibly_arr) == 1:
            return single_nominalize(possibly_arr[0]+0)
        else:
            return  np.vectorize(single_nominalize)(possibly_arr)
    except TypeError:
        return single_nominalize(possibly_arr)
       
                    