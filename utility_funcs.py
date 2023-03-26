# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 11:28:53 2023

@author: Sri
"""


import numpy as np
import math


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

        if any([np.isnan(r), np.isnan(theta), np.isnan(phi)]):
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
    