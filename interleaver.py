# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 13:38:01 2023

@author: Sri
"""


import itertools
import numpy as np
import math


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

def Interleave(list_):
    indice_max = len(list_)
    indice_order_gen = Bin_Order_N(indice_max)
    for i in indice_order_gen:
        yield list_[i]

'''
for n in range(1,N):
    l = list(Bin_Order_N(n, list_mode=list_mode))
    l.sort()
    if l != list(range(n)):
        print("Problemo")
        #breakpoint()
'''
