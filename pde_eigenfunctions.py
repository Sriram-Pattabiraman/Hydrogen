# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 10:02:24 2022

@author: Sri
"""


from icecream import ic


import functools

import numpy as np
import scipy as sp

from multiset import Multiset as multiset
from multiset import FrozenMultiset as frozenmultiset

class frozendict(dict): #taken from stack overflow, docstring and all. i changed the name to "frozendict" from "hashable_dict", including in the docstring.
    """
    frozendict implementation, suitable for use as a key into
    other dicts.

        >>> h1 = frozendict({"apples": 1, "bananas":2})
        >>> h2 = frozendict({"bananas": 3, "mangoes": 5})
        >>> h1+h2
        frozendict(apples=1, bananas=3, mangoes=5)
        >>> d1 = {}
        >>> d1[h1] = "salad"
        >>> d1[h1]
        'salad'
        >>> d1[h2]
        Traceback (most recent call last):
        ...
        KeyError: frozendict(bananas=3, mangoes=5)

    based on answers from
       http://stackoverflow.com/questions/1151658/python-hashable-dicts

    """
    def __key(self):
        return tuple(sorted(self.items()))
    def __repr__(self):
        return "{0}({1})".format(self.__class__.__name__,
            ", ".join("{0}={1}".format(
                    str(i[0]),repr(i[1])) for i in self.__key()))

    def __hash__(self):
        return hash(self.__key())
    def __setitem__(self, key, value):
        raise TypeError("{0} does not support item assignment"
                         .format(self.__class__.__name__))
    def __delitem__(self, key):
        raise TypeError("{0} does not support item assignment"
                         .format(self.__class__.__name__))
    def clear(self):
        raise TypeError("{0} does not support item assignment"
                         .format(self.__class__.__name__))
    def pop(self, *args, **kwargs):
        raise TypeError("{0} does not support item assignment"
                         .format(self.__class__.__name__))
    def popitem(self, *args, **kwargs):
        raise TypeError("{0} does not support item assignment"
                         .format(self.__class__.__name__))
    def setdefault(self, *args, **kwargs):
        raise TypeError("{0} does not support item assignment"
                         .format(self.__class__.__name__))
    def update(self, *args, **kwargs):
        raise TypeError("{0} does not support item assignment"
                         .format(self.__class__.__name__))
    # update is not ok because it mutates the object
    # __add__ is ok because it creates a new object
    # while the new object is under construction, it's ok to mutate it
    def __add__(self, right):
        result = frozendict(self)
        dict.update(result, right)
        return result



def Create_Ansatz(product_func_name, n_of_vars=1, var_names="DEFAULT"):
    if var_names == "DEFAULT":
        var_indices = range(n_of_vars)
        var_names = (f"x{i}" for i in var_indices)

    all_var_names = set()
    terms = multiset()
    for var_name in var_names:
        all_var_names.add(var_name)
        terms.add(frozendict({'type': 'Func', 'name': f"'{product_func_name}'_'{var_name}'", 'var_names': frozenset({var_name})}))

    out = frozendict({"type": "Product", 'name': f"{product_func_name}", "terms": frozenmultiset(terms), "var_names": frozenset(all_var_names)})
    return out

def Create_Basic_Differential_Operator(var_name):
    return frozendict({"type": "Differential_Operator", "name": f"partial_'{var_name}'", "var_names": frozenset({var_name})})

def Create_Applied_Operator(operator, applied_to_term):
    if operator['type'] != "Differential_Operator":
        raise(TypeError("Cannot apply non-differential operator!"))

    operator_name = operator['name']
    applied_to_term_name = applied_to_term['name']

    return frozendict({"type": "Applied_Differential_Operator", "name": f"Applied_'{operator_name}'_To_'{applied_to_term_name}'", "operator": operator, "applied_to_term": applied_to_term, "var_names": frozenset.union(applied_to_term['var_names'], operator['var_names'])})

def Negate(term):
    if term['type'] == "Sum":
        out_sum_terms = multiset()
        for summand in term['terms']:
            out_sum_terms.add(Negate(summand))

        out_sum_terms = frozenmultiset(out_sum_terms)
        return Term_Sum("Distributed_Negate_'%s'" % term['name'], out_sum_terms)
    elif term['type'] == "Negate":
        unnegated = term['terms']
        return unnegated
    elif type(term) == frozendict:
        return frozendict({'type': "Negate", "terms": term, "name": "Negative_%s" % term['name'], "var_names": term["var_names"]})
    elif type(term) == frozenmultiset:
        breakpoint()
    else:
        breakpoint()

#def Sum_Wrap(sum_func_name, multiset_of_term)
def Term_Sum(sum_func_name, terms):
    if type(terms) == dict or type(terms) == frozendict:
        return terms

    new_terms = multiset()
    need_to_recurse = False
    for term in terms:
        if term['type'] != "Sum":
            new_terms.add(term)
        else:
            new_terms.update(term['terms'])
            need_to_recurse = True

    if need_to_recurse:
        return Term_Sum(sum_func_name, new_terms)



    pos_terms = multiset()
    neg_terms = multiset()
    for term in terms:
        if term['type'] != "Negate":
            pos_terms.add(term)
        else:
            neg_terms.add(term['terms'])

    new_pos_terms = pos_terms - neg_terms
    new_neg_terms = neg_terms - pos_terms
    new_pos_terms = frozenmultiset(new_pos_terms)
    new_neg_terms = frozenmultiset(new_neg_terms)


    pos_var_names = set()
    for pos_term in new_pos_terms:
        pos_var_names.update(pos_term['var_names'])

    pos_var_names = frozenset(pos_var_names)

    neg_var_names = set()
    for neg_term in new_neg_terms:
        neg_var_names.update(neg_term['var_names'])

    neg_var_names = frozenset(neg_var_names)

    actually_negated_new_neg_terms = multiset()
    for new_neg_term in new_neg_terms:
        actually_negated_new_neg_terms.add(Negate(new_neg_term))
    actually_negated_new_neg_terms = frozenmultiset(actually_negated_new_neg_terms)

    out_sum = frozendict({'type': "Sum", "name": f"'{sum_func_name}'", "terms": frozenmultiset.union(new_pos_terms, actually_negated_new_neg_terms), "var_names": frozenset.union(pos_var_names, neg_var_names)})
    return out_sum

ansatz_3_terms = Create_Ansatz('psi',3)['terms']
unnegated_sum = Term_Sum("psi", ansatz_3_terms)
term1 = Negate(unnegated_sum)
term2 = Term_Sum('psi', Create_Ansatz('psi',2)['terms'])
out_sum = Term_Sum("out_sum", multiset([term1, term2]))

def Commute_Product_With_Applied_Differential_Operator(applied_operator):
    applied_operator_term_name = applied_operator["name"]
    if applied_operator["type"] != "Applied_Differential_Operator":
        raise(TypeError("Cannot commute with non-applied-differential-operator!"))

    operator = applied_operator["operator"]
    operator_name = operator["name"]
    prod_of_terms_term = applied_operator["applied_to_term"]

    if operator["type"] != "Differential_Operator":
        raise(TypeError("Cannot commute with non-differential operator!"))

    if prod_of_terms_term["type"] == "Applied_Differential_Operator":
        prod_of_terms_term = Commute_Product_With_Applied_Differential_Operator(prod_of_terms_term)
    elif prod_of_terms_term["type"] != "Product":
        raise(TypeError("Cannot commute non-product with differential operator!"))

    uncommutable_var_names = operator["var_names"]

    prod_of_terms_name = prod_of_terms_term["name"]
    terms = prod_of_terms_term["terms"]

    commuted_terms = multiset()
    commuted_var_names = set()
    uncommuted_terms = multiset()
    uncommuted_var_names = set()
    for term in terms:
        if frozenset(term["var_names"]).intersection(uncommutable_var_names) != frozenset():
            uncommuted_terms.add(term)
            uncommuted_var_names.update(term["var_names"])
        else:
            commuted_terms.add(term)
            commuted_var_names.update(term["var_names"])



    commuted_terms = frozenmultiset(commuted_terms)
    commuted_var_names = frozenset(commuted_var_names)
    uncommuted_terms = frozenmultiset(uncommuted_terms)
    uncommuted_var_names = frozenset(uncommuted_var_names)

    commuted_product = {"type": "Product", "name": f"Commuted_Parts_Of_'{prod_of_terms_name}'", "terms": commuted_terms, "var_names": commuted_var_names}
    out_applied_differential_operator = {"type": "Applied_Differential_Operator", "name": f"Applied_'{operator_name}'_To_Uncommuted_Parts_Of_'{prod_of_terms_name}'", "operator": operator, "applied_to_term": uncommuted_terms, "var_names": uncommuted_var_names}

    all_vars = frozenset.union(commuted_var_names, uncommuted_var_names)
    out_terms = frozenmultiset({frozendict(commuted_product), frozendict(out_applied_differential_operator)})
    return frozendict({"type": "Product", "name": f"Separated_'{applied_operator_term_name}'", "terms": out_terms, "var_names": all_vars, 'hints': frozendict({"commutation_hints": frozendict({"commuted_var_names": commuted_var_names, "uncommuted_var_names": uncommuted_var_names}) }) })

def Divide_Products(numerator_product, denominator_product): #!!!handle exponents
    if numerator_product["type"] != "Product":
        raise(TypeError("Cannot divide a non-product!"))
    if denominator_product["type"] != "Product":
        raise(TypeError("Cannot divide by a non-product!"))


    for term in denominator_product['terms']:
        if term in numerator_product['terms']:
            new_numerator_product_terms, new_denominator_product_terms = numerator_product['terms'], denominator_product['terms']

            new_numerator_product_terms = frozenmultiset.difference(numerator_product['terms'], denominator_product['terms'])
            new_denominator_product_terms = frozenmultiset.difference(denominator_product['terms'], numerator_product['terms'])

            numerator_product = dict(numerator_product)
            denominator_product = dict(denominator_product)

            numerator_product['terms'] = new_numerator_product_terms
            denominator_product['terms'] = new_denominator_product_terms

            numerator_product = frozendict(numerator_product)
            denominator_product = frozendict(denominator_product)

    new_numerator_var_names = set()
    for numerator_term in numerator_product['terms']:
        new_numerator_var_names.update(numerator_term['var_names'])

    new_denominator_var_names = set()
    for denominator_term in denominator_product['terms']:
        new_denominator_var_names.update(denominator_term['var_names'])

    new_numerator_var_names = frozenmultiset(new_numerator_var_names)
    new_denominator_var_names = frozenmultiset(new_denominator_var_names)

    numerator_product['var_names'] = new_numerator_var_names
    denominator_product['var_names'] = new_denominator_var_names
    all_vars = frozenset.union(new_numerator_var_names, new_denominator_var_names)

    return frozendict({"type": "Fraction", "name": f"Divide_'{numerator_product['name']}'_By_'{denominator_product['name']}'", 'numerator': numerator_product, 'denominator': denominator_product, 'var_names': all_vars})

def Isolate_Differential_Operator_Var_In_Commuted_Equation_With_Commutation_Hints(equation, var): #does no commutation
    if equation['type'] != "Equals":
        raise(TypeError("Cannot isolate in non-equation!"))

    left, right = equation['LHS'], equation['RHS']
    if left['type'] != "Sum":
        raise(TypeError("Cannot isolate LHS non-sum!"))

    if right['type'] != "Sum":
        raise(TypeError("Cannot isolate RHS non-sum!"))

    LHS_terms_without_var = multiset()
    RHS_terms_with_var = multiset()
    for left_term in left['terms']:
        if var not in left_term['hints']['commutation_hints']["uncommuted_var_names"]:
            LHS_terms_without_var.add(left_term)

    for right_term in right['terms']:
        if var in right_term['hints']['commutation_hints']["uncommuted_var_names"]:
            RHS_terms_with_var.add(right_term)

    LHS_terms_without_var = frozenmultiset(LHS_terms_without_var)
    RHS_terms_with_var = frozenmultiset(RHS_terms_with_var)
    left_without_sum = Term_Sum('left_to_remove', LHS_terms_without_var)
    right_with_sum = Term_Sum('right_to_remove', RHS_terms_with_var)

    new_LHS =  Term_Sum('isolated_lhs', frozenmultiset.union(left['terms'], Negate(left_without_sum)['terms'], right_with_sum['terms']))
    new_RHS = Term_Sum('isolated_rhs', frozenmultiset.union(right['terms'], left_without_sum['terms'], Negate(right_with_sum)['terms']))

    return frozendict({'type': "Equals", 'name': 'isolated_differential_operator_equation', 'LHS': new_LHS, 'RHS': new_RHS})





'''
def Bucket_Terms_By_Vars(terms):
    all_vars = terms['var_names']
    #init_dict
    buckets = {var: multiset() for var in all_vars}
    for term in terms:
        this_term_vars = term['var_names']
        for this_term_var in this_term_vars:
            buckets[this_term_var].add(term)

    for var in buckets.keys():
        buckets[var] = frozenset(buckets[var])

    return frozendict(buckets)

def in_exactly_one(set_family, element):
    in_count = 0
    for set_ in set_family:
        in_count += int(element in set_)
        if in_count > 1:
            return False

    if in_count == 1:
        return True

def find_isolatable_terms_and_vars(terms):
    buckets = Bucket_Terms_By_Vars(terms)
    isolatable_term_and_var = set()
    for var in buckets.keys():
        if len(buckets[var].distinct_elements()) == 1:
            term = list(buckets[var])[0]
            if in_exactly_one(buckets, term):
                isolatable_term_and_var.add([(term, buckets[var](term, 0)), var])
'''


'''
ansatz_prod = Create_Ansatz('psi', 1, var_names=["r", "theta", "phi"])

dr, dtheta, dphi = Create_Basic_Differential_Operator('r'), Create_Basic_Differential_Operator('theta'), Create_Basic_Differential_Operator('phi')
app_dr, app_dtheta, app_dphi = Create_Applied_Operator(dr, ansatz_prod), Create_Applied_Operator(dtheta, ansatz_prod), Create_Applied_Operator(dphi, ansatz_prod)

comm_r, comm_theta, comm_phi = Commute_Product_With_Applied_Differential_Operator(app_dr), Commute_Product_With_Applied_Differential_Operator(app_dtheta), Commute_Product_With_Applied_Differential_Operator(app_dphi)
sum_of_comm = Term_Sum('total_op', multiset([comm_r, comm_theta, comm_phi]))

minus_comm_r = Negate(comm_r)

result=Term_Sum('result', multiset([sum_of_comm, minus_comm_r]))

result_terms = result['terms']

result_terms_items = list(result_terms.items())

result_terms_item0, result_terms_item1 = result_terms_items[0][0], result_terms_items[1][0]
'''


ansatz_prod = Create_Ansatz('psi', 1, var_names=["r", "theta", "phi"])

dr, dtheta, dphi = Create_Basic_Differential_Operator('r'), Create_Basic_Differential_Operator('theta'), Create_Basic_Differential_Operator('phi')
app_dr, app_dtheta, app_dphi = Create_Applied_Operator(dr, ansatz_prod), Create_Applied_Operator(dtheta, ansatz_prod), Create_Applied_Operator(dphi, ansatz_prod)

comm_r, comm_theta, comm_phi = Commute_Product_With_Applied_Differential_Operator(app_dr), Commute_Product_With_Applied_Differential_Operator(app_dtheta), Commute_Product_With_Applied_Differential_Operator(app_dphi)
sum_of_comm = Term_Sum('total_op', multiset([comm_r, comm_theta, comm_phi]))

equation = frozendict({'type': 'Equals', 'name': 'in_equation', 'LHS': sum_of_comm, 'RHS': Term_Sum('zero', frozenmultiset())})
isolated_equation = Isolate_Differential_Operator_Var_In_Commuted_Equation_With_Commutation_Hints(equation, 'r')
