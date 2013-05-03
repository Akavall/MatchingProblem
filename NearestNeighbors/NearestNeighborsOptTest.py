# -*- coding: utf-8 -*-
"""
Created on Fri May 03 18:46:18 2013

@author: Kirill
"""

import cPickle as pickle
import numpy as np
from NearestNeighborsOpt import NearestNeighborsOpt

users, items, trans = pickle.load(open('users_items_trans.pkl', 'rb'))

class Schema(object):
    def __init__(self, users, items, trans):
        self.user2user_attr = users
        self.item2item_attr = items
        self.user2item = trans

schema = Schema(users, items, trans)

def test_most_popular_item():
    my_problem = NearestNeighborsOpt(schema)
    result = my_problem.get_most_pop_item()
    expected = np.array([3, 1, 5, 1, 1, 4, 3, 5])
    np.testing.assert_array_equal(result, expected, "Fail : most_popular_item")
    
def test_get_nearest_neighbors():
    my_problem = NearestNeighborsOpt(schema)
    result = my_problem.get_nearest_neighbors(5)
    expected = np.array([[7, 9, 0, 3, 5],
                         [6, 7, 9, 1, 3],
                         [5, 7, 9, 2, 4],
                         [0, 5, 7, 1, 3],
                         [5, 7, 9, 2, 4],
                         [7, 0, 3, 9, 5],
                         [8, 9, 1, 3, 6],
                         [2, 4, 5, 3, 7],
                         [5, 6, 7, 8, 9],
                         [2, 3, 4, 5, 9]])
    np.testing.assert_array_equal(result, expected, "Fail : get_nearest_neighbors")
    
def test_get_no_attr_users():
    my_problem = NearestNeighborsOpt(schema)
    result = my_problem.get_no_attr_users()[0]
    expected = np.array([8])
    np.testing.assert_array_equal(result, expected, "Fail : get_no_attr_users")
     
def test_get_recom_info():
     my_problem = NearestNeighborsOpt(schema)
     result = my_problem.get_recom_info(5)
     expected = np.array([[2, 0, 1, 1, 1, 3, 2, 2],
                          [1, 0, 2, 0, 1, 2, 1, 3],
                          [1, 1, 1, 0, 1, 1, 3, 3],
                          [2, 0, 2, 1, 0, 4, 1, 3],
                          [1, 1, 1, 0, 1, 1, 3, 3],
                          [2, 0, 1, 1, 1, 3, 2, 2],
                          [1, 0, 3, 0, 1, 2, 1, 2],
                          [1, 1, 1, 0, 0, 2, 2, 4],
                          [3, 1, 5, 1, 1, 4, 3, 5],
                          [0, 1, 1, 0, 1, 2, 3, 3]])
     np.testing.assert_array_equal(result, expected, "Fail : get_recom_info")




