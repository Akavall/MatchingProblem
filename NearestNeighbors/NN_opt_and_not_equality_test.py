# -*- coding: utf-8 -*-
"""
Created on Fri May 03 19:22:58 2013

@author: Lena
"""

import cPickle as pickle
import numpy as np
from NearestNeighborsOpt import NearestNeighborsOpt
from NearestNeighborsNotOpt import NearestNeighborsNotOpt

users, items, trans = pickle.load(open('users_items_trans.pkl', 'rb'))

class Schema(object):
    def __init__(self, users, items, trans):
        self.user2user_attr = users
        self.item2item_attr = items
        self.user2item = trans

schema = Schema(users, items, trans)

def test_result_equality():
     nn_opt = NearestNeighborsOpt(schema)
     nn_not_opt = NearestNeighborsNotOpt(schema)
     recoms_opt = nn_opt.get_recom_info(5)
     recoms_not_opt = nn_not_opt.get_recoms_info(5)
     np.testing.assert_array_equal(recoms_opt, recoms_not_opt,
                                   "Results are not equal.")

     
     
     

