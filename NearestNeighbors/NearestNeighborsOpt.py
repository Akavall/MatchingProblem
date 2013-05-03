# -*- coding: utf-8 -*-
"""
Created on Thu May 02 16:03:32 2013

@author: Lena
"""

"""
Created on Wed Apr 10 17:21:17 2013

@author: Kirill
"""

import logging
from new_engine.base_engine_model import BaseEngineModel
from django.conf import settings
if settings.USE_SCIPY_STATS:
    import numpy as np
    import scipy.sparse as scsp

import numpy as np
import scipy.sparse as scsp

class NearestNeighborsOpt(BaseEngineModel):
    def __init__(self, schema):
        '''
        Input:
        users:
        is a csc_matrix where users are rows
        and attributes associated with the users are
        columns. If a certain attribute belongs to
        a given user, then it is represented by 1, else
        0.
        items:
        is a csc_matrix where items are rows
        and attributes associated with the items are
        columns. If a certain attribute belongs to
        a given item, then it is represented by 1, else
        0.
        trans:
        is a csc_matrix of transactions, users are rows and
        items a columns. If a user did not purchase an item
        it is represented by 0, else it is represented by the
        number of items the use has purchased.
        '''
        
        self.schema=schema
        self.users = schema.user2user_attr.toarray()
        self.items = schema.item2item_attr.toarray()
        self.trans = schema.user2item.toarray()
        self.similarity = np.dot(self.users, self.users.T)
        
    def get_most_pop_item(self):
        return self.trans.sum(axis=0) 
        
    def get_nearest_neighbors(self, n):
        return self.similarity.argsort(axis=1)[:, -n:]
        
    def get_no_attr_users(self):
        return np.where(self.users.sum(axis=1) == 0)

    def get_recom_info(self, n):
        import ipdb;ipdb.set_trace()
        nn = self.get_nearest_neighbors(n)
        scores = self.trans[nn].sum(axis=1)
        no_attr = self.get_no_attr_users()
        most_pop = self.get_most_pop_item()
        scores[no_attr] = most_pop
        self.schema.zscore_user2item = np.matrix(scores)
        return np.matrix(scores)
        
    def run(self):
        return True, self.get_recom_info(25)