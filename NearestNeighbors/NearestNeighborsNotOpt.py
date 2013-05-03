# -*- coding: utf-8 -*-
"""
Created on Fri May 03 17:51:51 2013

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



class NearestNeighborsNotOpt(BaseEngineModel):
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
        self.users = schema.user2user_attr
        self.items = schema.item2item_attr
        self.trans = schema.user2item
        self.similarity = np.dot(self.users, self.users.T)
        self.trans_lil = self.trans.tolil()

    def most_popular_item(self):
        # Some users don't have any attributes at all,
        # so the only thing we can recommend them is
        # the most popular item.
        '''
        Returns: an index number of the item that has been puchased more than
        any item. (If two items have been puchased the same number
        of times, then an item with lower index will be chosen.)
        '''
        return self.trans.sum(axis=0)
    def get_nearest_neighbors(self, i, n, self_neighbor):
        '''
        Returns: numpy.ndarray of ione dimension and length of n. The elements are the
        indices of other users that are most similar to the user indicated
        by index 1.
        Input:
        i: index of a given user
        n: number of nearest neighbors that we want to find
        '''
        x = self.similarity.getcol(i).toarray()
        if not self_neighbor:
            x[i,0] = 0
        x = x.T[0]
        return x.argsort()[-n:]
    def best_item(self, nearest_neighbors):
        '''
        Returns: an index of the item that nearest neighbors of the the user
        have purchased the most.
        Input: numpy.ndarray() of 1 dim that contains indexes of the nearest neighbors
        '''
        items_bought_by_nn = self.trans_lil.getrow(nearest_neighbors[0])
        for neighbor in nearest_neighbors[1:]:
            items_bought_by_nn += self.trans_lil.getrow(neighbor)
        items_bought_array = items_bought_by_nn.toarray()
        return items_bought_array
    def get_recom(self, i, n, self_neighbor):
        '''
        Returns: an index of an item that we will recomed to the user of index i.
        Input:
        i: index of a given user
        n: number of nearest neighbors that we want to use in
        the estimation
        '''
        nn = self.get_nearest_neighbors(i, n, self_neighbor)
        return self.best_item(nn)
    def get_recoms_info(self, n, self_neighbor=True):
        '''
        Returns:
        a dictinary that contains two keys:
        'recom_dok_matrix':
        is a sparse dok_matrix
        that contains recommendations, users are
        rows and recommedations are columns.
        Each recommendations is indicated by one.
        Every user recieves one and only one
        recommendation.
        'recom_summary':
        is a collections.Counter
        which indicies of items as keys and number
        of times each item has been recommended as
        values
        Input:
        n: number of nearest neighbors that we want to use in
        the estimation
        '''
        import ipdb;ipdb.set_trace()
        recom_matrix = np.zeros((self.trans.shape))
        users_csr = self.users.tocsr()
        most_popular = self.most_popular_item()
        for i in xrange(self.similarity.shape[0]):
            recom = self.get_recom(i, n, self_neighbor)
            if users_csr.getrow(i).sum():
                recom_matrix[i] = recom
            else:
                recom_matrix[i] = most_popular
        self.schema.zscore_user2item = np.matrix(recom_matrix)
        return recom_matrix
#    def get_recom_info(self, n):
#        import ipdb;ipdb.set_trace()
#        nn = self.similarity.argsort(axis=1)[:, -n:]
#        scores = self.trans[nn].sum(axis=1)
#        no_attr = np.where(self.users.sum(axis=1) == 0)
#        most_pop = self.trans.sum(axis=0)
#        scores[no_attr] = most_pop
#        self.schema.zscore_user2item = np.matrix(scores)
#        return np.matrix(scores)
    def run(self):
        return True, self.get_recom_info(25)