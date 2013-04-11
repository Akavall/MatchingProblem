# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 17:21:17 2013

@author: Kirill
"""

import numpy as np
import scipy.sparse as scsp
from collections import Counter

class MatchingProblem(object):
    def __init__(self, users, items, trans):
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
        self.users = users
        self.items = items
        self.trans = trans
        self.similarity = users * users.T
        self.trans_lil = trans.tolil()
    def most_popular_item(self):
        # Some users don't have any attributes at all,
        # so the only thing we can recommend them is
        # the most popular item.
        '''
        Returns: an index number of the item that has been puchased more than
                 any item. (If two items have been puchased the same number
                 of times, then an item with lower index will be chosen.)
        '''
        purchases = [self.trans.getcol(i).sum() for i in xrange(self.trans.shape[1])]
        return np.array(purchases).argmax()
    def get_nearest_neighbors(self, i, n):
        '''
        Returns: numpy.ndarray of one dimension and length of n. The elements are the
                 indices of other users that are most similar to the user indicated
                  by index 1.   
        
        Input:
            
            i: index of a given user
            
            n: number of nearest neighbors that we want to find
        '''
        x = self.similarity.getcol(i).toarray()
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
        return items_bought_array.argmax()
    def get_recom(self, i, n):
        '''
        Returns: an index of an item that we will recomed to the user of index i.
        
        Input:
            
            i: index of a given user
            
            n: number of nearest neighbors that we want to use in
                the estimation
        '''
        nn = self.get_nearest_neighbors(i, n)
        return self.best_item(nn)
    def get_recom_info(self, n):
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
        recoms_made = []
        recom_matrix = scsp.dok_matrix(self.trans.shape, dtype = 'int8')
        users_csr = self.users.tocsr()
        most_popular = self.most_popular_item()
        for i in xrange(self.similarity.shape[0]):
            recom = self.get_recom(i, n)
            if users_csr.getrow(i).sum():
                recom_matrix[i, recom] = 1
                recoms_made.append(recom)
            else:
                recom_matrix[i, most_popular] = 1
                recoms_made.append(most_popular)
        recom_info = {'recom_dok_matrix' : recom_matrix,
         'recom_summary' : Counter(recoms_made)}
        return recom_info
    

    
    
