# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 21:06:56 2013

@author: Kirill
"""

import unittest

from MatchingProblem import MatchingProblem

import numpy as np
import scipy.sparse as scsp
from collections import Counter

#create items
rows = np.array(   [0,0,0,0,  1,1, 2,2,2,  3,3,3, 4,4,4,4,  5,5,5, 6,6,6,6,  7,7,7])
columns = np.array([0,2,7,11, 2,8, 0,7,11, 0,4,8, 1,7,9,10, 5,6,8, 1,6,7,11, 3,5,8])
values = np.ones(len(rows)).astype('int')

items = scsp.csc_matrix((values, (rows, columns)), shape=(8, 12))

# create users
rows = np.array(   [0,0, 1,1,1, 2,2, 3,3,3,3, 4,4, 5,5,5, 6, 7,7,7, 9,9])
columns = np.array([0,5, 0,1,3, 2,4, 0,1,3,5, 2,4, 0,4,5, 1, 2,3,5, 0,4])
values = np.ones(len(rows)).astype('int')

users = scsp.csc_matrix((values, (rows, columns)), shape=(10, 6))

# create transactions
rows = np.array(   [0,0,0,0, 1,1,1, 2,2, 3,3, 4,4,4, 5,5, 6, 7,7, 8,8, 9,9])
columns = np.array([0,2,3,5, 2,5,7, 1,7, 5,7, 2,6,7, 5,6, 2, 0,7, 0,2, 4,6])
values = np.ones(len(rows)).astype('int')

trans = scsp.csc_matrix((values, (rows, columns)), shape=(10, 8))

class MatchingProblemTest(unittest.TestCase):
    def test_most_popular_item(self):
        my_problem = MatchingProblem(users, items, trans)
        result = my_problem.most_popular_item()
        expected = 2
        self.assertEquals(result, expected)
        
    def test_get_nearest_neighbors(self):
        my_problem = MatchingProblem(users, items, trans)
        result = my_problem.get_nearest_neighbors(2, 3)
        expected = np.array([7, 9, 4])
        self.assertTrue(np.array_equal(result, expected))
        
    def test_best_item(self):
        my_problem = MatchingProblem(users, items, trans)
        result = my_problem.best_item(np.array([7,9,4]))
        expected = 6
        self.assertEqual(result, expected)
        
    def test_get_recom(self):
        my_problem = MatchingProblem(users, items, trans)
        result = my_problem.get_recom(2, 3)
        expected = 6
        self.assertEqual(result, expected)
        
    def test_get_recom_info_summary(self):
        my_problem = MatchingProblem(users, items, trans)
        result = my_problem.get_recom_info(3)['recom_summary']
        expected = Counter({5: 6, 7: 2, 2: 1, 6: 1})
        self.assertDictEqual(result, expected)
        
    def test_get_recom_info_matrix(self):
        my_problem = MatchingProblem(users, items, trans)
        result = my_problem.get_recom_info(3)['recom_dok_matrix'].toarray()
        
        expected = np.array([ [0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0]]).astype('int8')
                              
        self.assertTrue(np.array_equal(result, expected))
        
def main():
    unittest.main()
    
if __name__ == "__main__":
    main()
    

