# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 09:40:39 2019

@author: thomas.berthou
"""
import pickle as pkl
from R6C2g import R6C2
import gzip
from heat_district import heat_district
# to create test data
# =============================================================================

# inputs = [heat_district ,start, end, delta, city.iloc[index], tc_hot[:,index],tc_cold[:,index], internal_load[:,index], occ[:,index], te,solar_gain_in[:,index],solar_gain_out[:,index], opening[:,index]]
# names = ['system', 'start', 'end', 'delta', 'city', 'tc_hot', 'tc_cold', 'internal_load','occ', 'te', 'solar_gain_in' ,'solar_gain_out', 'opening']
#  
# inputs_r6c2 = dict()
# for key, value in zip(names, inputs ):
#     inputs_r6c2[key] = value
# pkl.dump(inputs_r6c2, gzip.open( 'inputs_r6c2.pklz', 'wb' ))
# =============================================================================

    

inputs_r6c2 = pkl.load( gzip.open('inputs_r6c2.pklz', 'rb' ))
(pa, ph, ti,p1_hot_inf,p1_hot_meca,p1_hot_win,p1_hot_wino,p1_hot_wa) = R6C2(**inputs_r6c2)
