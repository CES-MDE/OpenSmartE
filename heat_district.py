# -*- coding: utf-8 -*-
"""
Created on Thu Apr 07 14:36:23 2016

@author: tberthou
Simplified model of electic convector
to be update with radian and convective part
"""

import numpy as np
def heat_district(pcNom, p1_hot, system_efficiency,*args):
    
    ph = np.maximum(np.minimum(p1_hot,pcNom),0)
    pa = 1/system_efficiency*ph
    return pa, ph