# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 10:37:32 2019
air conditionning model from Perez 2017 : "Contribution à la conception énergétique de quartiers : simulation, optimisation et aide à la décision"
@author: thomas.berthou
"""
import numpy as np
def air_conditioning(p_max_cold, p1_cold, system_efficiency, te, tf):

    
    p1_cold = np.minimum(p1_cold, 0)
    p1_cold = np.abs(p1_cold)
    
    cop_cold = -0.1 * te + 0.12 * tf + 6.55
    cop_cold[cop_cold<1] = 1 #cop min is 1
    cop_cold[cop_cold>10] = 10 #cop max is 10
    
    ph = np.maximum(np.minimum(p1_cold,p_max_cold),0)
    ph[ph < 0.1 * p_max_cold] = 0 #la pac ne fonctionne pas à charge trop réduite (<10% de Pmax)
    p_max_corr = p_max_cold.copy()
    p_max_corr[ph == 0]=0
    pa = ph/cop_cold + 0.0125 * p_max_corr
    
    return pa, -ph