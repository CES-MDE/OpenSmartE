# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 15:14:51 2019

@author: thomas.berthou
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import numpy as np
def heat_pump(p_max_hot, p1_hot, system_efficiency, te, ti, tf):
    #Tevap = np.arange(-10,20)
    #Tcond = np.arange(30,55,5)
    #plt.figure()
    #for i in Tcond:
    #    cop = 0.1 * Tevap - 0.12 * i +7.55
    #    plt.plot(Tevap,cop)
    #    
    #plt.legend(['Tcondenseur = '+str(i) for i in Tcond])
    #plt.xlabel('Tevaporateur (°C) = Text')
    #plt.ylabel('COP en chauffage')
    #
    #
    #Tevap = np.arange(5,25,5)
    #Tcond = np.arange(15,40)
    #plt.figure()
    #for i in Tevap:
    #    cop =  -0.1 * Tcond + 0.12 * i + 6.55
    #    plt.plot(Tcond,cop)
    #    
    #plt.legend(['Tevaporateur = '+str(i) for i in Tevap])
    #plt.xlabel('Tcondenseur (°C) = Text')
    #plt.ylabel('COP en climatisation')
    #print('p1_hot ', p1_hot[0], p1_hot[-1])
    #print('tf ', tf[0], tf[-1])
    cop_hot = 0.1 * te - 0.12 * tf + 7.55
    
    cop_hot[cop_hot<1] = 1
    cop_hot[cop_hot>10] = 10
    
    #print('cop ',cop_hot[0], cop_hot[-1])
    ph = np.maximum(np.minimum(p1_hot,p_max_hot),0)
    ph[ph < 0.1 * p_max_hot] = 0 #la pac ne fonctionne pas à charge trop réduite (<10% de Pmax)
    #print('ph ',ph[0], ph[-1])
    #p_max_hot[ph == 0]=0
    p_max_corr = p_max_hot.copy()
    p_max_corr[ph == 0]=0
    pa = ph/cop_hot + 0.0125 * p_max_corr
    #print( 'pa' , pa[0], pa[-1])
    #time.sleep(1)
    return pa, ph