# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 09:41:15 2018
(Tsol) is considered to be a periodic function with a
phase shift, based on the average annual air temperature. The response to periodic temperature stress at a depth z is determined
using the theory of conduction heat diffusion in a semi infinite solid medium (Hagentoft, 2001)
@author: thomas.berthou
"""
import numpy as np
def ground_temperature(te):
    
    te_jour = te.reshape(365,24*6).mean(1)
    t_amp = np.abs(np.max(te_jour)-np.mean(te_jour))
    j0 = np.argmin(te_jour)+1
    asol = 1.5/(2085*1500) * 24 * 3600 #diffusivité thermique de l'argile, (m²/jour) 
    prof = 2 #profondeur (m)
    e = prof*(3.14/(365*asol))**0.5 #ground constante
    t_ground = np.mean(te) - (t_amp) * np.exp(-e) * np.cos((2*3.14)/365 * (np.arange(1,366)- j0 - e))
    t_ground = np.repeat(t_ground,24 * 6)
#    t_ground0 = np.mean(te) -3 * np.cos(np.linspace(1,13,365) * (2*3.14)/12)
    
#    plt.plot(te)
#    plt.plot(t_ground)
#    plt.xlabel('time step (10 min)')
#    plt.ylabel('temperature (°C)')
#    plt.legend(['outdoor air','ground'])
    return t_ground