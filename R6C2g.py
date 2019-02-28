 #-*- coding: utf-8 -*-
"""
Created on Wed Apr 09 15:01:10 2014
single zone building model for Smart-E 1.3, developed and validated in the theses of Thomas Berthou (2013)
article : Berthou et al.,Development and validation of a gray box model to predict thermal behavior of occupied office buildings, Energy and Buildings, 2014

This function aims to simulate the heating and cooling requirements and the average indoor temperature of each dwelling or building in the park.
It simulates several buildings (or dwellings) at the same time to speed-up the computation.
The R6C2 model is simulating "energy needs", electricity, gas, district heating or thermodynamics system consumption.
The systems are not directly integrated in the function but are “called” by the algorithm.
It can be operated as required or with regulation (perfect regulation with power limit in V1.2).
It is currently the most expensive function in Smart-E.
This bi-zone model is adapted from (Berthou, 2013).
In the bi-zone version two "R6C2" are connected at the air nodes by an "R2C1". The first zone represents the heated surfaces (bedrooms, living room, kitchen), a second zone represents the unheated or slightly heated surfaces (corridor, guest room, storage rooms).
In the bi-zone version, the internal gains are totally injected into the first zone.
In the single zone version, the heating set-point temperature is reduced to take into account not heating areas. 
This function also contains simplified models for closing shutters and opening windows.
For example, in summer if Tint>Text and Tint>Tcons then the occupants open the windows, this increases the fresh air flow by 1 Vol/h (realistic values).
Similarly in summer, if Tint<Text and the dwelling is occupied, the occupants close the shutters (and windows), reducing incoming solar gains by 90% (realistic values).
In order to initialize the temperature of the wall at t0, we simulate a few days (5 in V1.3) with the same weather conditions as the first 5 days taken in an antichronological direction.
In this function are also implemented dynamic DSM strategies such as those presented in Berthou (2015).
For heating systems with inertia, the emission system is represented by a R1C1 model from the work of Blervaque (2014).
The parameters of this model are adjusted in such a way that the maximum power calculated when sizing the systems can be supplied, and a time constant is associated with each transmitter (from Bézian (1997), 10 or 20 min).
These systems allow to calculate a necessary fluid temperature at the input of the HP model.
A "system" function, called at each simulation time step, allows to model the energy production systems.
There are constants efficiencies for gas boilers, electric radiators and substations with a maximum power limit.
For heat pump, the model presented in Perez theses (2017) is used.

INPUTS
start: Date where the simulation start (10 min time step) (s):INT
end	Date Date where the simulation stop (s):INT
delta: simulation time step (s):INT
city: Table for city (or district) description :DataFrame
tc_hot: Temperature set-point for heating (complete year) (°C) : :array
tc_cold: Temperature set-point for cooling (complete year) (°C):array
internal_load : internal load of the building/dwelling from occupancy and appliances (W) : array
occ: occupancy schedule: array
te: outdoor temperature: array
solar_gain_in : solar flux on external walls:(array)
solar_gain_out : solar flux on internal walls (from windows):(array)
opening : schedule for windows opening - fresh air (manually) : array
OUTPUTS
pa: electricity, gas or heat consumption (W): array 32b
ph: thermal needs (W)	 array 32b
ti	indoor temperature (°C)	array 32b


"""
#from numba import jit
import numpy as np
from ground_temperature import ground_temperature
from air_conditioning import air_conditioning
def R6C2(system, start, end, delta, city, tc_hot, tc_cold, internal_load,
            occ, te, solar_gain_in ,solar_gain_out, opening):
    #%%
    NB_dwe = len(city)    
    b_sur = city['S_total'].values.copy()
    v_air = city['V_air'].values.copy()
    u_roof = city['U_roof'].values.copy()
    u_floor = city['U_floor'].values.copy()
    u_win = city['U_window'].values.copy()
    u_wall = city['U_wall'].values.copy()
    s_out = city['S_vertical_out'].values.copy()
    s_win = city['S_window'].values.copy()
    s_roof = city['S_roof'].values.copy()
    s_floor = city['S_floor'].values.copy()
    nb_occ = city['N_occupant'].values.copy()
    d_inf = city['m_infiltration'].values.copy()
    d_ven_meca = city['d_ven_meca'].values.copy()
    d2 = city['R_no_heating'].values.copy()
    system_efficiency = city['system_efficiency'].values.copy()
    d = 1

    win_yn=city['win_yn'].values
    shutter_yn=city['shutter_yn'].values
    vmeca_yn=city['ven_yn'].values
    clim_yn=city['clim_yn'].values
    p_max_hot = city['P_heating'].values.copy()*d
    p_max_cold = city['P_cooling'].values.copy() * clim_yn
    
    
    tc_hot = (tc_hot + d2*(tc_hot-3))/(1+d2)
#    p_max_hot2 = city['P_heating'].values*d2
#    reduit = 3.
    ####Paramètres du modèle communs à tous les bâtiments####
    mob = 20000. # capacité du mobilier (par surface chauffée) J.K-1.m2
    h_int = 6. #Coeficient de convection interne (W.K^-1.m^-2)
    h_ext = 20. #Coeficient de convection externe (W.K^-1.m^-2)
    rho = 1.2 #masse volumique de l'air (kg.m-3)
    c_air = 1004. #Capacité thermique massique de l'air (J.K^-1.kg^-1)
    vol_h= 1. #vol/h   ventilation mécanique ou ouverture des fenètres, à affiner avec la vitesse du vent?
    #q_vent = 0.0125 #débit de ventilation par occupant (m3.s-1)
    #gi= 7 #watt/m2 #gains internes pas unité de surface (W.m-2)
    ####paramètres du modèle calculé pour chaque bâtiment####
    gr = 0.6 #Part radiative des gains internes (-)
    ci= rho * c_air * v_air + mob * b_sur
    ri = 1. / (h_int * (6*v_air**(2/3.)-s_win)) #Résistance de convection intérieure (K.W-1)
    rs = ri
    rt = (2/h_int + 0.05/0.35)/(s_out/4.) #np.maximum(d,d2)/np.minimum(d,d2) #5 cm de plâtre - connexion entre le zone chauffée et celle non chauffée
    re= 1. / (h_ext * (s_out-s_win + s_roof)) #Resistance de convection extérieure (K.W-1)
    rm0 = 1. / (u_roof * s_roof + u_floor * s_floor + u_wall * (s_out-s_win)) - ri - rs - re#Résistance de conduction des murs (isolation) (K.W-1)
    rm = np.maximum(rm0, re)
#    rinf = 1 / (rho * c_air * d_inf*v_air/3600.)
#    rmeca= 1 / (rho * c_air * d_meca*v_air/3600.)
#    d_inf_meca=d_inf+d_meca
#    rinf_meca = 1. / (rho * c_air * d_inf_meca*v_air/3600.)

#ajouts
    vinf = rho * c_air * d_inf*v_air/3600. #Conductance d'infiltration
    vmeca= rho * c_air * d_ven_meca*v_air/3600. #Conductance équivalente à la ventilation mecaniqu
    vwin=vol_h*v_air/3600*rho*c_air #ventilation par ouverture de fenêtre
    
    rfen = 1. / (u_win * s_win) # prend déja en compte les résistances convectives
    #rf = rinf_meca * rfen / (rinf_meca + rfen) #resistance équivalente aux infiltrations et aux fenètres (K.W-1) 
    
    rse = (city['T_max_flow'].values - 19)/city['P_heating'].values #resistance equivalente systeme-air
    cse = city['sys_time_constant'].values/rse #inertie système d'emission et réseau de distribution 1min, 10 min et 60 min?
    #construction du vecteur de paramètre du bâtiment
    x = {'ci' : ci * 4, 'cm' : city['inertia'].values, 'ri' : ri, 'rs' : rs, 'rm' : rm, 're' : re, 'rfen' : rfen, 'vinf': vinf,
         'vwin' : vwin, 'gr' : gr, 'vmeca': vmeca , 'rse' : rse, 'cse' : cse, 'rt': rt} #'rse' : rse, 'cse' : cse,
#    x2 = {'ci' : ci*4*d2, 'cm' : city['inertia'].values*d2, 'ri' : ri/d2, 'rs' : rs/d2, 'rm' : rm/d2, 're' : re/d2, 'rf' : rf/d2,
#         'vwin' : vwin*d2, 'gr' : gr, 'rse' : rse/d2, 'cse' : cse*d2 , 'rt': rt}


#%% prise en compte de la température du sol /sous sol/cave
    rfloor = 1. / (u_floor * s_roof)
    rwall = 1. / (u_wall * (s_out-s_win))
    rroof = 1. / (u_roof * s_roof)
    r_wr = rwall*rroof/(rwall + rroof)
    t_ground = ground_temperature(te)
    tec = (r_wr * t_ground[:,np.newaxis] + rfloor * te[:,np.newaxis]) / (r_wr + rfloor) # tepérature extèrieure qui prend en compte la température du sol

    #calcul des entrées (gains)

    f=solar_gain_in
    fm=solar_gain_out
    gain = internal_load #internal heat gain wrong occ value ?)
#initialisation de la simulation
    step = end-start
    ti, ti2 = 20, 20
    save_ph = [np.zeros(NB_dwe)]
    save_pa = [np.zeros(NB_dwe)]
    save_p1_hot_inf = [np.zeros(NB_dwe)]
    save_p1_hot_meca = [np.zeros(NB_dwe)]
    save_p1_hot_win = [np.zeros(NB_dwe)]
    save_p1_hot_wino = [np.zeros(NB_dwe)]
    save_p1_hot_wa = [np.zeros(NB_dwe)]
    save_ti = [20*np.ones(NB_dwe)]

    tm, tm2 = 17, 17
    tf, tf2 = 30*np.ones(NB_dwe), 30*np.ones(NB_dwe) #température de fluide
    p_min = 0
    ph, ph2 = 0, 0
    #Initialisation des états : simulation en besoin des 5 premiers jours de l'année, condition de symetrie à t0
    t_iner = 5*24*3600/delta 
    for i in range(int(start+t_iner),0,-1): #few days are simulated to initialised the stats
         th=(tm/x['rm']+ te[i+1]/x['re']+fm[i+1,:])/(1/x['rm']+1/x['re'])
         ts=(ti/x['ri']+tm/x['rs']+f[i+1,:]+x['gr']*gain[i,:])/(1/x['ri']+1/x['rs'])
         p1 = x['ci']*(tc_hot[i,:]-ti)/delta+(ti-ts)/x['ri']+(ti-te[i+1])/x['rfen'] + (ti-te[i+1])*x['vinf']-(1-x['gr'])*gain[i,:]  
         ph = np.maximum(np.minimum(p1,p_max_hot),p_min)
         tm = ((th-tm)/x['rm']+(ts-tm)/x['rs'])*delta/x['cm']+tm
         ti = ((ts-ti)/x['ri']+(te[i+1]-ti)/x['rfen'] + (te[i+1]-ti)*x['vinf']+ph+(1-x['gr'])*gain[i,:])*delta/x['ci']+ti
    #début de la simulation
    #%%
    
    for i in range(1,int(step)):
        
        #modèle simplifié de fermeture des volets
        shutter = ((ti > tc_cold[i-1,:]) & (occ[i-1,:] > 0)).astype(int)*0.9 * shutter_yn # closing shutters (stop 90% of the radiation)
        #modèle simplifié d'ouverture des fenètres sur un pas de temps été et hivers
        win = np.maximum(((ti > tc_cold[i-1,:]) & (te[i-1] < ti) & (occ[i-1,:] > 0)).astype(int)*1,opening[i,:]) #opening windows   
        #modèle simplifié de ventilation mecanique
        meca = ((occ[i-1,:] > 0)).astype(int)*1. * vmeca_yn
#        meca=1
        #calcul des noeuds:
        th = (tm/x['rm']+ tec[i-1]/x['re']+fm[i-1,:]*d)/(1/x['rm']+1/x['re'])
        ts = (ti/x['ri'] + tm/x['rs'] +  f[i-1,:]*d - shutter*f[i-1,:]*d + x['gr']*gain[i,:]) / (1/x['ri'] + 1/x['rs'])
        tm = ((th-tm)/x['rm']+(ts-tm)/x['rs'])*delta/x['cm']+tm
        #calcul de la puissance
        #p1_hot = x['ci']*(tc_hot[i,:]-ti)/delta+(ti-ts)/x['ri'] + (ti-te[i-1])/x['rf']-(1-x['gr'])*gain[i,:]+x['vwin']*win*(ti-te[i-1])
        p1_hot = x['ci']*(tc_hot[i,:]-ti)/delta+(ti-ts)/x['ri'] + (ti-te[i-1])/x['rfen'] + (ti-te[i-1])*x['vinf']\
        + (ti-te[i-1])*win*win_yn*x['vwin'] + (ti-te[i-1])*x['vmeca']*meca -(1-x['gr'])*gain[i,:]
        
        #p1_cold = x['ci']*(tc_cold[i,:]-ti)/delta+(ti-ts)/x['ri'] + (ti-te[i-1])/x['rf']-(1-x['gr'])*gain[i,:]+x['vwin']*win*(ti-te[i-1])
        p1_cold = x['ci']*(tc_cold[i,:]-ti)/delta+(ti-ts)/x['ri'] + (ti-te[i-1])/x['rfen'] + (ti-te[i-1])*x['vinf']\
                 + (ti-te[i-1])*win*win_yn*x['vwin'] + (ti-te[i-1])*x['vmeca']*meca -(1-x['gr'])*gain[i,:]
        
        p1_hot_inf = (ti-te[i-1])*x['vinf']
        p1_hot_meca = (ti-te[i-1])*x['vmeca']*meca #deperditions infiltration
        p1_hot_win = (ti-te[i-1])/x['rfen']#deperditions fenètres
        p1_hot_wino = (ti-te[i-1])*win*win_yn*x['vwin'] #deperdition ouverture des fenêtres
        p1_hot_wa = (tm-th)/x['rm']#parois
        # fonction CVC
        
        [pa, ph] = system(p_max_hot, p1_hot, system_efficiency, te[i], ti, tf)
        [pa_cold, ph_cold] = air_conditioning(p_max_cold, p1_cold, system_efficiency, te[i], tf)
        #modèle de d'émetteur générique
        tf = ((ti-tf)/x['rse'] + ph + ph_cold)*delta/x['cse'] + tf
               
        #%%zone vide
        #calcul des noeuds:
#        th2 = (tm2/x2['rm']+ te[i-1]/x2['re']+fm[i-1,:]*d2)/(1/x2['rm']+1/x2['re'])
#        ts2 = (ti2/x2['ri'] + tm2/x2['rs'] + f[i-1,:]*d2 - shutter*f[i-1,:]*d2)/(1/x2['ri'] + 1/x2['rs'])
#        tm2 = ((th2-tm2)/x2['rm']+(ts2-tm2)/x2['rs'])*delta/x2['cm']+tm2
#        #calcul de la puissance
#        p1_hot2 = x2['ci']*(tc_hot[i,:]-reduit-ti2)/delta+(ti2-ts2)/x2['ri'] + (ti2-te[i-1])/x2['rf'] + x2['vwin']*win*(ti2-te[i-1])
#        p1_cold2 = x2['ci']*(tc_cold[i,:]-ti2)/delta+(ti2-ts2)/x2['ri'] + (ti2-te[i-1])/x2['rf'] + x2['vwin']*win*(ti2-te[i-1])
#        # fonction CVC
#        #ph2 = np.maximum(np.minimum(p1_hot2,p_max_hot2),0)
#        #pa2 = ph2
#        [pa2, ph2] = system(p_max_hot2, p1_hot2, p1_cold2, system_efficiency, te[i], ti2, tf2)
#        #modèle de d'émetteur générique
#        tf2 = ((ti2-tf2)/x2['rse'] + ph2)*delta/x2['cse'] + tf2
#        #%% calcul des température intereures
#        ti2t = ti2
#        ti2 = ((ts2-ti2)/x2['ri'] + (te[i-1]-ti2)/x2['rf'] + (tf-ti2)/x2['rse'] + (ti-ti2)/x2['rt'] - x2['vwin']*win*(ti2-te[i-1]))*delta/x2['ci'] + ti2
#        
        #ti = ((ts-ti)/x['ri'] + (te[i-1]-ti)/x['rf'] + (tf-ti)/x['rse'] + (1-x['gr'])*gain[i,:] - x['vwin']*win*(ti-te[i-1]))*delta/x['ci'] + ti
        ti = ((ts-ti)/x['ri'] + (te[i-1]-ti)/x['rfen'] + (tf-ti)/x['rse'] + (te[i-1]-ti)*x['vinf'] + (te[i-1]-ti)*win*win_yn*x['vwin']\
          + (te[i-1]-ti)*x['vmeca']*vmeca_yn*meca +(1-x['gr'])*gain[i,:])*delta/x['ci']+ti
      
        
        #saugarde des données()
        save_ph.append(ph + ph_cold)
        save_pa.append(pa + pa_cold)
        save_ti.append(ti)

        save_p1_hot_meca.append(p1_hot_meca)
        save_p1_hot_inf.append(p1_hot_inf)
        save_p1_hot_win.append(p1_hot_win)
        save_p1_hot_wino.append(p1_hot_wino)
        save_p1_hot_wa.append(p1_hot_wa)

#        save_ph2.append(ph2)
#        save_pa2.append(pa2)
#        save_ti2.append(ti2)
        #save.append(tm)
        
    
    return np.array(save_pa, dtype=np.float32), np.array(save_ph,dtype=np.float32), np.array(save_ti, dtype=np.float32),\
	 np.array(save_p1_hot_inf, dtype=np.float32),np.array(save_p1_hot_meca, dtype=np.float32), np.array(save_p1_hot_win, dtype=np.float32), np.array(save_p1_hot_wino, dtype=np.float32), np.array(save_p1_hot_wa, dtype=np.float32)  #save_pa2, save_ph2, save_ti2#, tm, f, win, shutter
