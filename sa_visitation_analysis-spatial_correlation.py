#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pylab', 'inline')
import pickle
import pandas as pd
import csv
import re
import time
import os
import numpy as np
from math import radians, cos, sin, asin, sqrt
import itertools
from numpy import mean
from copy import deepcopy
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
import multiprocessing
from scipy import stats


# In[11]:


def haversine(lon1, lat1, lon2, lat2): # 经度1，纬度1，经度2，纬度2 （十进制度数）  
    """ 
    Calculate the great circle distance between two points  
    on the earth (specified in decimal degrees) 
    """  

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])  
    dlon = lon2 - lon1   
    dlat = lat2 - lat1   
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2  
    c = 2 * asin(sqrt(a))   
    r = 6371 
    return c * r 


# In[15]:


# plates_kinds = pickle.load(open('./plates_kinds_list_ee.pkl','rb'))
month = '07'
weekends = {'07':[2,3,9,10,16,17,23,24,30,31],
            '08':[6,7,13,14,20,21,27,28,31],
           '09': [3,4,10,11,17,18,24,25,31],
           '10': [1,2,8,9,15,16,22,23,29,30],
           '11': [5,6,12,13,19,20,26,27,31],
           '12': [3,4,10,11,17,18,24,25,31]}
weekdays = set(range(1,32)) - set(weekends[month])


# In[17]:


def still_filter(record_taxi_less):
#     print('still_filter processing..')
    full_dict={}
    timethresh=0
    counter=0
    for plate in record_taxi_less:
        full_dict[plate] = []
        for i in range(len(record_taxi_less[plate])):
            record=record_taxi_less[plate][i]
            distemp = []
            timegaptemp = []
            tripdis = []
            triptime = []
            trip = []
            buffer = []
            outliers = []
            timelength = 0
            for i in range(1,len(record)):#record: plate,latlgt,time,status,flag
                if((record[i][0] == record[i-1][0])and record[i-1][4] == 0 ):
            #             print('0')
                    if(record[i][2]==record[i-1][2] and record[i][1] == record[i-1][1]):
                        timelength = timelength + record[i][3]-record[i-1][3] 
                        rec=deepcopy(record[i])
                        rec.append(i)
                        buffer.append(rec)
                    if(timelength > timethresh and haversine(record[i][2],record[i][1],record[i-1][2],record[i-1][1]) != 0):#0 following 0's
                        outliers.append([buffer[0][1],buffer[0][2],buffer[0][-1],buffer[-1][-1],buffer[-1][3]-buffer[0][3]])#lat,lgt,srctime,desttime,timelength
            #                 record[i][-1] = 3 #divide the trip
                        buffer = []
                        timelength = 0
                    if(haversine(record[i][2],record[i][1],record[i-1][2],record[i-1][1]) != 0):
                        buffer = []
                        timelength = 0
            #         distemp.append(haversine(record[i][2],record[i][1],record[i-1][2],record[i-1][1]))
            #         timegaptemp.append(record[i][3]-record[i-1][3])
                if(record[i][4]!=record[i-1][4] or record[i][0] != record[i-1][0]) and record[i-1][4] == 0:
                    if(timelength > timethresh):
                        outliers.append([buffer[0][1],buffer[0][2],buffer[0][3],buffer[-1][3],buffer[-1][3]-buffer[0][3]])
                        buffer = []
                        timelength = 0
                    else:
                        buffer = []
                        timelength = 0
            filter_outlier=[]
            for out in outliers:
                if out[4]>3*60*60:
                    filter_outlier.append(out)
            full_dict[plate].append(filter_outlier)
        counter+=1
#         print(counter)
    return full_dict


# In[20]:


def extract_trajs(full_dict,record_taxi_less,valid_plates):
    seekings={}
    for po in valid_plates:
        seekings[po] = []
        for d in range(len(full_dict[po])):
            inds=[0]
            po_stay=full_dict[po][d]
            for stay in po_stay:
                inds.append(stay[2])
                inds.append(stay[3])
            inds.append(len(record_taxi_less[po][d])-1)
            seek=[]
            for i in range(len(inds)//2):
                seek.append(record_taxi_less[po][d][inds[2*i]:inds[2*i+1]])
            seekings[po].append(seek)

    trajs={}
    for po in valid_plates:
        trajs[po] = []
        for d in range(len(seekings[po])):
            for seek in seekings[po][d]:
    #             if len(seek)<=50:
    #                 continue
                traj=[]
                for row in seek:
                    if row[-2]==1:
                        continue
                    lgt = float(row[1])#>100
                    lat = float(row[2])#<30
                    plate = row[0][-6:]
                    locgrid = [int((lat-22.44)/0.009)+1,int((lgt-113.75)/0.01)+1]
                    time=row[3]%(24*3600)
                    timeslot = int((time/300)+1)
                    if len(traj)>0 and traj[-1][1]==locgrid[0] and traj[-1][2]==locgrid[1] and traj[-1][3]==timeslot:
                        continue
                    traj.append([plate,locgrid[0],locgrid[1],timeslot])
                if len(traj)>1:
                    trajs[po].append(traj)
    return trajs


# In[21]:


def extract_visitation_frequency(trajs,plate_index=0):
    grid_vf_plates = {}#grid_vf_plates[plate] = grid_vf, grid_vf[lat|lgt] = #visits
    for plate in trajs:
        grid_vf_plates[plate] = {}
        for traj in trajs[plate][:]:
            for i in range(len(traj)):
                grid='|'.join(map(str,traj[i][1:3]))
                if grid not in grid_vf_plates[plate].keys():
                    grid_vf_plates[plate][grid] = 1
                else:
                    grid_vf_plates[plate][grid] += 1
                    
    grid_vf_plate = grid_vf_plates[valid_plates[plate_index]]
    return grid_vf_plate


# In[22]:


def extract_familiar_unfamiliar_grids(num_weeks,trajs,plate_index=0,f_thresh_one_week = 2):
    grid_vf_plates = {}#grid_vf_plates[plate] = grid_vf, grid_vf[lat|lgt] = #visits
    for plate in trajs:
        grid_vf_plates[plate] = {}
        for traj in trajs[plate][-5*num_weeks:]:
            for i in range(len(traj)):
                grid='|'.join(map(str,traj[i][1:3]))
                if grid not in grid_vf_plates[plate].keys():
                    grid_vf_plates[plate][grid] = 1
                else:
                    grid_vf_plates[plate][grid] += 1
                    
    grid_vf_plate = grid_vf_plates[valid_plates[plate_index]]
    unfamiliar_grids = []
    familiar_grids = []
    for lat_grid in range(1,49):
        for lgt_grid in range(1,91):
            grid='|'.join(map(str,[lat_grid,lgt_grid]))
            if grid in grid_vf_plate.keys():
                if grid_vf_plate[grid]<=f_thresh_one_week*num_weeks:
                    unfamiliar_grids.append(grid)
                else:
                    familiar_grids.append(grid)
            else:
                unfamiliar_grids.append(grid)
    return familiar_grids,unfamiliar_grids


# In[24]:


def assign_action(trajs):
    '''
    Assign spatial actions
    '''
    actions = []
    for lat_ind in [-1,0,1]:
        for lgt_ind in [-1,0,1]:
            actions.append([lat_ind,lgt_ind])

    trajs_actions = {}
    for plate in trajs:
        trajs_actions[plate]  = []
        for i_traj in range(len(trajs[plate])):
#             trajs_actions[plate].append([])
            traj_actions = []
            traj = trajs[plate][i_traj]
            last_step = traj[0][1:3]#[lat index, lgt index]
            for j_step in range(1,len(traj)):
                current_step = traj[j_step][1:3]
                if(current_step != last_step):
                    action = list(np.array(current_step)-np.array(last_step))
                    if action in actions:
                        action_ind = actions.index(action)
                        last_step.append(action_ind)
                        traj_actions.append(last_step)
                    last_step = deepcopy(current_step)
            trajs_actions[plate].append(traj_actions)
    return trajs_actions
# trajs[valid_plates[0]][0][0:10]


# In[25]:


def extract_sa_visitation_frequency(trajs_actions):
    '''
    output: sa_vf_plates[plate] = {'lat|lgt|action':#visits,...}
    '''
    sa_vf_plates = {}
    for plate in trajs_actions:
        sa_vf_plates[plate] = {}
        for traj in trajs_actions[plate][:]:
            for i in range(len(traj)):
                sa='|'.join(map(str,traj[i][:]))
                if sa not in sa_vf_plates[plate].keys():
                    sa_vf_plates[plate][sa] = 1
                else:
                    sa_vf_plates[plate][sa] += 1
    return sa_vf_plates


# # Correlation analysis

# w1:
# Q(s,a): the expected earning efficiency within 5 hours after exiting grid s via action a.
# V(s): the expected earning efficiency within 5 hours after exiting grid s via any action.
# sa_vf: the visitation frequency of each state action pair.

# w2:
# sa_vf: the visitation frequency of each state action pair.

# In[27]:


def calculate_fee(dis):
    if(dis<2):
        return 10
    else:
        return (10+2.4*(dis-2))
def calculate_fee_ratio(exploring_trajs):
    expected_ee = {}
    for exploring_grid in exploring_trajs:
#         print(exploring_grid)
        fee_grid = 0
        time_grid = 0
        for traj in exploring_trajs[exploring_grid]:
            fee,t = calculate_fee_info(traj)
            fee_grid+=fee
            time_grid+=t
        if time_grid >0:
#             if(fee_grid/time_grid>200):
#                 print(exploring_grid)
            if np.isnan(fee_grid/time_grid):
                expected_ee[exploring_grid] = 0
            else:
                expected_ee[exploring_grid] = fee_grid/time_grid
        else:
            expected_ee[exploring_grid] = 0
    return expected_ee


# In[28]:


def calculate_fee_info(traj):
    trip = extract_trip(traj)
    trip = trip_speed_filter(trip,120)
    try:
        trippercar = conclude_taxi_day(trip)
    #     trippercar = taxi_day_time_filter(trippercar)
    #     len(trippercar)
        fee_info = calculate_ratio(trippercar)
#     print(len(fee_info))
    except:
#         print(len(trip))
        return 0,0
    return fee_info[0][1],fee_info[0][2]


# In[29]:


def extract_trip(record):#record: plate,latlgt,time,status,flag
#     print('trip extraction...')
    distemp = []
    timegaptemp = []
    tripdis = []
    triptime = []
    trip = []
    for i in range(1,len(record)):
#         if(i%(int(len(record)/100))==0):
#             print(i/len(record))
        if(record[i][0]==record[i-1][0]):
            distemp.append(haversine(record[i][2],record[i][1],record[i-1][2],record[i-1][1]))
            timegaptemp.append(record[i][3]-record[i-1][3])
    #     if(record[i][3]-record[i-1][3] < 0):
    #         print(record[i-1],"\n",record[i])
    #         break
            if((record[i][-2]!=record[i-1][-2]) and sum(timegaptemp)> 0) or i==len(record)-1:#next trip
                if(record[i-1][-2] == 0):
                    trip.append([record[i-1][0],sum(distemp),sum(timegaptemp),sum(distemp)*3600/sum(timegaptemp),record[i-1][-2],0])
                    distemp = []
                    timegaptemp = []
                if(record[i-1][-2] == 1):
                    trip.append([record[i-1][0],sum(distemp),sum(timegaptemp),sum(distemp)*3600/sum(timegaptemp),record[i-1][-2],calculate_fee(sum(distemp))])
                    distemp = []
                    timegaptemp = []
        elif(sum(timegaptemp)> 0):#next taxi
            if(record[i-1][-2] == 0):
                trip.append([record[i-1][0],sum(distemp),sum(timegaptemp),sum(distemp)*3600/sum(timegaptemp),record[i-1][-2],0])
                distemp = []
                timegaptemp = []
            if(record[i-1][-2] == 1):
                trip.append([record[i-1][0],sum(distemp),sum(timegaptemp),sum(distemp)*3600/sum(timegaptemp),record[i-1][-2],calculate_fee(sum(distemp))])
                distemp = []
                timegaptemp = []
#     print('#trips: ',len(trip)) 
    return(trip)
def trip_speed_filter(trip, speed_thresh):
    trip_filtered = []
    for i in range(len(trip)):
        if(trip[i][3]<speed_thresh):
            trip_filtered.append(trip[i])
    return trip_filtered
def conclude_taxi_day(trip):# trip: plate, distance, time length, speed, fee
#     print('conclude_taxi_day processing...')
    time0 = 0
    time1 = 0
    dis1 = 0
    fee1 = 0
    k0 = 0
    trippercar = []
    for i in range(0,len(trip)):
        if(trip[i][-2] == 0):
            time0 = time0 + trip[i][2]
        if(trip[i][-2] == 1):
            time1 = time1 + trip[i][2]
            dis1 = dis1 + trip[i][1]
            fee1 = fee1 + trip[i][-1]
    trippercar.append([trip[i][0],time0,time1,dis1,fee1])
#     print(k0,len(trip),len(trippercar))#65156 1356130 16931, 61840 1341939 17033
    return trippercar
def calculate_ratio(trippercar):
#     print('calculate ratio...')
    timeratio = []
    feeratio = []
    fee = []
    fee_info = []
    for i in range(len(trippercar)):
#         if(trippercar[i][2]+trippercar[i][1] == 0):
#             print(trippercar[i])
        timeratio.append(trippercar[i][2]/(trippercar[i][2]+trippercar[i][1]))
        feeratio.append(trippercar[i][-1]*60*60/(trippercar[i][2]+trippercar[i][1]))
        fee.append([trippercar[i][0],trippercar[i][-1]*60*60/(trippercar[i][2]+trippercar[i][1]),trippercar[i][2]/(trippercar[i][2]+trippercar[i][1]),trippercar[i][-1],trippercar[i][2]])
        fee_info.append([trippercar[i][0],trippercar[i][-1],(trippercar[i][2]+trippercar[i][1])/3600])
        #fee: plate,earning per hour, service time ratio, earnings in a day, service time in a day
    return fee_info


# In[30]:


def calculate_expected_ee(exploring_trajs_plates):
    expected_ee_plates = {}
    for plate in exploring_trajs_plates.keys():
#         print(plate)
        expected_ee = calculate_fee_ratio(exploring_trajs=exploring_trajs_plates[plate])
        expected_ee_plates[plate] = expected_ee
    return expected_ee_plates


# ## s-a visitation frequency

# In[31]:


def extract_q_sa_v_s_week(records_w1):
    '''
    Extract the Q(s,a) and the V(s) of each plate in a week.
    Output: qsa_plates[plate][sa_ind] = float 
            vs_plates[plate][s_ind] = float 
    '''
    qsa_exploring_trajs_plates = {}
    exploring_traj = []    
    vs_exploring_trajs_plates = {}
    history = []
    exploring_grids = []
    flag =1
    actions = []
    for lat_ind in [-1,0,1]:
        for lgt_ind in [-1,0,1]:
            actions.append([lat_ind,lgt_ind])
    for plate in records_w1:
        qsa_exploring_trajs = {}
        vs_exploring_trajs = {}
        for traj in records_w1[plate]:
            sign = True #the sign to indicate if all of the unfamilar grids have been recorded
            exploring_history = []
            k = 0
            while(sign):
                k+=1
                flag = 1 #
                last_history = deepcopy(exploring_history)
                for i in range(len(traj)-1): 
                    row = traj[i]
                    next_row = traj[i+1]
                    lgt = float(row[1])#>100
                    lat = float(row[2])#<30
                    plate = row[0][-6:]
                    locgrid = [int((lat-22.44)/0.009)+1,int((lgt-113.75)/0.01)+1]
                    time=row[3]%(24*3600)
                    timeslot = int((time/300)+1)
                    indicator = row[4] #passenger indicator
                    next_lgt = float(next_row[1])#>100
                    next_lat = float(next_row[2])#<30
                    next_locgrid = [int((next_lat-22.44)/0.009)+1,int((next_lgt-113.75)/0.01)+1]
                    action = list(np.array(next_locgrid) - np.array(locgrid))
                    grid='|'.join(map(str,locgrid))
                    next_grid = '|'.join(map(str,next_locgrid))
                    if(flag == 1):#check if exploring
                        if(next_grid != grid and action in actions and indicator == 0):#exiting current grid
                            if([grid,time] not in exploring_history):#if the grid has not been recorded, entering the exploring, else skip
                                flag = 0 #entering exploring
                                s_time = deepcopy(time)#exploring start time
                                exploring_grid = deepcopy(grid)
    #                             if(exploring_grid == '9|31' and s_time == 77708):
    #                                 print(row)
                                locgrid.append(actions.index(action))
                                grid_action = '|'.join(map(str,locgrid))#current state-action
                                exploring_traj.append(row)
                                exploring_history.append([exploring_grid,s_time])
                                exploring_grids.append(exploring_grid)
                                if grid_action not in qsa_exploring_trajs.keys():
                                    qsa_exploring_trajs[grid_action] = []
                                if exploring_grid not in vs_exploring_trajs.keys():
                                    vs_exploring_trajs[exploring_grid] = []

                    if(flag == 0):#in exploring
    #                     if(exploring_grid == '9|31' and s_time == 77708):
    #                         print(row)
                        if(time < (s_time+5*3600) and flag == 0):#within 5 hours after exploring
                            exploring_traj.append(row)
                        if(time >= (s_time + 5*3600) or i==(len(traj)-2) and flag == 0):
                            flag = 1 #end of sight
                            qsa_exploring_trajs[grid_action].append(exploring_traj)
                            vs_exploring_trajs[exploring_grid].append(exploring_traj)
                            exploring_traj = []
                if len(exploring_history) == len(last_history):
    #                 print(len(exploring_history),len(last_history))
                    sign = False
    #             print(exploring_history)
                history.append(deepcopy(exploring_history))
#             print(k)
        qsa_exploring_trajs_plates[plate] = deepcopy(qsa_exploring_trajs)
        vs_exploring_trajs_plates[plate] = deepcopy(vs_exploring_trajs)
        
        expected_qsa_plates = calculate_expected_ee(qsa_exploring_trajs_plates)
        expected_vs_plates = calculate_expected_ee(vs_exploring_trajs_plates)
        
    return expected_qsa_plates,expected_vs_plates


# In[32]:


def extract_r_sa_week(records_w1):
    '''
    Extract the r(s,a) of each plate in a week, 
    which is the average earning efficiency of each plate in the target state s' after (s,a).
    Output: r_sa_plates[plate][sa_ind] = float  
    '''
    rsa_exploring_trajs_plates = {}
    exploring_traj = []    
    vs_exploring_trajs_plates = {}
    history = []
    exploring_grids = []
    flag =1
    actions = []
    for lat_ind in [-1,0,1]:
        for lgt_ind in [-1,0,1]:
            actions.append([lat_ind,lgt_ind])
    for plate in records_w1:
        rsa_exploring_trajs = {}
        vs_exploring_trajs = {}
        for traj in records_w1[plate]:
            sign = True #the sign to indicate if all of the unfamilar grids have been recorded
            exploring_history = []
            k = 0
            while(sign):
                k+=1
                flag = 1 #
                last_history = deepcopy(exploring_history)
                for i in range(len(traj)-1): 
                    row = traj[i]
                    next_row = traj[i+1]
                    lgt = float(row[1])#>100
                    lat = float(row[2])#<30
                    plate = row[0][-6:]
                    locgrid = [int((lat-22.44)/0.009)+1,int((lgt-113.75)/0.01)+1]
                    time=row[3]%(24*3600)
                    timeslot = int((time/300)+1)
                    indicator = row[4] #passenger indicator
                    next_lgt = float(next_row[1])#>100
                    next_lat = float(next_row[2])#<30
                    next_locgrid = [int((next_lat-22.44)/0.009)+1,int((next_lgt-113.75)/0.01)+1]
                    action = list(np.array(next_locgrid) - np.array(locgrid))
                    grid='|'.join(map(str,locgrid))
                    next_grid = '|'.join(map(str,next_locgrid))
                    
                    if(flag == 1):#check if exploring
                        if(next_grid != grid and action in actions and indicator == 0):#exiting current grid
                            if([grid,time] not in exploring_history):#if the grid has not been recorded, entering the exploring, else skip
                                entering_grid = deepcopy(next_grid)#the grid that (s,a) enters
                                flag = 0 #entering exploring
                                s_time = deepcopy(time)#exploring start time
                                exploring_grid = deepcopy(grid)
    #                             if(exploring_grid == '9|31' and s_time == 77708):
    #                                 print(row)
                                locgrid.append(actions.index(action))
                                grid_action = '|'.join(map(str,locgrid))#current state-action
                                exploring_traj.append(row)
                                exploring_history.append([exploring_grid,s_time])
                                exploring_grids.append(exploring_grid)
                                if grid_action not in rsa_exploring_trajs.keys():
                                    rsa_exploring_trajs[grid_action] = []

                    if(flag == 0):#exploring next grid after excuting action a in state s
                        exploring_traj.append(row)
                        if(next_grid != entering_grid or i==(len(traj)-2) and flag == 0):#exiting the exploring grid
                            flag = 1 #end of sight
                            exploring_traj.append(row)
                            rsa_exploring_trajs[grid_action].append(exploring_traj)
                            exploring_traj = []
                if len(exploring_history) == len(last_history):
    #                 print(len(exploring_history),len(last_history))
                    sign = False
    #             print(exploring_history)
                history.append(deepcopy(exploring_history))
#             print(k)
        rsa_exploring_trajs_plates[plate] = deepcopy(rsa_exploring_trajs)
        
        expected_rsa_plates = calculate_expected_ee(rsa_exploring_trajs_plates)
        
    return expected_rsa_plates


# In[33]:


def extract_dsa_plates_week(records_w1):
    '''
    Extract the visitation frequency of each s-a pair for each plate in a week
    '''
    valid_plates = list(records_w1.keys())
    full_dict = still_filter(record_taxi_less=records_w1)
    trajs = extract_trajs(full_dict=full_dict,record_taxi_less=records_w1,valid_plates=valid_plates)
    trajs_actions = assign_action(trajs)
    sa_vf_plates_week = extract_sa_visitation_frequency(trajs_actions)
    return sa_vf_plates_week


# In[51]:



def calculate_A_sa_a3c(rsa_pl,expected_v_s):
    A_sa = {}
    for sa in rsa_pl:
        r_sa = rsa_pl[sa]
        sa_list = sa.split('|')
        grid = sa[:-2]
        grid_list = [int(sa_list[0]),int(sa_list[1])]
        action = int(sa_list[-1])
        actions = []
        for lat_ind in [-1,0,1]:
            for lgt_ind in [-1,0,1]:
                actions.append([lat_ind,lgt_ind])
        next_grid_list = np.array(grid_list)+np.array(actions[action])
        next_grid_str = '|'.join(map(str,next_grid_list))
        v_s = expected_v_s[grid]
        try:
            v_s_next = expected_v_s[next_grid_str]
        except:
            v_s_next = 0
        A_sa[sa] = r_sa + v_s_next - v_s
    return A_sa

def weight_2_grids(grid0_str,grid1_str):
    '''
    we employ the inverse of the Manhattan distance between each two grids as the weight 
    if the Manhattan distance is smaller than 3, 
    the weights for those grids whose Manhattan distances are greater than 2 will be 0.
    '''
    grid0 = grid0_str.split('|')
    grid1 = grid1_str.split('|')
    manh_dist = abs(int(grid0[0])-int(grid1[0]))+abs(int(grid0[1])-int(grid1[1]))
    if 0<manh_dist <3:
        return 1/(manh_dist+1)
    else:
        return 0

def Geary_C_test(x_dict):
    '''
    Input: x_dict[grid_str] = x
    Output: Geary's test score
    '''
    N = len(x_dict.keys())
    x_mean = mean(list(x_dict.values()))
    w_sum = 0
    top = 0
    bot = 0
    for grid0_str in x_dict.keys():
        x0 = x_dict[grid0_str]
        bot+=(x0-x_mean)**2
        for grid1_str in x_dict.keys():
            x1 = x_dict[grid1_str]
            w_01 = weight_2_grids(grid0_str,grid1_str)
            w_sum += w_01
            top+=w_01*(x0-x1)**2
    C = (N-1)*top/(2*w_sum*bot)
    return C

def Moran_I_test(x_dict):
    '''
    Input: x_dict[grid_str] = x
    Output: Moran's I test score
    '''
    N = len(x_dict.keys())
    x_mean = mean(list(x_dict.values()))
    w_sum = 0
    top = 0
    bot = 0
    sum_x2 = 0
    sum_x4 = 0
    sum_w01_10 = 0#sum(wij + wji)^2
    s2 = 0
    for grid0_str in x_dict.keys():
        x0 = x_dict[grid0_str]
        bot+=(x0-x_mean)**2
        sum_x2 += x0**2
        sum_x4 += x0**4
        sum_w0_ = 0
        sum_w_0 = 0
        for grid1_str in x_dict.keys():
            x1 = x_dict[grid1_str]
            w_01 = weight_2_grids(grid0_str,grid1_str)
            w_10 = weight_2_grids(grid1_str,grid0_str)
            
            sum_w01_10 +=(w_01+w_10)**2
            sum_w0_ += w_01
            sum_w_0 += w_10
            
            w_sum += w_01
            top+=w_01*(x0-x_mean)*(x1-x_mean)
        s2 += (sum_w0_+sum_w_0)**2
        
    I = (N)*top/(w_sum*bot)
    
    m2 = (sum_x2)/N
    m4 = sum_x4/N
    b2 = m4/(m2**2)
    s0 = w_sum
    s1 = sum_w01_10/2
    s2 = s2
    E_I = (-1)/(N-1)
    
    var_I = (N*((N**2-3*N+3)*s1-N*s2+3*(s0**2))-b2*((N**2-N)*s1-2*N*s2+6*s0**2))    /((N-1)*(N-2)*(N-3)*s0**2)-E_I**2
    
    Z_R = (I-E_I)/sqrt(var_I)
    p_values = stats.norm.sf(abs(Z_R))*2
    return I,N,p_values


# In[35]:


def spatial_norm_values(x_s_plates):
    #x_s_plates[plate][grid] = x
    norm_plates = {}
    for plate in x_s_plates:
        norm_plates[plate] = {}
        for grid in x_s_plates[plate]:
            value_grid = x_s_plates[plate][grid]
            grid_list = grid.split('|')
            grid_x = int(grid_list[0])
            grid_y = int(grid_list[1])
            norm_value = value_grid #norm value = value_grid + weight*value_neighbor
            for x_diff in range(-2,3):
                for y_diff in range(-2,3):
                    if(abs(x_diff)+abs(y_diff) < 3):
                        neighbor_grid = [grid_x+x_diff,grid_y+y_diff]
                        n_grid_str = '|'.join(map(str,neighbor_grid))
                        w_neighbor = weight_2_grids(grid,n_grid_str)
                        try:
                            value_neighbor = x_s_plates[plate][n_grid_str]
                        except:
                            value_neighbor = 0
                            
                        norm_value+=w_neighbor*value_neighbor
            norm_plates[plate][grid] = norm_value
    return norm_plates

