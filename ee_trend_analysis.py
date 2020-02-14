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
from  copy import deepcopy
import operator
from matplotlib import pyplot as plt
import scipy.stats


# In[3]:


weekends = {'08':[6,7,13,14,20,21,27,28,31],#
           '09': [3,4,10,11,17,18,24,25,31],
           '10': [1,2,8,9,15,16,22,23,29,30],
           '11': [5,6,12,13,19,20,26,27,31]}


# In[4]:


#2016.07
def calculate_week_fee_ratio(months = ['07','08','09','10','11','12'],plates = overlap_plates_6m):
    filepath = '../earning_efficiency/'
    weekends = {'07':[2,3,9,10,16,17,23,24,30,31],
                '08':[6,7,13,14,20,21,27,28,    31],
               '09': [3,4,10,11,17,18,24,25,31],
               '10': [1,2,8,9,15,16,22,23,29,30],
               '11': [5,6,12,13,19,20,26,27,31],
               '12': [3,4,10,11,17,18,24,25,31]}
    saturdays = {'07':[2,9,16,23,30],
                '08':[6,13,20,27],# as a flag to conclude week ratios
               '09': [3,10,17,24],
               '10': [1,8,15,22,29],
               '11': [5,12,19,26],
                '12':[3,10,17,24,31]}
    week_fee_ratio = {}
    week_fee_ratio_conclude={}#week_fee_ratio_conclude[month+day] = [fees for all plates]
    list_week_fee_ratio = []
    dict_week_fee_ratio = {}#dict_week_fee_ratio[plate] = week earning ratio,#working days,week saturday
    for month in months:
        for day in range(1,32):
            if day in saturdays[month]:#conclude week ratio
                week_index = month+str(day)
                week_fee_ratio_conclude[week_index] = []
                for plate in week_fee_ratio.keys():
                    if plate[-6:] not in dict_week_fee_ratio.keys():
                        dict_week_fee_ratio[plate[-6:]] = []
                    if(week_fee_ratio[plate][-1]>=1):
                        if not np.isnan(week_fee_ratio[plate][0]*60*60/week_fee_ratio[plate][1]):
                            week_fee_ratio_conclude[week_index].append(week_fee_ratio[plate][0]*60*60/week_fee_ratio[plate][1])
                            list_week_fee_ratio.append([plate,week_fee_ratio[plate][0]*60*60/week_fee_ratio[plate][1],week_fee_ratio[plate][2],month+str(day)])#plate,week earning ratio,#working days,week saturday
                            dict_week_fee_ratio[plate[-6:]].append([week_fee_ratio[plate][0]*60*60/week_fee_ratio[plate][1],week_fee_ratio[plate][2],month+str(day)])
                week_fee_ratio = {}
                
            if day in weekends[month]:#skip weekends
                continue
                
            print(day)
            day_str = '0'+str(day) if day<10 else str(day)
            try:
                day_fee_ratio = pickle.load(open(filepath+'./fee16'+month+day_str+'.pkl','rb'))
            except:
                print(month,day,'fee day not found')
            #fee: plate,earning per hour, service time ratio, earnings in a day, service time in a day
            for fee_plate in day_fee_ratio:
                plate = fee_plate[0]
                day_earning = fee_plate[-2]
                day_work_time = fee_plate[-1]/fee_plate[2]
                if plate not in week_fee_ratio.keys():
                    week_fee_ratio[plate] = [day_earning,day_work_time,1]# week earning, week working time, #week working days
                else:
                    week_fee_ratio[plate] = [week_fee_ratio[plate][0]+day_earning,week_fee_ratio[plate][1]+day_work_time,week_fee_ratio[plate][2]+1]
    return list_week_fee_ratio,dict_week_fee_ratio,week_fee_ratio_conclude


# In[6]:


def sgn(j,k):
    if(j>k):
        return 1
    elif(j==k):
        return 0
    else:
        return -1

def std_dev(nlist):
    narray=numpy.array(nlist)
    N = len(nlist)
    sum1=narray.sum()
    narray2=narray*narray
    sum2=narray2.sum()
    mean=sum1/N
    var=sum2/N-mean**2
    return var**0.5


# In[7]:


def mk_test(t_series,alpha):
    '''
    Conduct Mann-Kendall test on time series data for monotonic trend.
    input: (list)time_series, confidence level alpha
    output: pass or not indicator (1 pass, 0 fail)
            z value
            z threshold
    '''
    n = len(t_series)
    s = 0 #S
    for k in range(len(t_series)-1):
        for j in range(k+1,len(t_series)):
            s+=sgn(t_series[j],t_series[k])
    
    #calculate variance of S
    sorted_test = deepcopy(t_series)
    sorted_test = sorted(sorted_test)
    group = itertools.groupby(sorted_test)
    minus_term = 0
    for a,b in group:
        tp = len(list(b))
        minus_term+=(tp*(tp-1)*(2*tp+5))
    var = (n*(n-1)*(2*n+5)-minus_term)/18
    
    #calculate Z
    if s>0:
        z = (s-1)/(var**0.5)
    elif s<0:
        z = (s+1)/(var**0.5)
    else:
        z = 0
    
    z_thresh = scipy.stats.norm.ppf(1-alpha)
    if abs(z) > z_thresh:
        pass_indicator = 1 
    else:
        pass_indicator = 0
    return pass_indicator,z,z_thresh


# In[8]:


def correlation_coefficient_lr(t_series):
    '''
    Calculate the linear correlation coefficient of the time series.
    Output: the value of r. The trend is “strong” if the absolute value of r (which ranges from -1 to 1) is near one
    '''
    n = len(t_series)
    x = list(range(len(t_series)))
    x_m = mean(x)
    y_m = mean(t_series)
    y = t_series
    up_term = 0
    down_term_x = 0
    down_term_y = 0
    for i in range(n):
        up_term += (x[i]-x_m)*(y[i]-y_m)
        down_term_x += (x[i]-x_m)**2
        down_term_y += (y[i]-y_m)**2
    return up_term/((down_term_x*down_term_y)**0.5)


# In[9]:


def pettitt_test(t_series,alpha):
    '''
    Conduct Pettitt's test on time series data for change detection.
    Output: pass indicator(1 means there exists a change)
            Change point index
            increase indicator(1 means the change is increase)
    '''
    n = len(t_series)
    #calculate all U_t
    uts = []
    uts_abs = []
    for t in range(1,n):
        ut = 0
        for i in range(t):
            for j in range(t+1,n):
                ut+=sgn(t_series[i],t_series[j])
        uts_abs.append(abs(ut))
        uts.append(ut)
    k = max(uts_abs)
    p = 2*exp((-6*k**2)/(n**3+n**2))
    if p<alpha:
        pass_indicator = 1
    else:
        pass_indicator = 0
    change_point = uts_abs.index(k)
    increase_ind = 0
    if(uts[change_point]<0):
        increase_ind = 1
    else:
        increase_ind = 0
        
    #the change point in t_series should be t+1, and t = uts_abs.index(k)+1
    return pass_indicator,change_point+1,increase_ind 
    


# In[77]:


def Buishand_U_change_point_detection(inputdata):
    inputdata = np.array(inputdata)
    inputdata_mean = np.mean(inputdata)
    n  = inputdata.shape[0]
    k = range(n)
    Sk = [np.sum(inputdata[0:x+1] - inputdata_mean) for x in k]
    sigma = np.sqrt(np.sum((inputdata-np.mean(inputdata))**2)/(n-1))
    U = np.sum((Sk[0:(n - 2)]/sigma)**2)/(n * (n + 1))
    Ska = np.abs(Sk)
    S = np.max(Ska)
    K = list(Ska).index(S) + 1
    Skk = (Sk/sigma)
    return K

