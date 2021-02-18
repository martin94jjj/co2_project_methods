
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import csv
import sys
import glob
import pickle
from scipy.signal import lfilter,savgol_filter


def find_gas_change_time(gas_df,gas_switch_period = 7200,time_attribute='Datetime'):
    #
    #    Reads a dataset that has a time_attribute. This dataset should contain CO2 concentration change info
    #    Returns a dataset that contains the time that each gas switch is made.
    #Input:
    #      gas_df -> pandas DataFrame: the dataframe that contains gas information
    #      gas_switch_period -> int: period of gas change
    #      time_attribute -> gas_df's attribute that contains datetime information

    #Output:
    #      time_df: a dataframe that contains cycle number and the time for gas concentration change.
    i = 1
    cycle_number = 1
    low_co2 = True
    cycle_array = []
    low_to_high_array = []
    high_to_low_array = []
    while(i*gas_switch_period<len(gas_df)):
        if(low_co2):
            cycle_array.append(cycle_number)
            low_co2 = False
            low_to_high_array.append(gas_df.iloc[i*gas_switch_period][time_attribute])
            i+= 1
        else:
            cycle_number += 1
            low_co2 = True
            high_to_low_array.append(gas_df.iloc[i*gas_switch_period][time_attribute])
            i+= 1
    if len(high_to_low_array)<len(low_to_high_array):
        high_to_low_array.append('')
    time_df = pd.DataFrame({"Cycle":cycle_array,"low_to_high":
                            low_to_high_array,"high_to_low":high_to_low_array})
    
    return time_df
            


def create_baseline(gas_df,start,end,parameter = 'CO2_Flow',baseline_range = 100):
    #
    #
    #Reads a dataset created by **read_flow_co2**, 
    #and a start time and an end time from a process(capture process = deacidification + capture ;
    #release process = acidification + outgas) from **find_echem_time_period**. 
    #This method linearly fits the 20 points before the process start time and 20 points from process end time. 
    #Returns the linear fit parameters, the dataset index of the process start time and the dataset index of 
    #the process end time.
    # 
    #Input:
    #      gas_df -> pandas DataFrame: the dataframe that contains gas information
    #      start -> datetime: start time of capture or release
    #      end -> datetime: end time of capture or release
    #      parameter -> String: the dataset attribute used for baseline fitting
    #      baseline_range -> int: number of points used for baseline
    #Output:
    #      parameters: y=a1x+a2, returns (a1,a2)
    #
    point1 = gas_df[gas_df['Datetime']==start]
    point2 = gas_df[gas_df['Datetime']==end]
    index_1 = point1.index.values[0]
    index_2 = point2.index.values[0]

    x1 = gas_df.iloc[index_1-baseline_range:index_1]['Time_Delta'].values
    x2 = gas_df.iloc[index_2-baseline_range:index_2]['Time_Delta'].values
    y1 = gas_df.iloc[index_1-baseline_range:index_1][parameter].values
    y2 = gas_df.iloc[index_2-baseline_range:index_2][parameter].values
    #solve a1x+a2 = y
        
    x = np.hstack((x1,x2))
    y = np.hstack((y1,y2))
    
    fit = np.polyfit(x,y,1)
    
    return fit,index_1,index_2



def calculate_amount(gas_df,echem_time_df,gas_change_time_df,capture_parameter = 'Corrected_Flow_Right',outgas_parameter = 'Corrected_Flow_Right',baseline = 'Adaptive',cycle = 5, shift_periods = 0,baseline_range = 100):
    #
    #Reads a gas info dataset created by **read_flow_co2**, and a time period dataset by **find_echem_time_period**.
    #Calculates the amount of CO2 captured and released. Returns a dataset containing the cycle number, 
    #captured amount, outgas amount, average amount, fitting parameters for capture and outgas, 
    #and indices associated with the start and end of the capture/outgas processes.
    #
    #
    #Input:
    #      gas_df -> pandas DataFrame: dataset (created by **read_co2_flow**) that contains gas information
    #      echem_time_df -> pandas DataFrame: dataset (created by **find_echem_time_period**) that has process start and end
    #                time information
    #      gas_change_time_df -> pandas DataFrame: dataset (created by **find_gas_change_time**) that has 
    #                the time of gas composition change
    #      capture_parameter -> String: the dataset attribute used for capture baseline fitting
    #      outgas_parameter -> String: the dataset attribute used for outgas baseline fitting
    #      baseline -> String or float: enter a baseline value for CO2 flow or use the default adaptive baseline
    #
    #Output:
    #      dataset -> pandas.DataFrame: a dataset that contains the following attributes
    #
    #      dataset[Cycle_Number] -> int:cycle_number,
    #      dataset[Capture_Amount] -> float : captured amount in this cycle,
    #      dataset[Outgas_Amount] -> float : released amount in this cycle,
    #      dataset[Average_Amount] -> float: average captured/released amount in this cycle,
    #      dataset[c0] -> float: slope of the capture baseline,
    #      dataset[c1] -> float: intercept of the capture baseline,
    #      dataset[o0] -> float: slope the outgas baseline,
    #      dataset[o1] -> float: intercept of the outgas baseline
    #      dataset[c_start] -> int: index of the start of capture process
    #      dataset[c_end] -> int: index of the end of capture process
    #      dataset[o_start] -> int: index of the start of the release process
    #      dataset[o_end]  -> int: index of the end of the release process
    #
    #
    capture_amount_array = []
    outgas_amount_array = []
    cycle_number = []
    c0_array = []
    c1_array = []
    c_start_array = []
    c_end_array = []
    o0_array = []
    o1_array = []
    o_start_array = []
    o_end_array = []
    index = 0;

        
    for i in range(cycle):
        charge_start = echem_time_df.iloc[i]['Charge_Start_Time']
        capture_start = echem_time_df.iloc[i]['Capture_Start_Time']
        discharge_start = echem_time_df.iloc[i]['Discharge_Start_Time']
        outgas_start = echem_time_df.iloc[i]['Outgas_Start_Time']
        outgas_end = echem_time_df.iloc[i]['Outgas_End_Time']
        
        low_to_high_switch = gas_change_time_df.iloc[i]['low_to_high']
        high_to_low_switch = gas_change_time_df.iloc[i]['high_to_low']
        
        cycle_number.append(i+1)
        capture_amount = 0
        outgas_amount = 0
        
        #capture baseline parameter c,start_index and end_index
        c,c_start,c_end = create_baseline(gas_df,charge_start,low_to_high_switch,capture_parameter,baseline_range = baseline_range)
        
        #outgas baseline parameter o1,o2
        o,o_start,o_end = create_baseline(gas_df,discharge_start,high_to_low_switch,outgas_parameter,baseline_range = baseline_range)
        
        c0_array.append(c[0])
        c1_array.append(c[1])
        c_start_array.append(c_start)
        c_end_array.append(c_end)
        
        o0_array.append(o[0])
        o1_array.append(o[1])
        o_start_array.append(o_start)
        o_end_array.append(o_end)
        
        capture_df = gas_df[(gas_df['Datetime']>=charge_start) & (gas_df['Datetime']<low_to_high_switch)]
        outgas_df = gas_df[(gas_df['Datetime']>=discharge_start) & (gas_df['Datetime']<high_to_low_switch)]
        
        if baseline == 'Adaptive':
            #capture at intermediate CO2 partial pressure
            if capture_parameter == 'Corrected_Flow_Right':
                capture_amount = np.sum((capture_df.shift(periods=shift_periods)[capture_parameter]-(c[0]*capture_df['Time_Delta']+c[1])))/60.0
            else:
                capture_amount = np.sum((capture_df.shift(periods=shift_periods)[capture_parameter]-(c[0]*capture_df['Time_Delta']+c[1]))*11.7)/60.0
                
            outgas_amount = np.sum((outgas_df.shift(periods=shift_periods)[outgas_parameter]-(o[0]*outgas_df['Time_Delta']+o[1]))/60.0)
            
        else:
            capture_amount = sum((capture_df.shift(periods=shift_periods)[parameter]-baseline)/60.0*capture_df['Corrected_Flow_Right'])
            outgas_amount = sum((outgas_df.shift(periods=shift_periods)[parameter]-baseline)/60.0)



        capture_amount_array.append(capture_amount)
        outgas_amount_array.append(outgas_amount)
        average_amount_array = (np.abs(np.array(capture_amount_array))+np.array(outgas_amount_array))/2
    dataset = pd.DataFrame({'Cycle_Number':cycle_number,
                            'Capture_Amount':capture_amount_array,
                            'Outgas_Amount':outgas_amount_array,
                            'Average_Amount':average_amount_array,
                            'c0':c0_array,
                            'c1':c1_array,
                            'o0':o0_array,
                            'o1':o1_array,
                            'c_start':c_start_array,
                            'c_end':c_end_array,
                            'o_start':o_start_array,
                            'o_end':o_end_array})
    return dataset
