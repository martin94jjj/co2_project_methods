
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
    """    
            Reads a dataset that has a **time_attribute**. This dataset, usually created by `pd.read_csv()` or 
            `utils.merge_echem_gas_df()`, should contain CO2 concentration change info. 
            Returns a dataset that contains the time that each gas switch is made.

        :type gas_df: pd.DataFrame
        :param gas_df: the dataframe that contains gas data, i.e. pCO2, flowrate, MFC input, etc.

        :type gas_switch_peiord: int
        :param gas_switch_period: Time in seconds indicating the period of gas change.

        :type time_attribute: string
        :param time_attribute: **gas_df**'s attribute that contains datetime information, usually 'Datetime' or 'Time'.

        :rtype: *pd.DataFrame*
        :return:
              **time_df**: a dataframe that contains cycle number and the time for gas concentration change.

              time_df['Cycle'] -> (*int*): number of cycles\n
              time_df['low_to_high'] -> (*datetime.datetime*): The date and time that pCO2 is changed from low value to high value, e.g. from 0.1 bar to 1 bar\n
              time_df['high_to_low'] -> (*datetime.datetime*): The date and time that pCO2 is changed from high value to low value, e.g. from 1 bar to 0.1 bar\n

    """
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
            


def create_baseline(gas_df,start,end,parameter = 'CO2_Flow',baseline_range = 100,reverse_outgas_baseline_range=False):
    ''' 
    
    Reads a dataset created by `pd.read_csv()` on gas data or created by `utils.merge_echem_gas_df()`, 
    and a start time and an end time from a process(capture process = deacidification + capture ;
    release process = acidification + outgas) from `find_echem_time_period`. 
    This method linearly fits the **baseline_range** points before the process start time and **baseline_range** points from process end time. 
    Returns the linear fit parameters, the dataset index of the process start time and the dataset index of 
    the process end time.
     
    :type gas_df: pd.DataFrame
    :param gas_df: The dataframe that contains gas information, usually created by `pd.read_csv()` on gas data or `utils.merge_echem_gas_df()`

    :type start: datetime.datetime
    :param start: The start time of CO2 capture or release process

    :type end: datetime.datetime
    :param end: The end time of CO2 capture or release process

    :type parameter: string
    :param parameter: The dataset attribute used for baseline fitting, usually 'Corrected_Flow_Right'

    :type baseline_range: int
    :param baseline_range:  Number of points before capture start time and outgas end time used for baseline calculation. If **reverse_outgas_baseline_range** is True, use **baseline_range** points after outgas end time for baseline computation. Usually 500-3000, depending on the length of capture/release period

    :type reverse_outgas_baseline_range: boolean
    :param reverse_capture_baseline_range: Input True when you want to use **outgas_period** to calculate the outgased amount. 


    :rtype: *tuple*
    :return:
          **(fit, a1,a2)** : A tuple containing fit, created by `np.polyfit(x,y,1)`, and parameter a1 and a2 of y=a1x+a2, on indices 0,1,2 respectively.
    '''
    point1 = gas_df[gas_df['Datetime']==start]
    point2 = gas_df[gas_df['Datetime']==end]
    index_1 = point1.index.values[0]
    index_2 = point2.index.values[0]

    x1 = gas_df.iloc[index_1-baseline_range:index_1]['Time_Delta'].values
    if not reverse_outgas_baseline_range:
        x2 = gas_df.iloc[index_2-baseline_range:index_2]['Time_Delta'].values
    else:
        x2 = gas_df.iloc[index_2:index_2+baseline_range]['Time_Delta'].values
    y1 = gas_df.iloc[index_1-baseline_range:index_1][parameter].values
    y2 = gas_df.iloc[index_2-baseline_range:index_2][parameter].values
    #solve a1x+a2 = y
        
    x = np.hstack((x1,x2))
    y = np.hstack((y1,y2))
    
    fit = np.polyfit(x,y,1)
    
    return fit,index_1,index_2



def calculate_amount(gas_df,echem_time_df,gas_change_time_df,capture_parameter = 'Corrected_Flow_Right',outgas_parameter = 'Corrected_Flow_Right',
                        baseline = 'Adaptive',cycle = 5, shift_periods = 0,baseline_range = 100,capture_period=0,outgas_period=0,reverse_outgas_baseline_range=False):
    ''' 
    Reads a gas info dataset created by `pd.read_csv()` on gas data or `utils.merge_echem_gas_df()`, and a time period dataset by `find_echem_time_period`.
    Calculates the amount of CO2 captured and released. Returns a dataset containing the cycle number, 
    captured amount, outgas amount, average amount, fitting parameters for capture and outgas, 
    and indices associated with the start and end of the capture/outgas processes.
    
    
    :type gas_df: pd.DataFrame
    :param gas_df: dataset (created by `pd.read_csv()` on gas data or `utils.merge_echem_gas_df()`) that contains gas information

    :type echem_time_df: pd.DataFrame
    :param echem_time_df: dataset (created by `echem_method.find_echem_time_period`) that has process start and end
                    time information
    
    :type gas_change_time_df: pd.DataFrame
    :param gas_change_time_df: dataset (created by gas_methods.find_gas_change_time**) that has 
                    the time of gas composition change

    :type capture_parameter: string
    :param capture_parameter: The dataset attribute used for capture baseline fitting, usually "Corrected_Flow_Right"

    :type outgas_parameter: string
    :param outgas_parameter:  The dataset attribute used for outgas baseline fitting, usually "Corrected_Flow_Right"

    :type baseline: string or float
    :param baseline: Enter a fixed baseline value (*float*) for CO2 flow or use the default adaptive baseline

    :type cycle: int
    :param cycle: Enter the number of capture/release cycles.

    :type shift_periods: int
    :param shift_periods: (*Need to double check*)Number of indices to shift the capture/release parameter

    :type baseline_range: int
    :param baseline_range: Number of points before capture start time and outgas end time used for baseline calculation. If **reverse_outgas_baseline_range** is True, use **baseline_range** points after outgas end time for baseline computation

    :type capture_period: float
    :param capture_period: Use when default baseline that uses entire echem_time_df['Charge_Start_Time'] to gas_change_time_df['low_to_high'] fails. Fill in the value in seconds representing the capture period, 
    creates a baseline using the **baseline_range** values before echem_time_df['Charge_Start_Time'] and after the end of the capture period defined by echem_time_df['Charge_Start_Time']+**capture_period**.
    
    :type outgas_period: float
    :param outgas_period: Use when default baseline that uses entire echem_time_df['Discharge_Start_Time'] to gas_change_time_df['high_to_low'] fails. Fill in the value in seconds representing the outgas period, 
    creates a baseline using the **baseline_range** values before echem_time_df['Discharge_Start_Time'] and after the end of the outgas period defined by echem_time_df['Discharge_Start_Time']+**outgas_period**.
    
    :type reverse_outgas_baseline_range: boolean
    :param reverse_capture_baseline_range: Input True when you want to use **outgas_period** to calculate the outgased amount. 

    :rtype: *pd.DataFrame*
    :return: 
          **dataset**: a dataset that contains the following attributes
    
          dataset[Cycle_Number] -> (*int*):cycle_number\n
          dataset[Capture_Amount] -> (*float*): captured amount in this cycle\n
          dataset[Outgas_Amount] -> (*float*): released amount in this cycle\n
          dataset[Average_Amount] -> (*float*): average captured/released amount in this cycle\n
          dataset[c0] -> (*float*): slope of the capture baseline\n
          dataset[c1] -> (*float*): intercept of the capture baseline\n
          dataset[o0] -> (*float*): slope the outgas baseline\n
          dataset[o1] -> (*float*): intercept of the outgas baseline\n
          dataset[c_start] -> (*int*): index of the start of capture process\n
          dataset[c_end] -> (*int*): index of the end of capture process\n
          dataset[o_start] -> (*int*): index of the start of the release process\n
          dataset[o_end]  -> (*int*): index of the end of the release process\n
    
    '''

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

        
    for i in range(cycle):
        capture_start = echem_time_df.iloc[i]['Charge_Start_Time']
        outgas_start = echem_time_df.iloc[i]['Discharge_Start_Time']
        
        if not capture_period: 
            capture_end = gas_change_time_df.iloc[i]['low_to_high']
        else:
            capture_end = capture_start+datetime.timedelta(0,capture_period)
        if not outgas_period:
            outgas_end = gas_change_time_df.iloc[i]['high_to_low']
        else:
            outgas_end = outgas_start+datetime.timedelta(0,outgas_period)
        
        cycle_number.append(i+1)
        capture_amount = 0
        outgas_amount = 0
        
        #capture baseline parameter c,start_index and end_index
        c,c_start,c_end = create_baseline(gas_df,capture_start,capture_end,capture_parameter,baseline_range = baseline_range)
        
        #outgas baseline parameter o1,o2
        o,o_start,o_end = create_baseline(gas_df,outgas_start,outgas_end,outgas_parameter,baseline_range = baseline_range,reverse_outgas_baseline_range=reverse_outgas_baseline_range)
        
        c0_array.append(c[0])
        c1_array.append(c[1])
        c_start_array.append(c_start)
        c_end_array.append(c_end)
        
        o0_array.append(o[0])
        o1_array.append(o[1])
        o_start_array.append(o_start)
        o_end_array.append(o_end)
        
        capture_df = gas_df[(gas_df['Datetime']>=capture_start) & (gas_df['Datetime']<capture_end)]
        outgas_df = gas_df[(gas_df['Datetime']>=outgas_start) & (gas_df['Datetime']<outgas_end)]
        
        if baseline == 'Adaptive':
            #capture at intermediate CO2 partial pressure
            if capture_parameter == 'Corrected_Flow_Right':
                capture_amount = np.sum((capture_df.shift(periods=shift_periods)[capture_parameter]-(c[0]*capture_df['Time_Delta']+c[1])))/60.0
            else:
                capture_amount = np.sum((capture_df.shift(periods=shift_periods)[capture_parameter]-(c[0]*capture_df['Time_Delta']+c[1]))*11.7)/60.0
                
            outgas_amount = np.sum((outgas_df.shift(periods=shift_periods)[outgas_parameter]-(o[0]*outgas_df['Time_Delta']+o[1]))/60.0)
            
        else:
            capture_amount = sum((capture_df.shift(periods=shift_periods)[capture_parameter]-baseline)/60.0*capture_df['Corrected_Flow_Right'])
            outgas_amount = sum((outgas_df.shift(periods=shift_periods)[outgas_parameter]-baseline)/60.0)



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
