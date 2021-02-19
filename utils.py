
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import csv
import sys
import glob
from matplotlib.ticker import MultipleLocator
from scipy.fft import fft,ifft
import pickle
from scipy.signal import lfilter,savgol_filter
from scipy.optimize import fsolve,root_scalar,ridder,anderson,newton_krylov

def merge_echem_gas_df(echem_df,gas_df,co2_fit_path='../20210103_right_CO2_sensor_cubic_spline_fit',max_loop_num=11):
    '''
        Merge **echem_df**, created by `echem_methods.read_echem()` function and **gas_df**, created by `pd.read_csv()' on gas data, on newly created ['Time_Delta'] attribute.
        Add ['right_pco2'] attribute to the merged dataset. The ['right_pco2'] attribute is created by using a previously-prepared cubic spline fit that
        fits CO2 sensor analog signal to actual CO2 partial pressure. 

        :type echem_df: pd.DataFrame
        :param echem_df: A dataset created by `echem_methods.read_echem()` function that contains all electrochemistry data (voltage, current, pH, etc.)
        
        :type gas_df: pd.DataFrame
        :param gas_df: A dataset created by pd.read_csv()' on gas data that contains all gas data (pCO2, flow rate, mass flow controller input, etc.)

        :type co2_fit_path: string
        :param co2_fit_path: The path to the cublic spline fit made for fitting CO2 sensor analog input to actual CO2 partial pressure.

        :type max_loop_num: int
        :param max_loop_num: Maximum number of loops in Arduino when controlling gas input. Each loop accounts for one change of gas composition.
                            e.g. change from 0.1 bar pCO2 to 1 bar is one loop and changing from 1 bar to 0.1 bar is another loop. Usually one battery cycle
                            contains two gas-change loops.

        :rtype: *pd.DataFrame*
        :return: Merged dataset containing, original **echem_df** and **gas_df** datasets, plus extra ['right_pco2'] attribute.

    '''
    echem_df['Time_Delta']=(echem_df['Time']-gas_df['Datetime'].iloc[0]).apply(lambda x: x.days*24+x.seconds/3600)
    gas_df['Time_Delta'] = (gas_df['Datetime']-gas_df['Datetime'].iloc[0]).apply(lambda x: x.days*24+x.seconds/3600)

    total_df = gas_df.merge(echem_df,how='outer',on=['Time_Delta'])
    total_df = total_df[total_df['loop_num']<max_loop_num] #remove weird data

    with open(co2_fit_path,'rb') as f:
        right_co2_fit = pickle.load(f)
    
    n = 80  # the larger n is, the smoother curve will be
    b = [1.0 / n] * n
    a = 1
    filtered_right = lfilter(b,a,total_df['flow sensor right(sccm)'])
    total_df['right_pco2'] = right_co2_fit(total_df['CO2 sensor right(abs val)'])
    total_df['right_pco2'] = total_df['right_pco2'].apply(lambda x: np.where(x>0.90,1,x))
    
    filtered_co2_right = lfilter(b,a,total_df['right_pco2'])

    total_df['Corrected_Flow_Right']=total_df['flow sensor right(sccm)']*(total_df['right_pco2']*0.685+(1-total_df['right_pco2']))
    total_df['Corrected_Flow_Right_filtered'] = filtered_right*(filtered_co2_right*0.685+(1-filtered_co2_right))
    return total_df
