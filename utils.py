
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

def merge_echem_gas_df(echem_df,gas_df,max_loop_num=11):

    echem_df['Time_Delta']=(echem_df['Time']-gas_df['Datetime'].iloc[0]).apply(lambda x: x.days*24+x.seconds/3600)
    gas_df['Time_Delta'] = (gas_df['Datetime']-gas_df['Datetime'].iloc[0]).apply(lambda x: x.days*24+x.seconds/3600)

    total_df = gas_df.merge(echem_df,how='outer',on=['Time_Delta'])
    total_df = total_df[total_df['loop_num']<max_loop_num] #remove weird data

    with open('../20210103_right_CO2_sensor_cubic_spline_fit','rb') as f:
        right_co2_fit = pickle.load(f)
    
    n = 80  # the larger n is, the smoother curve will be
    b = [1.0 / n] * n
    a = 1
    filtered_right = lfilter(b,a,total_df['flow sensor right(sccm)'])
    total_df['right_pco2'] = right_co2_fit(total_df['CO2 sensor right(abs val)'])
    total_df['right_pco2'] = total_df['right_pco2'].apply(lambda x: np.where(x>0.9,1,x))
    
    filtered_co2_right = lfilter(b,a,total_df['right_pco2'])

    total_df['Corrected_Flow_Right']=total_df['flow sensor right(sccm)']*(total_df['right_pco2']*0.685+(1-total_df['right_pco2']))-0.5
    total_df['Corrected_Flow_Right_filtered'] = filtered_right*(filtered_co2_right*0.685+(1-filtered_co2_right))-0.5
    return total_df
