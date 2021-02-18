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

def find_date_time(file):
    
    #  Reads a Gamry file, finds its creation datetime, and returns a datetime variable.
    #
    #  Input:
    #        file -> String: the address of the file
    #
    #  Output:
    #        starting_date_time -> datetime.datetime: the datetime that the Gamry file was created(when this 
    #                                                  electrochemical method starts).
    #    
    for row in file:
        if row.startswith('DATE'):
            year = int(row.split()[2].split('/')[2])
            month = int(row.split()[2].split('/')[0])
            day = int(row.split()[2].split('/')[1])
        if row.startswith('TIME'):
            hour = int(row.split()[2].split(':')[0])
            minute = int(row.split()[2].split(':')[1])
            second = int(row.split()[2].split(':')[2])
        
    starting_date_time = datetime.datetime(year,month,day,hour,minute,second)
    return starting_date_time

def analyze_gamry_file(file,starting_date_time):
    
    #  Reads a Gamry file and the starting time of the first file 
    #  (e.g. the time of the creation or the last time point of previous half cycle). 
    #  Returns a dataset with continuous Time, Voltage, Current, pH and fitted pH for a half-cycle.
    #
    #  Inputs:
    #        file -> String: file path and name
    #        starting_date_time -> datetime.datetime: the initial datetime of the process
    #
    #  Output:
    #        dataset -> pandas.DataFrame: a dataset contains the following attributes
    #         
    #        dataset['Delta_T_s'] -> float: time in seconds since the start of the process
    #        dataset['Time'] -> datetime.datetime: datetime of the current datum point
    #        dataset['Voltage'] -> float: voltage data
    #        dataset['Current'] -> float: current data
    #        dataset['pH'] -> float: pH data
    #        dataset['fitted_pH'] -> float: fitted pH data. Fluctuations were removed.

    
    indicator = False #for indicating row # is not correct yet
    t_array = []
    pH_array_left = []
    fitted_pH_array_left = []
    pH_array_right = []
    fitted_pH_array_right = []
    voltage_array = []
    current_array = []
    total_time_array = []
    for row in file:
        #finds the first row where actual data is recorded
        if row.startswith('	0'):
            indicator = True

        # row.split()[0]! is to avoid the last row of string
        if indicator and row.split()[0]!='STARTTIMEOFFSET':

            #voltage array
            voltage_array.append(float(row.split()[2]))

            #time array(for the purpose of calculating capacity)
            time_delta_s = float(row.split()[1])
            t_array.append(time_delta_s)
            total_time_array.append(starting_date_time + datetime.timedelta(seconds=time_delta_s))

            #pH arrays
            pH_array_left.append((float(row.split()[11])*-17.602)+7.1728)       
            pH_array_right.append((float(row.split()[17])*-17.4)+7.7278)       

            #current array
            current_array.append(float(row.split()[3]))
            
    #remove unnecessary glitches on the pH data through fitting
    pH_fit_left = np.polyfit(t_array,pH_array_left,6)
    total_pH_fit_left = np.poly1d(pH_fit_left)
    fitted_pH_array_left = total_pH_fit_left(t_array)

    pH_fit_right = np.polyfit(t_array,pH_array_right,6)
    total_pH_fit_right = np.poly1d(pH_fit_right)
    fitted_pH_array_right = total_pH_fit_right(t_array)
    
    dataset = pd.DataFrame({'Delta_T_s':t_array,'Time':total_time_array,'Voltage':voltage_array,
                        'Current':current_array,'pH_left':pH_array_left,'pH_right':pH_array_right
                        ,'fitted_pH_left':fitted_pH_array_left,'fitted_pH_right':fitted_pH_array_right})
    
    
    return dataset


def read_echem(path,cycle_number=5,co2=True):
    
    #  Reads a Gamry file folder, utilizes **analyze_gamry_file** to get half-cycle data and puts everything together
    #  in a dataset with multi-cycle continuous Time,Voltage, Current,pH and fitted pH data.
    #
    #  Inputs:
    #        file -> String: file path and name
    #        cycle_number -> int: number of full electrochemical cycles performed.
    #
    #  Output:
    #        dataset -> pandas.DataFrame: a dataset contains the following attributes
    #         
    #        dataset['Time'] -> datetime.datetime: datetime of the current datum point
    #        dataset['Voltage'] -> float: voltage data
    #        dataset['Current'] -> float: current data
    #        dataset['pH'] -> float: pH data
    #        dataset['fitted_pH'] -> float: fitted pH data. Fluctuations were removed.
    
    
    #Import deacidification/acidification data
    srcdir = path+'CHARGE_DISCHARGE/'
    echem_files = glob.glob(srcdir+'*.DTA')

    #Import voltage-hold period (invasion and outgas) data
    voltage_hold_srcdir = path +'OTHER/'
    voltage_hold_files =glob.glob(voltage_hold_srcdir+'*.DTA')
    
    dataset = pd.DataFrame(columns = ['Time','Voltage','Current','pH_left','pH_right'
                                      ,'fitted_pH_left','fitted_pH_right'])

    ###initialize the starting time using the first charging file###
    with open(srcdir+'PWRCHARGE_#1.DTA','r') as initial_file:
        starting_date_time = find_date_time(initial_file)
        
    ###import electrochemistry data###
    
    # The value of i is the same as cycle_number -1
    # The idea here is: 
    # 1.Locate a particular cycle number i+1
    # 2.Append i+1 cycle's Charge Data
    # 3.Append i+1 cycle's CO2 infusing data
    # 4.Append i+1 cycle's Discharge Data
    # 5.Append i+1 cycle's CO2 outgasing data
    for i in range(cycle_number):
        
        ##Deacidification (Charge)
        with open(srcdir+'PWRCHARGE_#'+str(i+1)+'.DTA','r') as file:
            if i == 0:
                df = analyze_gamry_file(file,starting_date_time)
            else:
                df = analyze_gamry_file(file,dataset.iloc[-1]['Time'])
        
        
        dataset = dataset.append(df,ignore_index = True)

        if co2:
            #invasion data
            with open(voltage_hold_srcdir+'Invasion_#'+str(i+1)+'.DTA','r') as file:
                df = analyze_gamry_file(file,dataset.iloc[-1]['Time'])

            dataset = dataset.append(df,ignore_index = True)
        
        ##Discharge data
        with open(srcdir+'PWRDISCHARGE_#'+str(i+1)+'.DTA','r') as file:
            df = analyze_gamry_file(file,dataset.iloc[-1]['Time'])
        
        dataset = dataset.append(df,ignore_index = True)
        
        
        ##Outgass data
        if co2:
            with open(voltage_hold_srcdir+'Outgas_#'+str(i+1)+'.DTA','r') as file:
                df = analyze_gamry_file(file,dataset.iloc[-1]['Time'])
                #df['pH'] = df['pH'] - 0.1 

            dataset = dataset.append(df,ignore_index = True)
            
        dataset['Time_Delta'] = (dataset['Time']-dataset.iloc[0]['Time']).apply(lambda x: x.days*24+x.seconds/3600)

    return dataset

def cal_capacity_energy(path,cycle_number = 5):
    #  Reads a Gamry file folder, utilizes **analyze_gamry_file** to get half-cycle data and 
    #  puts everything together in a dataset with multi-cycle's cycle capacity,energy and efficiencies.
    #
    #  Inputs:
    #        file -> String: file path and name
    #        cycle_number -> int: number of full electrochemical cycles performed.
    #
    #  Output:
    #        dataset -> pandas.DataFrame: a dataset contains the following attributes
    #         
    #        dataset['Cycle'] -> datetime.datetime: datetime of the current datum point
    #        dataset['Charge_Capacity'] -> float: charge capacity
    #        dataset['Charge_Energy'] -> float: charge energy
    #        dataset['Discharge_Capacity'] -> float: discharge capacity
    #        dataset['Discharge_Energy'] -> float: discharge energy
    #        dataset['Coulombic_Efficiency'] -> float: coulombic energy
    #        dataset['Energy_Efficiency'] -> float: round-trip energy efficiency
    #
    
    
    #Import deacidification/acidification data
    srcdir = path+'CHARGE_DISCHARGE/'
    echem_files = glob.glob(srcdir+'*.DTA')

    #Import voltage-hold period (invasion and outgas) data
    voltage_hold_srcdir = path +'OTHER/'
    voltage_hold_files =glob.glob(voltage_hold_srcdir+'*.DTA')
    
    cycle_array = []
    charge_cap_array = []
    charge_energy_array = []
    discharge_cap_array = []
    discharge_energy_array = []
    coulombic_efficiency_array = []
    energy_efficiency_array = []
    #dataset = pd.DataFrame([0,0,0,0,0,0],columns = ['Cycle','Charge Capacity','Charge Energy',
    #                                   'Discharge Capacity','Discharge Energy','Coulombic Efficiency'])
    
    ###initialize the starting time using the first charging file###
    with open(srcdir+'PWRCHARGE_#1.DTA','r') as initial_file:
        starting_date_time = find_date_time(initial_file)
        
    for i in range(cycle_number):
        with open(srcdir+'PWRCHARGE_#'+str(i+1)+'.DTA','r') as file:
            
            charge_df = analyze_gamry_file(file,starting_date_time)

        
        with open(srcdir+'PWRDISCHARGE_#'+str(i+1)+'.DTA','r') as file:
            
            discharge_df = analyze_gamry_file(file,starting_date_time)
        
        cycle_array.append(i+1)
        charge_cap_array.append(np.sum(charge_df['Current']))
        charge_energy_array.append(np.sum(charge_df['Current']*charge_df['Voltage']))
        discharge_cap_array.append(np.sum(discharge_df['Current']))
        discharge_energy_array.append(np.sum(discharge_df['Current']*discharge_df['Voltage']))
        coulombic_efficiency_array.append(abs(discharge_cap_array[-1]/charge_cap_array[-1]))
        energy_efficiency_array.append(abs(discharge_energy_array[-1]/charge_energy_array[-1]))
    dataset = pd.DataFrame({'Cycle_Number':cycle_array,'Charge_Capacity':charge_cap_array,
                            'Charge_Energy':charge_energy_array,'Discharge_Capacity':discharge_cap_array,
                            'Discharge_Energy':discharge_energy_array
                            ,'Coulombic_Efficiency':coulombic_efficiency_array
                            ,'Energy_Efficiency':energy_efficiency_array})
    
    return dataset
        

def find_echem_time_period(path,co2=True,outgas_time = 60):
    
    #
    #  Utilizes find_date_time method, reads a Gamry folder and returns a dataset containing the start and end time 
    #  of deacidification, capture, acidification and outgas.
    #
    #  Input:
    #        path -> String: the address of the folder that contains the electrochemistry files.
    #        co2 -> Boolean: whether CO2 capture/release took place
    #        outgas_time -> float: time in minutes for the outgas period
    #
    #  Output:
    #        dataset -> pandas.DataFrame: a dataset that contains the following attributes
    #
    # 
    #        dataset['Cycle'] -> int: cycle number
    #        dataset['Charge_Start_Time'] -> datetime.datetime: 
    #        dataset['Capture_Start_Time'] -> datetime.datetime
    #
    #
    #
    #
    
    
    #Import electrochemistry data 
    srcdir = path + 'CHARGE_DISCHARGE/'
    files = glob.glob(srcdir + '*.DTA')
    #Import voltage hold data    
    voltage_hold_srcdir = path + 'OTHER/'
    voltage_hold_files = glob.glob(voltage_hold_srcdir + '*.DTA')
    #initialize the column needed
    cycle_array = []
    charge_start_time_array = []
    capture_start_time_array = []
    discharge_start_time_array = []
    outgas_start_time_array = []
    outgas_end_time_array = []
    
    for i in range(1,6):
        cycle_array.append(i)
        with open(srcdir+'PWRCHARGE_#'+str(i)+'.DTA',newline='') as charge_file:
            charge_start_time_array.append(find_date_time(charge_file))
        if co2:
            with open(voltage_hold_srcdir+'Invasion_#'+str(i)+'.DTA',newline = '') as capture_file:
                capture_start_time_array.append(find_date_time(capture_file))
        with open(srcdir+'PWRDISCHARGE_#'+str(i)+'.DTA',newline='') as discharge_file:
            discharge_start_time_array.append(find_date_time(discharge_file))
            if not co2:
                for row in discharge_file:
                    if row.startswith('DATE'):
                        year = int(row.split()[2].split('/')[2])
                        month = int(row.split()[2].split('/')[0])
                        day = int(row.split()[2].split('/')[1])
                    if row.startswith('TIME'):
                        hour = int(row.split()[2].split(':')[0])
                        minute = int(row.split()[2].split(':')[1])
                        second = int(row.split()[2].split(':')[2])
        if co2:
            with open(voltage_hold_srcdir+'Outgas_#'+str(i)+'.DTA',newline = '') as outgas_file:
                for row in outgas_file:
                    if row.startswith('DATE'):
                        year = int(row.split()[2].split('/')[2])
                        month = int(row.split()[2].split('/')[0])
                        day = int(row.split()[2].split('/')[1])
                    if row.startswith('TIME'):
                        hour = int(row.split()[2].split(':')[0])
                        minute = int(row.split()[2].split(':')[1])
                        second = int(row.split()[2].split(':')[2])
            
            outgas_start_time = datetime.datetime(year,month,day,hour,minute,second)
            outgas_start_time_array.append(outgas_start_time)
            minute = minute+int(outgas_time%60)
            hour = hour+int(outgas_time//60)
            if minute >= 60:
                minute = minute - 60
                hour = hour + 1
                if hour >= 24:
                    hour = hour-24
                    day = day+1
            outgas_end_time = datetime.datetime(year,month,day,hour,minute,second)
            outgas_end_time_array.append(outgas_end_time)
    if co2:
        dataset = pd.DataFrame({'Cycle':cycle_array,'Charge_Start_Time':charge_start_time_array,
                            'Capture_Start_Time':capture_start_time_array,
                            'Discharge_Start_Time':discharge_start_time_array,
                            'Outgas_Start_Time':outgas_start_time_array,
                            'Outgas_End_Time':outgas_end_time_array})
    else:
        dataset = pd.DataFrame({'Cycle':cycle_array,'Charge_Start_Time':charge_start_time_array,
                            'Discharge_Start_Time':discharge_start_time_array})
    return dataset

def create_echem_dfs(path,co2=False,cycle_number=5,outgas_time=107):
    
    #
    #  Combine the results of **read_echem**, **cal_capacity_energy**, 
    #  and **find_echem_time_period** to produce a dictionary of the three dfs.
    #
    #  Input:
    #        path -> String: the address of the folder that contains the electrochemistry files.
    #        co2 -> Boolean: whether CO2 capture/release took place
    #        outgas_time -> float: time in minutes for the outgas period
    #        cycle_number -> int: number of cycles
    #
    #  Output:
    #        a dictionary of echem_df,energy_df and time_df
    #
    #
    
    #Process Echem Data
    electrochem_path = path

    #create echem df
    echem_df = read_echem(electrochem_path,co2=co2,cycle_number =cycle_number)

    #concatenate all df and create a total echem df
    echem_df['Hours']=echem_df.index/3600

    #create energy df
    energy_df = cal_capacity_energy(electrochem_path,cycle_number=cycle_number)

    #create time df
    #107/120 because the voltage hold period is 107 minute, determined by deducting echem time from 2 hour.
    time_df = find_echem_time_period(electrochem_path,outgas_time=outgas_time,co2=co2)
    #display(time_40_df)
    
    return {"echem_df":echem_df,"energy_df":energy_df,"time_df":time_df}