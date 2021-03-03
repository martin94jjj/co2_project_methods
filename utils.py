
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

def merge_echem_gas_df(echem_df,gas_df,co2_fit_path='../20210103_right_CO2_sensor_cubic_spline_fit',max_loop_num=11,co2_heat_conductivity=0.685,flow_offset=0):
    
    '''

    Merge **echem_df**, created by `echem_methods.read_echem()` function and **gas_df**, created by `pd.read_csv()` on gas data, on newly created ['Time_Delta'] attribute.
    Add ['right_pco2'] attribute to the merged dataset. The ['right_pco2'] attribute is created by using a previously-prepared cubic spline fit that
    fits CO2 sensor analog signal to actual CO2 partial pressure. 

    :type echem_df: pd.DataFrame
    :param echem_df: A dataset created by `echem_methods.read_echem()` function that contains all electrochemistry data (voltage, current, pH, etc.)
    
    :type gas_df: pd.DataFrame
    :param gas_df: A dataset created by pd.read_csv()' on gas data that contains all gas data (pCO2, flow rate, mass flow controller input, etc.)

    :type co2_fit_path: string
    :param co2_fit_path: The path to the cublic spline fit made for fitting CO2 sensor analog input to actual CO2 partial pressure.

    :type max_loop_num: int
    :param max_loop_num: Maximum number of loops in Arduino when controlling gas input. Each loop accounts for one change of gas composition. e.g. change from 0.1 bar pCO2 to 1 bar is one loop and changing from 1 bar to 0.1 bar is another loop. Usually one battery cycle contains two gas-change loops.

    :type co2_heat_conductivity: float
    :param co2_heat_conductivity: Ratio of CO2 heat conductivity over N2 heat conductivity (flow sensor dependent). This value should be obtained through calibrating CO2 sensor and flow sensor. This value may change over time as sensor is corroded by humidity.

    :type flow_offset: float
    :param flow_offset: Offset in sccm. This is the difference between measured flow rate and the flow rate set by mass flow controllers. The offset is likely caused by accumulated problems of flow meter over time.

    :rtype: *pd.DataFrame*
    :return: Merged dataset containing, original **echem_df** and **gas_df** datasets, plus extra ['right_pco2'] attribute.
            Below is the list of additional attributes:

            dataset['Time_Delta']-> (*int*): Time difference with respect to the start of the experiment. \n
            dataset['right_pco2']-> (*float*): CO2 partial pressure in bar. Converted from **CO2 sensor right** by previously determined spline-fit on analog input and CO2 partial pressure
            dataset['Corrected_Flow_Right'] -> (*float*): Actual flow rate corrected from nominal flow rate and CO2 conductivity and offset value.
            dataset['Corrected_Flow_Right_filtered'] -> (*float*): Actual flow rate corrected from nominal flow rate and CO2 conductivity and offset value. `scipy.lfilter` is used to filtered the signal.
            dataset['CO2Flow]-> (*float*): Product of dataset['Corrected_Flow_Right'] and dataset['right_pco2']. The flow rate of pure CO2.

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
    
    total_df['Corrected_Flow_Right']=total_df['flow sensor right(sccm)']*(total_df['right_pco2']*co2_heat_conductivity+(1-total_df['right_pco2']))-flow_offset
    total_df['Corrected_Flow_Right_filtered'] = filtered_right*(filtered_co2_right*co2_heat_conductivity+(1-filtered_co2_right))-flow_offset
    total_df['CO2Flow'] = total_df['Corrected_Flow_Right']*total_df['right_pco2']

    return total_df




def merge_amount_dic_df(dic_df,amount_df,volume=0.01,pco2=0.5):

    '''
    Merge **amount_df** and states 1 and 3 of **dic_df** together so comparing experimental cpatured amount 
    and theoretical amount is easy.

    :type dic_df: pd.DataFrame
    :param dic_df: A dataset created by `calc_dic.calc_DIC()` function that contains DIC info at different state and different capture or release amount between states.

    :type amount_df: pd.DataFrame
    :param amount_df: A dataset created by `gas_methods.calculate_amount()` function. This dataset contains capture and release amount of CO2 in mL.

    :type volume: float
    :param volume: Volume in litre. Usually is 0.01 or 10 mL

    :type pco2: float
    :param pco2: pCO2 during capture. Used for correcting Delta_DIC_outgas_exp to effective Delta_DIC using an absolute measure (difference in DIC is only from co2aq difference at 1 bar pCO2 and **pco2** bar pCO2). 

    :rtype: *pd.DataFrame*
    :return: Merged dataset containing Delta_DIC of different source (experimental, TA, eq and theory,eq) 
    
            dataset['Delta_DIC_capture_exp'] -> (*float*): Delta DIC measured during capture according to gas flow \n
            dataset['Delta_DIC_outgas_exp'] -> (*float*): Delta DIC measured during outgas according to gas flow \n
            dataset['Delta_DIC_TA_capture'] -> (*float*): Delta DIC measured during capture according to TA and measured pH\n
            dataset['Delta_DIC_TA_outgas'] -> (*float*): Delta DIC measured during outgas according to TA and measured pH\n
            dataset['Delta_DIC_eq_capture'] -> (*float*): Delta DIC measured during capture according to measured pH and assuming gas solution equilibrium\n
            dataset['Delta_DIC_eq_outgas'] -> (*float*): Delta DIC measured during outgas according to measured pH and assuming gas solution equilibrium\n
            dataset['Delta_DIC_theory_capture'] -> (*float*): Delta DIC measured during capture according to TA and theoretical pH given TA and assuming gas solution equilibrium\n
            dataset['Delta_DIC_theory_outgas'] -> (*float*): Delta DIC measured during outgas according to TA and theoretical pH given TA and assuming gas solution equilibrium\n
            dataset['Delta_DIC_TA_effective'] -> (*float*): Effective Delta DIC TA (DIC_TA difference in state 1 and 3)\n
            dataset['Delta_DIC_eq_effective'] -> (*float*): Effective Delta DIC eq (DIC_eq difference in state 1 and 3)\n
            dataset['Delta_DIC_theory_effective'] -> (*float*): Effective Delta DIC theory (DIC_theory difference in state 1 and 3)\n
            dataset['Delta_DIC_exp_TA_effective'] -> (*float*): Effective Delta DIC_exp corrected by dataset['Delta_DIC_outgas_exp']*dataset['Delta_DIC_TA_effective']/dataset['Delta_DIC_TA_outgas'] (Assuming two types of measurements give similar results)\n
            dataset['Delta_DIC_exp_abs_effective'] -> (*float*): Effective Delta DIC_exp corrected by dataset['Delta_DIC_outgas_exp']-(1-**pco2**)*0.035 (Assume the difference in outgas Delta_DIC_measured and effective Delta_DIC is the difference in co2aq)\n

    '''

    dic_capture_df = dic_df[dic_df['State']=='1'].filter(['Cycle','Delta_DIC_TA','Delta_DIC_eq','Delta_DIC_theory']).reset_index(drop=True)
    dic_capture_df = np.abs(dic_capture_df.rename(columns={"Delta_DIC_TA":"Delta_DIC_TA_capture",'Delta_DIC_eq':'Delta_DIC_eq_capture',
                                   'Delta_DIC_theory':'Delta_DIC_theory_capture'}))
    
    
    dic_outgas_df = dic_df[dic_df['State']=='3'].filter(['Cycle','Delta_DIC_TA','Delta_DIC_eq','Delta_DIC_theory']).reset_index(drop=True)
    dic_outgas_df = np.abs(dic_outgas_df.rename(columns={"Delta_DIC_TA":"Delta_DIC_TA_outgas",'Delta_DIC_eq':'Delta_DIC_eq_outgas',
                                   'Delta_DIC_theory':'Delta_DIC_theory_outgas'}))
    
    amount_df = amount_df.rename(columns={'Cycle_Number':'Cycle'})
    amount_df['Delta_DIC_capture_exp'] = amount_df['Capture_Amount']/1000/24.01/volume
    amount_df['Delta_DIC_outgas_exp'] = amount_df['Outgas_Amount']/1000/24.01/volume
    
    merged_df = amount_df.merge(dic_capture_df,on='Cycle').merge(dic_outgas_df,on='Cycle')
        
    #correct the effective amount with the difference in DIC between state 1 and 1'
    dic_1_1prime_diff_df = np.abs(dic_df[dic_df['State']=='1\''].filter(['Cycle','Delta_DIC_TA','Delta_DIC_eq','Delta_DIC_theory']).reset_index(drop=True))
    merged_df['Delta_DIC_TA_effective'] = dic_outgas_df['Delta_DIC_TA_outgas']-dic_1_1prime_diff_df['Delta_DIC_TA']
    merged_df['Delta_DIC_eq_effective'] = dic_outgas_df['Delta_DIC_eq_outgas']-dic_1_1prime_diff_df['Delta_DIC_eq']
    merged_df['Delta_DIC_theory_effective'] = dic_outgas_df['Delta_DIC_theory_outgas']-dic_1_1prime_diff_df['Delta_DIC_theory']
    merged_df['Delta_DIC_exp_TA_effective'] = merged_df['Delta_DIC_outgas_exp']*merged_df['Delta_DIC_TA_effective']/merged_df['Delta_DIC_TA_outgas']
    merged_df['Delta_DIC_exp_abs_effective'] = merged_df['Delta_DIC_outgas_exp']-(1-pco2)*0.035

    return merged_df




def merge_echemEnergy_amountDIC_df(echem_energy_df,amount_DIC_df,volume=0.01):
    
    '''
    Merge **echem_energy_df** and **amount_DIC_df**, includes deacidification/acidification and cycle work/molCO2 capture/outgassed effectively 
    (Delta_DIC in bewtween states 1 and 3 (now only on outgasd amount)), and includes CO2 captured or released per electron.


    :type echem_energy_df: pd.DataFrame
    :param echem_energy_df: A dataset created by `echem_methods.cal_capacity_energy()` function. The dataset contains capacity and energy information.

    :type amonut_DIC_df: pd.DataFrame
    :param amount_DIC_df: A dataset created by `utils.merge_amount_dic_df()` function. The dataset contains Delta DIC calculated using various sources.
    
    :rtype: *pd.DataFrame*
    :return: A merged dataset that includes deacidification/acidification and cycle work/molCO2 capture/outgassed effectively (Delta_DIC in bewtween state3 1 and 3), and includes CO2 captured or released per electron.

            dataset['deacidification_work(kJ/molCO2)'] -> (*float*): deacidification work (kJ/molCO2), over Outgas_amount, 
            dataset['acidification_work(kJ/molCO2)'] -> (*float*): acidification work (kJ/molCO2), over Outgas_amount
            dataset['cycle_work(kJ/molCO2)'] -> (*float*): cycle work (kJ/molCO2), over Outgas_amount
            dataset['co2_captured/electron'] -> (*float*): mol of CO2 captured over mol of charge capacity
            dataset['co2_outgassed/electron'] -> (*float*): mol of CO2 outgassed over mol of discharge capacity

            dataset['deacidification_work(kJ/molCO2)_exp_TA'] -> (*float*): deacidification work (kJ/molCO2), over Delta_DIC_exp_TA_effective
            dataset['acidification_work(kJ/molCO2)_exp_TA'] -> (*float*): acidification work (kJ/molCO2), over Delta_DIC_exp_TA_effective
            dataset['cycle_work(kJ/molCO2)_exp_TA'] -> (*float*): cycle work (kJ/molCO2), over Delta_DIC_exp_TA_effective

            dataset['deacidification_work(kJ/molCO2)_exp_abs'] -> (*float*): deacidification work (kJ/molCO2), over Delta_DIC_exp_abs_effective
            dataset['acidification_work(kJ/molCO2)_exp_abs'] -> (*float*): acidification work (kJ/molCO2), over Delta_DIC_exp_abs_effective
            dataset['cycle_work(kJ/molCO2)_exp_abs'] -> (*float*): cycle work (kJ/molCO2), over Delta_DIC_exp_abs_effective
    '''

    echem_energy_df = echem_energy_df.rename(columns={'Cycle_Number':'Cycle'})
    energy_co2_df = echem_energy_df.merge(amount_DIC_df,on='Cycle')
    energy_co2_df['deacidification_work(kJ/molCO2)'] = energy_co2_df['Charge_Energy']/1000/(energy_co2_df['Outgas_Amount']/24.01/1000) 
    energy_co2_df['acidification_work(kJ/molCO2)'] = energy_co2_df['Discharge_Energy']/1000/(energy_co2_df['Outgas_Amount']/24.01/1000)
    energy_co2_df['cycle_work(kJ/molCO2)'] = energy_co2_df['deacidification_work(kJ/molCO2)']-energy_co2_df['acidification_work(kJ/molCO2)']
    energy_co2_df['co2_captured/electron'] = (energy_co2_df['Capture_Amount']/24.01/1000)/(energy_co2_df['Charge_Capacity']/96485)
    energy_co2_df['co2_outgassed/electron'] = (energy_co2_df['Outgas_Amount']/24.01/1000)/(energy_co2_df['Discharge_Capacity']/96485)

    energy_co2_df['deacidification_work(kJ/molCO2)_exp_TA'] = energy_co2_df['Charge_Energy']/1000/(energy_co2_df['Delta_DIC_exp_TA_effective']*volume) 
    energy_co2_df['acidification_work(kJ/molCO2)_exp_TA'] = energy_co2_df['Discharge_Energy']/1000/(energy_co2_df['Delta_DIC_exp_TA_effective']*volume)
    energy_co2_df['cycle_work(kJ/molCO2)_exp_TA'] = energy_co2_df['deacidification_work(kJ/molCO2)_exp_TA']+energy_co2_df['acidification_work(kJ/molCO2)_exp_TA']

    energy_co2_df['deacidification_work(kJ/molCO2)_exp_abs'] = energy_co2_df['Charge_Energy']/1000/(energy_co2_df['Delta_DIC_exp_abs_effective']*volume) 
    energy_co2_df['acidification_work(kJ/molCO2)_exp_abs'] = energy_co2_df['Discharge_Energy']/1000/(energy_co2_df['Delta_DIC_exp_abs_effective']*volume)
    energy_co2_df['cycle_work(kJ/molCO2)_exp_abs'] = energy_co2_df['deacidification_work(kJ/molCO2)_exp_abs']+energy_co2_df['acidification_work(kJ/molCO2)_exp_abs']
    
    return energy_co2_df