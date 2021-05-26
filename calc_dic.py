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

#define equilibrium constants
k1=1.1*10**-6 
k2=4.1*10**-10
kw=10**-14
henry_constant = 0.035
def dic(co2aq,pH,solve_value = 0):
    """
    Calculate DIC given aqueous co2 concentration (**co2aq**) and pH. **solve_value** is used when one tries to solve for **co2aq**
    or **pH** given DIC value (enter targeting DIC value for **solve_value**).

    :type co2aq: float
    :param co2aq: Aqueous co2 concentration

    :type pH: float
    :param pH: pH value

    :type solve_value: float
    :param solve_value: Target DIC value when using solvers from scipy.optimize

    :rtype: *float*
    :return: DIC value
    """
    return co2aq*(1 + k1/10**-pH + k1*k2/(10**-pH)**2)-solve_value

def hco3(co2aq,pH,solve_value = 0):
    """
    Calculate bicarbonate concentration given aqueous co2 concentration (**co2aq**) and pH. **solve_value** is used when one tries to solve for **co2aq**
    or **pH** given bicarbonate concentration value (enter targeting bicarbonate concentration for **solve_value**).

    :type co2aq: float
    :param co2aq: Aqueous co2 concentration

    :type pH: float
    :param pH: pH value

    :type solve_value: float
    :param solve_value: Target bicarbonate concentration value when using solvers from scipy.optimize

    :rtype: *float*
    :return: bicarbonate concentration
    """
    return dic(co2aq,pH)/(1+10**-pH/k1+k2/10**-pH)-solve_value

def co32(co2aq,pH,solve_value = 0):
    """
    Calculate carbonate concentration given aqueous co2 concentration (**co2aq**) and pH. **solve_value** is used when one tries to solve for **co2aq**
    or **pH** given carbonate concentration value (enter targeting carbonate concentration for **solve_value**).

    :type co2aq: float
    :param co2aq: Aqueous co2 concentration

    :type pH: float
    :param pH: pH value

    :type solve_value: float
    :param solve_value: Target carbonate concentration value when using solvers from scipy.optimize

    :rtype: *float*
    :return: carbonate concentration
    """
    return dic(co2aq,pH)/(1+10**-pH/k2+(10**-pH)**2/(k1*k2))-solve_value

def TA(co2aq,pH,solve_value = 0):
    """
    Calculate total alkalinity(TA) given aqueous co2 concentration (**co2aq**) and pH. **solve_value** is used when one tries to solve for **co2aq**
    or **pH** given TA value (enter targeting TA value for **solve_value**).

    :type co2aq: float
    :param co2aq: Aqueous co2 concentration

    :type pH: float
    :param pH: pH value

    :type solve_value: float
    :param solve_value: Target TA value when using solvers from scipy.optimize

    :rtype: *float*
    :return: TA value
    """
    return kw/(10**-pH)+hco3(co2aq,pH)+2*co32(co2aq,pH)+10**-pH-solve_value

def TA_pH_wrapper(co2aq,solve_value = 0):
    """A function wrapper used when using newton_krylov solver solving for pH given **co2aq** and **TA**, which doesn't take additional arguments
    
    :type co2aq: float
    :param co2aq: Aqueous co2 concentration

    :type solve_value: float
    :param solve_value: Target TA value when using solvers from scipy.optimize

    :rtype: *func*
    :return: a function used for newton_krylov solver to solve for pH, given **co2aq** and **solve_value** (TA concentration) value
    
    .. note::   Here is an example

                .. code-block:: python

                    TA_val=0.2
                    co2aq=0.0035
                    pH_measured=8
                    pH_func = TA_pH_wrapper(co2aq,solve_value = TA_val)
                    pH_theory = newton_krylov(pH_func,8)
                    pH_theory

                    >> array(7.69814971)

    
    """

    def func(pH):
        return kw/(10**-pH)+hco3(co2aq,pH)+2*co32(co2aq,pH)+10**-pH-solve_value
    return func

def calc_DIC(total_df,echem_time_df,gas_change_time_df,outgas_shift=20,volume=0.01,flag=0,solver="newton_krylov"):
    
    """
    Calculates DIC \ :sub:`TA`\, DIC \ :sub:`eq`\, pH  \ :sub:`theory,eq`\ and DIC \ :sub:`theory,eq`\, given the echem_gas_dataframe(**total_df**)
    , **echem_time_df**, which tells the start and end of each echem process, **gas_change_time_df**, which tells when atmosphere CO2 is changed, and the
    **volume** of the electrolyte in L.

    Values in State 3'i are calculaed based on initially measured pH (pH measured at state 3'i), pCO2, and assuming gas-solution equilibrium (co2aq=pCO2*0.035(Henry's constant))
    Other states' values are calculated given TA, and using other functions in this module.

    **outgas_shift** is necessary when the timing data in **gas_change_df** and **echem_time_df** are off by some time (usually within 30 seconds).

    :type total_df: pd.DataFrame
    :param total_df: A pandas dataframe, created by `utils.merge_echem_gas_df()` function, that contains echem and gas information

    :type echem_time_df: pd.DataFrame
    :param echem_time_df:  A pandas dataframe, created by `echem_method.find_time_period()` function, that contains the timing of the start and end of each echem process

    :type gas_change_time_df: pd.DataFrame
    :param gas_change_time_df: A pandas dataframe, created by 'gas_methods.find_gas_change_time()' function, that contains the timing of when atmosphere CO2 is changed

    :type outgas_shift: int
    :param outgas_shift: Time in seconds. Offsets inaccurate timing in **echem_time_df** or **gas_change_time_df**

    :type volume: float
    :param volume: Volume in litre. The volume of the electrolyte

    :type flag: int
    :param flag: flag for debug. 0 for not showing any message. 

    :type solver: string
    :param solver: name of scipy solver used for solving theoretical pH, given TA and co2aq (assuming equilibrium). The default solver is "newton_krylov". If encounter any issue, try "fsolve"

    :rtype: *pd.DataFrame*
    :return: A dataset that contains DIC \ :sub:`TA`\, DIC \ :sub:`eq`\, pH  \ :sub:`theory,eq`\ and DIC \ :sub:`theory,eq`\ for state 3'i, 1, 1', 3 and 3'f for each cycle.

        dataset['Cycle'] -> (*int*): The cycle number\n
        dataset['State'] -> (*string*): The state name, ranging from 3'i, 1, 1', 3 and 3'f\n
        dataset['pH_measured'] -> (*float*): The measured pH value\n
        dataset['pH_theory'] -> (*float*): The theoretical pH value given pCO2 and TA\n
        dataset['TA'] -> (*float*): The total alkalinity concentration in Molar\n
        dataset['DIC_TA'] -> (*float*): DIC \ :sub:`TA`\ value in Molar. DIC value calculated based on TA and measured pH, assuming no crossover of non-conservative ions.\n
        dataset['DIC_eq'] -> (*float*): DIC \ :sub:`eq`\ value in Molar. DIC value calculated based on measured pH and assuming gas-solution equilibrium, i.e. co2aq = pCO2*0.035 (Henry's constant)\n
        dataset['DIC_theory'] -> (*float*): DIC \ :sub:`theory,eq`\ value in Molar. DIC value calculated based on TA and theoretical pH.\n
        dataset['index'] -> (*int*): The index in **total_df** where each of the above value is calculated.\n
        dataset['Delta_DIC_TA'] -> (*float*) : The amount of DIC \ :sub:`TA`\ change in terms of Molar.\n
        dataset['Delta_DIC_eq'] -> (*float*) : The amount of DIC \ :sub:`eq`\ change in terms of Molar.\n
        dataset['Delta_DIC_theory'] -> (*float*) : The amount of DIC \ :sub:`theory,eq`\ change in terms of Molar.\n
   
    """

    cycle_num = len(echem_time_df)
    
    index_array = []
    cycle_array = []
    states_array = []
    pH_measured_array = []
    pH_theory_array = []
    TA_array = []
    DIC_TA_array = []
    DIC_eq_array = []
    DIC_theory_array = []
    Delta_DIC_TA_array = []
    Delta_DIC_eq_array = []
    Delta_DIC_theory_array = []
    
    
    cycle_num = len(echem_time_df)

    for i in range(cycle_num):
        for j in range(5):
            if j == 0:
                initial_index = (total_df[total_df['Datetime']==echem_time_df.iloc[i]['Charge_Start_Time']].index+1).values[0]
                initial_entry = total_df.iloc[initial_index]
                if i==0:
                    
                    initial_pH = initial_entry['pH_right']
                else:
                    #initial index and previous 20 indices for calculating a more reliable initial pH
                    initial_pH = np.average(total_df.iloc[initial_index-20:initial_index]['pH_right'])

                initial_pCO2 = initial_entry['CO2 input right(abs val)']/(initial_entry['CO2 input right(abs val)']+initial_entry['N2 input right(abs val)'])
                initial_co2aq = initial_pCO2*henry_constant
                initial_TA = TA(initial_co2aq,initial_pH)
                initial_DIC = dic(initial_co2aq,initial_pH)
                
                index_array.append(initial_index)
                cycle_array.append(i+1)
                states_array.append('3\'i')
                pH_measured_array.append(initial_pH)
                pH_theory_array.append(initial_pH)
                TA_array.append(initial_TA)
                DIC_TA_array.append(initial_DIC)
                DIC_eq_array.append(initial_DIC)
                DIC_theory_array.append(initial_DIC)
                Delta_DIC_TA_array.append(0)
                Delta_DIC_eq_array.append(0)
                Delta_DIC_theory_array.append(0)
                if(flag):
                    print("Cycle number:",i+1," state:",'3\'i', "co2aq: %0.3f"%initial_co2aq, "TA_val: %0.2f"%initial_TA,"pH measured: %0.2f"%initial_pH)
            else:
                if j == 1:
                    state = '1'
                    index = total_df[total_df['Datetime']==gas_change_time_df.iloc[i]['low_to_high']].index.values[0]
                elif j == 2:
                    state = '1\''
                    index = total_df[total_df['Datetime']==echem_time_df.iloc[i]['Discharge_Start_Time']].index.values[0]
                elif j == 3:
                    state = '3'
                    index = total_df[total_df['Datetime']==gas_change_time_df.iloc[i]['high_to_low']].index.values[0]
                elif j == 4:
                    state = '3\'f'
                    index = (total_df[total_df['Datetime']==echem_time_df.iloc[i]['Outgas_End_Time']].index-outgas_shift).values[0]

                entry = total_df.iloc[index]
                pH_measured = entry['pH_right']
                TA_val = TA_array[-1]+np.sum(total_df.iloc[index_array[-1]:index].Current)/96485/volume
                pCO2 = (entry['CO2 input right(abs val)']/(entry['CO2 input right(abs val)']+entry['N2 input right(abs val)']))
                co2aq = pCO2*henry_constant
                
                pH_func = TA_pH_wrapper(co2aq,solve_value = TA_val)
                if(flag):
                    print("Cycle number:",i+1," state:",state, "co2aq: %0.3f"%co2aq, "TA_val: %0.2f"%TA_val,"pH_func(pH_measured): %0.2f"%pH_func(pH_measured),"pH measured: %0.2f"%pH_measured)
                    #display(entry)
                if solver == "newton_krylov":
                    pH_theory = newton_krylov(pH_func,pH_measured)#use measured pH as the initial guess
                elif solver == "fsolve":
                    pH_theory = fsolve(pH_func,pH_measured)[0]
                #print(co2aq,TA_val,pH_func(pH_measured),pH_theory,pH_measured)

                co2aq_TA = fsolve(TA,co2aq,args=(pH_measured,TA_val))[0]# non-equilibrium co2aq, calculated from TA,use equilibrium co2aq as initial guess
                
                DIC_TA = dic(co2aq_TA,pH_measured)
                DIC_eq = dic(co2aq,pH_measured)
                DIC_theory = dic(co2aq,pH_theory)
                
                cycle_array.append(i+1)
                index_array.append(index)
                states_array.append(state)
                pH_measured_array.append(pH_measured)
                pH_theory_array.append(pH_theory)
                TA_array.append(TA_val)
                DIC_TA_array.append(DIC_TA)
                DIC_eq_array.append(DIC_eq)
                DIC_theory_array.append(DIC_theory)
                
                #calculate delta dic
                Delta_DIC_TA_array.append(DIC_TA-DIC_TA_array[-2])
                Delta_DIC_eq_array.append(DIC_eq-DIC_eq_array[-2])
                Delta_DIC_theory_array.append(DIC_theory-DIC_theory_array[-2])
    
    return pd.DataFrame({"Cycle":cycle_array,"State":states_array,"pH_measured" :pH_measured_array,
                         "pH_theory":pH_theory_array,"TA":TA_array,'DIC_TA':DIC_TA_array,
                         "DIC_eq":DIC_eq_array, "DIC_theory": DIC_theory_array,"index":index_array,
                         "Delta_DIC_TA":Delta_DIC_TA_array,"Delta_DIC_eq":Delta_DIC_eq_array,
                         "Delta_DIC_theory":Delta_DIC_theory_array
                        })


def create_theoretical_dic_pH_array(min_TA = 0,max_TA = 0.2,TA_points=100,capture_pco2 = 0.1,outgas_pco2=1.0,
                                    pco2_points=100, deacidification_pH_guess = 7.5,
                                    acidification_pH_guess = 7.5,pH_low_to_high_guess=7.5,
                                   pH_high_to_low_guess=7.5):
    
    """
    Create theoretical DIC and pH arrays given a max and min total alkalinity
    , capture and outgas co2 partial pressure. Useful for plotting the thermodynamics of DIC cycle.

    :type min_TA: float
    :param min_TA: Minimum total alkalinity or alkalinity in the discharged form

    :type max_TA: float
    :param max_TA: Maximum total alkalinity or alkalinity in the charged form

    :type TA_points: int
    :param TA_points: Number of points in the TA array. Used in np.linspace

    :type capture_pco2: float
    :param capture_pco2: CO2 partial pressure in bar during capture process

    :type outgas_pco2: float
    :param outgas_pco2: CO2 partial pressure in bar during outgas process

    :type pco2_points: int
    :param pco2_points: Number of points in the CO2 gas change array. Used in np.linspace

    :type deacidification_pH_guess: float
    :param deacidification_pH_guess: MUST BE A FLOAT. Initial guess of pH in deacidification for scipy.fsolve.

    :type acidification_pH_guess: float
    :param acidification_pH_guess: MUST BE A FLOAT. Initial guess of pH in acidification for scipy.fsolve.

    :type pH_low_to_high_guess: float 
    :param pH_low_to_high_guess: MUST BE A FLOAT. Initial guess of pH in gas change fomr low partial pressure to high partial pressure
                                 for scipy.fsolve.

    :type pH_high_to_low_guess: float
    :param pH_high_to_low_guess: MUST BE A FLOAT. Initial guess of pH in gas change fomr high partial pressure to low partial pressure
                                 for scipy.fsolve.

    :rtype: *dict*
    :return: a dictionary containing TA, DIC, pH array for various processes:

        dict['alkalinity'] -> (*list*): alkalinity array, e.g. 100 points between 0 and 0.2 M\n
        dict['change_gas'] -> (*list*): gas_change array, e.g. 100 points between 0.1 and 1 bar\n
        dict['pH_deacidification'] -> (*list*): pH array corresponding to TA and partial pressure in the deacidification process\n
        dict['pH_acidification'] -> (*list*): pH array corresponding to TA and partial pressure in the acidification process\n
        dict['dic_deacidification'] -> (*list*): DIC array corresponding to TA and partial pressure in the deacidification process\n
        dict['dic_acidification'] -> (*list*): DIC array corresponding to TA and partial pressure in the acidification process\n
        dict['pH_low_to_high'] -> (*list*): pH array corresponding to TA and partial pressure in the low to high gas change process\n
        dict['pH_high_to_low'] -> (*list*): pH array corresponding to TA and partial pressure in the high to low gas change process\n
        dict['dic_low_to_high'] -> (*list*): DIC array corresponding to TA and partial pressure in the low to high gas change process\n
        dict['dic_high_to_low'] -> (*list*): DIC array corresponding to TA and partial pressure in the high to low gas change process\n
        dict['capture_pco2'] -> (*float*): capture pCO2 as in the input\n
        dict['outgas_pco2'] -> (*float*): outgas pCO2 as in the input\n

    """
    alkalinity_array = np.linspace(min_TA,max_TA,TA_points)
    co2aq100p = 0.035
    co2aq_outgas = co2aq100p*outgas_pco2
    co2aq_capture = co2aq100p*capture_pco2
    pH_deacidification_array = []
    pH_acidification_array = []
    dic_deacidification_array = []
    dic_acidification_array = []

    change_gas_array = np.linspace(co2aq_capture,co2aq_outgas,100)

    pH_low_to_high_array = []
    pH_high_to_low_array = []
    dic_low_to_high_array = []
    dic_high_to_low_array = []
    
    #Calculate equilibrium pH and DIC at fixed pCO2 based on varying TA
    for i in range(len(alkalinity_array)):

        pH_func_deacidification = TA_pH_wrapper(co2aq_capture,solve_value=alkalinity_array[i])
        pH_deacidification = fsolve(pH_func_deacidification,deacidification_pH_guess)
        pH_func_acidification = TA_pH_wrapper(co2aq_outgas,solve_value=alkalinity_array[i])
        pH_acidification = fsolve(pH_func_acidification,acidification_pH_guess)

        dic_deacidification = dic(co2aq_capture,pH=pH_deacidification,solve_value = 0)
        dic_acidification = dic(co2aq_outgas,pH=pH_acidification,solve_value=0)
        pH_deacidification_array.append(pH_deacidification)
        pH_acidification_array.append(pH_acidification)
        dic_deacidification_array.append(dic_deacidification)
        dic_acidification_array.append(dic_acidification)

    #Calculate equilibrium pH and DIC at fixed TA based on varying pCO2
    for i in range(len(change_gas_array)):

        pH_func_low_to_high = TA_pH_wrapper(change_gas_array[i],solve_value=alkalinity_array[-1])
        pH_low_to_high = fsolve(pH_func_low_to_high,pH_low_to_high_guess)
        pH_func_high_to_low = TA_pH_wrapper(change_gas_array[i],solve_value=alkalinity_array[0])
        pH_high_to_low = fsolve(pH_func_high_to_low,pH_high_to_low_guess)

        dic_low_to_high = dic(change_gas_array[i],pH=pH_low_to_high,solve_value=0)
        dic_high_to_low = dic(change_gas_array[i],pH=pH_high_to_low,solve_value=0)

        pH_low_to_high_array.append(pH_low_to_high)
        pH_high_to_low_array.append(pH_high_to_low)
        dic_low_to_high_array.append(dic_low_to_high)
        dic_high_to_low_array.append(dic_high_to_low)
        
    
    return {"alkalinity":alkalinity_array,"change_gas":change_gas_array,"pH_deacidification":pH_deacidification_array
           ,"pH_acidification":pH_acidification_array,"dic_deacidification":dic_deacidification_array
           ,"dic_acidification":dic_acidification_array,"pH_low_to_high":pH_low_to_high_array
           ,"pH_high_to_low":pH_high_to_low_array,"dic_low_to_high":dic_low_to_high_array
           ,"dic_high_to_low":dic_high_to_low_array,"capture_pco2":capture_pco2,"outgas_pco2":outgas_pco2}