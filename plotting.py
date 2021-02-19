import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_baseline(amount_df,total_df,cycle=2,capture=True,baseline_range=100,capture_parameter = 'Corrected_Flow_Right_filtered',title='100 points baseline',ymin=-1,ymax=-1):
    '''    
        Plots the baseline of a given **amount_df**

    :type amount_df: pd.DataFrame
    :param amount_df: A dataset creates by `gas_methods.calculate_amount()` that contains captured/released CO2 amount

    :type total_df: pd.DataFrame
    :param total_df: A dataset creates by `utils.merge_echem_gas_df()` that contains gas and electrochem data.

    :type cycle: int
    :param cycle: The cycle number to plot

    :type capture: boolean
    :param capture: If True, plot capture process and baseline. If False, plot release process and baseline.
    
    :type baseline_range: int
    :param baseline_range: Number of extended points to plot the baseline. e.g. plot from **baseline_range** points before capture/release

    :type capture_parameter: string
    :param capture_parameter: The paramester used for calculating CO2 capture/release amount and baseline, usually 'Corrected_Flow_Right_filtered'.

    :type title: string
    :param title: The title for the output plot

    :type ymin: float
    :param ymin: y-axis lower limit

    :type ymax: float
    :param ymax: y-axis higher limit

    :rtype: None
    :return:
        A figure with capture/release process with baseline.
    
    '''
    if capture:
        start_index_key = 'c_start'
        end_index_key = 'c_end'
        start_fit0 = 'c0'
        end_fit0 = 'c1'
    else:
        start_index_key = 'o_start'
        end_index_key = 'o_end'
        start_fit0 = 'o0'
        end_fit0 = 'o1'
        

    plt.plot(total_df[capture_parameter].iloc[int(amount_df.iloc[cycle][start_index_key]-baseline_range):int(amount_df.iloc[cycle][end_index_key])],label = 'flow rate')
    plt.plot(np.poly1d([amount_df.iloc[cycle][start_fit0],  amount_df.iloc[cycle][end_fit0]])(total_df['Time_Delta']),label='fitted baseline')
    plt.xlim(int(amount_df.iloc[cycle][start_index_key]-baseline_range),int(amount_df.iloc[cycle][end_index_key]))
    plt.title(title)
    plt.ylabel('Flow Rate (sccm)')
    plt.xlabel('Time(seconds)')
    plt.legend(frameon=False)
    if ymin != -1:
        plt.ylim(ymin,ymax)
    plt.show()