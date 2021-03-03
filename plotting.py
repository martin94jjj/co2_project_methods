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




def plot_baseline_selection(total_df,capture=True,cycle=5,time_change_period=10800,start=3500,end=6500,vertical_offset=10,ymin=0,ymax=60,title="Capture Baseline Selection"):
    
    """
    Stack different cycles' capture or release period with a vertical offset value. Used for manual baseline selection

    :type total_df: pd.DataFrame
    :param total_df: A dataset created by `utils.merge_echem_gas_df`. It contains electrochemistry and gas flow data

    :type capture: boolean
    :param capture: *True* if capture periods are to be stacked. *False* if reduce periods are to be stacked.

    :type cycle: int
    :param cycle: Number of cycles.

    :type time_change_period: int
    :param time_change_period: The period in seconds that gas change takes place.

    :type start: int
    :param start: Index of the start of the baseline

    :type end: int
    :param end: Index of the end of the baseline

    :type vertical_offset: float
    :param vertical_offset: Offset values to separate each capture/outgas period on y-axis

    :type ymin: float
    :param ymin: Minimum of y-axis

    :type ymax: float
    :param ymax: Maximum of y-axis

    :type title: string
    :param title: Title of the plot

    :rtype: None
    :return: Plot an image of the stacked baseline 

    """

    fig, ax = plt.subplots(1,1,figsize=(5.941,4.630),dpi=400)
    if capture:
        for i in range(1,cycle+1):
            ax.plot(total_df.index.values[(i-1)*2*time_change_period:(2*i-1)*time_change_period]-time_change_period*2*(i-1),vertical_offset*(i-1)+total_df['Corrected_Flow_Right'].iloc[(i-1)*2*time_change_period:(2*i-1)*time_change_period])
    else:
        for i in range(0,cycle):
            ax.plot(total_df.index.values[(2*i+1)*time_change_period:(2*i+2)*time_change_period]-time_change_period*(2*i+1),vertical_offset*(i)+total_df['Corrected_Flow_Right'].iloc[(2*i+1)*time_change_period:(2*i+2)*time_change_period])

    ax.vlines(start,ymin=ymin,ymax=ymax,color='black')
    ax.vlines(end,ymin=ymin,ymax=ymax,color='black')


    size=15
    legend_size=11

    #ax[0].set_xlim(starting_time-delta,ending_time-delta)
    ax.tick_params(axis='x')
    ax.xaxis.set_tick_params(labelsize=size)
    ax.yaxis.set_tick_params(labelsize=size)
    ax.set_ylim(ymin,ymax)
    ax.set_ylabel('Flow rate (sccm) + #cycle+%2d'%vertical_offset,fontsize = size)
    ax.tick_params(axis='x',which='minor',direction='in',length=3)
    ax.tick_params(axis='x',which='major',direction='in',length=5)

    ax.tick_params(axis='y',which='minor',direction='in',length=3)
    ax.tick_params(axis='y',which='major',direction='in',length=5)
    ax.set_xlabel('Time (s)',fontsize = size)
    ax.set_title(title)

    plt.minorticks_on()
    plt.show()
