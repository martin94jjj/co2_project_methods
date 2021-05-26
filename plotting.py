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




def plot_baseline_selection(total_df,capture=True,cycle=5,time_change_period=10800,start=3500,end=6500,parameter= "Corrected_Flow_Right",vertical_offset=10,ymin=0,ymax=60,title="Capture Baseline Selection"):
    
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
            ax.plot(total_df.index.values[(i-1)*2*time_change_period:(2*i-1)*time_change_period]-time_change_period*2*(i-1),vertical_offset*(i-1)+total_df[parameter].iloc[(i-1)*2*time_change_period:(2*i-1)*time_change_period])
    else:
        for i in range(0,cycle):
            ax.plot(total_df.index.values[(2*i+1)*time_change_period:(2*i+2)*time_change_period]-time_change_period*(2*i+1),vertical_offset*(i)+total_df[parameter].iloc[(2*i+1)*time_change_period:(2*i+2)*time_change_period])

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


def plot_theoretical_dic_pH_TA(data_dict,x='TA',y='DIC',legend=True):
    """
    Plot relationship among TA, DIC and pH. Can change x and y freely depending on the content to be plotted.

    :type x: string
    :param x: The array used for x-axis. Choose from "TA","DIC" and "pH"

    :type y: string
    :param y: The array used for y-axis. Choose from "TA", "DIC" and "pH"

    :type legend: bool
    :param legend: *True* if you want to show the legend

    :rtype: None
    :return: Plot an image of DIC vs. pH (or other combinations of DIC, pH and TA) 
    """
    fig,ax=plt.subplots(1,1,figsize=(5.941,4.630),dpi=400)
    
    if x=='TA' or x =='ta':
        deacidification_x = data_dict['alkalinity']
        acidification_x = data_dict['alkalinity']
        low_to_high_x = [data_dict["alkalinity"][-1]]*len(data_dict["dic_low_to_high"])
        high_to_low_x = [data_dict["alkalinity"][-0]]*len(data_dict["dic_high_to_low"])
        xlabel='TA(M)'
    elif x=="DIC" or x == "dic":
        deacidification_x = data_dict["dic_deacidification"]
        acidification_x = data_dict['dic_acidification']
        low_to_high_x = data_dict["dic_low_to_high"]
        high_to_low_x = data_dict["dic_high_to_low"]
        xlabel='DIC(M)'
        
    elif x=="pH" or x == "ph":
        deacidification_x = data_dict["pH_deacidification"]
        acidification_x = data_dict['pH_acidification']
        low_to_high_x = data_dict["pH_low_to_high"]
        high_to_low_x = data_dict["pH_high_to_low"]
        xlabel ='pH'
        
    if y=='TA' or y =='ta':
        deacidification_y = data_dict['alkalinity']
        acidification_y = data_dict['alkalinity']
        low_to_high_y = [data_dict["alkalinity"][-1]]*len(data_dict["dic_low_to_high"])
        high_to_low_y = [data_dict["alkalinity"][-0]]*len(data_dict["dic_high_to_low"])
        ylabel = 'TA(M)'
    elif y=="DIC" or y == "dic":
        deacidification_y = data_dict["dic_deacidification"]
        acidification_y = data_dict['dic_acidification']
        low_to_high_y = data_dict["dic_low_to_high"]
        high_to_low_y = data_dict["dic_high_to_low"]
        ylabel = 'DIC(M)'
    elif y=="pH" or y == "ph":
        deacidification_y = data_dict["pH_deacidification"]
        acidification_y = data_dict['pH_acidification']
        low_to_high_y = data_dict["pH_low_to_high"]
        high_to_low_y = data_dict["pH_high_to_low"]
        ylabel = 'pH'
    
    low = data_dict['capture_pco2'] 
    high = data_dict['outgas_pco2']
    ax.plot(deacidification_x,deacidification_y,label='deacidification/CO$_2$ invasion',lw=2,color='#1f78b4')
    ax.plot(low_to_high_x,low_to_high_y,lw=2,label='{} to {} bar $p$CO$_2$'.format(low,high),color='#a6cee3')
    ax.plot(acidification_x,acidification_y,label='acidification/CO$_2$ outgassing',lw=2,color='#33a02c')
    ax.plot(high_to_low_x,high_to_low_y,label='{} to {} bar $p$CO$_2$'.format(high,low),lw=2,color='#b2df8a')
    
    if legend:
        ax.legend(frameon=False)
    ax.tick_params(axis='y',which='minor',right=True,direction='in',length=3)
    ax.tick_params(axis='y',which='major',right=True,direction='in',length=5.5)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x',which='minor',direction='in',bottom=True,length=3)
    ax.tick_params(axis='x',which='major',direction='in',bottom=True,length=5.5)
    ax.set_xlabel(xlabel)
    #ax.set_title("DIC vs. pH for Capture @ 0.1 bar pCO2 2M ΔTA")

    #ax.set_ylim(-0.3,2.2)
    #ax.set_xlim(3,10.1)
    plt.minorticks_on()
    plt.show()

