import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
#from matplotlib.colors import ListedColormap, LinearSegmentedColormap

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


def plot_theoretical_dic_pH_TA(data_dict,x='TA',y='DIC',legend=True,xlim_low = None,xlim_high=None,ylim_low=None,ylim_high=None,save_name=None):
    """
    Plot relationship among TA, DIC and pH. Can change x and y freely depending on the content to be plotted.

    :type x: string
    :param x: The array used for x-axis. Choose from "TA","DIC" and "pH"

    :type y: string
    :param y: The array used for y-axis. Choose from "TA", "DIC" and "pH"

    :type legend: bool
    :param legend: *True* if you want to show the legend

    :type xlim_low: float
    :param xlim_low: low xlim for `ax.set_xlim()`

    :type xlim_high: float
    :param xlim_high: high xlim for `ax.set_xlim()`

    :type ylim_low: float
    :param ylim_low: low ylim for `ax.set_ylim()`

    :type ylim_high: float
    :param ylim_high: high ylim for `ax.set_ylim()`

    :type save_name: string
    :param save_name: file name of the figure to be saved.

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
    if xlim_low:
        ax.set_xlim(xlim_low,xlim_high)
    if ylim_low:
        ax.set_ylim(ylim_low,ylim_high)
    #ax.set_title("DIC vs. pH for Capture @ 0.1 bar pCO2 2M ΔTA")
    if save_name:
        plt.savefig(save_name, dpi=400)
    #ax.set_ylim(-0.3,2.2)
    #ax.set_xlim(3,10.1)
    plt.minorticks_on()
    plt.show()


def plot_single_echem_cycles(df,x='Capacity',y='Voltage',cycle_number=None,colormap=None,legend=False,legend_size=10,legend_ncol=3,save_name=None,xlabel=None,ylabel=None,xlim = None, ylim=None):
    """
    Plot single cycle behaviors, e.g. voltage or pH change with capacity during charge and discharge.

    :type df: pd.DataFrame
    :param df: Echem dataframe generated by `echem_methods.analyze_gamry_file`

    :type x: string
    :param x: The array used for x-axis. Usually use "Capacity"

    :type y: string
    :param y: The array used for y-axis. Usually use "Voltage" or "pH_right"

    :type cycle_number: list
    :param cycle_number: A list of the specific cycle numbers to be plotted. If None, plot all cycles

    :type colormap: string
    :param colormap: A string for a Matplotlib built-in colormap. E.g. "winter", "hsv","rainbow". Default is 'rainbow'

    :type legend: bool
    :param legend: If True, show legend

    :type legend_size: float
    :param legend_size: Font size for legend, default is 10

    :type legend_ncol: int
    :param legend_ncol: Number of columns for legend

    :type save_name: string
    :param save_name: file name of the figure to be saved.

    :type xlabel: string
    :param xlabel: Label for x-axis. If "None", use value for "x".

    :type ylabel: string
    :param ylabel: Label for y-axis. If "None", use value for "y".

    :type xlim: tuple
    :param xlim: Range (xlim[0], xlim[1]) for x-axis

    :type ylim: tuple
    :param ylim: Range (ylim[0], ylim[1]) for y-axis

    :rtype: None
    :return: Plot an image of "x" vs. "y" in selected echem_cycles
    """
    
    if not colormap:
        colormap = cm.get_cmap('rainbow')
    else:
        colormap = cm.get_cmap(colormap)
        
    fig,ax=plt.subplots(1,1,figsize=(5.941,4.630),dpi=400)

    if cycle_number:
        df = df[df['Cycle_number'].isin(cycle_number)]
    else:
        cycle_number = np.arange(max(df['Cycle_number']))+1

    
    hue = colormap(np.linspace(0,0.9,len(cycle_number)))

    
    for i in tqdm(range(len(cycle_number))):
        ax.plot(np.abs(df[(df['Cycle_number']==cycle_number[i])&(df['Echem_process']=='PWRCHARGE')][x]),df[(df['Cycle_number']==cycle_number[i])&(df['Echem_process']=='PWRCHARGE')][y],label='Cycle {}'.format(cycle_number[i]),color=hue[i])
        ax.plot(np.abs(df[(df['Cycle_number']==cycle_number[i])&(df['Echem_process']=='PWRDISCHARGE')][x]),df[(df['Cycle_number']==cycle_number[i])&(df['Echem_process']=='PWRDISCHARGE')][y],color=hue[i])

    ax.tick_params(axis='y',which='minor',right=True,direction='in',length=3)
    ax.tick_params(axis='y',which='major',right=True,direction='in',length=5.5)
    ax.tick_params(axis='x',which='minor',direction='in',bottom=True,length=3)
    ax.tick_params(axis='x',which='major',direction='in',bottom=True,length=5.5)
    if not xlabel:
        ax.set_xlabel(x)
    else:
        ax.set_xlabel(xlabel)
    if not ylabel:
        ax.set_ylabel(y)
    else:
        ax.set_ylabel(ylabel)
    if legend:
        ax.legend(frameon=False,ncol=legend_ncol,fontsize=legend_size)
        
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    plt.minorticks_on()
    if save_name:
        plt.savefig(save_name, dpi=400)