import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_baseline(amount_df,cycle=2,capture=True,baseline_range=100,capture_parameter = 'Corrected_Flow_Right_filtered',title='100 points baseline',ymin=-1,ymax=-1):
    #
    # Plots the baseline of a given amount df
    #
    #
    #
    #
    #
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