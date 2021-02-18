# co2_project
Methods for Analyzing Data for Electrochemical CO2 Capture Project

Methods contain 

### 1.0 E-Chem Methods
### 2.0 Gas Flow Methods
### 3.0 Theoretical DIC Calculation Methods
### 4.0 Utility Methods
### 5.0 Plotting Methods

## 1.0 E-Chem Methods

This section contains the following methods:
**find_date_time**: Reads a Gamry file, finds its creation datetime, and returns a datetime variable.

**analyze_gamry_file**: Reads a Gamry file and the starting time of the first file (e.g. the time of the creation or the last time point of previous half cycle). Returns a dataset with continuous Time, Voltage, Current, pH and fitted pH for a half-cycle.

**read_echem**: Reads a Gamry file folder, utilizes **analyze_gamry_file** to get half-cycle data and puts everything together in a dataset with multi-cycle continuous Time,Voltage, Current,pH and fitted pH data.

**cal_capacity_energy**: Reads a Gamry file folder, utilizes **analyze_gamry_file** to get half-cycle data and puts everything together in a dataset with multi-cycle's cycle capacity,energy and efficiencies.

**find_echem_time_period**: Utilizes **find_date_time** method, reads a Gamry folder and returns a dataset containing the start and end time of deacidification, capture, acidification and outgas.

**create_echem_dfs**: Combine the results of **read_echem**, **cal_capacity_energy**, and **find_echem_time_period** to produce a dictionary of the three dfs.



## 2.0 Gas Flow Methods

**find_gas_change_time**: Reads a dataset that has a time_attribute. This dataset should contain CO2 concentration change info. Returns a dataset that contains the time that each gas switch is made.

**create_baseline**: Reads a dataset created by **pd.read_csv**, and a start time and an end time from a process(capture process = deacidification + capture ; release process = acidification + outgas) from **find_echem_time_period**. This method linearly fits the 10 points before the process start time and 10 points from process end time. Returns the linear fit parameters, the dataset index of the process start time and the dataset index of the process end time.

**calculate_amount**: Reads a gas info dataset created by **pd.read_csv**, and a time period dataset by **find_echem_time_period**. Calculates the amount of CO2 captured and released. Returns a dataset containing the cycle number, captured amount, outgas amount, average amount, fitting parameters for capture and outgas, and indices associated with the start and end of the capture/outgas processes


## 3.0 Theoretical DIC Calculation Methods

**dic**: Takes $CO_2(aq)$ in M and pH value and then return the dissolve inorganic carbon concentration in M; solve_value is used when **scipy.optimize.fsolve** is called in order to solve for $CO_2(aq)$ or pH given $DIC$.

**hco3**: Takes $CO_2(aq)$ in M and pH value and then return the $HCO_3^-$ concentration in M; solve_value is used when **scipy.optimize.fsolve** is called in order to solve for $CO_2(aq)$ or pH given hco3.

**co32**: Takes $CO_2(aq)$ in M and pH value and then return the $CO_3^{2-}$ concentration in M; solve_value is used when **scipy.optimize.fsolve** is called in order to solve for $CO_2(aq)$ or pH given carbonate concentration.

**TA**: Takes $CO_2(aq)$ in M and pH value and then return the total alkalinity concentration in M; solve_value is used when **scipy.optimize.fsolve** is called in order to solve for $CO_2(aq)$ or pH given $TA$ concentration.

**calc_DIC**: Takes `total_df`,`echem_time_df` and `gas_change_time_df`, obtains $pCO_2$, $pH_{measured}$, and $TA$ in (M) from the dataframes and return a table of all these value and $pH_{theorey,eq}$ (theoretical pH at gas/solution equilibrium given TA)$DIC_{TA}$ (DIC estimated from TA and measured pH), $DIC_{eq}$ (DIC estimated from measured pH while assuming gas/solution equilibrium) and DIC_{theory,eq} (theoretical DIC at gas/solution equilibrium given TA). For gas equilibrium I mean Gas/solution equilibrium: $CO_2(aq) =0.035 * pCO_2$


## 4.0 Utility Methods

**merge_echem_gas_df**: Combine datasets from 2.0 and 2.1 by merging on `Time_Delta`, which is time minus the first time stamp of `gas_df`. Remove weird data by removing data obtained beyond cycle 11. Fit co2 signal by cubic spline obtained on 20210103. Correct flow rate. Arbitrarily fix incorrect $CO_2$ percentage by setting $CO_2>0.9$ bar to 1.


## 5.0 Plotting Methods