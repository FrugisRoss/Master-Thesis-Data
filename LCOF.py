import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import gams.transfer as gt
import gams
import os
import seaborn as sns

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)



scenario = [
     ("Base Case", r"C:\Users\sigur\OneDrive\DTU\Run on HPC Polimi\Base_Case_RightOut\model")]

fuels = [
    "Ammonia_Eff", "AMMONIA_FLOW"
    "BioGasoline_Eff", "BIOGASOLINEFLOW_BJ_H2"
    "BioJet_Eff", "BIOJETFLOW_H2"
    "EME_Gasoline_Eff", "EME_GASOLINE_FLOW"
    "EME_Jet_Eff", "EME_JET_FLOW"
    "EME_LPG_Eff", "EME_LPG_FLOW"
]

fuel_to_processes = [
    (["AMMONIA_FLOW"], ["Nitrogen_Production", "Ammonia_Synthesis_50"]),
    (["BIOGASOLINEFLOW_BJ_H2","BIOJETFLOW_H2"], ["BioJet_H2_50"]),
    (["EME_GASOLINE_FLOW", "EME_JET_FLOW","EME_LPG_FLOW"], ["EMethanol_Synthesis_50", "EMethanol_Upgrade_50", "CO2_Sto", "CO2_DAC_50"]),

]

def Import_allendofmodel(file_path):
     main_results_path = os.path.join(file_path, "all_endofmodel.gdx")
     df = gt.Container(main_results_path)
     
     df_SOSIBU2INDIC = pd.DataFrame(df["SOSIBU2INDIC"].records)
     df_TRANSCOST = pd.DataFrame(df["TRANSCOST"].records)
     df_PROCDATA = pd.DataFrame(df["PROCDATA"].records)
     
     return df_SOSIBU2INDIC, df_TRANSCOST, df_PROCDATA


def Import_OptiflowMR(file_path):
    main_results_path = os.path.join(file_path, "Optiflow_MainResults.gdx")
    df = gt.Container(main_results_path)
    df_FLOWC = pd.DataFrame(df["VFLOW_Opti_C"].records)
    df_ECO_PROC_YCRAP = pd.DataFrame(df["ECO_PROC_YCRAP"].records)
    df_VFLOWTRANS_Opti_A = pd.DataFrame(df["VFLOWTRANS_Opti_A"].records)

    return df_FLOWC, df_ECO_PROC_YCRAP, df_VFLOWTRANS_Opti_A

def Import_BalmorelMR(file_path):
    main_results_path = os.path.join(file_path, "MainResults.gdx")
    df = gt.Container(main_results_path)
    
    df_EL_DEMAND_YCR = pd.DataFrame(df["EL_DEMAND_YCR"].records)
    df_H2_DEMAND_YCR = pd.DataFrame(df["H2_DEMAND_YCR"].records)
    df_H_DEMAND_YCRA = pd.DataFrame(df["H_DEMAND_YCRA"].records)
    df_EL_PRICE_YCR = pd.DataFrame(df["EL_PRICE_YCR"].records)
    df_H2_PRICE_YCR = pd.DataFrame(df["H2_PRICE_YCR"].records)
    df_H_PRICE_YCRA = pd.DataFrame(df["H_PRICE_YCRA"].records)

    return df_EL_DEMAND_YCR, df_H_DEMAND_YCRA, df_H2_DEMAND_YCR, df_EL_PRICE_YCR, df_H2_PRICE_YCR, df_H_PRICE_YCRA

#Average EL, H2 and H price per year

def Avg_yearly_price(scenario_path, commodity, year, country):
     """
     Calculate the average yearly price of a specified commodity (Electricity, H2, or Heat) 
     for a given year and country, weighted by the corresponding demand.
     This function imports data from the BalmorelMR model, filters the price and demand 
     data for the specified year and country, and computes the weighted average price 
     based on regional demand.
     Args:
          scenario_path (str): The file path to the scenario data for the BalmorelMR model.
          commodity (str): The type of commodity to analyze. Must be one of "Electricity", "H2", or "Heat".
          year (int): The year for which the average price is to be calculated.
          country (str): The country for which the average price is to be calculated.
     Returns:
          float: The weighted average price of the specified commodity in €/MWh.
                  If the denominator (total demand) is zero, the function will print a warning 
                  and return None.
     Raises:
          KeyError: If the required columns ('Y', 'C', 'RRR', 'value') are missing in the data.
          ValueError: If the specified commodity is not one of the allowed values.
     Notes:
          - The function assumes that the imported data contains columns 'Y' (year), 'C' (country), 
            'RRR' (region), and 'value' (price or demand values).
          - The demand values are scaled by a factor of 10^6 to match the price units.
     """

     
     df_EL_DEMAND_YCR, df_H_DEMAND_YCRA, df_H2_DEMAND_YCR, df_EL_PRICE_YCR, df_H2_PRICE_YCR, df_H_PRICE_YCRA = Import_BalmorelMR(scenario_path)
     if commodity == "Electricity":
          df_price = df_EL_PRICE_YCR
     elif commodity == "H2":
          df_price = df_H2_PRICE_YCR
     elif commodity == "Heat":
          df_price = df_H_PRICE_YCRA
     
     if commodity == "Electricity":
          df_demand = df_EL_DEMAND_YCR
     elif commodity == "H2":
          df_demand = df_H2_DEMAND_YCR
     elif commodity == "Heat":
          df_demand = df_H_DEMAND_YCRA



     # Filter the DataFrame for the specified year and country
     filtered_df_price = df_price[(df_price["Y"] == year) & (df_price["C"] == country)]
     filtered_df_demand = df_demand[(df_demand["Y"] == year) & (df_demand["C"] == country)]
     

     # Merge price and demand on 'RRR' (region)
     merged = pd.merge(filtered_df_price[['RRR', 'value']], 
                      filtered_df_demand[['RRR', 'value']], 
                      on='RRR', 
                      suffixes=('_price', '_demand'))
     
     print(merged)

     # Compute the weighted average
     numerator = (merged['value_price'] * merged['value_demand']*10**6).sum()
     denominator = merged['value_demand'].sum()*10**6*3.6
     
     if denominator == 0:
          print("Denominator is zero, cannot compute average price.")  
     else:
          avg_price = numerator / denominator
     
     print(f"Average {commodity} price in {country} for year {year}: {avg_price} €/GJ")

     return avg_price

Avg_yearly_price(r"C:\Users\sigur\OneDrive\DTU\Run on HPC Polimi\Base_Case_RightOut\model", "Electricity", "2050" , "DENMARK")
Avg_yearly_price(r"C:\Users\sigur\OneDrive\DTU\Run on HPC Polimi\Base_Case_RightOut\model", "H2", "2050" , "DENMARK")
Avg_yearly_price(r"C:\Users\sigur\OneDrive\DTU\Run on HPC Polimi\Base_Case_RightOut\model", "Heat", "2050" , "DENMARK")

          
def Avg_yearly_biomass_price (scenario_path, year, country):
    
    """
          Calculate the average yearly biomass price for a specific country and year.
          This function computes the weighted average price of biomass based on the 
          flow of biomass and its corresponding price data. It uses data from two 
          sources: `Import_OptiflowMR` and `Import_allendofmodel`.
          Parameters:
          -----------
          scenario_path : str
               The file path to the scenario data.
          year : int
               The year for which the average biomass price is to be calculated.
          country : str
               The country code for which the average biomass price is to be calculated.
          Returns:
          --------
          float
               The weighted average biomass price in €/GJ for the specified country and year.
          Notes:
          ------
          - The function filters biomass flow data for the specified year, country, 
            and biomass type ("Solid_Biomass").
          - It also filters price data for specific biomass flows 
            ("STRAW_FLOW", "WOOD_FLOW", "WOOD_PELLETS_GEN_FLOW").
          - If the denominator (total biomass flow) is zero, the function will print 
            a warning and avoid division by zero.
          - The function prints intermediate merged data and the computed average price.
          Raises:
          -------
          KeyError:
               If required columns are missing in the input dataframes.
          ValueError:
               If the input data is not properly formatted or contains invalid values.
          """

    df_FLOWC,_, _ =Import_OptiflowMR(scenario_path)
    df_SOSIBU2INDIC, _, _ = Import_allendofmodel(scenario_path)

    df_biomass= df_FLOWC[(df_FLOWC["Y"] == year) & (df_FLOWC["CCC"] == country) & (df_FLOWC["IPROCTO"] == "Solid_Biomass") ]
    df_price = df_SOSIBU2INDIC[
    (df_SOSIBU2INDIC["YYY"] == year) & 
    (df_SOSIBU2INDIC["FLOW"].isin(["STRAW_FLOW", "WOOD_FLOW", "WOOD_PELLETS_GEN_FLOW"]))
]
    # Merge biomass and prices by flow
    merged = pd.merge(df_biomass[['FLOW', 'value']], 
                      df_price[['FLOW', 'value']], 
                      on='FLOW', 
                      suffixes=('_bio', '_price'))
    
    print(merged)

    # Compute the weighted average
    numerator = (merged['value_bio'] * merged['value_price']*10**6).sum()
    denominator = merged['value_bio'].sum()*10**6

    if denominator == 0:
          print("Denominator is zero, cannot compute average price.")  
    else:
          avg_price = numerator / denominator

    print(f"Average biomass price in {country} for year {year}: {avg_price} €/GJ")

    return avg_price

Avg_yearly_biomass_price(r"C:\Users\sigur\OneDrive\DTU\Run on HPC Polimi\Base_Case_RightOut\model",  "2050" , "DENMARK")


def LCOF_calculation(scenario_path, fuels, fuel_to_processes, year, country):
    # Import data
    df_FLOWC, _, _ = Import_OptiflowMR(scenario_path)
    _, _, df_PROCDATA = Import_allendofmodel(scenario_path)

    # Dictionary to store results
    fuel_lifetime_tables = {}

    for fuel_group, processes in fuel_to_processes:
        # Filter PROCDATA for relevant processes and 'PROCLIFETIME'
        df_PROCDATA_filtered = df_PROCDATA[
            (df_PROCDATA["PROC"].isin(processes)) & 
            (df_PROCDATA["PROCDATASET"] == "PROCLIFETIME")
        ]

        if df_PROCDATA_filtered.empty:
            print(f"No PROCLIFETIME data found for processes: {processes}")
            continue

        # Find the maximum lifetime value
        max_lifetime = int(df_PROCDATA_filtered["value"].max())

        # Create a table: rows = 0 to max_lifetime (inclusive), 8 empty columns
        table = pd.DataFrame(
            index=range(max_lifetime + 1),   # +1 because you want inclusive
            columns=[
                "Year", 
                "TIC [M€]", 
                "O&M [M€]", 
                "FUEL COST [M€]", 
                "TOT YEARLY COST [M€]", 
                "TOT YEARLY COST ACT [M€]", 
                "YEARLY FUEL PRODUCTION [PJ]", 
                "YEARLY FUEL PRODUCTION ACT [PJ]"
            ]
        )

        # Fill the 'Year' column
        table["Year"] = range(max_lifetime + 1)

        # Save the table for this fuel group
        fuel_lifetime_tables[tuple(fuel_group)] = table

        
        print(f"Table for fuel group {fuel_group} (max lifetime {max_lifetime} years):")
        print(table)


    return fuel_lifetime_tables

LCOF_calculation(r"C:\Users\sigur\OneDrive\DTU\Run on HPC Polimi\Base_Case_RightOut\model", fuels, fuel_to_processes, "2050", "DENMARK")