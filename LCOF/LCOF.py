#This script can be used for the post processing of the results obtained from the Optiflow model.
#It calculates the Levelized Cost of Fuel (LCOF) for different fuel groups and transport sectors.
#Author: Rossella Frugis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import gams.transfer as gt
import gams
import os
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import glob
import math


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


#------------To be modified by the user-------------------

#Scenario is a tuple containing the referring scenario name and the pathway to the folder containing the Optiflow_MainResults.gdx and all_endofmodel.gdx files 

scenario = [
     
    # ("BASE BASELINE", r"C:\Users\sigur\OneDrive\DTU\Run on HPC Polimi\Base_Case_RightOut\model"),
    #  ("BASE FOSSIL ", r"C:\Users\sigur\OneDrive\DTU\Run on HPC Polimi\Base_Case_FOSSIL\model"),
    #  ("BASE NWI", r"C:\Users\sigur\OneDrive\DTU\Run on HPC Polimi\Base_Case_nowoodpellets\model"),
     
    # ("CO2 BASELINE", r"C:\Users\sigur\OneDrive\DTU\Run on HPC Polimi\CO2_Case_RLC\model"),
     #("CO2 NWI", r"C:\Users\sigur\OneDrive\DTU\Run on HPC Polimi\CO2_Case_RLC_nowoodpellets\model"),
    #  ("CO2 FOSSIL NWI", r"C:\Users\sigur\OneDrive\DTU\Run on HPC Polimi\CO2_Case_RLC_FOSSIL_nowoodpellets\model"),
    
   # ("BIODIVERSITY BASELINE", r"C:\Users\sigur\OneDrive\DTU\Run on HPC Polimi\Biodiversity_Case_RLC_nopf\model"),
   
    # ("BIODIVERSITY FOSSIL", r"C:\Users\sigur\OneDrive\DTU\Run on HPC Polimi\Biodiversity_Case_RLC_nopf_FOSSIL\model"),
    #("BIODIVERSITY NWI", r"C:\Users\sigur\OneDrive\DTU\Run on HPC Polimi\Biodiversity_Case_RLC_nopf_nowoodpellets\model"),
      ("BIODIVERSITY NWI FOSSIL", r"C:\Users\sigur\OneDrive\DTU\Run on HPC Polimi\Biodiversity_Case_RLC_nopf_FOSSIL_nowoodpellets\model"),
    #("BIODIVERSITY BASELINE", r"C:\Users\sigur\OneDrive\DTU\Run on HPC Polimi\Biodiversity_Case_RLC_nopf\model"),
   
    #("BIODIVERSITY FOSSIL", r"C:\Users\sigur\OneDrive\DTU\Run on HPC Polimi\Biodiversity_Case_RLC_nopf_FOSSIL\model"),
    #("BIODIVERSITY NWI", r"C:\Users\sigur\OneDrive\DTU\Run on HPC Polimi\Biodiversity_Case_RLC_nopf_nowoodpellets\model"),
    # ("BIODIVERSITY NWI FOSSIL", r"C:\Users\sigur\OneDrive\DTU\Run on HPC Polimi\Biodiversity_Case_RLC_nopf_FOSSIL_nowoodpellets\model"),

     #("CO2 BASELINE", r"C:\Users\sigur\OneDrive\DTU\Run on HPC Polimi\CO2_Case_RLC\model"),
     #("CO2 NWI", r"C:\Users\sigur\OneDrive\DTU\Run on HPC Polimi\CO2_Case_RLC_nowoodpellets\model"),
     ]


save_pdf = False

#Mapping of fuel flows to processes' pathways that produce them.
#Must be modified according to the processes defined in the Optiflow model.

fuel_to_processes = [
    (["AMMONIA_FLOW"], ["Nitrogen_Production", "Ammonia_Synthesis_50"]),
    (["BIOGASOLINEFLOW_BJ_H2","BIOJETFLOW_H2"], ["BioJet_H2_50"]),
    (["BIOGASOLINEFLOW_BJ_TG","BIOJETFLOW_TG"], ["BioJet_50"]),
    (["EME_GASOLINE_FLOW", "EME_JET_FLOW","EME_LPG_FLOW"], ["EMethanol_synthesis_50", "EMethanol_Upgrade_50", "CO2_DAC_50"]),
    (["KEROSENEFLOW"], ["KeroseneSource"]),

]

#List of tuples containing the process names and the corresponding fuel flows they produce.
#Used in the code to obtain the fuel quantity produced by each process.
#Must be modified according to the processes defined in the Optiflow model.

fuels = [
    ("Ammonia_Synthesis_50", "AMMONIA_FLOW"),
    ("BioJet_H2_50", "BIOGASOLINEFLOW_BJ_H2"),
    ("BioJet_H2_50", "BIOJETFLOW_H2"),
    ("BioJet_50", "BIOGASOLINEFLOW_BJ_TG"),
    ("BioJet_50", "BIOJETFLOW_TG"),
    ("EME_Upgrade_Sum", "EME_GASOLINE_FLOW"),
    ("EME_Upgrade_Sum", "EME_JET_FLOW"),
    ("EME_Upgrade_Sum", "EME_LPG_FLOW"),
    ('KeroseneSource','KEROSENEFLOW' )
]


# Table of Optiflow input fuel cost values for specific fossil fuels [€/GJ]
# The name of the fossil fuel must be expressed as the name of the Optiflow flow, so that it matches the other tuples

fossil_fuels_costs = {
    "DIESELFLOW": 45.63888889,
    "MDOFLOW": 18.38785047,
    "KEROSENEFLOW": 27.19546742
}

name_map = {
    "Ammonia_Eff": "AMMONIA_FLOW",
    "BioGasoline_Eff": "BIOGASOLINEFLOW",
    "EME_Gasoline_Eff": "EME_GASOLINE_FLOW",
    "EME_LPG_Eff": "EME_LPG_FLOW",
    "BioJet_Eff": "BIOJETFLOW",
    "EME_Jet_Eff": "EME_JET_FLOW",
    "KeroseneSource":"KEROSENEFLOW"
}

# Normalized name map for the plots by fuel group
normalized_name_map = {
     "AMMONIA_FLOW": "Ammonia",
     "BIOGASOLINEFLOW_BJ_H2": "Biofuels with H₂",
     "BIOGASOLINEFLOW_BJ_TG": "Biofuels",
     "EME_GASOLINE_FLOW": "E-Methanol Derived Fuels",
     "KEROSENEFLOW": "Kerosene",
     "MDOFLOW": "MDO",
     "DIESELFLOW": "Diesel"

}

sectors= { "Sea_fuels_sum":"Maritime",
        "Air_fuels_sum":"Aviation",
        "Road_fuels_sum":"Road",
    }

#-----------End of user modification section-------------------

# List of fuel groups to be deleted from the folder containing the CSV files.
# This is set to the left side (fuel group) of each tuple in fuel_to_processes.
fuel_groups = [tuple(fuel_group) for fuel_group, _ in fuel_to_processes]


# Delete all CSVs in the folder containing the fuel group names in their filenames
folder_path = os.path.dirname(__file__)
for fuel_group in fuel_groups:
    for fuel in fuel_group:
        csv_pattern = os.path.join(folder_path, f"*{fuel}*.csv")
        for file in glob.glob(csv_pattern):
            try:
                os.remove(file)
                print(f"Deleted: {file}")
            except OSError as e:
                print(f"Error deleting file {file}: {e}")


# Function to import all end-of-model results from the GDX file
def Import_allendofmodel(file_path):
    """
    Imports and returns key dataframes from the 'all_endofmodel.gdx' file located at the specified file path.
    Args:
        file_path (str): The directory path where the 'all_endofmodel.gdx' file is located.
    Returns:
        tuple: A tuple containing the following pandas DataFrames:
            - df_SOSIBU2INDIC: DataFrame of SOSIBU2INDIC records.
            - df_TRANSCOST: DataFrame of TRANSCOST records.
            - df_PROCDATA: DataFrame of PROCDATA records.
            - df_DISCOUNTRATE: DataFrame of DISCOUNTRATE records.
            - df_TRANSDIST: DataFrame of TRANSDIST records.
    """
    main_results_path = os.path.join(file_path, "all_endofmodel.gdx")
    df = gt.Container(main_results_path)
    
    df_SOSIBU2INDIC = pd.DataFrame(df["SOSIBU2INDIC"].records)
    df_TRANSCOST = pd.DataFrame(df["TRANSCOST"].records)
    df_PROCDATA = pd.DataFrame(df["PROCDATA"].records)
    df_DISCOUNTRATE = pd.DataFrame(df["DISCOUNTRATE"].records)
    df_TRANSDIST = pd.DataFrame(df["TRANSDIST"].records)
    
    return df_SOSIBU2INDIC, df_TRANSCOST, df_PROCDATA, df_DISCOUNTRATE, df_TRANSDIST

# Function to import Optiflow Main Results from the GDX file
def Import_OptiflowMR(file_path):
    main_results_path = os.path.join(file_path, "Optiflow_MainResults.gdx")
    df = gt.Container(main_results_path)
    df_FLOWC = pd.DataFrame(df["VFLOW_Opti_C"].records)
    df_ECO_PROC_YCRAP = pd.DataFrame(df["ECO_PROC_YCRAP"].records)
    df_VFLOWTRANS_Opti_A = pd.DataFrame(df["VFLOWTRANS_Opti_A"].records)
    df_FLOWA = pd.DataFrame(df["VFLOW_Opti_A"].records)

    return df_FLOWC, df_ECO_PROC_YCRAP, df_VFLOWTRANS_Opti_A, df_FLOWA


# Function to import BalmorelMR results from the GDX file
def Import_BalmorelMR(file_path):
    """
    Imports and processes Balmorel model results from a specified directory.
    Args:
        file_path (str): The directory path containing the 'MainResults.gdx' file.
    Returns:
        tuple: A tuple containing six pandas DataFrames in the following order:
            - df_EL_DEMAND_YCR: Electricity demand by year, country, and region.
            - df_H_DEMAND_YCRA: Heat demand by year, country, region, and area.
            - df_H2_DEMAND_YCR: Hydrogen demand by year, country, and region.
            - df_EL_PRICE_YCR: Electricity price by year, country, and region.
            - df_H2_PRICE_YCR: Hydrogen price by year, country, and region.
            - df_H_PRICE_YCRA: Heat price by year, country, region, and area.
    """
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
    for a given year and country, weighted by the corresponding demand, in €/PJ.
    This function imports price and demand data from the BalmorelMR model, filters for the
    specified year and country, and computes the weighted average price based on regional demand.
    Args:
        scenario_path (str): Path to the scenario data directory.
        commodity (str): "Electricity", "H2", or "Heat".
        year (str or int): Year for which the average price is calculated.
        country (str): Country code.
    Returns:
        float: Weighted average price of the specified commodity in €/PJ,
            or None if total demand is zero.
    Notes:
        - The function expects columns 'Y' (year), 'C' (country), 'RRR' (region), and 'value' in the data.
        - Demand values are scaled by 10^6 to match price units.
        - If total demand is zero, a warning is printed and None is returned.
    """

    df_EL_DEMAND_YCR, df_H_DEMAND_YCRA, df_H2_DEMAND_YCR, df_EL_PRICE_YCR, df_H2_PRICE_YCR, df_H_PRICE_YCRA = Import_BalmorelMR(scenario_path)
    if commodity == "Electricity":
        df_price = df_EL_PRICE_YCR
    elif commodity == "H2":
        df_price = df_H2_PRICE_YCR
    elif commodity == "Heat":
        df_price = df_H_PRICE_YCRA
    else:
        raise ValueError(f"Unknown commodity: {commodity}")

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

    # Compute the weighted average
    numerator = (merged['value_price'] * merged['value_demand'] * 10**6).sum()
    denominator = merged['value_demand'].sum() * 3.6

    if denominator == 0:
        print("Denominator is zero, cannot compute average price.")
        avg_price = None
    else:
        avg_price = numerator / denominator

    print(f"Average {commodity} price in {country} for year {year}: {avg_price} €/PJ")

    return avg_price



         
def Avg_yearly_biomass_price (scenario_path, year, country):
    
    """
    Calculate the average yearly price of biomass for a given year and country, weighted by the corresponding biomass flow, in €/PJ.
    This function imports biomass flow and price data from the OptiflowMR and all_endofmodel models, filters for the
    specified year and country, and computes the weighted average price based on biomass flow, including transport costs.
    Args:
        scenario_path (str): Path to the scenario data directory.
        year (str or int): Year for which the average price is calculated.
        country (str): Country code.
    Returns:
        float: Weighted average price of biomass in €/PJ, including transport costs,
            or None if total biomass flow is zero.
    Notes:
        - The function expects columns 'Y' (year), 'CCC' (country), 'IPROCTO' (process), 'FLOW', and 'value' in the data.
        - Biomass flow values are scaled by 10^6 to match price units.
        - If total biomass flow is zero, a warning is printed and None is returned.
    """

    df_FLOWC,_, df_VFLOWTRANS_Opti_A, _ =Import_OptiflowMR(scenario_path)
    df_SOSIBU2INDIC, df_TRANSCOST, _, _, df_TRANSDIST= Import_allendofmodel(scenario_path)

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
    
    #print(merged)

    # Compute the weighted average
    numerator = (merged['value_bio'] * merged['value_price']*10**6).sum()
    denominator = merged['value_bio'].sum()

    if denominator == 0:
          print("Denominator is zero, cannot compute average price.")  
    else:
          avg_price = numerator / denominator

    df_transquantity = df_VFLOWTRANS_Opti_A[
          (df_VFLOWTRANS_Opti_A["Y"] == year) &
          (df_VFLOWTRANS_Opti_A["PROC"] != "ProcEximTruckC") 
     ][["IAAAE", "IAAAI", "PROC", "value"]]

    df_transcost = df_TRANSCOST[
          (df_TRANSCOST["FLOWINDIC"] == "OPERATIONCOST")
     ][["PROC", "value"]]

     # Merge the two DataFrames on the 'PROC' column
    df_transport = pd.merge(df_transquantity, df_transcost, on="PROC", how="left", suffixes=('_quantity', '_price'))

     # Join df_transquantity with df_TRANSDIST on 'IAAAE' and 'IAAAI'
    df_transport = pd.merge(df_transport, df_TRANSDIST, on=["IAAAE", "IAAAI", "PROC"], how="left")

    df_transport = df_transport.rename(columns={'value': 'valuedist_dist'})

    df_transport["value"] = df_transport["value_quantity"] * df_transport["value_price"] * df_transport["valuedist_dist"] * 10**6 # €/GJ/km * PJ * km *10^6 GJ/PJ = €

    tran_numerator = df_transport["value"].sum()
    tran_denominator = df_transport["value_quantity"].sum() #PJ

    avg_transport_price = tran_numerator / tran_denominator if tran_denominator != 0 else 0 #€/PJ

    avg_price_notransport = avg_price 

    avg_price += avg_transport_price
    
    
    


    print(f"Average biomass price excluding transport costs in {country} for year {year}: {avg_price_notransport} €/PJ")
    print(f"Average biomass price in {country} for year {year}: {avg_price} €/PJ")

    return avg_price


def LCOF_calculation(scenario_path, fuels, fuel_to_processes, year, country):
    """
    Calculate the Levelized Cost of Fuel (LCOF) for each fuel group and process pathway for a given scenario, year, and country.
    This function imports process, cost, and flow data, computes investment, O&M, and feedstock costs, and calculates the LCOF
    by discounting costs and fuel production over the process lifetime. Results are saved as CSV files for each fuel group.
    Args:
        scenario_path (str): Path to the scenario data directory.
        fuels (list): List of tuples mapping process names to their produced fuel flows.
        fuel_to_processes (list): List of tuples mapping fuel flow groups to process pathways.
        year (str or int): Year for which the LCOF is calculated.
        country (str): Country code.
    Returns:
        fuel_lcof (dict): Calculated LCOF (€/GJ) for each fuel group.
    Notes:
        - The function expects relevant columns in the imported dataframes, including process, cost, and flow information.
        - Feedstock prices are obtained using helper functions for electricity, hydrogen, heat, and biomass.
        - Discount rate is taken from the imported data.
        - Results are saved as CSV files in the same directory as the script.
    """
    # Import data
    df_FLOWC, df_ECO_PROC_YCRAP, _, df_FLOWA = Import_OptiflowMR(scenario_path)
    _, _, df_PROCDATA, df_DISCOUNTRATE, _ = Import_allendofmodel(scenario_path)

    # Dictionary to store results
    fuel_lifetime_tables = {}
    fuel_lcof = {}

    el_price = Avg_yearly_price(scenario_path, "Electricity", "2050", "DENMARK")
    h2_price = Avg_yearly_price(scenario_path, "H2", "2050", "DENMARK")
    heat_price = Avg_yearly_price(scenario_path, "Heat", "2050", "DENMARK")
    biomass_price = Avg_yearly_biomass_price(scenario_path, "2050", "DENMARK")

    # DISCOUNT RATE
    disc_rate = df_DISCOUNTRATE.values[0]

    for fuel_group, processes in fuel_to_processes:
        # LIFETIME DATA
        df_PROCDATA_lifetime = df_PROCDATA[
            (df_PROCDATA["PROC"].isin(processes)) & 
            (df_PROCDATA["PROCDATASET"] == "PROCLIFETIME")
        ]

        if df_PROCDATA_lifetime.empty:
            print(f"No PROCLIFETIME data found for processes: {processes}")
            continue

        # Find the maximum lifetime value
        max_lifetime = int(df_PROCDATA_lifetime["value"].max())

        # INVESTMENT COST DATA

        # Identify cities where all processes are present for this fuel group
        # Filter df_FLOWA for year and relevant processes
        filtered = df_FLOWA[
            (df_FLOWA["Y"] == year) &
            (df_FLOWA["IPROCFROM"].isin(processes))
        ]

        # Group by city and collect unique IPROCFROM values per city
        cities_with_processes = filtered.groupby("AAA")["IPROCFROM"].apply(set)

        # Keep only cities where all required processes are present
        valid_cities = cities_with_processes[cities_with_processes.apply(lambda x: set(processes).issubset(x))].index.tolist()

        # Apply this city filter to investment and fixed O&M costs
        df_ECO_PROC_YCRAP_investment = df_ECO_PROC_YCRAP[
            (df_ECO_PROC_YCRAP["Y"] == year) &
            (df_ECO_PROC_YCRAP["C"] == country) &
            (df_ECO_PROC_YCRAP["PROC"].isin(processes)) &
            (df_ECO_PROC_YCRAP["COST_TYPE"] == "INVESTMENT_NA") &
            (df_ECO_PROC_YCRAP["AAA"].isin(valid_cities))
        ]

        print(f"The df_ECO_PROC_YCRAP_investment for the fuel_group {fuel_group} is:")
        print(df_ECO_PROC_YCRAP_investment)

        df_ECO_PROC_YCRAP_fixedop = df_ECO_PROC_YCRAP[
            (df_ECO_PROC_YCRAP["Y"] == year) &
            (df_ECO_PROC_YCRAP["C"] == country) &
            (df_ECO_PROC_YCRAP["PROC"].isin(processes)) &
            (df_ECO_PROC_YCRAP["COST_TYPE"] == "FIXED") &
            (df_ECO_PROC_YCRAP["AAA"].isin(valid_cities))
        ]

        print(f"The df_ECO_PROC_YCRAP_fixedop for the fuel_group {fuel_group} is:")
        print(df_ECO_PROC_YCRAP_fixedop)
        
        # VARIABLE O&M COST DATA

        df_FLOWA_variableop = df_FLOWA[
            (df_FLOWA["Y"] == year) & 
            (df_FLOWA["IPROCFROM"].isin(processes)) & 
            (df_FLOWA["FLOW"] == "OPERATIONCOST") &
            (df_FLOWA["AAA"].isin(valid_cities))
        ]
        
        print(f"The df_FLOWA_variableop for the fuel_group {fuel_group} is:")
        print(df_FLOWA_variableop)
        # FEEDSTOCK DATA

        net_consumed_electricity = 0
        net_consumed_h2 = 0
        net_consumed_heat = 0
        net_consumed_biomass = 0

        # FEEDSTOCK DATA

        feedstock_consumption = pd.DataFrame(index=["Electricity", "Hydrogen", "Heat", "Biomass"], columns=processes)

        for proc in processes:
            # Electricity
            consumed_electricity = df_FLOWA.loc[
                (df_FLOWA["Y"] == year) & 
                (df_FLOWA["IPROCTO"] == proc) & 
                (df_FLOWA["IPROCFROM"] == "ElecBuffer_GJ") &
                (df_FLOWA["AAA"].isin(valid_cities))
            ]["value"].sum()

            produced_electricity = df_FLOWA.loc[
                (df_FLOWA["Y"] == year) & 
                (df_FLOWA["IPROCFROM"] == proc) & 
                (df_FLOWA["IPROCTO"] == "EL_Opti_to_Bal_Conv") &
                (df_FLOWA["AAA"].isin(valid_cities))
            ]["value"].sum()

            net_consumed_electricity += consumed_electricity - produced_electricity

            # Hydrogen
            consumed_h2 = df_FLOWA.loc[
                (df_FLOWA["Y"] == year) & 
                (df_FLOWA["IPROCTO"] == proc) & 
                (df_FLOWA["IPROCFROM"] == "Hydrogen_Use") &
                (df_FLOWA["AAA"].isin(valid_cities))
            ]["value"].sum()

            net_consumed_h2 += consumed_h2

            # Heat
            consumed_heat = df_FLOWA.loc[
                (df_FLOWA["Y"] == year) &
                (df_FLOWA["IPROCTO"] == proc) & 
                (df_FLOWA["IPROCFROM"] == "HeatBuffer_GJ") &
                (df_FLOWA["AAA"].isin(valid_cities))
            ]["value"].sum()

            produced_heat = df_FLOWA.loc[
                (df_FLOWA["Y"] == year) & 
                (df_FLOWA["IPROCFROM"] == proc) & 
                (df_FLOWA["IPROCTO"] == "Heat_Opti_to_Bal_Conv") &
                (df_FLOWA["AAA"].isin(valid_cities))
            ]["value"].sum()

            net_consumed_heat += consumed_heat - produced_heat

            # Biomass
            consumed_biomass = df_FLOWA.loc[
                (df_FLOWA["Y"] == year) & 
                (df_FLOWA["IPROCTO"] == proc) & 
                (df_FLOWA["IPROCFROM"] == "Biomass_for_use") &
                (df_FLOWA["AAA"].isin(valid_cities))
            ]["value"].sum()

            net_consumed_biomass += consumed_biomass

            # Fill the DataFrame
            feedstock_consumption.at["Electricity", proc] = consumed_electricity - produced_electricity
            feedstock_consumption.at["Hydrogen", proc] = consumed_h2
            feedstock_consumption.at["Heat", proc] = consumed_heat - produced_heat
            feedstock_consumption.at["Biomass", proc] = consumed_biomass

        # Save feedstock consumption table for this fuel group
        feedstock_csv_path = os.path.join(os.path.dirname(__file__), f"feedstock_consumption_{'_'.join(fuel_group)}.csv")
        feedstock_consumption.to_csv(feedstock_csv_path)

        print(f"Feedstock consumption table for fuel group {fuel_group}:")
        print(feedstock_consumption)

        # FUEL PRODUCTION DATA

        # Filter the fuels list to only those relevant to the current fuel_group
        filtered_pairs = [(proc, flow) for proc, flow in fuels if flow in fuel_group]

        # Create a mask that checks for each (proc, flow) pair
        mask = False
        for proc, flow in filtered_pairs:
            mask |= ((df_FLOWA["IPROCFROM"] == proc) & (df_FLOWA["FLOW"] == flow) & (df_FLOWA["AAA"].isin(valid_cities)))

        # Now select with additional conditions
        yearly_fuel_production = df_FLOWA.loc[
            (df_FLOWA["Y"] == year) &
            mask
        ]["value"].sum()

        # Create a table: rows = 0 to max_lifetime (inclusive), 8 empty columns
        table = pd.DataFrame(
            index=range(max_lifetime + 1),   # +1 to include the last year
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

        for t in range(max_lifetime + 1):
            # TOTAL INVESTMENT COST COLUMN
            if t == 0:
                # For the first year, use the investment cost
                table.at[t, "TIC [M€]"] = df_ECO_PROC_YCRAP_investment["value"].sum()
            else:
                table.at[t, "TIC [M€]"] = 0

            for proc in processes:
                try:
                    # Try to get the lifetime for this process
                    lifetime = df_PROCDATA_lifetime.loc[df_PROCDATA_lifetime["PROC"] == proc, "value"].values[0]
                    # If current year is lifetime + 1, add investment
                    if t == lifetime + 1:
                        investment_value = df_ECO_PROC_YCRAP_investment.loc[df_ECO_PROC_YCRAP_investment["PROC"] == proc, "value"].values[0]
                        table.at[t, "TIC [M€]"] += investment_value
                except IndexError:
                    print(f"Warning: Process {proc} not found in lifetime data.")

            # O&M COST COLUMN
            if t == 0:
                table.at[t, "O&M [M€]"] = 0
            else:
                table.at[t, "O&M [M€]"] = df_ECO_PROC_YCRAP_fixedop["value"].sum() + df_FLOWA_variableop["value"].sum()

            # FUEL COST COLUMN
            if t == 0:
                table.at[t, "FUEL COST [M€]"] = 0
            else:
                table.at[t, "FUEL COST [M€]"] = (net_consumed_electricity * el_price + net_consumed_h2 * h2_price + net_consumed_heat * heat_price + net_consumed_biomass * biomass_price) * 10**-6

            # TOTAL YEARLY COST COLUMN
            table.at[t, "TOT YEARLY COST [M€]"] = table.at[t, "TIC [M€]"] + table.at[t, "O&M [M€]"] + table.at[t, "FUEL COST [M€]"]

            # TOTAL YEARLY COST ACTUALIZED COLUMN
            table.at[t, "TOT YEARLY COST ACT [M€]"] = table.at[t, "TOT YEARLY COST [M€]"] / ((1 + disc_rate) ** t)

            # TOTAL YEARLY FUEL PRODUCTION COLUMN
            if t == 0:
                table.at[t, "YEARLY FUEL PRODUCTION [PJ]"] = 0
            else:
                table.at[t, "YEARLY FUEL PRODUCTION [PJ]"] = yearly_fuel_production

            # TOTAL YEARLY FUEL PRODUCTION ACTUALIZED COLUMN
            table.at[t, "YEARLY FUEL PRODUCTION ACT [PJ]"] = table.at[t, "YEARLY FUEL PRODUCTION [PJ]"] / ((1 + disc_rate) ** t)

        lcof = (table["TOT YEARLY COST ACT [M€]"].sum() / table["YEARLY FUEL PRODUCTION ACT [PJ]"].sum())

        # Save the table for this fuel group
        fuel_lifetime_tables[tuple(fuel_group)] = table
        fuel_lcof[tuple(fuel_group)] = lcof

        print(f"LCOF for fuel group {fuel_group}: {lcof} €/GJ")

        # Save the table as a CSV file
        fuel_group_name = "_".join(fuel_group)  # Create a name from the fuel group
        csv_file_path = os.path.join(os.path.dirname(__file__), f"{fuel_group_name}.csv")
        table.to_csv(csv_file_path, index=False)

    return fuel_lifetime_tables,fuel_lcof




def plot_LCOF_bysector(scenario_path, lcof_fuels, sectors, name_map, year, country, scenario_name):
    """
    Plots the Levelized Cost of Fuel (LCOF) by transport sector for a given scenario, year, and country.
    This function imports fuel flow data, combines specific fuel groups if present, calculates the weighted average LCOF for each sector, 
    and generates a bar plot showing the LCOF for each transport sector. The plot is customized with axis labels, title, and formatting.
    Args:
        scenario_path (str): Path to the scenario data directory.
        lcof_fuels (dict): Dictionary mapping fuel group tuples to their LCOF values.
        sectors (dict): Dictionary mapping sector codes to sector names.
        name_map (dict): Dictionary mapping fuel codes to readable fuel names.
        year (int): The year for which to plot the LCOF.
        country (str): The country code to filter the data.
        scenario_name (str): Name of the scenario to display as the plot title.
    Returns:
        plotly.graph_objs._figure.Figure: The generated Plotly figure object.
    """


    # Import data
    df_FLOWC, _, _, _ = Import_OptiflowMR(scenario_path)

    # Combine specific fuel groups if they exist in lcof_fuels
    group1 = ["BIOGASOLINEFLOW_BJ_H2", "BIOJETFLOW_H2"]
    group2 = ["BIOGASOLINEFLOW_BJ_TG", "BIOJETFLOW_TG"]
    combined_group = ["BIOGASOLINEFLOW", "BIOJETFLOW"]

    if any(fuel in lcof_fuels for fuel in [tuple(group1), tuple(group2)]):
        # Calculate the weighted average LCOF for the combined group
        total_quantity_group1 = df_FLOWC[
            (df_FLOWC["IPROCTO"].isin(["BioGasoline_SUM", "BioJet_SUM"])) &
            (df_FLOWC["FLOW"].isin(group1)) &
            (df_FLOWC["Y"] == year) &
            (df_FLOWC["CCC"] == country)
        ]["value"].sum()

        total_quantity_group2 = df_FLOWC[
            (df_FLOWC["IPROCTO"].isin(["BioGasoline_SUM", "BioJet_SUM"])) &
            (df_FLOWC["FLOW"].isin(group2)) &
            (df_FLOWC["Y"] == year) &
            (df_FLOWC["CCC"] == country)
        ]["value"].sum()

        lcof_group1 = lcof_fuels.get(tuple(group1), 0)
        lcof_group2 = lcof_fuels.get(tuple(group2), 0)

        if math.isnan(lcof_group1) and total_quantity_group1 == 0.0:
            lcof_group1 = 0.0
        if math.isnan(lcof_group2) and total_quantity_group2 == 0.0:
            lcof_group2 = 0.0
        print(f"lcof_group1: {lcof_group1}")
        print(f"lcof_group2: {lcof_group2}")

        total_quantity_combined = total_quantity_group1 + total_quantity_group2
        print(f"Total quantity for group 1: {total_quantity_group1}")
        print(f"Total quantity for group 2: {total_quantity_group2}")
        print(total_quantity_combined)
        if total_quantity_combined > 0:
            combined_lcof = (
                (lcof_group1 * total_quantity_group1) +
                (lcof_group2 * total_quantity_group2)
            ) / total_quantity_combined
        else:
            combined_lcof = None

        print(combined_lcof)

        # Update lcof_fuels with the combined group
        lcof_fuels[tuple(combined_group)] = combined_lcof

        # Remove the original groups from lcof_fuels
        lcof_fuels.pop(tuple(group1), None)
        lcof_fuels.pop(tuple(group2), None)

    # Filter and map
    filtered_df_FLOWC = df_FLOWC[
        (df_FLOWC["Y"] == year) &
        (df_FLOWC["CCC"] == country) &
        (df_FLOWC["IPROCTO"].isin(sectors.keys()))
    ].copy()

    filtered_df_FLOWC["Sector"] = filtered_df_FLOWC["IPROCTO"].map(sectors)
    filtered_df_FLOWC["Fuel"] = filtered_df_FLOWC["IPROCFROM"].map(name_map)

    def assign_lcof(row):
        for fuel_group, lcof_value in lcof_fuels.items():
            if row["Fuel"] in fuel_group:
                return lcof_value * 1e6
        return None

    filtered_df_FLOWC["Fuel LCOF"] = filtered_df_FLOWC.apply(assign_lcof, axis=1)

    # Calculate sectoral LCOF
    sector_lcof = {}
    for sector in filtered_df_FLOWC["Sector"].unique():
        sector_df = filtered_df_FLOWC[filtered_df_FLOWC["Sector"] == sector]
        numerator = (sector_df["value"] * sector_df["Fuel LCOF"]).sum()
        denominator = sector_df["value"].sum() * 1e6
        sector_lcof[sector] = numerator / denominator if denominator != 0 else None

    # Prepare data for plot
    sectors = list(sector_lcof.keys())
    lcof_values = list(sector_lcof.values())

    fig = go.Figure()

    fig.add_trace(go.Bar(
          x=sectors,
          y=lcof_values,
          marker=dict(color='#1baee3'),  # Blueish color
          name="LCOF"
     ))

    fig.update_layout(
        xaxis=dict(
            title="Transport Sector",
            tickangle=0,
            showline=True,
            linewidth=1.2,
            linecolor='black',
            mirror=True,
            tickfont=dict(size=14)
        ),
        yaxis=dict(
            title="LCOF (€/GJ)",
            range=[0, 30],  # Set y-axis range to go up to 20
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='lightgray',
            zerolinewidth=0.6,
            linecolor='black',
            linewidth=1.2,
            mirror=True,
            tickfont=dict(size=12)
        ),
        font=dict(
            family="DejaVu Sans, sans-serif",
            size=14,
            color="black"
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        autosize=False,
        width=800,
        height=600,
        margin=dict(l=100, r=100, t=80, b=80)  # Adjusted top margin for title
    )

    # Add scenario name as a title
    fig.update_layout(
        title=dict(
            text=f"{scenario_name}",
            x=0.5,
            y=0.90,
            xanchor='center',
            yanchor='top',
            font=dict(size=20, family="DejaVu Sans, sans-serif", color="black")
        )
    )

    fig.show()

    return fig


# Normalize keys
def normalize_key(k):
     if isinstance(k, (tuple, list)):
          k = k[0]
     return k.strip().upper()

def plot_lcof_bar(fuels_lcof, normalized_name_map, scenario_name):
    """
    Plots a bar chart of LCOF values for each fuel group, excluding those with LCOF == 0.

    Args:
        fuels_lcof (dict): Dictionary mapping fuel group keys to LCOF values.
        normalized_name_map (dict): Mapping from normalized fuel group keys to readable names.
        scenario_name (str): Title for the plot.

    Returns:
        plotly.graph_objs._figure.Figure: The generated Plotly figure object.
    """
    def normalize_key(k):
        if isinstance(k, (tuple, list)):
            k = k[0]
        return k.strip().upper()

    # Prepare plot data, only include fuels with LCOF != 0 and not None/nan
    x_labels = []
    y_values = []
    bar_colors = []
    legend_labels = []

    for k, v in fuels_lcof.items():
        if v == 0 or v is None or (isinstance(v, float) and (np.isnan(v) or abs(v) < 1e-8)):
            continue
        norm_key = normalize_key(k)

        readable = None
        for key, name in normalized_name_map.items():
            if key in norm_key:
                readable = name
                break
        if not readable:
            print(f"Warning: Unmapped key '{norm_key}' — using fallback label.")
            readable = norm_key.replace("_", " ").title()

        x_labels.append(readable)
        y_values.append(v)

        if any(fossil in norm_key.upper() for fossil in ["KEROSENE", "MDO", "DIESEL"]):
            bar_colors.append("#e31b1b")  # Red
            legend_labels.append("Fossil Fuel Price")
        else:
            bar_colors.append("#e3a41b")  # Orange
            legend_labels.append("LCOF")
            
    # Create Plotly bar chart with custom legend
    fig = go.Figure()

    # Plot LCOF bars (orange)
    for i in range(len(x_labels)):
        if legend_labels[i] == "LCOF":
            fig.add_trace(go.Bar(
                x=[x_labels[i]],
                y=[y_values[i]],
                marker=dict(color=bar_colors[i]),
                name="LCOF",
                showlegend=not any(t.name == "LCOF" for t in fig.data)
            ))
        else:
            fig.add_trace(go.Bar(
                x=[x_labels[i]],
                y=[y_values[i]],
                marker=dict(color=bar_colors[i]),
                name="Fossil Fuel Price",
                showlegend=not any(t.name == "Fossil Fuel Price" for t in fig.data)
            ))

    fig.update_layout(
        xaxis=dict(
            title="Fuel Group",
            tickangle=0,
            showline=True,
            linewidth=1,
            linecolor='black',
            tickfont=dict(size=14)
        ),
        yaxis=dict(
            title="[€/GJ]",
            range=[0, 30],  # Set y-axis range to go up to 30
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='lightgray',
            zerolinewidth=0.6,
            linecolor='black',
            linewidth=1,
            tickfont=dict(size=12)
        ),
        font=dict(
            family="DejaVu Sans, sans-serif",
            size=14,
            color="black"
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=100, r=220, t=100, b=100),  # Extra right margin for legend
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.05,
            font=dict(size=14),
            bordercolor="black",
            borderwidth=2,
            bgcolor="white"
        ),
        autosize=False,
        width=900,
        height=600,
    )

    # Add a full rectangle border around the plot
    fig.update_layout(
        xaxis=dict(showline=True, mirror=True),
        yaxis=dict(showline=True, mirror=True),
    )

    # Add scenario name as a title
    fig.update_layout(
        title=dict(
            text=f"{scenario_name}",
            x=0.5,
            y=0.90,
            xanchor='center',
            yanchor='top',
            font=dict(size=20, family="DejaVu Sans, sans-serif", color="black")
        )
    )

    fig.show()
    return fig


# Calculate LCOF and store results
fuel_lifetime_tables,fuels_lcof = LCOF_calculation(
    scenario[0][1], 
    fuels, 
    fuel_to_processes, 
    "2050", 
    "DENMARK"
)

# Overwrite LCOF for fossil fuels if present in fuel_to_processes
for fuel_group, _ in fuel_to_processes:
    for fossil_fuel, lcof_value in fossil_fuels_costs.items():
        # Import df_FLOWC to check if fossil_fuel is present in the 'FLOW' column
        df_FLOWC, _, _, _ = Import_OptiflowMR(scenario[0][1])
        if fossil_fuel in df_FLOWC['FLOW'].values:
            # Assign the fixed value for this fuel group
            fuels_lcof[tuple([fossil_fuel])] = lcof_value


fig = plot_lcof_bar(fuels_lcof,
                    normalized_name_map, 
                    scenario[0][0])


if save_pdf=='True':
    fig.write_image(rf"C:\Users\sigur\OneDrive\DTU\Pictures for report polimi\Results\FuelsLCOE_{scenario[0][0]}.pdf", engine='kaleido')

fig= plot_LCOF_bysector(
    scenario[0][1], 
    fuels_lcof, 
    sectors, 
    name_map, 
    "2050", 
    "DENMARK", 
    scenario[0][0]
)

if save_pdf=='True':
    fig.write_image(rf"C:\Users\sigur\OneDrive\DTU\Pictures for report polimi\Results\LCOF_bysector_{scenario[0][0]}.pdf", engine='kaleido')


