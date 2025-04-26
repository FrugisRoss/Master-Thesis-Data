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

def Import_OptiflowMR(file_path):
    main_results_path = os.path.join(file_path, "Optiflow_MainResults.gdx")
    df = gt.Container(main_results_path)
    df_FLOWC = pd.DataFrame(df["VFLOW_Opti_C"].records)
    df_ECO_PROC_YCRAP = pd.DataFrame(df["ECO_PROC_YCRAP"].records)

    return df_FLOWC, df_ECO_PROC_YCRAP

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
     denominator = merged['value_demand'].sum()*10**6
     
     if denominator == 0:
          print("Denominator is zero, cannot compute average price.")  
     else:
          avg_price = numerator / denominator
     
     print(f"Average {commodity} price in {country} for year {year}: {avg_price} €/MWh")

     return avg_price

Avg_yearly_price(r"C:\Users\sigur\OneDrive\DTU\Run on HPC Polimi\Base_Case_RightOut\model", "Electricity", "2050" , "DENMARK")
Avg_yearly_price(r"C:\Users\sigur\OneDrive\DTU\Run on HPC Polimi\Base_Case_RightOut\model", "H2", "2050" , "DENMARK")
Avg_yearly_price(r"C:\Users\sigur\OneDrive\DTU\Run on HPC Polimi\Base_Case_RightOut\model", "Heat", "2050" , "DENMARK")

          

    
    