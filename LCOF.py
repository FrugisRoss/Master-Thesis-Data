#%%
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


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)



scenario = [
     ("Base Case", r"C:\Users\sigur\OneDrive\DTU\Run on HPC Polimi\Base_Case_RightOut\model")]

fuels = [
    ("Ammonia_Synthesis_50", "AMMONIA_FLOW"),
    ("BioJet_H2_50", "BIOGASOLINEFLOW_BJ_H2"),
    ("BioJet_H2_50", "BIOJETFLOW_H2"),
    ("EME_Upgrade_Sum", "EME_GASOLINE_FLOW"),
    ("EME_Upgrade_Sum", "EME_JET_FLOW"),
    ("EME_Upgrade_Sum", "EME_LPG_FLOW")
]

fuel_to_processes = [
    (["AMMONIA_FLOW"], ["Nitrogen_Production", "Ammonia_Synthesis_50"]),
    (["BIOGASOLINEFLOW_BJ_H2","BIOJETFLOW_H2"], ["BioJet_H2_50"]),
    (["EME_GASOLINE_FLOW", "EME_JET_FLOW","EME_LPG_FLOW"], ["EMethanol_synthesis_50", "EMethanol_Upgrade_50", "CO2_Sto", "CO2_DAC_50"]),

]

def Import_allendofmodel(file_path):
     main_results_path = os.path.join(file_path, "all_endofmodel.gdx")
     df = gt.Container(main_results_path)
     
     df_SOSIBU2INDIC = pd.DataFrame(df["SOSIBU2INDIC"].records)
     df_TRANSCOST = pd.DataFrame(df["TRANSCOST"].records)
     df_PROCDATA = pd.DataFrame(df["PROCDATA"].records)
     df_DISCOUNTRATE = pd.DataFrame(df["DISCOUNTRATE"].records)
     df_TRANSDIST = pd.DataFrame(df["TRANSDIST"].records)
     
     return df_SOSIBU2INDIC, df_TRANSCOST, df_PROCDATA, df_DISCOUNTRATE, df_TRANSDIST


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
     for a given year and country, weighted by the corresponding demand in €/PJ.
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
     
     #print(merged)

     # Compute the weighted average
     numerator = (merged['value_price'] * merged['value_demand']*10**6).sum()
     denominator = merged['value_demand'].sum()
     
     if denominator == 0:
          print("Denominator is zero, cannot compute average price.")  
     else:
          avg_price = numerator / denominator
     
     print(f"Average {commodity} price in {country} for year {year}: {avg_price} €/PJ")

     return avg_price

# Avg_yearly_price(r"C:\Users\sigur\OneDrive\DTU\Run on HPC Polimi\Base_Case_RightOut\model", "Electricity", "2050" , "DENMARK")
# Avg_yearly_price(r"C:\Users\sigur\OneDrive\DTU\Run on HPC Polimi\Base_Case_RightOut\model", "H2", "2050" , "DENMARK")
# Avg_yearly_price(r"C:\Users\sigur\OneDrive\DTU\Run on HPC Polimi\Base_Case_RightOut\model", "Heat", "2050" , "DENMARK")

          
def Avg_yearly_biomass_price (scenario_path, year, country):
    
    """
          Calculate the average yearly biomass price for a specific country and year in €/PJ.
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

    df_FLOWC,_, df_VFLOWTRANS_Opti_A =Import_OptiflowMR(scenario_path)
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

#avg_price= Avg_yearly_biomass_price(r"C:\Users\sigur\OneDrive\DTU\Run on HPC Polimi\Base_Case_RightOut\model",  "2050" , "DENMARK")

#%%
def LCOF_calculation(scenario_path, fuels, fuel_to_processes, year, country):
    # Import data
    df_FLOWC, df_ECO_PROC_YCRAP, _ = Import_OptiflowMR(scenario_path)
    _, _, df_PROCDATA, df_DISCOUNTRATE, _ = Import_allendofmodel(scenario_path)

    # Dictionary to store results
    fuel_lifetime_tables = {}
    fuel_lcof = {}


    el_price=Avg_yearly_price(r"C:\Users\sigur\OneDrive\DTU\Run on HPC Polimi\Base_Case_RightOut\model", "Electricity", "2050" , "DENMARK")
    h2_price=Avg_yearly_price(r"C:\Users\sigur\OneDrive\DTU\Run on HPC Polimi\Base_Case_RightOut\model", "H2", "2050" , "DENMARK")
    heat_price=Avg_yearly_price(r"C:\Users\sigur\OneDrive\DTU\Run on HPC Polimi\Base_Case_RightOut\model", "Heat", "2050" , "DENMARK")
    biomass_price=Avg_yearly_biomass_price(r"C:\Users\sigur\OneDrive\DTU\Run on HPC Polimi\Base_Case_RightOut\model",  "2050" , "DENMARK")

    #DISCOUNT RATE
    disc_rate= df_DISCOUNTRATE.values[0]


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

        #INVESTMENT COST DATA

        df_ECO_PROC_YCRAP_investment= df_ECO_PROC_YCRAP[
               (df_ECO_PROC_YCRAP["Y"] == year) & 
               (df_ECO_PROC_YCRAP["C"] == country) & 
               (df_ECO_PROC_YCRAP["PROC"].isin(processes)) & 
               (df_ECO_PROC_YCRAP["COST_TYPE"] == "INVESTMENT_NA")
          ]
        
        #FIXED O&M COST DATA

        df_ECO_PROC_YCRAP_fixedop= df_ECO_PROC_YCRAP[
               (df_ECO_PROC_YCRAP["Y"] == year) & 
               (df_ECO_PROC_YCRAP["C"] == country) & 
               (df_ECO_PROC_YCRAP["PROC"].isin(processes)) & 
               (df_ECO_PROC_YCRAP["COST_TYPE"] == "FIXED")
          ]
        
        #VARIABLE O&M COST DATA

        df_FLOWC_variableop= df_FLOWC[
               (df_FLOWC["Y"] == year) & 
               (df_FLOWC["CCC"] == country) & 
               (df_FLOWC["IPROCFROM"].isin(processes)) & 
               (df_FLOWC["FLOW"] == "OPERATIONCOST")
          ]
        
        #FEEDSTOCK DATA

        net_consumed_electricity= 0
        net_consumed_h2=0
        net_consumed_heat=0
        net_consumed_biomass=0

        # FEEDSTOCK DATA

        feedstock_consumption = pd.DataFrame(index=["Electricity", "Hydrogen", "Heat", "Biomass"], columns=processes)

        for proc in processes:
            # Electricity
            consumed_electricity = df_FLOWC.loc[
                (df_FLOWC["Y"] == year) & 
                (df_FLOWC["CCC"] == country) & 
                (df_FLOWC["IPROCTO"] == proc) & 
                (df_FLOWC["IPROCFROM"] == "ElecBuffer_GJ")
            ]["value"].sum()

            produced_electricity = df_FLOWC.loc[
                (df_FLOWC["Y"] == year) & 
                (df_FLOWC["CCC"] == country) & 
                (df_FLOWC["IPROCFROM"] == proc) & 
                (df_FLOWC["IPROCTO"] == "EL_Opti_to_Bal_Conv")
            ]["value"].sum()

            net_consumed_electricity += consumed_electricity - produced_electricity

            # Hydrogen
            consumed_h2 = df_FLOWC.loc[
                (df_FLOWC["Y"] == year) & 
                (df_FLOWC["CCC"] == country) & 
                (df_FLOWC["IPROCTO"] == proc) & 
                (df_FLOWC["IPROCFROM"] == "Hydrogen_Use")
            ]["value"].sum()

            net_consumed_h2 += consumed_h2

            # Heat
            consumed_heat = df_FLOWC.loc[
                (df_FLOWC["Y"] == year) & 
                (df_FLOWC["CCC"] == country) & 
                (df_FLOWC["IPROCTO"] == proc) & 
                (df_FLOWC["IPROCFROM"] == "HeatBuffer_GJ")
            ]["value"].sum()

            produced_heat = df_FLOWC.loc[
                (df_FLOWC["Y"] == year) & 
                (df_FLOWC["CCC"] == country) & 
                (df_FLOWC["IPROCFROM"] == proc) & 
                (df_FLOWC["IPROCTO"] == "Heat_Opti_to_Bal_Conv")
            ]["value"].sum()

            net_consumed_heat += consumed_heat - produced_heat

            # Biomass
            consumed_biomass = df_FLOWC.loc[
                (df_FLOWC["Y"] == year) & 
                (df_FLOWC["CCC"] == country) & 
                (df_FLOWC["IPROCTO"] == proc) & 
                (df_FLOWC["IPROCFROM"] == "Biomass_for_use")
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

        #FUEL PRODUCTION DATA

        # Filter the fuels list to only those relevant to the current fuel_group
        filtered_pairs = [(proc, flow) for proc, flow in fuels if flow in fuel_group]

        # Create a mask that checks for each (proc, flow) pair
        mask = False
        for proc, flow in filtered_pairs:
               mask |= ((df_FLOWC["IPROCFROM"] == proc) & (df_FLOWC["FLOW"] == flow))

        # Now select with additional conditions
        yearly_fuel_production = df_FLOWC.loc[
        (df_FLOWC["Y"] == year) &
        (df_FLOWC["CCC"] == country) &
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
             
             #TOTAL INVESTMENT COST COLUMN
             
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
             
             #O&M COST COLUMN

             if t==0:
                  table.at[t, "O&M [M€]"] = 0
             else:
                  table.at[t, "O&M [M€]"] = df_ECO_PROC_YCRAP_fixedop["value"].sum() + df_FLOWC_variableop["value"].sum()

            #FUEL COST COLUMN
             if t==0:
                 table.at[t, "FUEL COST [M€]"] = 0
             else:
                  table.at[t, "FUEL COST [M€]"] = (net_consumed_electricity * el_price + net_consumed_h2 * h2_price + net_consumed_heat * heat_price + net_consumed_biomass * biomass_price) * 10**-6

            #TOTAL YEARLY COST COLUMN
                 
             table.at[t, "TOT YEARLY COST [M€]"] = table.at[t, "TIC [M€]"] + table.at[t, "O&M [M€]"] + table.at[t, "FUEL COST [M€]"]
            
            #TOTAL YEARLY COST ACTUALIZED COLUMN

             table.at[t, "TOT YEARLY COST ACT [M€]"] = table.at[t, "TOT YEARLY COST [M€]"] / ((1 + disc_rate) ** t)

            #TOTAL YEARLY FUEL PRODUCTION COLUMN
             if t==0:
                  table.at[t, "YEARLY FUEL PRODUCTION [PJ]"] = 0
             else:
                  table.at[t, "YEARLY FUEL PRODUCTION [PJ]"] = yearly_fuel_production 

            #TOTAL YEARLY FUEL PRODUCTION ACTUALIZED COLUMN
             table.at[t, "YEARLY FUEL PRODUCTION ACT [PJ]"] = table.at[t, "YEARLY FUEL PRODUCTION [PJ]"] / ((1 + disc_rate) ** t)

        lcof= (table["TOT YEARLY COST ACT [M€]"].sum() / table["YEARLY FUEL PRODUCTION ACT [PJ]"].sum()) 

        # Save the table for this fuel group
        fuel_lifetime_tables[tuple(fuel_group)] = table
        fuel_lcof[tuple(fuel_group)] = lcof

        
     #    print(f"Table for fuel group {fuel_group} (max lifetime {max_lifetime} years):")
        
     #    print(table)

        # Save the table as a CSV file
        fuel_group_name = "_".join(fuel_group)  # Create a name from the fuel group
        csv_file_path = os.path.join(os.path.dirname(__file__), f"{fuel_group_name}.csv")
        table.to_csv(csv_file_path, index=False)

        

        print(f"LCOF for fuel group {fuel_group}: {lcof} €/GJ")


    return fuel_lifetime_tables, fuel_lcof


#%%

# Calculate LCOF and store results
first_try_table, first_try_lcof = LCOF_calculation(
     r"C:\Users\sigur\OneDrive\DTU\Run on HPC Polimi\Base_Case_RightOut\model", 
     fuels, 
     fuel_to_processes, 
     "2050", 
     "DENMARK"
)

fuel_group_name_map = {
     ("AMMONIA_FLOW",): "Ammonia",
     ("BIOGASOLINEFLOW_BJ_H2", "BIOJETFLOW_H2"): "Biofuels (Jet/Gasoline via H₂)",
     ("EME_GASOLINE_FLOW", "EME_JET_FLOW", "EME_LPG_FLOW"): "E-Methanol Derived Fuels",
}

# Normalized name map (can expand this as needed)
normalized_name_map = {
      "AMMONIA_FLOW": "Ammonia",
      "BIOGASOLINEFLOW_BJ_H2": "Biofuels (Jet/Gasoline via H₂)",
      "EME_GASOLINE_FLOW": "E-Methanol Derived Fuels",
}
# Normalize keys
def normalize_key(k):
      if isinstance(k, (tuple, list)):
            k = k[0]
      return k.strip().upper()

# Prepare plot data
x_labels = []
y_values = []

for k, v in first_try_lcof.items():
      norm_key = normalize_key(k)
      readable = normalized_name_map.get(norm_key)
      if not readable:
            print(f"⚠️ Warning: Unmapped key '{norm_key}' — using fallback label.")
            readable = norm_key.replace("_", " ").title()
      x_labels.append(readable)
      y_values.append(v)

# Create Plotly bar chart
fig = go.Figure()

fig.add_trace(go.Bar(
      x=x_labels,
      y=y_values,
      marker=dict(color='skyblue'),
      name="LCOF"
))

fig.update_layout(
      title="Levelized Cost of Fuel (LCOF) per Fuel Group",
      xaxis=dict(
            title="Fuel Group",
            tickangle=0,  # Ensure labels are not tilted
            showline=True,
            linewidth=1,
            linecolor='black',
            tickfont=dict(size=12)
      ),
      yaxis=dict(
            title="LCOF (€/GJ)",
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
      margin=dict(l=60, r=30, t=80, b=80),
      showlegend=False,  # Ensure the legend is displayed
)

# Add a full rectangle border around the plot
fig.update_layout(
      xaxis=dict(showline=True, mirror=True),  # Mirror x-axis lines
      yaxis=dict(showline=True, mirror=True),  # Mirror y-axis lines
)


fig.show()