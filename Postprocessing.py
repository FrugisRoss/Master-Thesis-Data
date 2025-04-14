
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import gams.transfer as gt
#import plots_pybalmorel as pyb
import gams
import os
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.graph_objects as go


#%%
# ------------------------------------------------------------------------------
# A) DEFINE YOUR SCENARIOS
# Each scenario is a tuple: (scenario_name, scenario_path)
# ------------------------------------------------------------------------------
scenario_list = [
     ("Base Case", r"C:\Users\sigur\OneDrive\DTU\Run on HPC Polimi\Base_Case_ModOut\model"),
     ("Base Case DK", r"C:\Users\sigur\OneDrive\DTU\Run on HPC Polimi\Base_Case_DK_ModOut\model"),
     ("CO2 Scenario", r"C:\Users\sigur\OneDrive\DTU\Run on HPC Polimi\CO2_Case_RLC_ModOut\model"),
     ("CO2 Scenario DK", r"C:\Users\sigur\OneDrive\DTU\Run on HPC Polimi\CO2_Case_RLC_DK_ModOut\model"),
     ("Biodiversity+CO2 Scenario", r"C:\Users\sigur\OneDrive\DTU\Run on HPC Polimi\Biodiversity_Case_RLC_ModOut\model"),
     
     
     ("Biodiversity+CO2 Scenario DK", r"C:\Users\sigur\OneDrive\DTU\Run on HPC Polimi\Biodiversity_Case_RLC_DK_ModOut\model"),     
     #("Biodiversity+CO2 Fossil ", r"C:\Users\sigur\OneDrive\DTU\Run on HPC Polimi\Biodiversity_Case_RLC_FOSSIL\model"),
    

]

Resource_name = {
    'Biomass_for_use': 'Biomass',
    'Hydrogen_Use': 'Hydrogen'
}

Demands_name = {
    'Sea_fuels_sum': 'Maritime Demand',
    'Road_fuels_sum': 'Road Demand',
    'Air_fuels_sum': 'Aviation Demand'
}

Fuels_name = {
    'Ammonia_Eff': 'Ammonia',
    'Methanol_Eff': 'Methanol',
    'EMethanol_Eff': 'E-Methanol',
    'EME_Gasoline_Eff': 'E-Gasoline',
    'EME_LPG_Eff': 'E-LPG',
    'BioGasoline_Eff': 'Biogasoline',
    'E_FT_Diesel_Eff': 'E-Diesel',
    'BioJet_Eff': 'Biojet',
    'E_FT_Jet_Eff': 'E-Jet-FT',
    'EME_Jet_Eff': 'E-Jet-ME',
    'MDOSource': 'Marine Shipping Diesel Oil',
    'KeroseneSource': 'Kerosene',
    'DieselSource': 'Diesel'
}

# ------------------------------------------------------------------------------
# B) LOADING FUNCTIONS
# ------------------------------------------------------------------------------
def Import_OptiflowMR(file_path):
    df = gt.Container(file_path)
    df_FLOWA = pd.DataFrame(df["VFLOW_Opti_A"].records)
    df_FLOWC = pd.DataFrame(df["VFLOW_Opti_C"].records)
    df_EMI_YCRAG = pd.DataFrame(df["EMI_YCRAG"].records)
    df_EMI_PROC = pd.DataFrame(df["EMI_PROC"].records)
    df_VOBJ = pd.DataFrame(df["VOBJ"].records)
    return df_FLOWA, df_FLOWC, df_EMI_YCRAG, df_EMI_PROC, df_VOBJ

def Import_BalmorelMR(file_path):
    df = gt.Container(file_path)
    df_CC_YCRAG = pd.DataFrame(df["CC_YCRAG"].records)
    df_F_CONS_YCRA = pd.DataFrame(df["F_CONS_YCRA"].records)
    df_EMI_YCRAG = pd.DataFrame(df["EMI_YCRAG"].records)
    df_G_CAP_YCRAF = pd.DataFrame(df["G_CAP_YCRAF"].records)
    df_PRO_YCRAGF = pd.DataFrame(df["PRO_YCRAGF"].records)
    df_OBJ_YCR = pd.DataFrame(df["OBJ_YCR"].records)
    df_EL_DEMAND_YCR = pd.DataFrame(df["EL_DEMAND_YCR"].records)
    df_H2_DEMAND_YCR = pd.DataFrame(df["H2_DEMAND_YCR"].records)
    df_H_DEMAND_YCRA = pd.DataFrame(df["H_DEMAND_YCRA"].records)
    df_X_FLOW_YCR = pd.DataFrame(df["X_FLOW_YCR"].records)
    df_XH2_FLOW_YCR = pd.DataFrame(df["XH2_FLOW_YCR"].records)
    df_XH_FLOW_YCA = pd.DataFrame(df["XH_FLOW_YCA"].records)
    return df_CC_YCRAG, df_F_CONS_YCRA, df_EMI_YCRAG, df_G_CAP_YCRAF, df_PRO_YCRAGF, df_OBJ_YCR, df_EL_DEMAND_YCR, df_H_DEMAND_YCRA, df_H2_DEMAND_YCR, df_X_FLOW_YCR, df_XH2_FLOW_YCR, df_XH_FLOW_YCA

# ------------------------------------------------------------------------------
# C) DATA-PROCESSING FUNCTIONS
# ------------------------------------------------------------------------------
def group_EMI_YCRAG(df_EMI_YCRAG, df_FLOWC, df_EMI_PROC):
    df_emi = df_EMI_YCRAG.copy()
    df_emi['Category'] = None

    # Define masks for categorization in df_EMI_YCRAG
    mask_ind_emi  = df_emi['AAA'].str.contains('IND',  case=False, na=False)
    mask_idvu_emi = df_emi['AAA'].str.contains('IDVU', case=False, na=False)
    mask_chp_emi  = df_emi['AAA'].str.contains('DK1_A|DK2_A', case=False, na=False)
    mask_positive = df_emi['value'] > 0
    mask_negative = df_emi['value'] < 0

    # Positive emissions
    df_emi.loc[mask_positive & mask_ind_emi,  'Category'] = 'Industry Heating'
    df_emi.loc[mask_positive & mask_idvu_emi, 'Category'] = 'Individual Users Heating'
    df_emi.loc[mask_positive & mask_chp_emi,  'Category'] = 'CHP Generation'

    # Negative emissions (BECCS)
    df_emi.loc[mask_negative & mask_ind_emi,  'Category'] = 'Industry with BECCS'
    df_emi.loc[mask_negative & mask_idvu_emi, 'Category'] = 'Individual with BECCS'
    df_emi.loc[mask_negative & mask_chp_emi,  'Category'] = 'CHP with BECCS'

    # Define the mapping based on IPROCFROM
    co2_map = {
        'CO2_DAC_Total': 'DAC',
        'CO2_Biochar_Sum': 'Biochar Sequestration',
        'Untouched_Forest_HOV': 'New Protected Forest',
        'Untouched_Forest_SJA': 'New Protected Forest',
        'Untouched_Forest_SYD': 'New Protected Forest',
        'Untouched_Forest_MID': 'New Protected Forest',
        'Untouched_Forest_NOR': 'New Protected Forest',
        'C_Rich_Soils_Extraction_HOV': 'Carbon Rich Soils Extraction',
        'C_Rich_Soils_Extraction_SJA': 'Carbon Rich Soils Extraction',
        'C_Rich_Soils_Extraction_SYD': 'Carbon Rich Soils Extraction',
        'C_Rich_Soils_Extraction_MID': 'Carbon Rich Soils Extraction',
        'C_Rich_Soils_Extraction_NOR': 'Carbon Rich Soils Extraction',
        'New_Productive_Forest_HOV': 'New Productive Forest',
        'New_Productive_Forest_SJA': 'New Productive Forest',
        'New_Productive_Forest_SYD': 'New Productive Forest',
        'New_Productive_Forest_MID': 'New Productive Forest',
        'New_Productive_Forest_NOR': 'New Productive Forest'
    }

    # Define list of tuples (IPROCFROM, IPROCTO_filter)
    # For tuples where the second element is None, only IPROCFROM is checked.
    # Otherwise, the IPROCTO column must contain the provided substring (case-insensitive).
    co2_tuples = [
        ('CO2_Biochar_Sum', 'CO2_Biochar_Buffer'),
        ('Untouched_Forest_HOV', 'CO2_Land'),
        ('Untouched_Forest_SJA', 'CO2_Land'),
        ('Untouched_Forest_SYD', 'CO2_Land'),
        ('Untouched_Forest_MID', 'CO2_Land'),
        ('Untouched_Forest_NOR', 'CO2_Land'),
        ('C_Rich_Soils_Extraction_HOV', 'CO2_Land'),
        ('C_Rich_Soils_Extraction_SJA', 'CO2_Land'),
        ('C_Rich_Soils_Extraction_SYD', 'CO2_Land'),
        ('C_Rich_Soils_Extraction_MID', 'CO2_Land'),
        ('C_Rich_Soils_Extraction_NOR', 'CO2_Land'),
        ('New_Productive_Forest_HOV', 'CO2_Land'),
        ('New_Productive_Forest_SJA', 'CO2_Land'),
        ('New_Productive_Forest_SYD', 'CO2_Land'),
        ('New_Productive_Forest_MID', 'CO2_Land'),
        ('New_Productive_Forest_NOR', 'CO2_Land'),
        # For DAC rows, filter on IPROCTO containing 'CO2_Seq'
        ('CO2_DAC_Total', 'CO2_Seq_50')
    ]

    # Function to check if a row in df_FLOWC matches any tuple criteria.
    def matches_tuple(row):
        for iprocfrom, iprocto_filter in co2_tuples:
            if row['IPROCFROM'] == iprocfrom:
                if iprocto_filter is None:
                    return True
                # Check if IPROCTO contains the substring (case-insensitive)
                if pd.notnull(row['IPROCTO']) and iprocto_filter.lower() in row['IPROCTO'].lower():
                    return True
        return False

    # Apply the filter on df_FLOWC using the tuple criteria
    df_co2 = df_FLOWC[df_FLOWC.apply(matches_tuple, axis=1)].copy()
    # Map the category based on IPROCFROM using co2_map
    df_co2['Category'] = df_co2['IPROCFROM'].map(co2_map)
    df_co2['value'] = df_co2['value'] * -1000  # Convert from Mt to kton and apply negative sign

    # Process df_EMI_PROC as before for fuel sources
    if df_EMI_PROC.empty or 'PROC' not in df_EMI_PROC.columns:
        print("Warning: df_EMI_PROC is empty or missing 'PROC' column.")
        df_fuel = pd.DataFrame(columns=['Category', 'value'])  # Empty DataFrame
    else:
        fuel_map = {
            'MDOSource': 'Marine Shipping Diesel Oil',
            'KeroseneSource': 'Kerosene',
            'DieselSource': 'Diesel'
        }
        fuel_list = ['MDOSource', 'KeroseneSource', 'DieselSource']
        df_fuel = df_EMI_PROC[df_EMI_PROC['PROC'].isin(fuel_list)].copy()
        df_fuel['Category'] = df_fuel['PROC'].map(fuel_map)

    # Combine all dataframes
    df_final = pd.concat([df_emi, df_co2, df_fuel], ignore_index=True)
    # Aggregate by category
    df_agg = df_final.groupby('Category', as_index=False)['value'].sum()

    return df_agg

def process_flows_and_consumption(df_FLOWC, df_F_CONS_YCRA):
    # Filter df_FLOWC
    mask_iprocto = df_FLOWC['IPROCTO'].isin(['Straw_for_Energy','Wood_for_Energy'])
    mask_iprocfrom_flow = (
        df_FLOWC['IPROCFROM'].isin(['Straw_for_Energy', 'Wood_for_Energy'])
        & df_FLOWC['FLOW'].isin(['STRAW_FLOW', 'WOOD_FLOW'])
    )
    mask_iprocfrom_pellets = (df_FLOWC['IPROCFROM'] == 'Wood_Pellets_Gen')

    df_flowc_filtered = df_FLOWC[mask_iprocto | mask_iprocfrom_flow | mask_iprocfrom_pellets].copy()

    # Filter df_F_CONS_YCRA
    df_f_cons_filtered = df_F_CONS_YCRA[
        (df_F_CONS_YCRA['C'] == 'DENMARK') &
        (df_F_CONS_YCRA['FFF'].isin(['WOODCHIPS','STRAW','WOODPELLETS']))
    ].copy()
    df_f_cons_filtered['value'] *= 3.6

    # Assign categories
    df_f_cons_filtered['Category'] = 'CHP Generation'
    mask_ind = df_f_cons_filtered['AAA'].str.contains('IND', case=False, na=False)
    mask_idv = df_f_cons_filtered['AAA'].str.contains('IDVU', case=False, na=False)
    df_f_cons_filtered.loc[mask_ind,'Category'] = 'Industry Heating'
    df_f_cons_filtered.loc[mask_idv,'Category'] = 'Individual Users Heating'

    return df_flowc_filtered, df_f_cons_filtered

# ------------------------------------------------------------------------------
# D) MULTI-SCENARIO PLOTTING FUNCTIONS (EXISTING PLOTS)
# ------------------------------------------------------------------------------
def multi_scenario_fuel_supply(
    scenarios,
    Demands_name,
    Fuels_name,
    Resource_name,
    year,
    plot_title="Fuel Demand Supply"
):
    fig = make_subplots(
        rows=1,
        cols=len(scenarios),
        shared_yaxes=True,
        subplot_titles=[s[0] for s in scenarios]
    )

    color_map = {
        'Ammonia': '#1f77b4',
        'Methanol': '#ff7f0e',
        'E-Methanol': '#2ca02c',
        'E-Gasoline': '#e31087',
        'E-LPG': '#9467bd',
        'Biogasoline': '#8c564b',
        'E-Diesel': '#e377c2',
        'Biojet': '#7f7f7f',
        'E-Jet-FT': '#bcbd22',
        'E-Jet-ME': '#17becf',
        'Biomass': '#006600',
        'Hydrogen': '#009999',
        'CO2': '#4D4D4D',
        'Marine Shipping Diesel Oil': '#4D4D4D',
        'Kerosene':  '#d90d17',
        'Diesel': '#fcd90f',
    }
    fallback_color = '#333333'

    # A set to track which legend names have been shown
    encountered_legends = set()

    for idx, (scenario_name, scenario_path) in enumerate(scenarios):
        main_results_path = os.path.join(scenario_path, "MainResults.gdx")
        optiflow_path     = os.path.join(scenario_path, "Optiflow_MainResults.gdx")

        # Unpack all four returned values
        df_FLOWA, df_FLOWC, df_EMI_YCRAG, df_EMI_PROC, df_VOBJ = Import_OptiflowMR(optiflow_path)
        df_CC_YCRAG, df_F_CONS_YCRA, df_EMI_YCRAG, df_G_CAP_YCRAF, df_PRO_YCRAGF, df_OBJ_YCR, df_EL_DEMAND_YCR, df_H_DEMAND_YCRA, df_H2_DEMAND_YCR, df_X_FLOW_YCR, df_XH2_FLOW_YCR, df_XH_FLOW_YCA = Import_BalmorelMR(main_results_path) 

        # Filter by year
        df_FLOWC = df_FLOWC[df_FLOWC['Y'] == str(year)]

        # 1) Fuel supply
        filtered_df = df_FLOWC[
            df_FLOWC['IPROCTO'].isin(Demands_name.keys()) &
            df_FLOWC['IPROCFROM'].isin(Fuels_name.keys())
        ].copy()
        filtered_df['Demand'] = filtered_df['IPROCTO'].map(Demands_name)
        filtered_df['Fuel']   = filtered_df['IPROCFROM'].map(Fuels_name)
        aggregated_df = filtered_df.groupby(['Demand','Fuel'])['value'].sum().reset_index()

        # 2) Resource use
        resource_df = df_FLOWC[df_FLOWC['IPROCFROM'].isin(Resource_name.keys())].copy()
        resource_df['Resource'] = resource_df['IPROCFROM'].map(Resource_name)
        aggregated_resource_df = resource_df.groupby('Resource')['value'].sum().reset_index()

        # Build x-axis categories
        demands_unique   = list(aggregated_df['Demand'].unique())
        resources_unique = list(aggregated_resource_df['Resource'].unique())
        x_categories = demands_unique + [""] + resources_unique

        # Demand->Fuel map
        demand_fuel_map = {d: {} for d in demands_unique}
        for row in aggregated_df.itertuples():
            demand_fuel_map[row.Demand][row.Fuel] = row.value

        # Resource map
        resource_map = {}
        for row in aggregated_resource_df.itertuples():
            resource_map[row.Resource] = row.value

        all_fuels = sorted(list(aggregated_df['Fuel'].unique()))
        c_col = idx+1

        # --- Add bar traces for each Fuel
        for fuel in all_fuels:
            y_vals = []
            for xcat in x_categories:
                if xcat in demands_unique:
                    y_vals.append(demand_fuel_map[xcat].get(fuel, 0.0))
                else:
                    y_vals.append(0.0)

            # Decide if we show the legend for this fuel
            show_legend = (fuel not in encountered_legends)
            if show_legend:
                encountered_legends.add(fuel)

            fig.add_trace(
                go.Bar(
                    x=x_categories,
                    y=y_vals,
                    name=fuel,
                    marker_color=color_map.get(fuel, fallback_color),
                    showlegend=show_legend,
                ),
                row=1, col=c_col
            )

        # --- Add bar traces for resources (negative)
        for resource in resources_unique:
            y_vals = []
            for xcat in x_categories:
                if xcat == resource:
                    y_vals.append(-resource_map[resource])
                else:
                    y_vals.append(0)

            show_legend = (resource not in encountered_legends)
            if show_legend:
                encountered_legends.add(resource)

            fig.add_trace(
                go.Bar(
                    x=x_categories,
                    y=y_vals,
                    name=resource,
                    marker_color=color_map.get(resource, fallback_color),
                    showlegend=show_legend,
                ),
                row=1, col=c_col
            )

        # Axis styling
        fig.update_xaxes(
            title_text='',
            showline=True,
            mirror=True,
            linewidth=1,
            linecolor='black',
            showgrid=False,
            row=1, col=c_col
        )

        y_title = '[PJ]' if idx==0 else ''
        fig.update_yaxes(
            title_text=y_title,
            showline=True,
            mirror=True,
            linewidth=1,
            linecolor='black',
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=0.6,
            zeroline=True,
            zerolinewidth=0.6,
            zerolinecolor='lightgray',
            tickmode='linear',
            tick0=0,
            dtick=25,
            row=1, col=c_col
        )
    
    # Layout
    fig.update_layout(
        title=plot_title,
        barmode='stack',
        bargap=0.3,
        font=dict(family='DejaVu Sans, sans-serif', size=14, color='black'),
        paper_bgcolor='white',
        plot_bgcolor='white',
        legend=dict(
            bgcolor='white',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=12)
        ),
        margin=dict(l=50, r=20, t=80, b=60),
    )

    fig.show()
    #fig.write_image('Fuel_Supply.svg')

def multi_scenario_stacked_emissions(scenarios, plot_title="Stacked Emissions by Scenario"):
    """
    Creates a single figure with subplots in columns (1 x #scenarios).
    Each subplot is stacked emissions for that scenario.
    """
    # Create subplots: 1 row, N columns
    fig = make_subplots(
        rows=1,
        cols=len(scenarios),
        subplot_titles=[s[0] for s in scenarios],
        shared_yaxes=True
    )

    # Example color map
    color_map = {
        "Industry Heating":            "#065903",
        "Industry with BECCS":         "#84ad63",
        "Individual Users Heating":    "#4c53b5",
        "Individual with BECCS":       "#9b989c",
        "CHP Generation":              "#91066c",
        "CHP with BECCS":              "#f27ed3",
        "Biogas":                      "#c47900",
        "DAC":                         "#8900c4",
        "Biochar Sequestration":       "#5c3f3d",
        "Marine Shipping Diesel Oil":  "#0057b8",
        "Kerosene":                    "#ff4500",
        "Diesel":                      "#ffa500",
        "Carbon Rich Soils Extraction": "#ffff00",  # yellow 
        "New Productive Forest":       "#f4a460"   # sandy brown
        }
    fallback_color = "#999999"

    encountered_categories = set()

    for idx, (scenario_name, scenario_path) in enumerate(scenarios):
        main_results_path = os.path.join(scenario_path, "MainResults.gdx")
        optiflow_path     = os.path.join(scenario_path, "Optiflow_MainResults.gdx")

        df_FLOWA, df_FLOWC, df_EMI_opt, df_EMI_PROC, df_VOBJ = Import_OptiflowMR(optiflow_path)
        df_CC_YCRAG, df_F_CONS_YCRA, df_EMI_YCRAG, df_G_CAP_YCRAF, df_PRO_YCRAGF, df_OBJ_YCR, df_EL_DEMAND_YCR, df_H_DEMAND_YCRA, df_H2_DEMAND_YCR, df_X_FLOW_YCR, df_XH2_FLOW_YCR, df_XH_FLOW_YCA  = Import_BalmorelMR(main_results_path)

        df_agg = group_EMI_YCRAG(df_EMI_opt, df_FLOWC, df_EMI_PROC)

        df_pos = df_agg[df_agg["value"] >= 0].copy()
        df_neg = df_agg[df_agg["value"] < 0].copy()

        df_pos.sort_values("value", inplace=True)
        df_neg.sort_values("value", ascending=False, inplace=True)

        df_pos["base"] = df_pos["value"].cumsum() - df_pos["value"]
        df_neg["base"] = df_neg["value"].cumsum() - df_neg["value"]

        x_val = 0
        c_col = idx + 1

        # Positive bars
        for _, rowP in df_pos.iterrows():
            cat  = rowP["Category"]
            val  = rowP["value"]
            base = rowP["base"]

            show_legend = (cat not in encountered_categories)
            if show_legend:
                encountered_categories.add(cat)

            fig.add_trace(
                go.Bar(
                    x=[x_val],
                    y=[val],
                    base=[base],
                    name=cat,
                    marker_color=color_map.get(cat, fallback_color),
                    showlegend=show_legend
                ),
                row=1, col=c_col
            )

        # Negative bars
        for _, rowN in df_neg.iterrows():
            cat  = rowN["Category"]
            val  = rowN["value"]
            base = rowN["base"]

            show_legend = (cat not in encountered_categories)
            if show_legend:
                encountered_categories.add(cat)

            fig.add_trace(
                go.Bar(
                    x=[x_val],
                    y=[val],
                    base=[base],
                    name=cat,
                    marker_color=color_map.get(cat, fallback_color),
                    showlegend=show_legend
                ),
                row=1, col=c_col
            )

        fig.update_xaxes(
            type="linear",
            range=[-1, 1],
            showticklabels=False,
            showline=True,
            mirror=True,
            linewidth=1,
            linecolor="black",
            showgrid=False,
            title_text='',
            row=1, col=c_col
        )

        y_title = '[ktons]' if idx == 0 else ''
        fig.update_yaxes(
            title_text=y_title,
            showline=True,
            mirror=True,
            linewidth=1,
            linecolor="black",
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=0.6,
            zeroline=True,
            zerolinewidth=0.6,
            zerolinecolor="lightgray",
            row=1, col=c_col
        )

    fig.update_layout(
        title=plot_title,
        barmode="overlay",
        font=dict(
            family="DejaVu Sans, sans-serif",
            size=14,
            color="black"
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=12)
        ),
        margin=dict(l=50, r=20, t=80, b=60)
    )

    fig.show()
    # fig.write_image('CO2_Emissions.svg')

def multi_scenario_biomass_consumption(scenarios, plot_title="Biomass Consumption Comparison"):
    fig = make_subplots(
        rows=1,
        cols=len(scenarios),
        shared_yaxes=True,
        subplot_titles=[s[0] for s in scenarios],
    )

    fuel_colors = {
        'Wood Chips':   '#1f77b4',
        'Wood Pellets': '#ff7f0e',
        'Straw':        '#2ca02c'
    }
    fallback_color = 'gray'

    encountered_legends = set()

    for idx, (scenario_name, scenario_path) in enumerate(scenarios):
        main_results_path = os.path.join(scenario_path, "MainResults.gdx")
        optiflow_path     = os.path.join(scenario_path, "Optiflow_MainResults.gdx")

        df_FLOWA, df_FLOWC, df_EMI_opt, df_EMI_PROC, df_VOBJ = Import_OptiflowMR(optiflow_path)
        df_CC_YCRAG, df_F_CONS_YCRA, df_EMI_YCRAG, df_G_CAP_YCRAF, df_PRO_YCRAGF, df_OBJ_YCR, df_EL_DEMAND_YCR, df_H_DEMAND_YCRA, df_H2_DEMAND_YCR, df_X_FLOW_YCR, df_XH2_FLOW_YCR, df_XH_FLOW_YCA = Import_BalmorelMR(main_results_path)

        df_flowc_filtered, df_f_cons_filtered = process_flows_and_consumption(df_FLOWC, df_F_CONS_YCRA)

        df_cat_fuel = df_f_cons_filtered.groupby(['Category','FFF'], as_index=False)['value'].sum()

        def map_fff_to_fuel(fff):
            if fff == 'WOODCHIPS':   return 'Wood Chips'
            if fff == 'WOODPELLETS': return 'Wood Pellets'
            if fff == 'STRAW':       return 'Straw'
            return None

        df_cat_fuel['Fuel'] = df_cat_fuel['FFF'].apply(map_fff_to_fuel)

        def map_category_to_colidx(cat):
            if cat == 'Industry Heating':        return 0
            if cat == 'Individual Users Heating':return 1
            if cat == 'CHP Generation':          return 2
            return None

        fuels = ['Wood Chips','Wood Pellets','Straw']
        fuel_arrays = {f: [0]*5 for f in fuels}

        for _, row2 in df_cat_fuel.iterrows():
            cidx = map_category_to_colidx(row2['Category'])
            f    = row2['Fuel']
            v    = row2['value']
            if cidx is not None and f in fuel_arrays:
                fuel_arrays[f][cidx] += v

        df_flowc_4th = df_flowc_filtered[
            df_flowc_filtered['IPROCFROM'].isin(['Wood_Pellets_Gen','Straw_for_Energy','Wood_for_Energy'])
        ].copy()

        def map_iproc_to_fuel(iproc):
            if iproc == 'Wood_Pellets_Gen':  return 'Wood Pellets'
            if iproc == 'Straw_for_Energy':  return 'Straw'
            if iproc == 'Wood_for_Energy':   return 'Wood Chips'
            return None

        df_flowc_4th['Fuel'] = df_flowc_4th['IPROCFROM'].apply(map_iproc_to_fuel)
        df_flowc_4th_agg = df_flowc_4th.groupby('Fuel', as_index=False)['value'].sum()

        for _, r4 in df_flowc_4th_agg.iterrows():
            f = r4['Fuel']
            v = r4['value']
            if f in fuel_arrays:
                fuel_arrays[f][3] += v

        df_flowc_5th = df_flowc_filtered[
            (df_flowc_filtered['IPROCFROM'] == 'Wood_Pellets_Gen') |
            (df_flowc_filtered['IPROCTO'].isin(['Straw_for_Energy','Wood_for_Energy']))
        ].copy()
        df_flowc_5th['Fuel_from'] = df_flowc_5th['IPROCFROM'].apply(map_iproc_to_fuel)
        df_flowc_5th['Fuel_to']   = df_flowc_5th['IPROCTO'].apply(map_iproc_to_fuel)

        flowc_5th_from = df_flowc_5th.groupby('Fuel_from')['value'].sum().reset_index()
        flowc_5th_from.rename(columns={'Fuel_from':'Fuel','value':'FlowC_5th_from'}, inplace=True)
        flowc_5th_to = df_flowc_5th.groupby('Fuel_to')['value'].sum().reset_index()
        flowc_5th_to.rename(columns={'Fuel_to':'Fuel','value':'FlowC_5th_to'}, inplace=True)
        flowc_5th_combined = pd.merge(flowc_5th_from, flowc_5th_to, on='Fuel', how='outer').fillna(0)
        flowc_5th_combined['FlowC_5th_total'] = flowc_5th_combined['FlowC_5th_from'] + flowc_5th_combined['FlowC_5th_to']

        df_f_cons_wp = df_f_cons_filtered[df_f_cons_filtered['FFF'] == 'WOODPELLETS'].copy()
        df_f_cons_wp['Fuel'] = df_f_cons_wp['FFF'].apply(map_fff_to_fuel)
        fcons_wp_val = df_f_cons_wp['value'].sum()

        for _, r5 in flowc_5th_combined.iterrows():
            f  = r5['Fuel']
            v5 = r5['FlowC_5th_total']
            if f in fuel_arrays:
                fuel_arrays[f][4] -= v5

        if 'Wood Pellets' in fuel_arrays:
            fuel_arrays['Wood Pellets'][4] -= fcons_wp_val

        x_labels = [
            "Industry Heating",
            "Individual Users",
            "CHP Generation",
            "Fuels Production",
            "Total Biomass"
        ]

        c_col = idx+1
        for f in fuels:
            show_legend = (f not in encountered_legends)
            if show_legend:
                encountered_legends.add(f)

            fig.add_trace(
                go.Bar(
                    x=x_labels,
                    y=fuel_arrays[f],
                    name=f,
                    marker_color=fuel_colors.get(f, fallback_color),
                    showlegend=show_legend,
                ),
                row=1, col=c_col
            )

        fig.update_xaxes(
            title_text='',
            showline=True,
            mirror=True,
            linewidth=1,
            linecolor='black',
            showgrid=False,
            row=1, col=c_col
        )

        y_title = '[PJ]' if idx==0 else ''
        fig.update_yaxes(
            title_text=y_title,
            showline=True,
            mirror=True,
            linewidth=1,
            linecolor='black',
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=0.6,
            zeroline=True,
            zerolinewidth=0.6,
            zerolinecolor='lightgray',
            row=1, col=c_col
        )

    fig.update_layout(
        title=plot_title,
        barmode='stack',
        font=dict(family='DejaVu Sans, sans-serif', size=14, color='black'),
        paper_bgcolor='white',
        plot_bgcolor='white',
        legend=dict(
            bgcolor='white',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=12)
        ),
        margin=dict(l=50, r=20, t=80, b=60)
    )

    fig.show()
    # fig.write_image('Biomass_Use.svg')


def multi_scenario_gcap_histogram(scenarios, country = 'DENMARK', plot_title="Installed Capacity By Scenario"):
    """
    For each scenario, this function plots a histogram using the df_G_CAP_YCRAF data.
    The x-axis contains one column per unique 'COMMODITY', and for each commodity the bar is stacked by 'TECH_TYPE'
    (with friendly technology names and fixed colors shown in the legend). Rows where 'TECH_TYPE'
    contains 'STORAGE' are excluded.
    Only values higher than 10e-10 (i.e. >= 1e-9) are plotted.
    """
    # Unified mapping from technology codes to friendly names.
    tech_name_map = {
        "CHP-BACK-PRESSURE": "CHP Back-Pressure",
        "ELECT-TO-HEAT": "Electric to Heat",
        "INTERSEASONAL-HEAT-STORAGE": "Interseasonal Heat Storage",
        "INTRASEASONAL-HEAT-STORAGE": "Intraseasonal Heat Storage",
        "SOLAR-PV": "Solar PV",
        "HYDRO-RUN-OF-RIVER": "Hydro Run-of-River",
        "WIND-ON": "Wind Onshore",
        "WIND-OFF": "Wind Offshore",
        "ELECTROLYZER": "Electrolyzer",
        "H2-STORAGE": "H2 Storage",
        "BOILERS": "Boilers",
        "CHP-EXTRACTION": "CHP Extraction",
        "INTRASEASONAL-ELECT-STORAGE": "Intraseasonal Electro Storage",
        "HYDRO-RESERVOIRS": "Hydro Reservoirs",
        "CONDENSING": "Condensing",
        "FUELCELL": "Fuel Cell",
        "STEAMREFORMING": "Steam Reforming",
    }
    tech_color_map = {
        "CHP Back-Pressure": "#8B4513",         # brown
        "Electric to Heat": "#d62728",           # red
        "Interseasonal Heat Storage": "#2ca02c",
        "Intraseasonal Heat Storage": "#d62728",
        "Solar PV": "#ffeb3b",                   # yellow
        "Hydro Run-of-River": "#005f73",         # dark-blue/teal
        "Wind Onshore": "#87CEFA",               # light blue
        "Wind Offshore": "#ADD8E6",              # light blue
        "Electrolyzer": "#32CD32",               # lighter, vivid green
        "H2 Storage": "#17becf",
        "Boilers": "#FFA500",                    # orange
        "CHP Extraction": "#556B2F",             # dark olive green
        "Intraseasonal Electro Storage": "#98df8a",
        "Hydro Reservoirs": "#003f5c",           # dark blue
        "Condensing": "#c5b0d5",
        "Fuel Cell": "#c49c94",
        "Steam Reforming": "#f7b6d2"
    }

    # Fallback color palette if a tech is not in tech_color_map.
    color_palette = px.colors.qualitative.Plotly
    color_index = {}

    # Set threshold: Only values >= 10e-10 (1e-9) will be plotted.
    threshold = 1e-9

    fig = make_subplots(
        rows=1,
        cols=len(scenarios),
        shared_yaxes=True,
        subplot_titles=[s[0] for s in scenarios]
    )

    # Global set to track legend entries (across scenarios)
    encountered_legends = set()

    for idx, (scenario_name, scenario_path) in enumerate(scenarios):
        main_results_path = os.path.join(scenario_path, "MainResults.gdx")
        # Load Balmorel results.
        df_CC_YCRAG, df_F_CONS_YCRA, df_EMI_YCRAG, df_G_CAP_YCRAF, df_PRO_YCRAGF, df_OBJ_YCR, df_EL_DEMAND_YCR, df_H_DEMAND_YCRA, df_H2_DEMAND_YCR, df_X_FLOW_YCR, df_XH2_FLOW_YCR, df_XH_FLOW_YCA = Import_BalmorelMR(main_results_path)

        if country != 'all':
            df_G_CAP_YCRAF = df_G_CAP_YCRAF[df_G_CAP_YCRAF['C'] == country]
        # Exclude rows where 'TECH_TYPE' contains 'STORAGE'
        if 'TECH_TYPE' in df_G_CAP_YCRAF.columns:
            df_G_CAP_YCRAF = df_G_CAP_YCRAF[~df_G_CAP_YCRAF['TECH_TYPE'].astype(str).str.contains('STORAGE', case=False, na=False)]

        # Group by 'COMMODITY' and 'TECH_TYPE'
        grouped = df_G_CAP_YCRAF.groupby(['COMMODITY', 'TECH_TYPE'], as_index=False)['value'].sum()
        commodities = grouped['COMMODITY'].unique().tolist()
        tech_list = grouped['TECH_TYPE'].unique().tolist()

        for tech in tech_list:
            friendly_name = tech_name_map.get(tech, tech)
            color = tech_color_map.get(friendly_name)
            if color is None:
                if tech not in color_index:
                    color_index[tech] = color_palette[len(color_index) % len(color_palette)]
                color = color_index[tech]

            y_vals = []
            for com in commodities:
                # Retrieve value and apply threshold filtering.
                val_series = grouped.loc[(grouped['COMMODITY'] == com) & (grouped['TECH_TYPE'] == tech), 'value']
                value = val_series.values[0] if not val_series.empty else 0
                if value < threshold:
                    value = 0
                y_vals.append(value)
            # Skip trace if total is zero after threshold filtering.
            if sum(y_vals) == 0:
                continue
            if tech not in encountered_legends:
                show_legend = True
                encountered_legends.add(tech)
            else:
                show_legend = False

            fig.add_trace(
                go.Bar(
                    x=commodities,
                    y=y_vals,
                    name=friendly_name,
                    marker_color=color,
                    showlegend=show_legend,
                ),
                row=1, col=idx+1
            )

        fig.update_xaxes(
            title_text='COMMODITY',
            showline=True,
            mirror=True,
            linewidth=1,
            linecolor='black',
            showgrid=False,
            row=1, col=idx+1
        )
        y_title = '[GW]' if idx == 0 else ''
        fig.update_yaxes(
            title_text=y_title,
            showline=True,
            mirror=True,
            linewidth=1,
            linecolor='black',
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=0.6,
            zeroline=True,
            zerolinewidth=0.6,
            zerolinecolor='lightgray',
            row=1, col=idx+1
        )

    fig.update_layout(
        title=plot_title,
        barmode='stack',
        font=dict(family='DejaVu Sans, sans-serif', size=14, color='black'),
        paper_bgcolor='white',
        plot_bgcolor='white',
        legend=dict(
            bgcolor='white',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=12)
        ),
        margin=dict(l=50, r=20, t=80, b=60)
    )
    fig.show()

    # fig.write_image('G_CAP_YCRAF_Histogram.svg')


def multi_scenario_pro_histogram(scenarios, country, plot_title="Energy Production by Scenario (in Balmorel)", demand_markers=True):
    """
    For each scenario, this function plots a histogram using the df_PRO_YCRAGF data.
    The x-axis contains one column per unique 'COMMODITY', and for each commodity the bar is stacked by 'TECH_TYPE'
    (with friendly technology names and fixed colors shown in the legend). Rows where 'TECH_TYPE'
    contains 'STORAGE' are excluded.
    Only values higher than 10e-9 (i.e. ≥ 1e-8) are plotted.

    :param demand_markers: If set to True, shows the demand markers for Domestic and Exogeneous Demand
    """
    demand_shown = False
    exo_demand_shown = False
    threshold_pro = 1e-8

    tech_name_map = {
        "CHP-BACK-PRESSURE": "CHP Back-Pressure",
        "ELECT-TO-HEAT": "Electric to Heat",
        "INTERSEASONAL-HEAT-STORAGE": "Interseasonal Heat Storage",
        "INTRASEASONAL-HEAT-STORAGE": "Intraseasonal Heat Storage",
        "SOLAR-PV": "Solar PV",
        "HYDRO-RUN-OF-RIVER": "Hydro Run-of-River",
        "WIND-ON": "Wind Onshore",
        "WIND-OFF": "Wind Offshore",
        "ELECTROLYZER": "Electrolyzer",
        "H2-STORAGE": "H2 Storage",
        "BOILERS": "Boilers",
        "CHP-EXTRACTION": "CHP Extraction",
        "INTRASEASONAL-ELECT-STORAGE": "Intraseasonal Electro Storage",
        "HYDRO-RESERVOIRS": "Hydro Reservoirs",
        "CONDENSING": "Condensing",
        "FUELCELL": "Fuel Cell",
        "STEAMREFORMING": "Steam Reforming",
    }

    tech_color_map = {
        "CHP Back-Pressure": "#8B4513",
        "Electric to Heat": "#d62728",
        "Interseasonal Heat Storage": "#ba55d3",
        "Intraseasonal Heat Storage": "#c8a2c8",
        "Solar PV": "#ffeb3b",
        "Hydro Run-of-River": "#005f73",
        "Wind Onshore": "#87CEFA",
        "Wind Offshore": "#ADD8E6",
        "Electrolyzer": "#32CD32",
        "H2 Storage": "#17becf",
        "Boilers": "#FFA500",
        "CHP Extraction": "#556B2F",
        "Intraseasonal Electro Storage": "#98df8a",
        "Hydro Reservoirs": "#003f5c",
        "Condensing": "#c5b0d5",
        "Fuel Cell": "#c49c94",
        "Steam Reforming": "#f7b6d2"
    }

    fig = make_subplots(
        rows=1,
        cols=len(scenarios),
        shared_yaxes=True,
        subplot_titles=[s[0] for s in scenarios]
    )

    encountered_legends = set()

    for idx, (scenario_name, scenario_path) in enumerate(scenarios):
        main_results_path = os.path.join(scenario_path, "MainResults.gdx")
        df_CC_YCRAG, df_F_CONS_YCRA, df_EMI_YCRAG, df_G_CAP_YCRAF, df_PRO_YCRAGF, df_OBJ_YCR, df_EL_DEMAND_YCR, df_H_DEMAND_YCRA, df_H2_DEMAND_YCR, df_X_FLOW_YCR, df_XH2_FLOW_YCR, df_XH_FLOW_YCA = Import_BalmorelMR(main_results_path)

        if country != 'all':
            df_PRO_YCRAGF = df_PRO_YCRAGF[df_PRO_YCRAGF['C'] == country]

        if 'TECH_TYPE' in df_PRO_YCRAGF.columns:
            # Keep HEAT rows as-is (including STORAGE)
            df_heat = df_PRO_YCRAGF[df_PRO_YCRAGF['COMMODITY'] == 'HEAT']

            # For other commodities, exclude STORAGE tech types
            df_other = df_PRO_YCRAGF[df_PRO_YCRAGF['COMMODITY'] != 'HEAT']
            df_other = df_other[~df_other['TECH_TYPE'].astype(str).str.contains('STORAGE', case=False, na=False)]

            # Concatenate back
            df_PRO_YCRAGF = pd.concat([df_heat, df_other], ignore_index=True)
        # Group and filter
        grouped = df_PRO_YCRAGF.groupby(['COMMODITY', 'TECH_TYPE'], as_index=False)['value'].sum()
        grouped = grouped[grouped['value'] >= threshold_pro]

        commodities = grouped['COMMODITY'].unique().tolist()
        tech_list = grouped['TECH_TYPE'].unique().tolist()

        for tech in tech_list:
            friendly_name = tech_name_map.get(tech, tech)
            color = tech_color_map.get(friendly_name)
            if color is None:
                if tech not in color_index:
                    color_index[tech] = color_palette[len(color_index) % len(color_palette)]
                color = color_index[tech]

            y_vals = []
            for com in commodities:
                val_series = grouped.loc[(grouped['COMMODITY'] == com) & (grouped['TECH_TYPE'] == tech), 'value']
                value = val_series.values[0] if not val_series.empty else 0
                # Only include values above threshold
                if value < threshold_pro:
                    value = 0
                y_vals.append(value)
            # If the trace has no value above threshold, skip it.
            if sum(y_vals) == 0:
                continue

            if tech not in encountered_legends:
                show_legend = True
                encountered_legends.add(tech)
            else:
                show_legend = False

            fig.add_trace(
                go.Bar(
                    x=commodities,
                    y=y_vals,
                    name=friendly_name,
                    marker_color=color,
                    showlegend=show_legend,
                ),
                row=1, col=idx+1
            )

        # Add domestic demand and EXO demand marker per COMMODITY
        if demand_markers:
            demand_sources = {
                "ELECTRICITY": df_EL_DEMAND_YCR,
                "HEAT": df_H_DEMAND_YCRA,
                "HYDROGEN": df_H2_DEMAND_YCR
            }

            for com in commodities:
                demand_df = demand_sources.get(com)
                if demand_df is None or demand_df.empty:
                    continue

                # Domestic demand (as before)
                if 'C' in demand_df.columns:
                    demand_value = demand_df[demand_df['C'] == country]['value'].sum()
                else:
                    demand_value = 0

                if demand_value >= threshold_pro:
                    fig.add_trace(
                        go.Scatter(
                            x=[com],
                            y=[demand_value],
                            mode='markers',
                            name="Total Domestic Demand",
                            marker=dict(
                                symbol='line-ew-open',
                                size=35,
                                color='red',
                                line=dict(width=3)
                            ),
                            showlegend=False if demand_shown else True,
                            legendgroup='domesticdemand',
                            legendrank=1
                        ),
                        row=1, col=idx + 1
                    )
                    demand_shown = True

                if 'C' in demand_df.columns and 'VARIABLE_CATEGORY' in demand_df.columns:
                    exo_demand = demand_df[
                        (demand_df['C'] == country) &
                        (demand_df['VARIABLE_CATEGORY'].astype(str).str.contains('EXO', case=False, na=False))
                    ]['value'].sum()
                else:
                    exo_demand = 0

                if exo_demand >= threshold_pro:
                    fig.add_trace(
                        go.Scatter(
                            x=[com],
                            y=[exo_demand],
                            mode='markers',
                            name="Exogeous Domestic Demand",
                            marker=dict(
                                symbol='line-ew-open',
                                size=35,
                                color='blue',
                                line=dict(width=3)
                            ),
                            showlegend=False if exo_demand_shown else True,
                            legendgroup='domesticdemand',
                            legendrank=2
                        ),
                        row=1, col=idx + 1
                    )
                    exo_demand_shown = True

        # X-axis
        fig.update_xaxes(
            title_text='COMMODITY',
            showline=True,
            mirror=True,
            linewidth=1,
            linecolor='black',
            showgrid=False,
            row=1, col=idx+1
        )

        # Y-axis
        y_title = '[TWh]' if idx == 0 else ''
        fig.update_yaxes(
            title_text=y_title,
            showline=True,
            mirror=True,
            linewidth=1,
            linecolor='black',
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=0.6,
            zeroline=True,
            zerolinewidth=0.6,
            zerolinecolor='lightgray',
            row=1, col=idx+1
        )

    fig.update_layout(
        title=plot_title,
        barmode='stack',
        font=dict(family='DejaVu Sans, sans-serif', size=14, color='black'),
        paper_bgcolor='white',
        plot_bgcolor='white',
        legend=dict(
        bgcolor='white',
        bordercolor='black',
        borderwidth=1,
        )
    )
    fig.show()

def multi_scenario_impexp_histogram(scenarios, country, marker_length=110, plot_title="Energy Imports and Exports by Scenario"):
    import os
    import math
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    threshold = 1e-8

    market_areas = {
        'ALBANIA': ['AL'], 'AUSTRIA': ['AT'], 'BELGIUM': ['BE'], 'BOSNIA_AND_HERZEGOVINA': ['BA'],
        'BULGARIA': ['BG'], 'CROATIA': ['HR'], 'CYPRUS': ['CY'], 'CZECH_REPUBLIC': ['CZ'],
        'DENMARK': ['DK1', 'DK2'], 'ESTONIA': ['EE'], 'FINLAND': ['FIN'], 'FRANCE': ['FR'],
        'GERMANY': ['DE4-E', 'DE4-N', 'DE4-S', 'DE4-W'], 'GREECE': ['GR'], 'HUNGARY': ['HU'],
        'IRELAND': ['IE'], 'ITALY': ['IT'], 'LATVIA': ['LV'], 'LITHUANIA': ['LT'],
        'LUXEMBOURG': ['LU'], 'MALTA': ['MT'], 'MONTENEGRO': ['ME'], 'NETHERLANDS': ['NL'],
        'NORTH_MACEDONIA': ['MK'], 'NORWAY': ['NO1', 'NO2', 'NO3', 'NO4', 'NO5'],
        'POLAND': ['PL'], 'PORTUGAL': ['PT'], 'ROMANIA': ['RO'], 'SERBIA': ['RS'],
        'SLOVAKIA': ['SK'], 'SLOVENIA': ['SI'], 'SPAIN': ['ES'],
        'SWEDEN': ['SE1', 'SE2', 'SE3', 'SE4'], 'SWITZERLAND': ['CH'], 'TURKEY': ['TR'], 'UNITED_KINGDOM': ['UK']
    }

    # One color per direction
    color_map = {
        'Import': '#4C78A8',
        'Export': '#ebde34'
    }

    valid_scenarios = []
    scenario_data = []
    net_export_shown = False
    imp_shown=False
    exp_shown=False

    market_area_list = market_areas.get(country, [])

    for scenario_name, scenario_path in scenarios:
        main_results_path = os.path.join(scenario_path, "MainResults.gdx")
        _, _, _, _, _, _, _, _, _, df_X_FLOW_YCR, df_XH2_FLOW_YCR, _ = Import_BalmorelMR(main_results_path)

        if (df_X_FLOW_YCR is None or df_X_FLOW_YCR.empty) and (df_XH2_FLOW_YCR is None or df_XH2_FLOW_YCR.empty):
            continue

        total_val = 0
        for com, df in {'ELECTRICITY': df_X_FLOW_YCR, 'HYDROGEN': df_XH2_FLOW_YCR}.items():
            if df is None or df.empty:
                continue
            imports = df[(df['C'] != country) & (df['IRRRI'].isin(market_area_list))]
            exports = df[(df['C'] == country) & (~df['IRRRI'].isin(market_area_list))]
            total_val += abs(imports['value'].sum()) + abs(exports['value'].sum())

        if total_val > threshold:
            valid_scenarios.append((scenario_name, scenario_path))
            scenario_data.append((df_X_FLOW_YCR, df_XH2_FLOW_YCR))

    if not valid_scenarios:
        print("No scenarios have significant import/export values.")
        return

    fig = make_subplots(
        rows=1,
        cols=len(valid_scenarios),
        shared_yaxes=True,
        subplot_titles=[s[0] for s in valid_scenarios]
    )

    for idx, ((scenario_name, _), (df_X_FLOW_YCR, df_XH2_FLOW_YCR)) in enumerate(zip(valid_scenarios, scenario_data)):
        flow_data = {
            'HYDROGEN': df_XH2_FLOW_YCR,
            'ELECTRICITY': df_X_FLOW_YCR
        }

        for com in ['HYDROGEN', 'ELECTRICITY']:
            df = flow_data[com]
            if df is None or df.empty:
                continue

            imports = df[(df['C'] != country) & (df['IRRRI'].isin(market_area_list))]
            exports = df[(df['C'] == country) & (~df['IRRRI'].isin(market_area_list))]

            import_val = imports['value'].sum()
            export_val = exports['value'].sum()

            # Plot Import (negative)
            if abs(import_val) > threshold:
                fig.add_trace(
                    go.Bar(
                        x=[com],
                        y=[-import_val],
                        name='Imports',
                        marker_color=color_map['Import'],
                        showlegend=False if imp_shown else True,
                    ),
                    row=1, col=idx + 1
                )
            imp_shown = True

            # Plot Export (positive)
            if abs(export_val) > threshold:
                fig.add_trace(
                    go.Bar(
                        x=[com],
                        y=[export_val],
                        name='Exports',
                        marker_color=color_map['Export'],
                        showlegend=False if exp_shown else True,
                        
                    ),
                    row=1, col=idx + 1
                )
            exp_shown = True

            # Net export marker
            net_export = export_val - import_val
            fig.add_trace(
                go.Scatter(
                    x=[com],
                    y=[net_export],
                    mode='markers',
                    marker=dict(symbol='line-ew-open', size=marker_length, color='red', line=dict(width=3)),
                    name='Net Exports',
                    showlegend=False if net_export_shown else True,
                    legendgroup='netexports',
                    legendrank=1
                ),
                row=1, col=idx + 1
            )
            net_export_shown = True

        fig.update_xaxes(
            title_text='COMMODITY',
            showline=True,
            mirror=True,
            linewidth=1,
            linecolor='black',
            showgrid=False,
            row=1, col=idx + 1
        )

    # Determine max y-axis value for neat tick marks
    y_max = 0
    for (df_X_FLOW_YCR, df_XH2_FLOW_YCR) in scenario_data:
        for com_df in [df_X_FLOW_YCR, df_XH2_FLOW_YCR]:
            if com_df is not None and not com_df.empty:
                y_max = max(y_max, abs(com_df['value'].sum()))

    y_limit = math.ceil(y_max / 25) * 25
    tickvals = list(range(-y_limit, y_limit + 1, 25))
    ticktext = [str(abs(val)) for val in tickvals]  # Always positive labels

    for idx in range(len(valid_scenarios)):
        fig.update_yaxes(
            title_text='[TWh]' if idx == 0 else '',
            showline=True,
            mirror=True,
            linewidth=1,
            linecolor='black',
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=0.6,
            zeroline=True,
            zerolinewidth=0.6,
            zerolinecolor='lightgray',
            tickvals=tickvals,
            ticktext=ticktext,
            row=1, col=idx + 1
        )

    fig.update_layout(
        title=plot_title,
        barmode='relative',
        font=dict(family='DejaVu Sans, sans-serif', size=14, color='black'),
        paper_bgcolor='white',
        plot_bgcolor='white',
        legend=dict(
            bgcolor='white',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=12),
            groupclick="toggleitem",
            traceorder="reversed"
        ),
        margin=dict(l=50, r=20, t=80, b=60)
    )

    fig.show()

    
def multi_scenario_fuel_share_histogram(scenario_list, country, plot_title="Production Fuel Share Histogram"):
    """
    For each scenario in scenario_list, this function plots a histogram where each column corresponds to a 
    'COMMODITY' (except Hydrogen) and the bar is stacked by fuel type (from the 'FFF' column). Each fuel's 
    bar height is its share of the total production for that commodity.
    
    Each scenario in scenario_list is a tuple of (scenario_name, folder_path) where folder_path contains 
    the file 'MainResults.gdx'.
    
    Global fuel mappings (based on provided fuel list):
      - NATGAS      -> Natural Gas
      - ELECTRIC    -> Electric
      - HEAT        -> Heat
      - SUN         -> Solar
      - WATER       -> Water
      - MUNIWASTE   -> Municipal Waste
      - WIND        -> Wind
      - HYDROGEN    -> Hydrogen
      - LIGHTOIL    -> Light Oil
      - COAL        -> Coal
      - LIGNITE     -> Lignite
      - WOODCHIPS   -> Woodchips
      - WOODPELLETS -> Wood Pellets
      - FUELOIL     -> Fuel Oil
      - WOOD        -> Wood
      - STRAW       -> Straw
    
    In addition, rows where 'TECH_TYPE' contains 'STORAGE' (case insensitive) are excluded from both 
    the total production and the plotted shares.
    
    Only fuel shares higher than 10e-2 (i.e. 0.1) are plotted; values below this threshold are set to zero.
    
    The produced commodity "HYDROGEN" is omitted entirely from the plot.
    
    Parameters:
    - scenario_list: List of tuples, where each tuple is (scenario_name, folder_path)
    - plot_title: Title of the overall plot
    """

    # Threshold for plotting share values.
    threshold = 10e-4  # 0.1

    # Global mapping: fuel code -> friendly name
    fuel_name_map = {
        'NATGAS': 'Natural Gas',
        'ELECTRIC': 'Electric',
        'HEAT': 'Heat',
        'SUN': 'Solar',
        'WATER': 'Water',
        'MUNIWASTE': 'Municipal Waste',
        'WIND': 'Wind',
        'HYDROGEN': 'Hydrogen',
        'LIGHTOIL': 'Light Oil',
        'COAL': 'Coal',
        'LIGNITE': 'Lignite',
        'WOODCHIPS': 'Woodchips',
        'WOODPELLETS': 'Wood Pellets',
        'FUELOIL': 'Fuel Oil',
        'WOOD': 'Wood',
        'STRAW': 'Straw'
    }
    
    # Global fixed color mapping for each friendly fuel name.
    fuel_color_map = {
        'Natural Gas': '#d62728',    # red
        'Electric': '#1f77b4',         # blue
        'Heat': '#ff7f0e',             # orange
        'Solar': '#ffeb3b',            # yellow
        'Water': '#17becf',            # cyan
        'Municipal Waste': '#8c564b',  # brownish
        'Wind': '#2ca02c',             # green
        'Hydrogen': '#9467bd',         # purple
        'Light Oil': '#e377c2',        # pink
        'Coal': '#7f7f7f',             # grey
        'Lignite': '#bcbd22',          # olive
        'Woodchips': '#8c6d31',        # darker brown
        'Wood Pellets': '#b5cf6b',     # light green
        'Fuel Oil': '#e7ba52',         # mustard
        'Wood': '#cedb9c',             # pale green
        'Straw': '#cedb9c'
    }
    
    # Create one subplot per scenario.
    fig = make_subplots(
        rows=1,
        cols=len(scenario_list),
        shared_yaxes=True,
        subplot_titles=[s[0] for s in scenario_list]
    )
    
    # Track fuels already added to the legend.
    encountered_legends = set()
    
    for idx, (scenario_name, scenario_folder) in enumerate(scenario_list):
        # Build the path to the main results file inside the given folder.
        main_results_path = os.path.join(scenario_folder, "MainResults.gdx")
        
        # Check if MainResults.gdx exists in the folder.
        if not os.path.exists(main_results_path):
            raise FileNotFoundError(f"MainResults.gdx not found in folder: {scenario_folder}")
        
        # Import data (assuming Import_BalmorelMR returns a tuple where the 5th element is df_PRO_YCRAGF).
        df_CC_YCRAG, df_F_CONS_YCRA, df_EMI_YCRAG, df_G_CAP_YCRAF, df_PRO_YCRAGF, df_OBJ_YCR, df_EL_DEMAND_YCR, df_H_DEMAND_YCRA, df_H2_DEMAND_YCR, df_X_FLOW_YCR, df_XH2_FLOW_YCR, df_XH_FLOW_YCA = Import_BalmorelMR(main_results_path)

        if country != 'all':
            df_PRO_YCRAGF = df_PRO_YCRAGF[df_PRO_YCRAGF['C'] == country]
        
        # Exclude rows where 'TECH_TYPE' contains 'STORAGE' (if that column exists).
        if 'TECH_TYPE' in df_PRO_YCRAGF.columns:
            df_PRO_YCRAGF = df_PRO_YCRAGF[~df_PRO_YCRAGF['TECH_TYPE'].astype(str).str.contains('STORAGE', case=False, na=False)]
        
        # Group by 'COMMODITY' and 'FFF' to sum production values.
        grouped = df_PRO_YCRAGF.groupby(['COMMODITY', 'FFF'], as_index=False)['value'].sum()
        
        # Remove rows where the produced commodity is Hydrogen (case insensitive).
        grouped = grouped[~grouped['COMMODITY'].str.upper().eq("HYDROGEN")]
        
        # Compute the total production per commodity.
        totals = grouped.groupby('COMMODITY')['value'].sum().reset_index().rename(columns={'value': 'total_value'})
        
        # Merge the totals back with the grouped data.
        merged = pd.merge(grouped, totals, on='COMMODITY')
        
        # Calculate the share for each fuel type.
        merged['share'] = merged['value'] / merged['total_value']
        
        # Get unique commodities and fuels present in this scenario.
        commodities = merged['COMMODITY'].unique().tolist()
        fuels_in_scenario = merged['FFF'].unique().tolist()

        # For each fuel in the scenario, use the global maps for friendly name and color.
        for fuel in fuels_in_scenario:
            friendly_name = fuel_name_map.get(fuel, fuel)
            color = fuel_color_map.get(friendly_name)
            if color is None:
                color_palette = px.colors.qualitative.Plotly
                color = color_palette[hash(friendly_name) % len(color_palette)]
            
            # Prepare the share values per commodity, applying the threshold.
            fuel_shares = []
            for com in commodities:
                share_series = merged.loc[(merged['COMMODITY'] == com) & (merged['FFF'] == fuel), 'share']
                if not share_series.empty:
                    share_val = share_series.values[0]
                    # Only include the share if it is above the threshold; otherwise set to 0.
                    fuel_shares.append(share_val if share_val >= threshold else 0)
                else:
                    fuel_shares.append(0)
            
            # Only add the trace if there is any production for this fuel in the scenario.
            if sum(fuel_shares) > 0:
                fig.add_trace(
                    go.Bar(
                        x=commodities,
                        y=fuel_shares,
                        name=friendly_name,
                        marker_color=color,
                        # Only show legend once (in the first subplot) for each fuel.
                        showlegend=True if friendly_name not in encountered_legends else False,
                    ),
                    row=1,
                    col=idx+1
                )
                encountered_legends.add(friendly_name)
        
        # Update subplot x-axes.
        fig.update_xaxes(
            title_text='COMMODITY',
            showline=True,
            mirror=True,
            linewidth=1,
            linecolor='black',
            showgrid=False,
            row=1, col=idx+1
        )
        # Only label the y-axis in the first subplot.
        y_title = 'Share of Production' if idx == 0 else ''
        fig.update_yaxes(
            title_text=y_title,
            showline=True,
            mirror=True,
            linewidth=1,
            linecolor='black',
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=0.6,
            zeroline=True,
            zerolinewidth=0.6,
            zerolinecolor='lightgray',
            row=1, col=idx+1,
            tickformat=".0%"
        )
    
    # Update overall layout.
    fig.update_layout(
        title=plot_title,
        barmode='stack',
        font=dict(family='DejaVu Sans, sans-serif', size=14, color='black'),
        paper_bgcolor='white',
        plot_bgcolor='white',
        legend=dict(
            bgcolor='white',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=12)
        ),
        margin=dict(l=50, r=20, t=80, b=60)
    )
    
    fig.show()


def multi_scenario_objective_histogram_simple(scenario_list, country, plot_title="Objective Breakdown by Scenario"):
    """
    Creates one histogram per scenario, with one bar per SUBCATEGORY
    from df_OBJ_YCR, filtered by the specified country. 
    Bars are not stacked by OBJECTIVE—only total values per SUBCATEGORY are shown.
    
    Parameters:
    - scenario_list: List of tuples (scenario_name, scenario_folder)
    - country: Country to filter by (set to 'all' to skip filtering)
    - plot_title: Title of the whole figure
    """
    
    # Set up the subplots: one per scenario
    fig = make_subplots(
        rows=1,
        cols=len(scenario_list),
        shared_yaxes=True,
        subplot_titles=[s[0] for s in scenario_list]
    )

    for idx, (scenario_name, scenario_folder) in enumerate(scenario_list):
        # Load MainResults.gdx path
        main_results_path = os.path.join(scenario_folder, "MainResults.gdx")
        if not os.path.exists(main_results_path):
            raise FileNotFoundError(f"MainResults.gdx not found in folder: {scenario_folder}")
        
        # Load data (adjust index as needed for your Import_BalmorelMR)
        df_CC_YCRAG, df_F_CONS_YCRA, df_EMI_YCRAG, df_G_CAP_YCRAF, df_PRO_YCRAGF, df_OBJ_YCR, df_EL_DEMAND_YCR, df_H_DEMAND_YCRA, df_H2_DEMAND_YCR, df_X_FLOW_YCR, df_XH2_FLOW_YCR, df_XH_FLOW_YCA = Import_BalmorelMR(main_results_path)

        # Filter by country
        if country != 'all':
            df_OBJ_YCR = df_OBJ_YCR[df_OBJ_YCR['C'] == country]

        # Check if required columns exist
        if not {'SUBCATEGORY', 'value'}.issubset(df_OBJ_YCR.columns):
            raise KeyError("Expected columns 'SUBCATEGORY' and 'value' not found in df_OBJ_YCR.")

        # Group by SUBCATEGORY (ignoring OBJECTIVE)
        grouped = df_OBJ_YCR.groupby('SUBCATEGORY', as_index=False)['value'].sum()

        # Sort for consistent bar order
        grouped = grouped.sort_values('value', ascending=False)

        # Add the bar trace to the current subplot
        fig.add_trace(
            go.Bar(
                x=grouped['SUBCATEGORY'],
                y=grouped['value'],
                marker_color='steelblue',
                showlegend=False
            ),
            row=1,
            col=idx+1
        )

        # X-axis formatting
        fig.update_xaxes(
            title_text='SUBCATEGORY',
            tickangle=45,
            showgrid=False,
            row=1,
            col=idx+1
        )

        # Y-axis only on first plot
        fig.update_yaxes(
            title_text='M€' if idx == 0 else '',
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='lightgray',
            row=1,
            col=idx+1
        )

    # Final layout tweaks
    fig.update_layout(
        title=plot_title,
        font=dict(size=14),
        barmode='group',
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=50, r=20, t=80, b=60)
    )

    fig.show()

def stacked_objective_by_subcategory(scenario_list, country, plot_title="Stacked Objective Breakdown by Scenario"):
    """
    Creates a single bar chart with one bar per scenario, stacked by SUBCATEGORY
    using the df_OBJ_YCR DataFrame (after filtering by country).
    
    Parameters:
    - scenario_list: List of tuples (scenario_name, scenario_folder)
    - country: Country to filter by (set to 'all' to skip filtering)
    - plot_title: Title of the plot
    """


    # Dictionary to hold data: {scenario_name: {subcategory: value}}
    scenario_data = {}

    all_subcategories = set()

    for scenario_name, scenario_folder in scenario_list:
        # Path to data
        main_results_path = os.path.join(scenario_folder, "MainResults.gdx")
        if not os.path.exists(main_results_path):
            raise FileNotFoundError(f"MainResults.gdx not found in folder: {scenario_folder}")

        # Load data
        df_CC_YCRAG, df_F_CONS_YCRA, df_EMI_YCRAG, df_G_CAP_YCRAF, df_PRO_YCRAGF, df_OBJ_YCR, df_EL_DEMAND_YCR, df_H_DEMAND_YCRA, df_H2_DEMAND_YCR, df_X_FLOW_YCR, df_XH2_FLOW_YCR, df_XH_FLOW_YCA = Import_BalmorelMR(main_results_path)

        # Filter by country
        if country != 'all':
            df_OBJ_YCR = df_OBJ_YCR[df_OBJ_YCR['C'] == country]

        # Check required columns
        if not {'SUBCATEGORY', 'value'}.issubset(df_OBJ_YCR.columns):
            raise KeyError("Expected columns 'SUBCATEGORY' and 'value' not found in df_OBJ_YCR.")

        # Group by SUBCATEGORY
        grouped = df_OBJ_YCR.groupby('SUBCATEGORY', as_index=False)['value'].sum()

        # Store values
        scenario_data[scenario_name] = dict(zip(grouped['SUBCATEGORY'], grouped['value']))
        all_subcategories.update(grouped['SUBCATEGORY'])

    # Sort subcategories for consistency
    all_subcategories = sorted(all_subcategories)
    scenario_names = [s[0] for s in scenario_list]

    # Create one Bar per SUBCATEGORY (stacked)
    fig = go.Figure()

    for subcat in all_subcategories:
        y_vals = [scenario_data.get(scenario, {}).get(subcat, 0) for scenario in scenario_names]
        fig.add_trace(
            go.Bar(
                x=scenario_names,
                y=y_vals,
                name=subcat
            )
        )

    # Final layout
    fig.update_layout(
        title=plot_title,
        barmode='stack',
        font=dict(size=14),
        paper_bgcolor='white',
        plot_bgcolor='white',
        xaxis_title='Scenario',
        yaxis_title='M€',
        margin=dict(l=50, r=20, t=80, b=60)
    )

    fig.show()


def multi_scenario_objective(scenario_list,  plot_title="Objective Function Value"):
    """
    Plots the contenent of VOBJ from Optiflow_MainResults.gdx for each scenario.
    
    """


    # Dictionary to hold data: {scenario_name: {subcategory: value}}
    scenario_data = {}



    for scenario_name, scenario_folder in scenario_list:
        # Path to data
        main_results_path = os.path.join(scenario_folder, "Optiflow_MainResults.gdx")
        if not os.path.exists(main_results_path):
            raise FileNotFoundError(f"Optiflow_MainResults.gdx not found in folder: {scenario_folder}")

        # Load data
        df_FLOWA, df_FLOWC, df_EMI_YCRAG, df_EMI_PROC, df_VOBJ = Import_OptiflowMR(main_results_path)

        # Check if required columns exist
        if not {'level'}.issubset(df_VOBJ.columns):
            raise KeyError("Expected column 'Level' not found in df_VOBJ.")

        # Extract the value for the scenario
        total_value = df_VOBJ['level'].sum()

        # Store the total value for the scenario
        scenario_data[scenario_name] = total_value

    # Create a bar plot
    fig = go.Figure()

    # Add a bar for each scenario
    for scenario_name, value in scenario_data.items():
        fig.add_trace(
            go.Bar(
                x=[scenario_name],
                y=[value],
                name=scenario_name,
                marker_color='steelblue'
            )
        )

    # Final layout
    fig.update_layout(
    title=plot_title,
    xaxis_title='Scenario',
    yaxis_title='M€',
    font=dict(size=14),
    paper_bgcolor='white',
    plot_bgcolor='white',
    margin=dict(l=50, r=20, t=80, b=60)
    )

    fig.show()

       
# ------------------------------------------------------------------------------
# F) Execute plotting functions and save the plots as needed
# ------------------------------------------------------------------------------
multi_scenario_fuel_supply(
    scenario_list,
    Demands_name,
    Fuels_name,
    Resource_name,
    year=2050,
    plot_title="Fuels Demand Supply"
)

multi_scenario_stacked_emissions(
    scenario_list,
    plot_title="CO2 Emissions "
)

multi_scenario_biomass_consumption(
    scenario_list,
    plot_title="Biomass Consumption"
)

multi_scenario_gcap_histogram(
    scenario_list,
    country='DENMARK',
    plot_title="Total Installed Capacity by Scenario"
)

multi_scenario_pro_histogram(
    scenario_list,
    country='DENMARK',
    plot_title="Production by Scenario (in Balmorel)",
    demand_markers=False
)

multi_scenario_impexp_histogram(scenario_list,
                                country='DENMARK', 
                                marker_length=110,
                                plot_title="Imports and Exports by Scenario")

multi_scenario_fuel_share_histogram(
    scenario_list,
    country='DENMARK',
    plot_title="Energy Production by Source"
)

multi_scenario_objective_histogram_simple(
        scenario_list, 
        country='DENMARK', 
        plot_title="Objective Breakdown by Scenario")

stacked_objective_by_subcategory(scenario_list, 
                                 country='all', 
                                 plot_title="Stacked Objective Breakdown by Scenario")

multi_scenario_objective(scenario_list, 
                         plot_title="Objective Function Value")
