
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import gams.transfer as gt
import plots_pybalmorel as pyb
import gams
import os
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ------------------------------------------------------------------------------
# A) DEFINE YOUR SCENARIOS
#    Each scenario is a tuple: (scenario_name, scenario_path)
# ------------------------------------------------------------------------------
scenario_list = [
    ("Base Case", r"C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Run_on_HPC\Balmorel\Base_Case\model"),
    ("CO2 Scenario", r"C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Run_on_HPC\Balmorel\CO2_Case\model"),
    ("Biodiversity+CO2 Scenario", r"C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Run_on_HPC\Balmorel\Biodiversity_Case_RLC\model"),

    ("Biodiversity+CO2 Fossil Scenario", r"C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Run_on_HPC\Balmorel\Biodiversity_Case_FOSSIL\model"),
#    ("Biodiversity+CO2 with Fossil Scenario", r"C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Balmorel\Biodiversity_Case_RLC_FOSSIL\model"),

]

Resource_name = {
    'Biomass_for_use':'Biomass',
    'Hydrogen_Use':'Hydrogen'
}

Demands_name = {
    'Sea_fuels_sum':'Maritime demand',
    'Road_fuels_sum':'Road demand',
    'Air_fuels_sum':'Air demand'
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
    return df_FLOWA, df_FLOWC, df_EMI_YCRAG, df_EMI_PROC

def Import_BalmorelMR(file_path):
    df = gt.Container(file_path)
    df_CC_YCRAG = pd.DataFrame(df["CC_YCRAG"].records)
    df_F_CONS_YCRA = pd.DataFrame(df["F_CONS_YCRA"].records)
    df_EMI_YCRAG = pd.DataFrame(df["EMI_YCRAG"].records)
    return df_CC_YCRAG, df_F_CONS_YCRA, df_EMI_YCRAG

# ------------------------------------------------------------------------------
# C) DATA-PROCESSING FUNCTIONS
# ------------------------------------------------------------------------------
def group_EMI_YCRAG(df_EMI_YCRAG, df_FLOWC, df_EMI_PROC):
    df_emi = df_EMI_YCRAG.copy()
    df_emi['Category'] = None

    # Define masks for categorization
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

    # Process CO2 flows from df_FLOWC
    co2_map = {
        'CO2_DAC_Total': 'DAC',
        'CO2_Land': 'Land Solutions',
        'CO2_Biochar_Sum': 'Biochar Sequestration',
        'CO2_Biogas': 'Biogas'
    }
    co2_list = ['CO2_Land', 'CO2_Biochar_Sum', 'CO2_Biogas']
    df_co2 = df_FLOWC[df_FLOWC['IPROCFROM'].isin(co2_list)].copy()
    df_co2['Category'] = df_co2['IPROCFROM'].map(co2_map)
    df_co2['value'] = df_co2['value'] * -1000  # Convert from Mt to kton and apply negative sign

    # DAC -> Sequestration
    df_co2_dac = df_FLOWC[
        (df_FLOWC['IPROCFROM'] == 'CO2_DAC_Total') &
        (df_FLOWC['IPROCTO'].str.contains('CO2_Seq', case=False, na=False))
    ].copy()
    df_co2_dac['Category'] = df_co2_dac['IPROCFROM'].map(co2_map)
    df_co2_dac['value'] = df_co2_dac['value'] * -1000

    # Check if df_EMI_PROC is empty or lacks the 'PROC' column
    if df_EMI_PROC.empty or 'PROC' not in df_EMI_PROC.columns:
        print("Warning: df_EMI_PROC is empty or missing 'PROC' column.")
        df_fuel = pd.DataFrame(columns=['Category', 'value'])  # Empty DataFrame
    else:
        # Add Marine Shipping Diesel Oil, Kerosene, and Diesel from df_EMI_PROC
        fuel_map = {
            'MDOSource': 'Marine Shipping Diesel Oil',
            'KeroseneSource': 'Kerosene',
            'DieselSource': 'Diesel'
        }
        fuel_list = ['MDOSource', 'KeroseneSource', 'DieselSource']
        df_fuel = df_EMI_PROC[df_EMI_PROC['PROC'].isin(fuel_list)].copy()
        df_fuel['Category'] = df_fuel['PROC'].map(fuel_map)

    # Combine all dataframes
    df_final = pd.concat([df_emi, df_co2, df_co2_dac, df_fuel], ignore_index=True)

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
    mask_idv = df_f_cons_filtered['AAA'].str.contains('IDVU',case=False, na=False)
    df_f_cons_filtered.loc[mask_ind,'Category'] = 'Industry Heating'
    df_f_cons_filtered.loc[mask_idv,'Category'] = 'Individual Users Heating'

    return df_flowc_filtered, df_f_cons_filtered
  
# ------------------------------------------------------------------------------
# D) MULTI-SCENARIO PLOTTING FUNCTIONS (X-axis title REMOVED)
# ------------------------------------------------------------------------------
def multi_scenario_fuel_supply(
    scenarios,
    Demands_name,
    Fuels_name,
    Resource_name,
    year,
    plot_title="Fuel Supply Comparison"
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
        'E-Gasoline': '#d62728',
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
        df_FLOWA, df_FLOWC, df_EMI_YCRAG, df_EMI_PROC = Import_OptiflowMR(optiflow_path)
        df_CC, df_F_CONS, df_EMI_b = Import_BalmorelMR(main_results_path)

    
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

        y_title = 'PJ' if idx==0 else ''
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

def multi_scenario_stacked_emissions(scenarios, plot_title="Stacked Emissions by Scenario"):
    """
    Creates a single figure with subplots in columns (1 x #scenarios).
    Each subplot is stacked emissions for that scenario.

    - The scenario name is shown in the subplot title (top).
    - We use a numeric x-axis at x=0 (type="linear").
    - We hide all tick labels with showticklabels=False.
    - That way, no scenario labels appear at the bottom.
    """

    # Create subplots: 1 row, N columns
    fig = make_subplots(
        rows=1,
        cols=len(scenarios),
        # put scenario names in subplot titles (top)
        subplot_titles=[s[0] for s in scenarios],
        shared_yaxes=True  # share scale across subplots
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
        "Land Solutions":              "#D9D9D9",
        "Biochar Sequestration":       "#5c3f3d",
        "Marine Shipping Diesel Oil":  "#0057b8",
        "Kerosene":                    "#ff4500",
        "Diesel":                      "#ffa500"
    }
    fallback_color = "#999999"

    # Track categories encountered so we show each in the legend only once
    encountered_categories = set()

    for idx, (scenario_name, scenario_path) in enumerate(scenarios):

        # Load data
        main_results_path = os.path.join(scenario_path, "MainResults.gdx")
        optiflow_path     = os.path.join(scenario_path, "Optiflow_MainResults.gdx")

        df_FLOWA, df_FLOWC, df_EMI_opt, df_EMI_PROC = Import_OptiflowMR(optiflow_path)
        df_CC, df_F_CONS, df_EMI_bal   = Import_BalmorelMR(main_results_path)

        # Process for stacked emissions
        df_agg = group_EMI_YCRAG(df_EMI_opt, df_FLOWC, df_EMI_PROC)

        # Split positive & negative
        df_pos = df_agg[df_agg["value"] >= 0].copy()
        df_neg = df_agg[df_agg["value"] < 0].copy()

        # Sort for stacking
        df_pos.sort_values("value", inplace=True)
        df_neg.sort_values("value", ascending=False, inplace=True)

        # Manual stack base
        df_pos["base"] = df_pos["value"].cumsum() - df_pos["value"]
        df_neg["base"] = df_neg["value"].cumsum() - df_neg["value"]

        # Instead of using scenario_name for x, we use numeric 0
        # so there's no category text to show
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
                    x=[x_val],  # numeric array
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

        # Style the x-axis to remove ticks/labels
        fig.update_xaxes(
            type="linear",           # numeric axis
            range=[-1, 1],          # small range around 0 (just so bars are visible)
            showticklabels=False,    # remove tick labels
            showline=True,
            mirror=True,
            linewidth=1,
            linecolor="black",
            showgrid=False,
            title_text='',           # no axis title
            row=1, col=c_col
        )

        # Only put Y-axis label on first subplot
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

    # Final layout
    fig.update_layout(
        title=plot_title,
        barmode="overlay",  # manual stacking
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

def multi_scenario_biomass_consumption(scenarios, plot_title="Biomass Consumption Comparison"):
    fig = make_subplots(
        rows=1,
        cols=len(scenarios),
        shared_yaxes=True,
        subplot_titles=[s[0] for s in scenarios],
    )

    # Fuel color map
    fuel_colors = {
        'Wood Chips':   '#1f77b4',
        'Wood Pellets': '#ff7f0e',
        'Straw':        '#2ca02c'
    }
    fallback_color = 'gray'

    # Keep track of which fuel names have been used in the legend
    encountered_legends = set()

    for idx, (scenario_name, scenario_path) in enumerate(scenarios):
        main_results_path = os.path.join(scenario_path, "MainResults.gdx")
        optiflow_path     = os.path.join(scenario_path, "Optiflow_MainResults.gdx")

        df_FLOWA, df_FLOWC, df_EMI_opt, df_EMI_PROC = Import_OptiflowMR(optiflow_path)
        df_CC, df_F_CONS, df_EMI_bal   = Import_BalmorelMR(main_results_path)

        # 1) Process flows & consumption
        df_flowc_filtered, df_f_cons_filtered = process_flows_and_consumption(df_FLOWC, df_F_CONS)

        # 2) Summation by (Category, FFF)
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
            return None  # fallback

        fuels = ['Wood Chips','Wood Pellets','Straw']
        fuel_arrays = {f: [0]*5 for f in fuels}  # 5 columns

        # Columns 0..2 => from df_cat_fuel
        for _, row2 in df_cat_fuel.iterrows():
            cidx = map_category_to_colidx(row2['Category'])
            f    = row2['Fuel']
            v    = row2['value']
            if cidx is not None and f in fuel_arrays:
                fuel_arrays[f][cidx] += v

        # 4th column => df_flowc_4th
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

        # 5th column => negative
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

        # Subtract from 5th col
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

        # X-axis labels
        x_labels = [
            "Industry Heating",
            "Individual Users",
            "CHP Generation",
            "Fuels Production",
            "Total Biomass"
        ]

        c_col = idx+1
        # Now we add stacked bars for each fuel in fuel_arrays
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

        # Style
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


multi_scenario_fuel_supply(
    scenario_list,
    Demands_name,
    Fuels_name,
    Resource_name,
    year=2050,
    plot_title="Fuel Supply across Scenarios"
)

multi_scenario_stacked_emissions(
    scenario_list,
    plot_title="Stacked Emissions across Scenarios"
)

multi_scenario_biomass_consumption(
    scenario_list,
    plot_title="Biomass Consumption across Scenarios"
)


#%%
def plot_municipalities(df, shapefile_path, column_municipality, column_value, filter_column, filter_value, plot_title, cmap):
    # Municipality name mapping
    municipality_name_mapping = {
        "Albertslund": "Albertslund",
        "Alleroed": "Allerød",
        "Assens": "Assens",
        "Ballerup": "Ballerup",
        "Billund": "Billund",
        "Bornholm": "Bornholm",
        "Broendby": "Brøndby",
        "Broenderslev": "Brønderslev",
        "Christiansoe": "Christiansø",
        "Dragoer": "Dragør",
        "Egedal": "Egedal",
        "Esbjerg": "Esbjerg",
        "Fanoe": "Fanø",
        "Favrskov": "Favrskov",
        "Faxe": "Faxe",
        "Fredensborg": "Fredensborg",
        "Fredericia": "Fredericia",
        "Frederiksberg": "Frederiksberg",
        "Frederikshavn": "Frederikshavn",
        "Frederikssund": "Frederikssund",
        "Furesoe": "Furesø",
        "Faaborg-Midtfyn": "Faaborg-Midtfyn",
        "Gentofte": "Gentofte",
        "Gladsaxe": "Gladsaxe",
        "Glostrup": "Glostrup",
        "Greve": "Greve",
        "Gribskov": "Gribskov",
        "Guldborgsund": "Guldborgsund",
        "Haderslev": "Haderslev",
        "Halsnaes": "Halsnæs",
        "Hedensted": "Hedensted",
        "Helsingoer": "Helsingør",
        "Herlev": "Herlev",
        "Herning": "Herning",
        "Hilleroed": "Hillerød",
        "Hjoerring": "Hjørring",
        "Holbaek": "Holbæk",
        "Holstebro": "Holstebro",
        "Horsens": "Horsens",
        "Hvidovre": "Hvidovre",
        "Hoeje_Taastrup": "Høje-Taastrup",
        "Hoersholm": "Hørsholm",
        "Ikast-Brande": "Ikast-Brande",
        "Ishoej": "Ishøj",
        "Jammerbugt": "Jammerbugt",
        "Kalundborg": "Kalundborg",
        "Kerteminde": "Kerteminde",
        "Kolding": "Kolding",
        "Koebenhavn": "København",
        "Koege": "Køge",
        "Langeland": "Langeland",
        "Lejre": "Lejre",
        "Lemvig": "Lemvig",
        "Lolland": "Lolland",
        "Lyngby-Taarbaek": "Lyngby-Taarbæk",
        "Laesoe": "Læsø",
        "Mariagerfjord": "Mariagerfjord",
        "Middelfart": "Middelfart",
        "Morsoe": "Morsø",
        "Norddjurs": "Norddjurs",
        "Nordfyns": "Nordfyns",
        "Nyborg": "Nyborg",
        "Naestved": "Næstved",
        "Odder": "Odder",
        "Odense": "Odense",
        "Odsherred": "Odsherred",
        "Randers": "Randers",
        "Rebild": "Rebild",
        "Ringkoebing-Skjern": "Ringkøbing-Skjern",
        "Ringsted": "Ringsted",
        "Roskilde": "Roskilde",
        "Rudersdal": "Rudersdal",
        "Roedovre": "Rødovre",
        "Samsoe": "Samsø",
        "Silkeborg": "Silkeborg",
        "Skanderborg": "Skanderborg",
        "Skive": "Skive",
        "Slagelse": "Slagelse",
        "Solroed": "Solrød",
        "Soroe": "Sorø",
        "Stevns": "Stevns",
        "Struer": "Struer",
        "Svendborg": "Svendborg",
        "Syddjurs": "Syddjurs",
        "Soenderborg": "Sønderborg",
        "Thisted": "Thisted",
        "Toender": "Tønder",
        "Taarnby": "Tårnby",
        "Vallensbaek": "Vallensbæk",
        "Varde": "Varde",
        "Vejen": "Vejen",
        "Vejle": "Vejle",
        "Vesthimmerlands": "Vesthimmerlands",
        "Viborg": "Viborg",
        "Vordingborg": "Vordingborg",
        "Aeroe": "Ærø",
        "Aabenraa": "Aabenraa",
        "Aalborg": "Aalborg",
        "Aarhus": "Aarhus",
    }

        # Filter the DataFrame based on the specified value in the filter_column
    filtered_df = df[df[filter_column] == filter_value]
    
    # Rename municipalities in the DataFrame according to the mapping
    filtered_df[column_municipality] = filtered_df[column_municipality].map(municipality_name_mapping)
    
    # Load the shapefile
    Municipality_gdf = gpd.read_file(shapefile_path)
    
    # Ensure the column names match
    column_municipality_gdf = 'LAU_NAME'  # Replace with the actual column name in your shapefile
    
    # Merge the filtered DataFrame with the GeoDataFrame
    merged = Municipality_gdf.merge(filtered_df, left_on=column_municipality_gdf, right_on=column_municipality, how='left')
    
    # Fill NaN values with 0
    merged[column_value] = merged[column_value].fillna(0)
    
    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_facecolor('lightgrey')  # Set the background color to greyish
    merged.plot(column=column_value, ax=ax, legend=True, cmap=cmap, edgecolor='black',
                legend_kwds={'label': plot_title,
                             'orientation': "horizontal"})
    
    # Remove axes
    ax.set_axis_off()
    
    # Add title and show plot
    plt.title(plot_title)
    plt.show()

df_FLOWA, df_FLOWC, df_EMI_YCRAG = Import_OptiflowMR(OptiflowMR_path)
plot_municipalities( df_FLOWA, shapefile_path=r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\LAU_RG_01M_2021_3035.shp\Administrative_DK.shp', column_municipality='AAA', column_value='value', filter_column='IPROCFROM',filter_value='Straw_for_Energy', plot_title='Straw for Energy in Denmark [PJ]', cmap='Reds')
plot_municipalities( df_FLOWA, shapefile_path=r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\LAU_RG_01M_2021_3035.shp\Administrative_DK.shp', column_municipality='AAA', column_value='value', filter_column='IPROCFROM', filter_value='Wood_for_Energy', plot_title='Wood for Energy in Denmark  [PJ]', cmap='Greens')
plot_municipalities( df_FLOWA, shapefile_path=r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\LAU_RG_01M_2021_3035.shp\Administrative_DK.shp', column_municipality='AAA', column_value='value', filter_column='IPROCFROM', filter_value='Land_for_Wood_Production', plot_title='Productive Forest in Denmark [Mha]', cmap='Blues')
plot_municipalities( df_FLOWA, shapefile_path=r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\LAU_RG_01M_2021_3035.shp\Administrative_DK.shp', column_municipality='AAA', column_value='value', filter_column='IPROCFROM', filter_value='Agriculture', plot_title='Agricultural Land in Denmark [Mha]', cmap='Purples')


# %%
# Filter the DataFrame for the specified IPROCFROM values
filtered_df = df_FLOWC[df_FLOWC['IPROCFROM'].isin(['Agricultural_Land_Gen', 'Land_for_Wood_Production', 'Wood_for_Energy', 'Straw_for_Energy','Wood_for_Use', 'Straw_for_Use'])]

# Print the 'value' column for the filtered rows
print(filtered_df[['IPROCFROM', 'value']])
# %%
# Clear all variables



# # %%

# %%
