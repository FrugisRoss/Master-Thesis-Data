
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
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.graph_objects as go


# ------------------------------------------------------------------------------
# A) DEFINE YOUR SCENARIOS
# Each scenario is a tuple: (scenario_name, scenario_path)
# ------------------------------------------------------------------------------
scenario_list = [
    # ("CO2 Scenario", r"C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Run_on_HPC\Balmorel\CO2_Case_RLC\model"),
    # ("CO2 Scenario +150% LC", r"C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Run_on_HPC\Balmorel\CO2_Case_RLC+150cost\model"),
    # ("CO2 Scenario +300% LC", r"C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Run_on_HPC\Balmorel\CO2_Case_RLC+300cost\model"),
    ("Base Case", r"C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Run_on_HPC\Balmorel\Base_Case\model"),
    ("CO2 Scenario", r"C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Run_on_HPC\Balmorel\CO2_Case_RLC\model"),
     ("Biodiversity+CO2 Scenario", r"C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Run_on_HPC\Balmorel\Biodiversity_Case_RLC\model"),
     ("Biodiversity+CO2 Fossil Scenario", r"C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Run_on_HPC\Balmorel\Biodiversity_Case_RLC_FOSSIL\model"),
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
    return df_FLOWA, df_FLOWC, df_EMI_YCRAG, df_EMI_PROC

def Import_BalmorelMR(file_path):
    df = gt.Container(file_path)
    df_CC_YCRAG = pd.DataFrame(df["CC_YCRAG"].records)
    df_F_CONS_YCRA = pd.DataFrame(df["F_CONS_YCRA"].records)
    df_EMI_YCRAG = pd.DataFrame(df["EMI_YCRAG"].records)
    df_G_CAP_YCRAF = pd.DataFrame(df["G_CAP_YCRAF"].records)
    df_PRO_YCRAGF = pd.DataFrame(df["PRO_YCRAGF"].records)
    return df_CC_YCRAG, df_F_CONS_YCRA, df_EMI_YCRAG, df_G_CAP_YCRAF, df_PRO_YCRAGF

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
        df_FLOWA, df_FLOWC, df_EMI_YCRAG, df_EMI_PROC = Import_OptiflowMR(optiflow_path)
        df_CC, df_F_CONS, df_EMI_b, df_G_CAP_YCRAF, df_PRO_YCRAGF = Import_BalmorelMR(main_results_path) 

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
        "Land Solutions":              "#D9D9D9",
        "Biochar Sequestration":       "#5c3f3d",
        "Marine Shipping Diesel Oil":  "#0057b8",
        "Kerosene":                    "#ff4500",
        "Diesel":                      "#ffa500"
    }
    fallback_color = "#999999"

    encountered_categories = set()

    for idx, (scenario_name, scenario_path) in enumerate(scenarios):
        main_results_path = os.path.join(scenario_path, "MainResults.gdx")
        optiflow_path     = os.path.join(scenario_path, "Optiflow_MainResults.gdx")

        df_FLOWA, df_FLOWC, df_EMI_opt, df_EMI_PROC = Import_OptiflowMR(optiflow_path)
        df_CC, df_F_CONS, df_EMI_bal, df_G_CAP_YCRAF, df_PRO_YCRAGF   = Import_BalmorelMR(main_results_path)

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

        df_FLOWA, df_FLOWC, df_EMI_opt, df_EMI_PROC = Import_OptiflowMR(optiflow_path)
        df_CC, df_F_CONS, df_EMI_bal, df_G_CAP_YCRAF, df_PRO_YCRAGF   = Import_BalmorelMR(main_results_path)

        df_flowc_filtered, df_f_cons_filtered = process_flows_and_consumption(df_FLOWC, df_F_CONS)

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


def multi_scenario_gcap_histogram(scenarios, plot_title="G_CAP_YCRAF Histogram by Scenario"):
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
        df_CC, df_F_CONS, df_EMI, df_G_CAP_YCRAF, df_PRO_YCRAGF = Import_BalmorelMR(main_results_path)

        # Filter for rows where 'C' is 'DENMARK'
        df_G_CAP_YCRAF = df_G_CAP_YCRAF[df_G_CAP_YCRAF['C'] == 'DENMARK']
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


def multi_scenario_pro_histogram(scenarios, plot_title="PRO_YCRAGF Histogram by Scenario"):
    """
    For each scenario, this function plots a histogram using the df_PRO_YCRAGF data.
    The x-axis contains one column per unique 'COMMODITY', and for each commodity the bar is stacked by 'TECH_TYPE'
    (with friendly technology names and fixed colors shown in the legend). Rows where 'TECH_TYPE'
    contains 'STORAGE' are excluded.
    Only values higher than 10e-9 (i.e. ≥ 1e-8) are plotted.
    """

    # Set threshold for PRO_YCRAGF histogram: only values >= 10e-9 (i.e. 1e-8) are plotted.
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

    color_palette = px.colors.qualitative.Plotly
    color_index = {}

    fig = make_subplots(
        rows=1,
        cols=len(scenarios),
        shared_yaxes=True,
        subplot_titles=[s[0] for s in scenarios]
    )

    encountered_legends = set()

    for idx, (scenario_name, scenario_path) in enumerate(scenarios):
        main_results_path = os.path.join(scenario_path, "MainResults.gdx")
        df_CC, df_F_CONS, df_EMI, df_G_CAP_YCRAF, df_PRO_YCRAGF = Import_BalmorelMR(main_results_path)

        df_PRO_YCRAGF = df_PRO_YCRAGF[df_PRO_YCRAGF['C'] == 'DENMARK']
        if 'TECH_TYPE' in df_PRO_YCRAGF.columns:
            df_PRO_YCRAGF = df_PRO_YCRAGF[~df_PRO_YCRAGF['TECH_TYPE'].astype(str).str.contains('STORAGE', case=False, na=False)]

        grouped = df_PRO_YCRAGF.groupby(['COMMODITY', 'TECH_TYPE'], as_index=False)['value'].sum()
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

        fig.update_xaxes(
            title_text='COMMODITY',
            showline=True,
            mirror=True,
            linewidth=1,
            linecolor='black',
            showgrid=False,
            row=1, col=idx+1
        )
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
            font=dict(size=12)
        ),
        margin=dict(l=50, r=20, t=80, b=60)
    )
    fig.show()


def multi_scenario_fuel_share_histogram(scenario_list, plot_title="Production Fuel Share Histogram"):
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
        _, _, _, _, df_PRO_YCRAGF = Import_BalmorelMR(main_results_path)
        
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



# ------------------------------------------------------------------------------
# F) Execute plotting functions and save the plots as needed
# ------------------------------------------------------------------------------
multi_scenario_fuel_supply(
    scenario_list,
    Demands_name,
    Fuels_name,
    Resource_name,
    year=2050,
    plot_title="Fuels Demand Supply across Scenarios"
)

multi_scenario_stacked_emissions(
    scenario_list,
    plot_title="CO2 Emissions across Scenarios"
)

multi_scenario_biomass_consumption(
    scenario_list,
    plot_title="Biomass Consumption across Scenarios"
)

multi_scenario_gcap_histogram(
    scenario_list,
    plot_title="Total Installed Capacity by Scenario"
)

multi_scenario_pro_histogram(
    scenario_list,
    plot_title="Production by Scenario"
)

multi_scenario_fuel_share_histogram(
    scenario_list,
    plot_title="REP by Scenario"
)
#%%

# Set a system font to avoid missing font warnings
plt.rcParams['font.family'] = 'Arial'

# === Municipality Name Mapping ===
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

# === Import Function ===
def Import_OptiflowMR(file_path):
    df = gt.Container(file_path)
    df_FLOWA = pd.DataFrame(df["VFLOW_Opti_A"].records)
    # Ensure that the 'value' column is numeric.
    df_FLOWA["value"] = pd.to_numeric(df_FLOWA["value"], errors='coerce')
    # Apply the municipality mapping.
    df_FLOWA["AAA"] = df_FLOWA["AAA"].replace(municipality_name_mapping)
    
    df_FLOWC = pd.DataFrame(df["VFLOW_Opti_C"].records)
    df_EMI_YCRAG = pd.DataFrame(df["EMI_YCRAG"].records)
    df_EMI_PROC = pd.DataFrame(df["EMI_PROC"].records)
    return df_FLOWA, df_FLOWC, df_EMI_YCRAG, df_EMI_PROC

# === Function to Plot Maps ===
def plot_municipalities(df, shapefile_path, column_municipality, column_value,
                        filter_column_from, filter_column_to, filter_pairs, cmap,
                        scenario_name, ax, norm):
    """Plot a map of Denmark's municipalities with aggregated values based on multiple (IPROCFROM, IPROCTO) pairs."""
    
    # Load the shapefile.
    Municipality_gdf = gpd.read_file(shapefile_path)
    column_municipality_gdf = "LAU_NAME"  # adjust as needed

    # Normalize names and filtering columns.
    df[column_municipality] = df[column_municipality].astype(str).str.strip()
    Municipality_gdf[column_municipality_gdf] = Municipality_gdf[column_municipality_gdf].astype(str).str.strip()
    df[filter_column_from] = df[filter_column_from].astype(str).str.strip()
    df[filter_column_to] = df[filter_column_to].astype(str).str.strip()
    
    # Use case-insensitive filtering and allow the target string to start with the expected value.
    mask = df.apply(lambda row: any(
        (row[filter_column_from].upper() == pair[0].strip().upper() and
         row[filter_column_to].upper().startswith(pair[1].strip().upper()))
         for pair in filter_pairs), axis=1)
    
    filtered_df = df[mask]
    aggregated_df = filtered_df.groupby("AAA", observed=False)["value"].sum().reset_index()

    # Merge with municipality shapefile.
    merged = Municipality_gdf.merge(aggregated_df, left_on=column_municipality_gdf, right_on="AAA", how="left")
    merged[column_value] = merged[column_value].fillna(0)

    # Plot with a thin grey border.
    merged.plot(column=column_value, ax=ax, cmap=cmap, edgecolor="grey", linewidth=0.5, norm=norm)
    ax.set_title(scenario_name, fontsize=12,  family="Arial")
    ax.set_axis_off()

    return merged[[column_municipality_gdf, column_value]]  # Return merged data for debugging.

# === List of Scenarios ===
scenario_list = [
#    ("Base Case", r"C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Run_on_HPC\Balmorel\Base_Case\model\Optiflow_MainResults.gdx"),
#     ("CO2 Scenario", r"C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Run_on_HPC\Balmorel\CO2_Case_RLC\model\Optiflow_MainResults.gdx"),
#     ("Biodiversity+CO2 Scenario", r"C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Run_on_HPC\Balmorel\Biodiversity_Case_RLC\model\Optiflow_MainResults.gdx"),
#     ("Biodiversity+CO2 Fossil Scenario", r"C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Run_on_HPC\Balmorel\Biodiversity_Case_RLC_FOSSIL\model\Optiflow_MainResults.gdx"),
    ("Biodiversity+CO2 Fossil -50% Transport Costs", r"C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Run_on_HPC\Balmorel\Biodiversity_Case_RLC-50tc\model\Optiflow_MainResults.gdx"),

    ("Biodiversity+CO2 Fossil", r"C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Run_on_HPC\Balmorel\Biodiversity_Case_RLC\model\Optiflow_MainResults.gdx"),
    ("Biodiversity+CO2 Fossil +50% Transport Costs", r"C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Run_on_HPC\Balmorel\Biodiversity_Case_RLC+50tc\model\Optiflow_MainResults.gdx"),
#         ("CO2 Scenario", r"C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Run_on_HPC\Balmorel\CO2_Case_RLC\model\Optiflow_MainResults.gdx"),
#         ("CO2 Scenario +150% LC", r"C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Run_on_HPC\Balmorel\CO2_Case_RLC+150cost\model\Optiflow_MainResults.gdx"),
#        ("CO2 Scenario +300% LC", r"C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Run_on_HPC\Balmorel\CO2_Case_RLC+300cost\model\Optiflow_MainResults.gdx"),
]

# === List of Filters to Apply ===
# For the "New Productive Forest [Mha]" plot we wish to capture rows with IPROCFROM "Agricultural_Land"
# and IPROCTO values that start with "New_Productive_Forest"
plot_filters = [
    # ([("Agricultural_Land", "C_Rich_Soils_Extraction_HOV"),
    # ("Agricultural_Land", "C_Rich_Soils_Extraction_SJA"),
    # ("Agricultural_Land", "C_Rich_Soils_Extraction_SYD"),
    # ("Agricultural_Land", "C_Rich_Soils_Extraction_MID"),
    # ("Agricultural_Land", "C_Rich_Soils_Extraction_NOR")],
    # "Carbon Rich Soil Extraction", "Oranges"),

    ([("Agricultural_Land", "New_Productive_Forest_HOV"),
    ("Agricultural_Land", "New_Productive_Forest_SJA"),
    ("Agricultural_Land", "New_Productive_Forest_SYD"),
    ("Agricultural_Land", "New_Productive_Forest_MID"),
    ("Agricultural_Land", "New_Productive_Forest_NOR")],
     "New Productive Forest", "Greens"),

    # ([("Productive_Forest", "Untouched_Forest_HOV"),
    # ("Productive_Forest", "Untouched_Forest_SJA"),
    # ("Productive_Forest", "Untouched_Forest_SYD"),
    # ("Productive_Forest", "Untouched_Forest_MID"),
    # ("Productive_Forest", "Untouched_Forest_NOR")],
    # "New Protected Forest", "Purples"),

    #  ([("Agricultural_Land", "Agriculture")],"Agricultural Land", "Greens"),
    #  ([("Land_for_Wood_Production", "Wood_Production")],"Productive Forest", "Oranges"),
#       ([("CO2_Source_DAC", "CO2_DAC_50"),("CO2_Source_Biogen", "CO2_BIOGEN_TOT"),("CO2_Source_Fossil", "CO2_FOS_TOT")],"Total CO2 Resource[Mton]", "Purples"),
      ([("Air_fuels_sum", "AirBuffer"),
        ("Road_fuels_sum", "RoadBuffer"),
        ("Sea_fuels_sum", "SeaBuffer")],"Renewable Fuel Production", "Reds"),

        ([("BioJet_Eff", "Air_fuels_sum"),("BioGasoline_Eff", "Road_fuels_sum"),],"Biofuels Fuel Production", "Reds"),

 ]

shapefile_path = r"C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\LAU_RG_01M_2021_3035.shp\Administrative_DK.shp"

# === Main Loop Over Each Filter ===
for filter_pairs, plot_title, cmap in plot_filters:
    fig, axes = plt.subplots(1, len(scenario_list), figsize=(20, 5))
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    df_all_scenarios = pd.DataFrame()

    # --- Step 1: Determine Global Color Scale (Min & Max) ---
    global_min, global_max = np.inf, -np.inf

    for scenario_name, file_path in scenario_list:
        df_FLOWA, df_FLOWC, df_EMI_YCRAG, df_EMI_PROC = Import_OptiflowMR(file_path)
        df_FLOWA["AAA"] = df_FLOWA["AAA"].astype(str).str.strip().replace(municipality_name_mapping)
        df_FLOWA["IPROCFROM"] = df_FLOWA["IPROCFROM"].astype(str).str.strip()
        df_FLOWA["IPROCTO"] = df_FLOWA["IPROCTO"].astype(str).str.strip()
        
        mask = df_FLOWA.apply(lambda row: any(
            (row["IPROCFROM"].upper() == pair[0].strip().upper() and
             row["IPROCTO"].upper().startswith(pair[1].strip().upper()))
             for pair in filter_pairs), axis=1)
        filtered_df = df_FLOWA[mask]

        aggregated_df = filtered_df.groupby("AAA", observed=False)["value"].sum().reset_index()

        if aggregated_df.empty:
            aggregated_df["value"] = 0
        else:
            min_val, max_val = aggregated_df["value"].min(), aggregated_df["value"].max()
            global_min, global_max = min(global_min, min_val), max(global_max, max_val)

        aggregated_df.rename(columns={"value": scenario_name}, inplace=True)
        if df_all_scenarios.empty:
            df_all_scenarios = aggregated_df
        else:
            df_all_scenarios = pd.merge(df_all_scenarios, aggregated_df, on="AAA", how="outer")

    df_all_scenarios.fillna(0, inplace=True)
    print(f"\n📊 Data Table for {plot_title}")
    pd.set_option('display.max_rows', None)
    print(df_all_scenarios)
    
    # --- Step 2: Create Shared Color Scale ---
    norm = mcolors.Normalize(vmin=global_min, vmax=global_max)

    # --- Step 3: Generate and Display Plots ---
    for idx, (scenario_name, file_path) in enumerate(scenario_list):
        print(f"\n🔹 Processing {scenario_name} for {filter_pairs}")
        df_FLOWA, df_FLOWC, df_EMI_YCRAG, df_EMI_PROC = Import_OptiflowMR(file_path)
        
        df_FLOWA["AAA"] = df_FLOWA["AAA"].astype(str).str.strip().replace(municipality_name_mapping)
        df_FLOWA["IPROCFROM"] = df_FLOWA["IPROCFROM"].astype(str).str.strip()
        df_FLOWA["IPROCTO"] = df_FLOWA["IPROCTO"].astype(str).str.strip()
        
        plot_municipalities(
            df_FLOWA,
            shapefile_path,
            column_municipality="AAA",
            column_value="value",
            filter_column_from="IPROCFROM",
            filter_column_to="IPROCTO",
            filter_pairs=filter_pairs,
            cmap=cmap,
            scenario_name=scenario_name,
            ax=axes[idx],
            norm=norm
        )

    plt.suptitle(plot_title, fontsize=16, family="Arial")
    plt.tight_layout(rect=[0.1, 0, 1, 0.95])

    # Add Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.99, 0.15, 0.02, 0.7])
    fig.colorbar(sm, cax=cbar_ax, orientation="vertical").set_label("[Mha]", fontsize=14, family="Arial")

    plt.show()
    plt.savefig('Multiple_Scen_Map.svg')

# %%
df_FLOWA, df_FLOWC, df_EMI_YCRAG, df_EMI_PROC=Import_OptiflowMR(r"C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Run_on_HPC\Balmorel\Base_Case\model\Optiflow_MainResults.gdx")

# Filter the DataFrame for the specified IPROCFROM values
filtered_df = df_FLOWC[df_FLOWC['IPROCFROM'].isin(['Agricultural_Land_Gen', 'Land_for_Wood_Production', 'Wood_for_Energy', 'Straw_for_Energy','Wood_for_Use', 'Straw_for_Use'])]

# Print the 'value' column for the filtered rows
print(filtered_df[['IPROCFROM', 'value']])
# %%
# Clear all variables



# # %%

# %%
