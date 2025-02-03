
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

MainResults_path=r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Balmorel\Biodiversity_Case_Prova\model\MainResults.gdx'
OptiflowMR_path=r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Balmorel\Biodiversity_Case_Prova\model\Optiflow_MainResults.gdx'

Resource_name = {'Biomass_for_use':'Biomass',
               'Hydrogen_Use':'Hydrogen'}

Demands_name= {'Sea_fuels_sum':'Maritime demand',
               'Road_fuels_sum':'Road demand',
               'Air_fuels_sum':'Air demand'}

Fuels_name = {
    'Ammonia_Eff': 'Ammonia',
    'Methanol_Eff': 'Methanol',
    'EMethanol_Eff': 'E-Methanol',
    'EME_Gasoline_Eff': 'E-Gasoline',
    'EME_LPG_Eff': 'E-LPG',
    'BioGasoline_Eff': 'Biogasoline',  
    'E_FT_Diesel_Eff': 'E-Diesel',
    'BioJet_Eff': 'Biojet',
    'E-FT-Jet_Eff': 'E-Jet-FT',
    'EME_Jet_Eff': 'E-Jet-ME'
}

# Function to import the table from the MainResults file
def Import_OptiflowMR(file_path):
    df = gt.Container(file_path)
    df_FLOWA = pd.DataFrame(df["VFLOW_Opti_A"].records)
    df_FLOWC = pd.DataFrame(df["VFLOW_Opti_C"].records)
    df_EMI_YCRAG = pd.DataFrame(df["EMI_YCRAG"].records)
    return df_FLOWA, df_FLOWC, df_EMI_YCRAG

def Import_BalmorelMR(file_path):
    df = gt.Container(file_path)
    df_CC_YCRAG = pd.DataFrame(df["CC_YCRAG"].records)
    df_F_CONS_YCRA = pd.DataFrame(df["F_CONS_YCRA"].records)
    df_EMI_YCRAG = pd.DataFrame(df["EMI_YCRAG"].records)
    return df_CC_YCRAG, df_F_CONS_YCRA, df_EMI_YCRAG

df_FLOWA, df_FLOWC, df_EMI_YCRAG=Import_OptiflowMR(OptiflowMR_path)
df_CC_YCRAG, df_F_CONS_YCRA, df_EMI_YCRAG=Import_BalmorelMR(MainResults_path)

def Plot_fuel_supply(MainResults_path, OptiflowMR_path, Demands_name, Fuels_name, Resource_name, year, plot_title):
    df_FLOWA, df_FLOWC, df_EMI_YCRAG = Import_OptiflowMR(OptiflowMR_path)

    # Filter by the specified year
    df_FLOWC = df_FLOWC[df_FLOWC['Y'] == str(year)]

    # Filter for fuel supply
    filtered_df = df_FLOWC[
        df_FLOWC['IPROCTO'].isin(Demands_name.keys())
        & df_FLOWC['IPROCFROM'].isin(Fuels_name.keys())
    ]
    filtered_df['Demand'] = filtered_df['IPROCTO'].map(Demands_name)
    filtered_df['Fuel']   = filtered_df['IPROCFROM'].map(Fuels_name)
    aggregated_df = filtered_df.groupby(['Demand', 'Fuel'])['value'].sum().reset_index()

    # Filter for resource use
    resource_df = df_FLOWC[df_FLOWC['IPROCFROM'].isin(Resource_name.keys())]
    resource_df['Resource'] = resource_df['IPROCFROM'].map(Resource_name)
    aggregated_resource_df = resource_df.groupby('Resource')['value'].sum().reset_index()

    # Define colors
    color_map = {
        'Ammonia'   : '#1f77b4',
        'Methanol'  : '#ff7f0e',
        'E-Methanol': '#2ca02c',
        'E-Gasoline': '#d62728',
        'E-LPG'     : '#9467bd',
        'Biogasoline': '#8c564b',
        'E-Diesel'  : '#e377c2',
        'Biojet'    : '#7f7f7f',
        'E-Jet-FT'  : '#bcbd22',
        'E-Jet-ME'  : '#17becf',
        'Biomass'   : '#006600',
        'Hydrogen'  : '#009999',
        'CO2'       : '#4D4D4D'
    }

    # Create subplots (shared Y-axis), no automatic subplot titles
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True)

    # --- Fuel Supply bars (left subplot)
    for fuel in aggregated_df['Fuel'].unique():
        df_fuel = aggregated_df[aggregated_df['Fuel'] == fuel]
        fig.add_trace(
            go.Bar(
                x=df_fuel['Demand'],
                y=df_fuel['value'],
                name=fuel,
                marker_color=color_map.get(fuel, '#333333'),
                width=0.4,  # narrower bars
            ),
            row=1, col=1
        )

    # --- Resource Use bars (right subplot)
    for resource in aggregated_resource_df['Resource'].unique():
        df_resource = aggregated_resource_df[aggregated_resource_df['Resource'] == resource]
        fig.add_trace(
            go.Bar(
                x=[resource],
                # Negative for consumption
                y=[-df_resource['value'].values[0]],
                name=resource,
                marker_color=color_map.get(resource, '#333333'),
                width=0.4,
            ),
            row=1, col=2
        )

    # We skip adding any annotation text for "Fuel Supply" / "Resource Use"
    # to remove those subplot labels.

    # Overall layout
    fig.update_layout(
        title=plot_title,
        barmode='stack',
        bargap=0.3,  # spacing between bars
        font=dict(
            family='DejaVu Sans Bold, DejaVu Sans, sans-serif',
            size=14,
            color='black'
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        legend=dict(
            bgcolor='white',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=12)
        ),
        margin=dict(l=60, r=20, t=80, b=60),
    )

    # Left subplot axes
    fig.update_xaxes(
        title_text='Demand',
        showline=True,
        mirror=True,
        linewidth=1,
        linecolor='black',
        showgrid=False,           # no vertical lines
        row=1, col=1
    )
    fig.update_yaxes(
        title_text='PJ',
        showline=True,
        mirror=True,
        linewidth=1,
        linecolor='black',
        showgrid=True,            # keep horizontal lines
        gridcolor='lightgray',
        gridwidth=0.6,
        zeroline=True,
        zerolinewidth=0.6,
        zerolinecolor='lightgray',
        tickmode='linear',
        tick0=0,
        dtick=25,
        row=1, col=1
    )

    # Right subplot axes
    # Because of shared_yaxes, we typically only need to style y once.
    # But let's do it explicitly here to ensure the black box on the right:
    fig.update_xaxes(
        title_text='Resource',
        showline=True,
        mirror=True,
        linewidth=1,
        linecolor='black',
        showgrid=False,          # no vertical lines
        row=1, col=2
    )
    fig.update_yaxes(
        showline=True,
        mirror=True,
        linewidth=1,
        linecolor='black',
        showgrid=True,           # keep horizontal lines
        gridcolor='lightgray',
        gridwidth=0.6,
        zeroline=True,
        zerolinewidth=0.6,
        zerolinecolor='lightgray',
        tickmode='linear',
        tick0=0,
        dtick=25,
        row=1, col=2
    )

    # Render
    fig.show()

Plot_fuel_supply(MainResults_path, OptiflowMR_path, Demands_name, Fuels_name,Resource_name, 2050, 'Base Case')
#%%

def group_EMI_YCRAG(df_EMI_YCRAG, df_FLOWC):
    """
    1) From df_EMI_YCRAG, select only rows where 'value' > 0.
       Classify them into: 
         - 'Industry Heating'
         - 'Individual Users Heating'
         - 'CHP Generation'
       based on substring matching in column 'AAA':
         - if 'AAA' contains 'IND'  => 'Industry Heating'
         - if 'AAA' contains 'IDVU' => 'Individual Users Heating'
         - else => 'CHP Generation

    2) From df_FLOWC, take rows where 'IPROCFROM' is in 
       ['CO2_Source_DAC','CO2_Land','CO2_Biochar_Sum','CO2_Biogas'].
       - Multiply their 'value' by -1000 (negative sign + convert from Mt to kton).
       - Map IPROCFROM to human-readable Category names: 
         DAC, Land Solutions, Biochar Sequestration, Biogas.

    Returns a DataFrame with columns ['Category','value'] 
    where 'value' is summed per Category.
    """

    # ------------------------------------------------
    # A) Process df_EMI_YCRAG (only positive values)
    # ------------------------------------------------
    df_emi = df_EMI_YCRAG.copy()
    df_emi['Category'] = None

    # Substring matching for 'IND' or 'IDVU' in column 'AAA'
    mask_ind_emi  = df_emi['AAA'].str.contains('IND',  case=False, na=False)
    mask_idvu_emi = df_emi['AAA'].str.contains('IDVU', case=False, na=False)
    mask_chp_emi  = ~(mask_ind_emi | mask_idvu_emi)  # everything else is CHP

    # Apply different masking based on the 'value' column
    mask_positive = df_emi['value'] > 0
    mask_negative = df_emi['value'] < 0

    # For positive values
    df_emi.loc[mask_positive & mask_ind_emi,  'Category'] = 'Industry Heating'
    df_emi.loc[mask_positive & mask_idvu_emi, 'Category'] = 'Individual Users Heating'
    df_emi.loc[mask_positive & mask_chp_emi,  'Category'] = 'CHP Generation'

    # For negative values
    df_emi.loc[mask_negative & mask_ind_emi,  'Category'] = 'Industry with BECCS'
    df_emi.loc[mask_negative & mask_idvu_emi, 'Category'] = 'Individual with BECCS'
    df_emi.loc[mask_negative & mask_chp_emi,  'Category'] = 'CHP with BECCS'

    # ------------------------------------------------
    # B) Process df_FLOWC for CO2 items
    # ------------------------------------------------
    co2_list = ['CO2_Land', 'CO2_Biochar_Sum', 'CO2_Biogas']
    co2_map = {
        'CO2_DAC_Total': 'DAC',
        'CO2_Land': 'Land Solutions',
        'CO2_Biochar_Sum': 'Biochar Sequestration',
        'CO2_Biogas': 'Biogas'
    }

    df_co2 = df_FLOWC[df_FLOWC['IPROCFROM'].isin(co2_list)].copy()
    df_co2['Category'] = df_co2['IPROCFROM'].map(co2_map)

    # Multiply the 'value' by -1000 => negative sign + convert from Mt to kton
    df_co2['value'] = df_co2['value'] * -1000

    # ------------------------------------------------
    # Filter df_FLOWC for CO2_DAC_Total and CO2_Seq
    # ------------------------------------------------
    df_co2_dac = df_FLOWC[
        (df_FLOWC['IPROCFROM'] == 'CO2_DAC_Total') &
        (df_FLOWC['IPROCTO'].str.contains('CO2_Seq', case=False, na=False))
    ].copy()
    df_co2_dac['Category'] = df_co2_dac['IPROCFROM'].map(co2_map)

    # Multiply the 'value' by -1000 => negative sign + convert from Mt to kton
    df_co2_dac['value'] = df_co2_dac['value'] * -1000

    # ------------------------------------------------
    # D) Combine and group
    # ------------------------------------------------
    # Merge all dataframes (positive EMI, negative CC, negative CO2, and CO2 DAC)
    df_final = pd.concat([df_emi, df_co2, df_co2_dac], ignore_index=True)

    # Group by Category and sum 'value'
    df_agg = df_final.groupby('Category', as_index=False)['value'].sum()

    return df_agg


df_FLOWA, df_FLOWC, df_EMI_YCRAG = Import_OptiflowMR(OptiflowMR_path)

df_agg = group_EMI_YCRAG(df_EMI_YCRAG, df_FLOWC)
print("Sum of all values in 'value' column:", df_agg['value'].sum())


def plot_stacked_histogram(df_agg, plot_title="Stacked Emissions"):
    """
    Creates a single-column bar chart where:
      - Positive 'value' segments stack upward from zero.
      - Negative 'value' segments stack downward from zero.
    
    Assumes df_agg has columns ['Category', 'value'].
    
    If df_agg has duplicate categories that you wish to aggregate, you can uncomment the groupby.
    """
    # --- Optional aggregation if needed ---
    # df_agg = df_agg.groupby("Category", as_index=False)["value"].sum()

    # Separate positive and negative values.
    df_pos = df_agg[df_agg["value"] >= 0].copy()
    df_neg = df_agg[df_agg["value"] < 0].copy()

    # Sort so that segments closest to zero are added first.
    # For positives, smallest positive first so that the lowest segment starts at zero.
    df_pos.sort_values("value", inplace=True)
    # For negatives, sort in descending order (largest negative, i.e. closest to zero, first).
    df_neg.sort_values("value", ascending=False, inplace=True)

    # Calculate the base for positive segments:
    # The first positive segment will have a base of 0, the second will start where the first ended, etc.
    # For example, if positives are 10, 20, 30 then:
    #   base for 10 = 0, base for 20 = 10, base for 30 = 10+20 = 30.
    df_pos["base"] = df_pos["value"].cumsum() - df_pos["value"]

    # Similarly for negative segments:
    # For negatives (e.g., -5, -15), we want:
    #   base for -5 = 0, base for -15 = -5.
    df_neg["base"] = df_neg["value"].cumsum() - df_neg["value"]

    # Define a color map (with a fallback color) for consistency.
    color_map = {
        "Industry Heating":            "#065903",
        "Industry Heating CC":         "#84ad63",
        "Individual Users Heating":    "#4c53b5",
        "Individual Users Heating CC": "#9b989c",
        "CHP Generation":              "#91066c",
        "CHP Generation CC":           "#f27ed3",
        "Biogas":                      "#c47900",
        "DAC":                         "#8900c4",
        "Land Solutions":              "#D9D9D9",
        "Biochar Sequestration":       "#5c3f3d"
    }
    fallback_color = "#999999"

    # Create a figure.
    # (We use a single x-axis category so that all segments appear in one stacked column.)
    x_val = "Total"
    fig = go.Figure()

    # Add positive segments with the computed base.
    for _, row in df_pos.iterrows():
        cat  = row["Category"]
        val  = row["value"]
        base = row["base"]
        fig.add_trace(go.Bar(
            x=[x_val],
            y=[val],
            base=[base],
            name=cat,
            marker=dict(color=color_map.get(cat, fallback_color))
        ))

    # Add negative segments with the computed base.
    for _, row in df_neg.iterrows():
        cat  = row["Category"]
        val  = row["value"]
        base = row["base"]
        fig.add_trace(go.Bar(
            x=[x_val],
            y=[val],
            base=[base],
            name=cat,
            marker=dict(color=color_map.get(cat, fallback_color))
        ))

    # Set layout.
    # We use 'overlay' mode so that our manually calculated base positions are honored.
    fig.update_layout(
        title=plot_title,
        barmode="overlay",  # Do not let Plotly compute its own stacking positions.
        font=dict(
            family="DejaVu Sans Bold, DejaVu Sans, sans-serif",
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
        margin=dict(l=60, r=20, t=80, b=60)
    )

    # X-axis styling: draw the axis line, no vertical grid lines.
    fig.update_xaxes(
        showline=True,
        mirror=True,
        linewidth=1,
        linecolor="black",
        showgrid=False
    )

    # Y-axis styling: add horizontal grid lines, a zero line, and a label.
    fig.update_yaxes(
        title_text="[ktons]",
        showline=True,
        mirror=True,
        linewidth=1,
        linecolor="black",
        showgrid=True,
        gridcolor="lightgray",
        gridwidth=0.6,
        zeroline=True,
        zerolinewidth=0.6,
        zerolinecolor="lightgray"
    )

    fig.show()



plot_stacked_histogram(df_agg, 'Emissions by Category')



#%%
def process_flows_and_consumption(df_FLOWC, df_F_CONS_YCRA):
    """
    1) From df_FLOWC, select rows that satisfy ANY of:
       - IPROCTO ∈ ['Straw_for_Energy','Wood_for_Energy']
       - IPROCFROM ∈ ['Straw_for_Energy','Wood_for_Energy'] 
         AND FLOW ∈ ['STRAW_FLOW','WOOD_FLOW']
       - IPROCFROM = 'Wood_Pellets_Gen'

       => Return them as df_flowc_filtered.

    2) From df_F_CONS_YCRA, select rows where 'C' == 'DENMARK' and FFF ∈ ['WOODCHIPS','STRAW','WOODPELLETS'].
       - Multiply the 'value' column by 3.6 for these filtered rows.
       - Add a column 'Category':
           - 'Industry Heating'         if AAA contains 'IND'
           - 'Individual Users Heating' if AAA contains 'IDVU'
           - 'CHP Generation'           otherwise

       => Return as df_f_cons_filtered (WITHOUT aggregation).

    Returns:
      df_flowc_filtered, df_f_cons_filtered
    """

    # -------------------------------------
    # A) Filter df_FLOWC
    # -------------------------------------
    mask_iprocto = df_FLOWC['IPROCTO'].isin([
        'Straw_for_Energy', 
        'Wood_for_Energy'
    ])
    
    mask_iprocfrom_flow = (
        df_FLOWC['IPROCFROM'].isin(['Straw_for_Energy', 'Wood_for_Energy'])
        & df_FLOWC['FLOW'].isin(['STRAW_FLOW', 'WOOD_FLOW'])
    )
    
    mask_iprocfrom_pellets = (df_FLOWC['IPROCFROM'] == 'Wood_Pellets_Gen')
    
    # Combine conditions with OR
    df_flowc_filtered = df_FLOWC[
        mask_iprocto | mask_iprocfrom_flow | mask_iprocfrom_pellets
    ].copy()

    # -------------------------------------
    # B) Filter df_F_CONS_YCRA
    #     * Now also require df_F_CONS_YCRA['C'] == 'DENMARK'
    # -------------------------------------
    df_f_cons_filtered = df_F_CONS_YCRA[
        (df_F_CONS_YCRA['C'] == 'DENMARK') &
        (df_F_CONS_YCRA['FFF'].isin(['WOODCHIPS', 'STRAW', 'WOODPELLETS']))
    ].copy()

    # -------------------------------------
    # C) Multiply 'value' by 3.6
    # -------------------------------------
    df_f_cons_filtered['value'] *= 3.6

    # -------------------------------------
    # D) Assign categories based on 'AAA'
    # -------------------------------------
    df_f_cons_filtered['Category'] = 'CHP Generation'  # default

    mask_ind = df_f_cons_filtered['AAA'].str.contains('IND',  case=False, na=False)
    mask_idv = df_f_cons_filtered['AAA'].str.contains('IDVU', case=False, na=False)

    df_f_cons_filtered.loc[mask_ind, 'Category'] = 'Industry Heating'
    df_f_cons_filtered.loc[mask_idv, 'Category'] = 'Individual Users Heating'

    # -------------------------------------
    # E) Return results (no aggregation)
    # -------------------------------------
    return df_flowc_filtered, df_f_cons_filtered

df_flowc_filtered, df_f_cons_agg = process_flows_and_consumption(df_FLOWC, df_F_CONS_YCRA)


def plot_five_column_histogram(df_flowc_filtered, df_f_cons_filtered):
    """
    Build a 5-column stacked histogram with the 3 fuels in the legend:
      1) Industry Heating         (sums from df_f_cons_filtered, Category='Industry Heating')
      2) Individual Users Heating (sums from df_f_cons_filtered, Category='Individual Users Heating')
      3) CHP Generation           (sums from df_f_cons_filtered, Category='CHP Generation')
      4) FlowC_4th: df_flowc_filtered with IPROCFROM in ['Wood_Pellets_Gen','Straw_for_Energy','Wood_for_Energy']
      5) FlowC_5th: df_flowc_filtered with IPROCFROM='Wood_Pellets_Gen' & IPROCTO in ['Straw_for_Energy','Wood_for_Energy']
                    PLUS df_f_cons_filtered with 'FFF'='WOODPELLETS'
      The 5th column is negative in the final plot. 
      
      Legend only has 3 fuels: 'Wood Chips', 'Wood Pellets', 'Straw'.
    """

    #------------------------------------------------------------------------------
    # 1) Re-group df_f_cons_filtered by [Category, FFF] to get separate fuels 
    #    for each category. This is more granular than df_f_cons_agg.
    #------------------------------------------------------------------------------
    # We'll assume df_f_cons_filtered has columns: ['AAA','FFF','value','Category', ...]
    # and FFF can be in ['WOODCHIPS','WOODPELLETS','STRAW'].
    df_cat_fuel = df_f_cons_filtered.groupby(['Category','FFF'], as_index=False)['value'].sum()

    # We'll map the FFF strings to "Wood Chips", "Wood Pellets", "Straw"
    def map_fff_to_fuel(fff):
        if fff == 'WOODCHIPS':
            return 'Wood Chips'
        elif fff == 'WOODPELLETS':
            return 'Wood Pellets'
        elif fff == 'STRAW':
            return 'Straw'
        else:
            return None

    df_cat_fuel['Fuel'] = df_cat_fuel['FFF'].apply(map_fff_to_fuel)

    # A helper to map Category => which of the first 3 columns
    def map_category_to_colidx(category):
        if category == 'Industry Heating':
            return 0
        elif category == 'Individual Users Heating':
            return 1
        elif category == 'CHP Generation':
            return 2
        else:
            return None  # we only have 3 categories we care about

    #------------------------------------------------------------------------------
    # 2) Prepare blank arrays for each of the 3 fuels, spanning 5 columns
    #------------------------------------------------------------------------------
    fuels = ['Wood Chips', 'Wood Pellets', 'Straw']
    # We'll hold y-values in a dict: fuel_arrays['Wood Chips'] = [0,0,0,0,0], etc.
    fuel_arrays = {}
    for f in fuels:
        fuel_arrays[f] = [0]*5  # 5 columns

    #------------------------------------------------------------------------------
    # 3) Fill columns 1,2,3 from df_cat_fuel 
    #------------------------------------------------------------------------------
    for _, row in df_cat_fuel.iterrows():
        col_idx = map_category_to_colidx(row['Category'])
        fuel = row['Fuel']
        val = row['value']

        if col_idx is not None and fuel in fuel_arrays:
            fuel_arrays[fuel][col_idx] += val

    #------------------------------------------------------------------------------
    # 4) Build data for the 4th column: lines in df_flowc_filtered with 
    #    IPROCFROM in ['Wood_Pellets_Gen','Straw_for_Energy','Wood_for_Energy']
    #------------------------------------------------------------------------------
    df_flowc_4th = df_flowc_filtered[
        df_flowc_filtered['IPROCFROM'].isin(['Wood_Pellets_Gen','Straw_for_Energy','Wood_for_Energy'])
    ].copy()

    # We'll map IPROCFROM to a "Fuel" as well
    def map_iproc_to_fuel(iproc):
        if iproc == 'Wood_Pellets_Gen':
            return 'Wood Pellets'
        elif iproc == 'Straw_for_Energy':
            return 'Straw'
        elif iproc == 'Wood_for_Energy':
            return 'Wood Chips'
        else:
            return None

    df_flowc_4th['Fuel'] = df_flowc_4th['IPROCFROM'].apply(map_iproc_to_fuel)

    # Summation by Fuel for the 4th column
    df_flowc_4th_agg = df_flowc_4th.groupby('Fuel', as_index=False)['value'].sum()

    for _, row in df_flowc_4th_agg.iterrows():
        f = row['Fuel']
        v = row['value']
        if f in fuel_arrays:
            fuel_arrays[f][3] += v   # 4th column index is 3

    # Step 5) Build data for the 5th column
    # -------------------------------------

    # Apply the logical OR filter as per user instruction
    df_flowc_5th = df_flowc_filtered[
        (df_flowc_filtered['IPROCFROM'] == 'Wood_Pellets_Gen') |
        (df_flowc_filtered['IPROCTO'].isin(['Straw_for_Energy','Wood_for_Energy']))
    ].copy()

    # Verify that df_flowc_5th contains data
    print(f"Number of rows in df_flowc_5th: {len(df_flowc_5th)}")
    print("Sample rows from df_flowc_5th:")
    print(df_flowc_5th.head())

    # Apply the mapping function to 'IPROCFROM' and 'IPROCTO'
    df_flowc_5th['Fuel_from'] = df_flowc_5th['IPROCFROM'].apply(map_iproc_to_fuel)
    df_flowc_5th['Fuel_to'] = df_flowc_5th['IPROCTO'].apply(map_iproc_to_fuel)

    # Sum the 'value' for each fuel from 'Fuel_from' and 'Fuel_to'
    flowc_5th_from = df_flowc_5th.groupby('Fuel_from')['value'].sum().reset_index()
    flowc_5th_to = df_flowc_5th.groupby('Fuel_to')['value'].sum().reset_index()

    # Rename columns for clarity
    flowc_5th_from.rename(columns={'Fuel_from': 'Fuel', 'value': 'FlowC_5th_from'}, inplace=True)
    flowc_5th_to.rename(columns={'Fuel_to': 'Fuel', 'value': 'FlowC_5th_to'}, inplace=True)

    # Merge the two dataframes to get total FlowC_5th per fuel
    flowc_5th_combined = pd.merge(flowc_5th_from, flowc_5th_to, on='Fuel', how='outer').fillna(0)

    # Calculate total FlowC_5th per fuel
    flowc_5th_combined['FlowC_5th_total'] = flowc_5th_combined['FlowC_5th_from'] + flowc_5th_combined['FlowC_5th_to']

    # Display the combined FlowC_5th data
    print("FlowC_5th Aggregated Data:")
    print(flowc_5th_combined)

    # Process df_f_cons_wp
    df_f_cons_wp = df_f_cons_filtered[df_f_cons_filtered['FFF'] == 'WOODPELLETS'].copy()

    # Map 'FFF' to 'Fuel' using the same mapping function
    df_f_cons_wp['Fuel'] = df_f_cons_wp['FFF'].apply(map_iproc_to_fuel)

    # Sum the 'value' for 'Wood Pellets' consumption
    fcons_wp_val = df_f_cons_wp['value'].sum()
    print(f"Wood Pellets Consumption Value: {fcons_wp_val}")

    # Assign negative values to the fifth column per fuel
    for _, row in flowc_5th_combined.iterrows():
        fuel = row['Fuel']
        flowc_val = row['FlowC_5th_total']
        if fuel in fuel_arrays:
            fuel_arrays[fuel][4] -= flowc_val  # Assign as negative

    # Assign negative consumption for 'Wood Pellets'
    if 'Wood Pellets' in fuel_arrays:
        fuel_arrays['Wood Pellets'][4] -= fcons_wp_val

    # Display the fuel_arrays for verification
    print("Fuel Arrays after assigning 5th column:")
    for fuel, values in fuel_arrays.items():
        print(f"{fuel}: {values}")
    print("\n")

    #------------------------------------------------------------------------------
    # 6) Plot with Plotly as a stacked histogram with 5 columns on the x-axis
    #------------------------------------------------------------------------------
    x_labels = [
        "Industry Heating",
        "Individual Users Heating",
        "CHP Generation",
        "Fuels Production",
        "Total Biomass Consumed"
    ]

    # Define custom colors for each fuel
    fuel_colors = {
        'Wood Chips': '#1f77b4',      # Blue
        'Wood Pellets': '#ff7f0e',    # Orange
        'Straw': '#2ca02c'            # Green
    }

    fig = go.Figure()

    # Create one stacked trace per fuel with custom colors
    for fuel in fuels:
        fig.add_trace(go.Bar(
            x=x_labels,
            y=fuel_arrays[fuel],
            name=fuel,
            marker_color=fuel_colors.get(fuel, 'gray')  # Default to gray if not specified
        ))

    # Stacked layout & styling
    fig.update_layout(
        title="Biomass Consumption",
        barmode='stack',  # stack categories at each of the 5 x positions
        font=dict(
            family='DejaVu Sans Bold, DejaVu Sans, sans-serif',
            size=14,
            color='black'
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        legend=dict(
            bgcolor='white',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=12)
        ),
        margin=dict(l=60, r=20, t=80, b=60)
    )

    # X-axis: bounding box, no vertical grid
    fig.update_xaxes(
        showline=True,
        mirror=True,
        linewidth=1,
        linecolor='black',
        showgrid=False
    )

    # Y-axis: bounding box, horizontal grid, zero line, labeled "[PJ]"
    fig.update_yaxes(
        title_text='[PJ]',
        showline=True,
        mirror=True,
        linewidth=1,
        linecolor='black',
        showgrid=True,
        gridcolor='lightgray',
        gridwidth=0.6,
        zeroline=True,
        zerolinewidth=0.6,
        zerolinecolor='lightgray'
    )

    fig.show()
plot_five_column_histogram(df_flowc_filtered, df_f_cons_agg)

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

#%%
plot_municipalities( df_FLOWA, shapefile_path=r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\LAU_RG_01M_2021_3035.shp\Administrative_DK.shp', column_municipality='AAA', column_value='value', filter_column='IPROCFROM', filter_value='CO2_Source_Biogen', plot_title='Point Source Biogen CO2 [Mtons]', cmap='Greens')
plot_municipalities( df_FLOWA, shapefile_path=r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\LAU_RG_01M_2021_3035.shp\Administrative_DK.shp', column_municipality='AAA', column_value='value', filter_column='IPROCFROM', filter_value='CO2_Source_Fossil', plot_title='Point Source Fossil CO2 [Mtons]', cmap='Greys')
plot_municipalities( df_FLOWA, shapefile_path=r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\LAU_RG_01M_2021_3035.shp\Administrative_DK.shp', column_municipality='AAA', column_value='value', filter_column='IPROCFROM', filter_value='CO2_Source_DAC', plot_title='Point Source DAC CO2 [Mtons]', cmap='Purples')
plot_municipalities( df_FLOWA, shapefile_path=r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\LAU_RG_01M_2021_3035.shp\Administrative_DK.shp', column_municipality='AAA', column_value='value', filter_column='IPROCFROM', filter_value='CO2_Seq_sum', plot_title='Sequestered CO2 [Mtons]', cmap='Blues')
# %%
# Filter the DataFrame for the specified IPROCFROM values
filtered_df = df_FLOWC[df_FLOWC['IPROCFROM'].isin(['Agricultural_Land_Gen', 'Land_for_Wood_Production', 'Wood_for_Energy', 'Straw_for_Energy','Wood_for_Use', 'Straw_for_Use'])]

# Print the 'value' column for the filtered rows
print(filtered_df[['IPROCFROM', 'value']])
# %%
# Clear all variables



# %%
