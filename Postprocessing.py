#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import gams.transfer as gt
import pybalmorel as pyb
import gams
import os
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


MainResults_path=r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Balmorel\Wood_Agri_Biomass\model\MainResults.gdx'
OptiflowMR_path=r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Balmorel\Wood_Agri_Biomass\model\Optiflow_MainResults.gdx'

Demands_name= {'Sea_fuels_sum':'Maritime demand',
               'Road_fuels_sum':'Road demand',
               'Air_fuels_sum':'Air demand'}

Fuels_name = {
    'Ammonia_Eff': 'Ammonia',
    'Methanol_Eff': 'Methanol',
    'E_Methanol_Eff': 'E-Methanol',
    'EME_Gasoline_Eff': 'E-Gasoline',
    'EME_LPG_Eff': 'E-LPG',
    'BioGasoline_Eff': 'Biogasoline',  # Updated key
    'E_FT_Diesel_Eff': 'E-Diesel',
    'BioJet_Eff': 'Biojet',
    'E-FT-Jet_Eff': 'E-Jet-FT',
    'EME_Jet_Eff': 'E-Jet-ME'
}

# Function to import the table from the MainResults file
def Import_OptiflowMR(file_path):
    df = gt.Container(file_path)
    df_FLOWA = pd.DataFrame(df["VFLOW_Opti_C"].records)
    df_FLOWC = pd.DataFrame(df["VFLOW_Opti_A"].records)

    return df_FLOWA, df_FLOWC

df_FLOWA, df_FLOWC = Import_OptiflowMR(OptiflowMR_path)

#%%
# Filter df_FLOWC to include only the rows with 'IPROCTO' in Demands_name and 'IPROCFROM' in Fuels_name
filtered_df = df_FLOWC[df_FLOWC['IPROCTO'].isin(Demands_name.keys()) & df_FLOWC['IPROCFROM'].isin(Fuels_name.keys())]

# Map the 'IPROCTO' values to their corresponding demand names
filtered_df['Demand'] = filtered_df['IPROCTO'].map(Demands_name)

# Map the 'IPROCFROM' values to their corresponding fuel names
filtered_df['Fuel'] = filtered_df['IPROCFROM'].map(Fuels_name)

# Aggregate the values by 'Demand' and 'Fuel'
aggregated_df = filtered_df.groupby(['Demand', 'Fuel'])['value'].sum().reset_index()

# Define a color map for the fuels
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
    'E-Jet-ME': '#17becf'
}

# Plot the stacked bar chart with custom colors
fig = px.bar(aggregated_df, x='Demand', y='value', color='Fuel', title='Stacked Demand Histogram', labels={'value': 'Value', 'Fuel': 'Fuel'}, color_discrete_map=color_map)
fig.show()
# %%
