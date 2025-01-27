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

MainResults_path=r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Run_on_HPC\Balmorel\Base_Case\model\MainResults.gdx'
OptiflowMR_path=r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Run_on_HPC\Balmorel\Base_Case\model\Optiflow_MainResults.gdx'

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

    return df_FLOWA, df_FLOWC

def Plot_fuel_supply(MainResults_path, OptiflowMR_path, Demands_name, Fuels_name, year):
    df_FLOWA, df_FLOWC = Import_OptiflowMR(OptiflowMR_path)
    print (df_FLOWC)

    # Filter df_FLOWC to include only the rows with 'Y' equal to the specified year
    df_FLOWC = df_FLOWC[df_FLOWC['Y'] == str(year)]

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
    # Create the figure
    fig = go.Figure()

    # Add traces to the figure
    for fuel in aggregated_df['Fuel'].unique():
        df_fuel = aggregated_df[aggregated_df['Fuel'] == fuel]
        fig.add_trace(go.Bar(
            x=df_fuel['Sector'],
            y=df_fuel['Fuel Demand'],
            name=fuel,
            marker_color=color_map.get(fuel, '#333333')
        ))

    # Update the layout of the figure
    fig.update_layout(
        title=f'Fuel supply year {year}',
        xaxis_title='Demand',
        yaxis_title='Demand',
        barmode='stack'
    )

    # Show the figure
    fig.show()


Plot_fuel_supply(MainResults_path, OptiflowMR_path, Demands_name, Fuels_name, 2050 )
# %%

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

df_FLOWA, df_FLOWC = Import_OptiflowMR(OptiflowMR_path)
plot_municipalities( df_FLOWA, shapefile_path=r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\LAU_RG_01M_2021_3035.shp\Administrative_DK.shp', column_municipality='AAA', column_value='value', filter_column='IPROCFROM',filter_value='Straw_for_Energy', plot_title='Straw for Energy in Denmark', cmap='Reds')
plot_municipalities( df_FLOWA, shapefile_path=r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\LAU_RG_01M_2021_3035.shp\Administrative_DK.shp', column_municipality='AAA', column_value='value', filter_column='IPROCFROM', filter_value='Wood_for_Energy', plot_title='Wood for Energy in Denmark', cmap='Greens')
plot_municipalities( df_FLOWA, shapefile_path=r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\LAU_RG_01M_2021_3035.shp\Administrative_DK.shp', column_municipality='AAA', column_value='value', filter_column='IPROCFROM', filter_value='Productive_Forest_Land_Gen', plot_title='Productive Forest in Denmark', cmap='Blues')
plot_municipalities( df_FLOWA, shapefile_path=r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\LAU_RG_01M_2021_3035.shp\Administrative_DK.shp', column_municipality='AAA', column_value='value', filter_column='IPROCFROM', filter_value='Agricultural_Land_Gen', plot_title='Agricultural Land in Denmark', cmap='Grays')

# %%
# Filter the DataFrame for the specified IPROCFROM values
filtered_df = df_FLOWC[df_FLOWC['IPROCFROM'].isin(['Agricultural_Land_Gen', 'Productive_Forest_Land', 'Wood_for_Energy', 'Straw_for_Energy'])]

# Print the 'value' column for the filtered rows
print(filtered_df[['IPROCFROM', 'value']])
# %%
