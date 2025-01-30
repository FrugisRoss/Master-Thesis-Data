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

MainResults_path=r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Run_on_HPC\Balmorel\Base_Case\model\MainResults.gdx'
OptiflowMR_path=r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Run_on_HPC\Balmorel\Base_Case\model\Optiflow_MainResults.gdx'

Resource_name = {'Biomass_for_use':'Biomass',
               'Hydrogen_Use':'Hydrogen',
               'CO2_Use':'CO2'}

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
    return df_CC_YCRAG

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
# %%


def group_EMI_YCRAG(df_EMI_YCRAG, df_CC_YCRAG, df_FLOWC):
    """
    1) From df_EMI_YCRAG, select only rows where 'value' > 0.
       Classify them into: 
         - 'Industry Heating'
         - 'Individual Users Heating'
         - 'CHP Generation'
       based on substring matching in column 'AAA':
         - if 'AAA' contains 'IND'  => 'Industry Heating'
         - if 'AAA' contains 'IDVU' => 'Individual Users Heating'
         - else => 'CHP Generation'

    2) From df_CC_YCRAG, take all rows, multiply 'value' by -1, 
       then classify them into:
         - 'Industry Heating CC'
         - 'Individual Users Heating CC'
         - 'CHP Generation CC'
       based on substring matching in column 'IPROCFROM':
         - if 'IPROCFROM' contains 'IND'  => 'Industry Heating CC'
         - if 'IPROCFROM' contains 'IDVU' => 'Individual Users Heating CC'
         - else => 'CHP Generation CC'

    3) From df_FLOWC, take rows where 'IPROCFROM' is in 
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
    df_emi = df_EMI_YCRAG[df_EMI_YCRAG['value'] > 0].copy()
    df_emi['Category'] = None

    # Substring matching for 'IND' or 'IDVU' in column 'AAA'
    mask_ind_emi  = df_emi['AAA'].str.contains('IND',  case=False, na=False)
    mask_idvu_emi = df_emi['AAA'].str.contains('IDVU', case=False, na=False)
    mask_chp_emi  = ~(mask_ind_emi | mask_idvu_emi)  # everything else is CHP

    df_emi.loc[mask_ind_emi,  'Category'] = 'Industry Heating'
    df_emi.loc[mask_idvu_emi, 'Category'] = 'Individual Users Heating'
    df_emi.loc[mask_chp_emi,  'Category'] = 'CHP Generation'

    # ------------------------------------------------
    # B) Process df_CC_YCRAG (multiply by -1)
    # ------------------------------------------------
    df_cc = df_CC_YCRAG.copy()
    df_cc['value'] = df_cc['value'] * -1  # negative CC values
    df_cc['Category'] = None

    # Substring matching for 'IND' or 'IDVU' in column 'IPROCFROM'
    mask_ind_cc  = df_cc['AAA'].str.contains('IND',  case=False, na=False)
    mask_idvu_cc = df_cc['AAA'].str.contains('IDVU', case=False, na=False)
    mask_chp_cc  = ~(mask_ind_cc | mask_idvu_cc)  # everything else is CHP

    df_cc.loc[mask_ind_cc,  'Category'] = 'Industry Heating CC'
    df_cc.loc[mask_idvu_cc, 'Category'] = 'Individual Users Heating CC'
    df_cc.loc[mask_chp_cc,  'Category'] = 'CHP Generation CC'

    # ------------------------------------------------
    # C) Process df_FLOWC for CO2 items
    # ------------------------------------------------
    co2_list = ['CO2_Source_DAC', 'CO2_Land', 'CO2_Biochar_Sum', 'CO2_Biogas']
    co2_map = {
        'CO2_Source_DAC': 'DAC',
        'CO2_Land': 'Land Solutions',
        'CO2_Biochar_Sum': 'Biochar Sequestration',
        'CO2_Biogas': 'Biogas'
    }

    df_co2 = df_FLOWC[df_FLOWC['IPROCFROM'].isin(co2_list)].copy()
    df_co2['Category'] = df_co2['IPROCFROM'].map(co2_map)

    # Multiply the 'value' by -1000 => negative sign + convert from Mt to kton
    df_co2['value'] = df_co2['value'] * -1000

    # ------------------------------------------------
    # D) Combine and group
    # ------------------------------------------------
    # Merge all dataframes (positive EMI, negative CC, negative CO2)
    df_final = pd.concat([df_emi, df_cc, df_co2], ignore_index=True)

    # Group by Category and sum 'value'
    df_agg = df_final.groupby('Category', as_index=False)['value'].sum()

    return df_agg



def plot_3col_stacked_no_sum(df_agg, plot_title="Three-Column Emissions"):
    """
    Creates 3 columns on the x-axis:
      1) "Total"        -> use abs(value) for each row BUT only if that row's
                           Category is in `allowed_in_total`; otherwise 0.
      2) "Not Captured" -> use value if >0 else 0 (for all categories)
      3) "Captured"     -> use value if <0 else 0 (for all categories)

    Each row in df_agg is a separate stacked segment 
    (so each 'Category' from df_agg appears in the legend).

    Assumes df_agg has columns ['Category','value'].
    No additional summing or grouping is performed.

    In the first column, only the absolute values of:
      - Industry Heating
      - Industry Heating CC
      - Individual Users Heating
      - Individual Users Heating CC
      - CHP Generation
      - CHP Generation CC
      - Biogas
    will be stacked. Rows with other Categories get 0 in the first column.
    """

    # Categories allowed in the first column (the "Total" bar)
    allowed_in_total = {
        "Industry Heating",
        "Industry Heating CC",
        "Individual Users Heating",
        "Individual Users Heating CC",
        "CHP Generation",
        "CHP Generation CC",
        "Biogas"
    }

    # Copy the data so we don't modify the original
    df_plot = df_agg.copy()

    # For each row, define columns:
    #  abs_val => absolute value if the Category is in allowed_in_total, else 0
    #  pos_val => value if > 0 else 0 (unchanged for all categories)
    #  neg_val => value if < 0 else 0 (unchanged for all categories)
    df_plot['abs_val'] = df_plot.apply(
        lambda row: abs(row['value']) if row['Category'] in allowed_in_total else 0,
        axis=1
    )
    df_plot['pos_val'] = df_plot['value'].clip(lower=0)  # anything <0 => 0
    df_plot['neg_val'] = df_plot['value'].clip(upper=0)  # anything >0 => 0

    # A new palette with greens, greys, blues, purples, etc.
    # (These are distinct from the default Plotly palette.)
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
    # Any category not in the map will fall back to this color:
    fallback_color = "#999999"

    # We'll create a figure with 3 x positions
    x_labels = ["Total", "Not Caputured", "Captured"]
    fig = go.Figure()

    # Each row in df_plot => one stacked trace
    for _, row in df_plot.iterrows():
        cat = row['Category']
        abs_val = row['abs_val']  # only non-zero if in allowed_in_total
        pos_val = row['pos_val']
        neg_val = row['neg_val']
        
        fig.add_trace(go.Bar(
         x=x_labels,
         y=[abs_val, pos_val, neg_val],
         name=cat,
        marker=dict(color=color_map.get(cat, fallback_color))
        ))

    # Stacked layout & styling
    fig.update_layout(
        title=plot_title,
        barmode='stack',  # stack categories at each of the 3 x positions
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

    # Y-axis: bounding box, horizontal grid, zero line, labeled "[ktons]"
    fig.update_yaxes(
        title_text='[ktons]',
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

df_CC_YCRAG = Import_BalmorelMR(MainResults_path)
df_FLOWA, df_FLOWC, df_EMI_YCRAG = Import_OptiflowMR(OptiflowMR_path)
df_agg=group_EMI_YCRAG(df_EMI_YCRAG, df_CC_YCRAG, df_FLOWC)
plot_3col_stacked_no_sum(df_agg, 'Emissions by Category')

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
plot_municipalities( df_FLOWA, shapefile_path=r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\LAU_RG_01M_2021_3035.shp\Administrative_DK.shp', column_municipality='AAA', column_value='value', filter_column='IPROCFROM', filter_value='Productive_Forest_Land_Gen', plot_title='Productive Forest in Denmark [Mha]', cmap='Blues')
plot_municipalities( df_FLOWA, shapefile_path=r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\LAU_RG_01M_2021_3035.shp\Administrative_DK.shp', column_municipality='AAA', column_value='value', filter_column='IPROCFROM', filter_value='Agricultural_Land_Gen', plot_title='Agricultural Land in Denmark [Mha]', cmap='Purples')

#%%
plot_municipalities( df_FLOWA, shapefile_path=r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\LAU_RG_01M_2021_3035.shp\Administrative_DK.shp', column_municipality='AAA', column_value='value', filter_column='IPROCFROM', filter_value='Co2_Source_Biogen', plot_title='Point Source Biogen CO2 [Mtons]', cmap='Greens')
plot_municipalities( df_FLOWA, shapefile_path=r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\LAU_RG_01M_2021_3035.shp\Administrative_DK.shp', column_municipality='AAA', column_value='value', filter_column='IPROCFROM', filter_value='Co2_Source_Fossil', plot_title='Point Source Fossil CO2 [Mtons]', cmap='Greys')
plot_municipalities( df_FLOWA, shapefile_path=r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\LAU_RG_01M_2021_3035.shp\Administrative_DK.shp', column_municipality='AAA', column_value='value', filter_column='IPROCFROM', filter_value='Co2_Source_DAC', plot_title='Point Source DAC CO2 [Mtons]', cmap='Purples')
plot_municipalities( df_FLOWA, shapefile_path=r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\LAU_RG_01M_2021_3035.shp\Administrative_DK.shp', column_municipality='AAA', column_value='value', filter_column='IPROCFROM', filter_value='Co2_Seq_sum', plot_title='Sequestered CO2 [Mtons]', cmap='Blues')
# %%
# Filter the DataFrame for the specified IPROCFROM values
filtered_df = df_FLOWC[df_FLOWC['IPROCFROM'].isin(['Agricultural_Land_Gen', 'Land_for_Wood_Production', 'Wood_for_Energy', 'Straw_for_Energy','Wood_for_Use', 'Straw_for_Use'])]

# Print the 'value' column for the filtered rows
print(filtered_df[['IPROCFROM', 'value']])
# %%
