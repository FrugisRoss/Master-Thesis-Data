#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import gams.transfer as gt
import sys
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

# === List of Scenarios ===
scenario_list = [
     ("Base Case", r"C:\Users\sigur\OneDrive\DTU\Run on HPC Polimi\Base_Case\model\Optiflow_MainResults.gdx"),
     ("CO2 Scenario", r"C:\Users\sigur\OneDrive\DTU\Run on HPC Polimi\CO2_Case_RLC\model\Optiflow_MainResults.gdx"),
     ("Biodiversity+CO2 Scenario", r"C:\Users\sigur\OneDrive\DTU\Run on HPC Polimi\Biodiversity_Case_RLC\model\Optiflow_MainResults.gdx"),
     ("Biodiversity+CO2 Fossil ", r"C:\Users\sigur\OneDrive\DTU\Run on HPC Polimi\Biodiversity_Case_RLC_FOSSIL\model\Optiflow_MainResults.gdx"),
    

]

# === List of Filters to Apply ===

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
       ([("CO2_Source_DAC", "CO2_DAC_50"),("CO2_Source_Biogen", "CO2_BIOGEN_TOT"),("CO2_Source_Fossil", "CO2_FOS_TOT")],"Total CO2 Resource[Mton]", "Purples"),
      ([("Air_fuels_sum", "AirBuffer"),
        ("Road_fuels_sum", "RoadBuffer"),
        ("Sea_fuels_sum", "SeaBuffer")],"Renewable Fuel Production", "Reds"),



 ]

# === Shapefile for administrative boundaries ===
shapefile_path = r"C:\Users\sigur\OneDrive\DTU\Input Data\QGIS data\LAU_RG_01M_2021_3035.shp\Administrative_DK.shp"

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


# Set a system font to avoid missing font warnings
plt.rcParams['font.family'] = 'Arial'

def Import_OptiflowMR_geo(file_path):
    """ 
    Import Optiflow Main Results from GDX file and return DataFrames for different components.
    """
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

#    return merged[[column_municipality_gdf, column_value]]  # Return merged data for debugging.



def optiflow_maps(scenario_list, plot_filters, shapefile_path, municipality_name_mapping):
    """
    Plots the sum of the values of multiple (IPROCFROM, IPROCTO) pairs for each scenario on a vector layer of shapefile_path.

    Parameters:
        scenario_list (list): List of (scenario_name, gdx_file_path) tuples.
        plot_filters (list): List of filters to apply, each as (filter_pairs, plot_title, colormap).
        shapefile_path (str): Path to the shapefile containing Danish municipalities.
        municipality_name_mapping (dict): Mapping from GAMS names to shapefile names.
    """
    for filter_pairs, plot_title, cmap in plot_filters:
        fig, axes = plt.subplots(1, len(scenario_list), figsize=(20, 5))
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]

        df_all_scenarios = pd.DataFrame()
        global_min, global_max = np.inf, -np.inf

        for scenario_name, file_path in scenario_list:
            df_FLOWA, *_ = Import_OptiflowMR_geo(file_path)
            df_FLOWA["AAA"] = df_FLOWA["AAA"].astype(str).str.strip().replace(municipality_name_mapping)
            df_FLOWA["IPROCFROM"] = df_FLOWA["IPROCFROM"].astype(str).str.strip()
            df_FLOWA["IPROCTO"] = df_FLOWA["IPROCTO"].astype(str).str.strip()

            mask = df_FLOWA.apply(lambda row: any(
                (row["IPROCFROM"].upper() == pair[0].strip().upper() and
                 row["IPROCTO"].upper().startswith(pair[1].strip().upper()))
                 for pair in filter_pairs), axis=1)
            filtered_df = df_FLOWA[mask]

            aggregated_df = filtered_df.groupby("AAA", observed=False)["value"].sum().reset_index()

            if not aggregated_df.empty:
                min_val, max_val = aggregated_df["value"].min(), aggregated_df["value"].max()
                global_min = min(global_min, min_val)
                global_max = max(global_max, max_val)

            aggregated_df.rename(columns={"value": scenario_name}, inplace=True)
            df_all_scenarios = pd.merge(df_all_scenarios, aggregated_df, on="AAA", how="outer") if not df_all_scenarios.empty else aggregated_df

        df_all_scenarios.fillna(0, inplace=True)
        print(f"\n📊 Data Table for {plot_title}")
        pd.set_option('display.max_rows', None)
        print(df_all_scenarios)

        norm = mcolors.Normalize(vmin=global_min, vmax=global_max)

        for idx, (scenario_name, file_path) in enumerate(scenario_list):
            print(f"\n🔹 Processing {scenario_name} for {plot_title}")
            df_FLOWA, *_ = Import_OptiflowMR_geo(file_path)
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

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar_ax = fig.add_axes([0.99, 0.15, 0.02, 0.7])
        fig.colorbar(sm, cax=cbar_ax, orientation="vertical").set_label("[Mha]", fontsize=14, family="Arial")

        plt.show()

optiflow_maps(scenario_list, plot_filters, shapefile_path, municipality_name_mapping)


 #%%


# # --- 1) Municipality -> Region Dictionary (All 98 Municipalities) ---
# municipality_to_region = {
#     # Region Hovedstaden (29)
#     "Albertslund": "Region Hovedstaden",
#     "Allerød": "Region Hovedstaden",
#     "Ballerup": "Region Hovedstaden",
#     "Bornholm": "Region Hovedstaden",
#     "Brøndby": "Region Hovedstaden",
#     "Dragør": "Region Hovedstaden",
#     "Egedal": "Region Hovedstaden",
#     "Fredensborg": "Region Hovedstaden",
#     "Frederiksberg": "Region Hovedstaden",
#     "Frederikssund": "Region Hovedstaden",
#     "Furesø": "Region Hovedstaden",
#     "Gentofte": "Region Hovedstaden",
#     "Gladsaxe": "Region Hovedstaden",
#     "Glostrup": "Region Hovedstaden",
#     "Gribskov": "Region Hovedstaden",
#     "Halsnæs": "Region Hovedstaden",
#     "Helsingør": "Region Hovedstaden",
#     "Herlev": "Region Hovedstaden",
#     "Hillerød": "Region Hovedstaden",
#     "Hørsholm": "Region Hovedstaden",
#     "Høje-Taastrup": "Region Hovedstaden",
#     "Hvidovre": "Region Hovedstaden",
#     "Ishøj": "Region Hovedstaden",
#     "København": "Region Hovedstaden",
#     "Lyngby-Taarbæk": "Region Hovedstaden",
#     "Rudersdal": "Region Hovedstaden",
#     "Rødovre": "Region Hovedstaden",
#     "Tårnby": "Region Hovedstaden",
#     "Vallensbæk": "Region Hovedstaden",

#     # Region Sjælland (17)
#     "Faxe": "Region Sjælland",
#     "Greve": "Region Sjælland",
#     "Guldborgsund": "Region Sjælland",
#     "Holbæk": "Region Sjælland",
#     "Kalundborg": "Region Sjælland",
#     "Køge": "Region Sjælland",
#     "Lejre": "Region Sjælland",
#     "Lolland": "Region Sjælland",
#     "Næstved": "Region Sjælland",
#     "Odsherred": "Region Sjælland",
#     "Ringsted": "Region Sjælland",
#     "Roskilde": "Region Sjælland",
#     "Slagelse": "Region Sjælland",
#     "Solrød": "Region Sjælland",
#     "Sorø": "Region Sjælland",
#     "Stevns": "Region Sjælland",
#     "Vordingborg": "Region Sjælland",

#     # Region Syddanmark (22)
#     "Assens": "Region Syddanmark",
#     "Billund": "Region Syddanmark",
#     "Esbjerg": "Region Syddanmark",
#     "Fanø": "Region Syddanmark",
#     "Fredericia": "Region Syddanmark",
#     "Faaborg-Midtfyn": "Region Syddanmark",
#     "Haderslev": "Region Syddanmark",
#     "Kerteminde": "Region Syddanmark",
#     "Kolding": "Region Syddanmark",
#     "Langeland": "Region Syddanmark",
#     "Middelfart": "Region Syddanmark",
#     "Nordfyns": "Region Syddanmark",
#     "Nyborg": "Region Syddanmark",
#     "Odense": "Region Syddanmark",
#     "Svendborg": "Region Syddanmark",
#     "Sønderborg": "Region Syddanmark",
#     "Tønder": "Region Syddanmark",
#     "Varde": "Region Syddanmark",
#     "Vejen": "Region Syddanmark",
#     "Vejle": "Region Syddanmark",
#     "Ærø": "Region Syddanmark",
#     "Aabenraa": "Region Syddanmark",

#     # Region Midtjylland (19)
#     "Favrskov": "Region Midtjylland",
#     "Hedensted": "Region Midtjylland",
#     "Herning": "Region Midtjylland",
#     "Holstebro": "Region Midtjylland",
#     "Horsens": "Region Midtjylland",
#     "Ikast-Brande": "Region Midtjylland",
#     "Lemvig": "Region Midtjylland",
#     "Norddjurs": "Region Midtjylland",
#     "Odder": "Region Midtjylland",
#     "Randers": "Region Midtjylland",
#     "Ringkøbing-Skjern": "Region Midtjylland",
#     "Samsø": "Region Midtjylland",
#     "Silkeborg": "Region Midtjylland",
#     "Skanderborg": "Region Midtjylland",
#     "Skive": "Region Midtjylland",
#     "Struer": "Region Midtjylland",
#     "Syddjurs": "Region Midtjylland",
#     "Viborg": "Region Midtjylland",
#     "Aarhus": "Region Midtjylland",

#     # Region Nordjylland (11)
#     "Brønderslev": "Region Nordjylland",
#     "Frederikshavn": "Region Nordjylland",
#     "Hjørring": "Region Nordjylland",
#     "Jammerbugt": "Region Nordjylland",
#     "Læsø": "Region Nordjylland",
#     "Mariagerfjord": "Region Nordjylland",
#     "Morsø": "Region Nordjylland",
#     "Rebild": "Region Nordjylland",
#     "Thisted": "Region Nordjylland",
#     "Vesthimmerlands": "Region Nordjylland",
#     "Aalborg": "Region Nordjylland",
# }

# # --- 2) Region-Level Land Costs [M€/ha] (Example values) ---
# region_land_costs_m_eur_ha = {
#     "Region Hovedstaden": 0.0322,
#     "Region Sjælland": 0.01745,
#     "Region Syddanmark": 0.0305,
#     "Region Midtjylland": 0.0283,
#     "Region Nordjylland": 0.0227
# }

# # Multiply by 1000 to get k€/ha
# region_land_costs_k_eur_ha = {
#     region: cost * 1000 for region, cost in region_land_costs_m_eur_ha.items()
# }

# # --- 3) Load the municipality shapefile ---
# shapefile_path = r"C:\Users\sigur\OneDrive\DTU\Input Data\QGIS data\LAU_RG_01M_2021_3035.shp\Administrative_DK.shp"
# gdf = gpd.read_file(shapefile_path)

# # The column in your shapefile that holds the municipality name
# column_municipality_gdf = "LAU_NAME"

# # Make sure the municipality names match the dictionary keys (strip whitespace, etc.)
# gdf[column_municipality_gdf] = gdf[column_municipality_gdf].astype(str).str.strip()

# # --- 4) Map each municipality to its region ---
# gdf["Region"] = gdf[column_municipality_gdf].map(municipality_to_region)

# # --- 5) Dissolve by region to create a single polygon per region ---
# gdf_regions = gdf.dissolve(by="Region", as_index=False)

# # --- 6) Attach the land costs (now in k€/ha) to the dissolved regions ---
# df_region_costs = pd.DataFrame(list(region_land_costs_k_eur_ha.items()), columns=["Region", "LandCost"])
# gdf_regions = gdf_regions.merge(df_region_costs, on="Region", how="left")

# # Any region not found in region_land_costs would get NaN; fill with 0 if desired
# gdf_regions["LandCost"] = gdf_regions["LandCost"].fillna(0)

# # --- 7) Plot the map, coloring each region by LandCost (k€/ha) ---
# fig, ax = plt.subplots(figsize=(8, 8))

# # Determine min/max for color scaling
# min_val = gdf_regions["LandCost"].min()
# max_val = gdf_regions["LandCost"].max()
# norm = mcolors.Normalize(vmin=min_val, vmax=max_val)

# # Choose a colormap
# cmap = plt.cm.Oranges

# # Plot the region polygons
# gdf_regions.plot(
#     column="LandCost",
#     cmap=cmap,
#     linewidth=0.5,
#     edgecolor="black",
#     legend=False,  # We'll add a custom colorbar below
#     ax=ax,
#     norm=norm
# )

# ax.set_title("Cost of Land [k€/ha] by Region", fontsize=14)
# ax.set_axis_off()

# # Add a colorbar
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
# cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
# cbar.set_label("Land Cost [k€/ha]", fontsize=12)

# plt.tight_layout()
# plt.show()

# # %%
# df_FLOWA, df_FLOWC, df_EMI_YCRAG, df_EMI_PROC=Import_OptiflowMR_geo(r"C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Run_on_HPC\Balmorel\Base_Case\model\Optiflow_MainResults.gdx")

# # Filter the DataFrame for the specified IPROCFROM values
# filtered_df = df_FLOWC[df_FLOWC['IPROCFROM'].isin(['Agricultural_Land_Gen', 'Land_for_Wood_Production', 'Wood_for_Energy', 'Straw_for_Energy','Wood_for_Use', 'Straw_for_Use'])]

# # Print the 'value' column for the filtered rows
# print(filtered_df[['IPROCFROM', 'value']])
# # %%

# # Prepare the data
# data = {
#     'Urban Area'       : [8.04, 8.04, 8.04, 8.04],
#     'Agricultural Area ': [61.52, 48.95, 50.09, 50.09],
#     'Forest Area'      : [7.73, 20.30, 13.77, 13.77],
#     'Wetlands Area'    : [0.75, 0.75, 4.72, 4.72],
#     'Other'            : [13.10, 13.10, 13.10, 13.10],
#     'Protected Area'   : [8.86, 8.86, 10.28, 10.28]
# }

# # Labels for the y-axis (the scenarios)
# index_labels = [
#     'Base Case',
#     'CO2 Case',
#     'Biodiversity+CO2 Case',
#     'Biodiversity+CO2 with Fossil Case'
# ]

# # Create a DataFrame
# df = pd.DataFrame(data, index=index_labels)

# # Create a stacked HORIZONTAL bar chart
# ax = df.plot(
#     kind='barh',
#     stacked=True,
#     figsize=(10, 6)
# )

# # Label axes and title
# plt.xlabel('Area [%]')
# plt.ylabel('Scenario')
# plt.title('Land-Use Areas by Scenario')

# # Turn on vertical grid lines
# ax.grid(axis='x', linestyle='--', alpha=0.7)

# # Move legend to the right side
# plt.legend(
#     bbox_to_anchor=(1.05, 1),  # slightly outside the plot area
#     loc='upper left'
# )

# plt.tight_layout()
# plt.show()

# %%

