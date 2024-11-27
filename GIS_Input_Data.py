# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 10:25:22 2024

@author: Rossella Frugis
"""
#%%

import geopandas as gpd
import pandas as pd
import rioxarray as rxr
from shapely.geometry import shape
import rasterio
from rasterio.mask import mask
from rasterstats import zonal_stats
from IPython.display import display
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid



import os

#%%
#Obtains different files for different Land Cover (aggregated) Class

# Load the DK land Area with administrative boundaries vector layer and merge the administrative boundaries (https://data.humdata.org/dataset/kontur-boundaries-denmark)
DKLand_adm_gdf = gpd.read_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\OCHA_Administrative_Boundaries\Danish_Land.shp')
merged_geometry = DKLand_adm_gdf.union_all()
DKLand_gdf=gpd.GeoDataFrame(geometry=[merged_geometry], crs=DKLand_adm_gdf.crs)
DKLand_gdf.to_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\OCHA_Administrative_Boundaries\Danish_Land_merged.shp')

# Load the WDPA layers and merge them (https://www.protectedplanet.net/en/thematic-areas/wdpa?tab=WDPA)
WDPA0_gdf = gpd.read_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\WDPA_WDOECM_Nov2024_Public_DNK_shp\WDPA_WDOECM_Nov2024_Public_DNK_shp_0\WDPA_WDOECM_Nov2024_Public_DNK_shp-polygons.shp')
WDPA1_gdf = gpd.read_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\WDPA_WDOECM_Nov2024_Public_DNK_shp\WDPA_WDOECM_Nov2024_Public_DNK_shp_1\WDPA_WDOECM_Nov2024_Public_DNK_shp-polygons.shp')
merged_gdf = pd.concat([WDPA0_gdf, WDPA1_gdf], ignore_index=True)
merged_geometry = merged_gdf.union_all()
WDPA_merged_gdf=gpd.GeoDataFrame(geometry=[merged_geometry], crs=DKLand_adm_gdf.crs)
WDPA_merged_gdf.to_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\WDPA_WDOECM_Nov2024_Public_DNK_shp\merged_WDPA.shp')

#Subtracting Protected areas from Danish land 
if DKLand_gdf.crs != WDPA_merged_gdf.crs:
    WDPA_merged_gdf = WDPA_merged_gdf.to_crs(DKLand_gdf.crs)
DKLand_subtracted_gdf= DKLand_gdf['geometry'].difference(WDPA_merged_gdf)
DKLand_notprotected_gdf = gpd.GeoDataFrame(DKLand_gdf.drop(columns='geometry'), geometry=DKLand_subtracted_gdf, crs=DKLand_gdf.crs)
DKLand_notprotected_gdf.to_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\OCHA_Administrative_Boundaries\Danish_Land_notprotected.shp')

# Load the CLC shapefile (https://land.copernicus.eu/en/products/corine-land-cover)
CLC_gdf = gpd.read_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC\Results\U2018_CLC2018_V2020_20u1.shp')

# Keep only the areas that are not protected
if DKLand_notprotected_gdf.crs != CLC_gdf.crs:
    CLC_gdf = CLC_gdf.to_crs(DKLand_notprotected_gdf.crs)

# Truncate the 'Shape_Area' and 'Shape_Leng' fields to 10 characters
if 'Shape_Area' in CLC_gdf.columns:
    CLC_gdf['Shape_Area'] = CLC_gdf['Shape_Area'].astype(str).str[:10]

if 'Shape_Leng' in CLC_gdf.columns:
    CLC_gdf['Shape_Leng'] = CLC_gdf['Shape_Leng'].astype(str).str[:10]

# Perform the intersection
CLC_gdf = gpd.overlay(CLC_gdf, DKLand_notprotected_gdf, how='intersection')

# Save the output to a new shapefile
output_path_CLC = r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\CLC_notprotected.shp'
CLC_gdf.to_file(output_path)

#%%

#Importing the file with municipalities of Denmark (https://ec.europa.eu/eurostat/web/gisco/geodata/statistical-units/local-administrative-units)
output_path_CLC = r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\CLC_notprotected.shp'
CLC_gdf=gpd.read_file(output_path_CLC)
Municipality_gdf= gpd.read_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\LAU_RG_01M_2021_3035.shp\Administrative_DK.shp')

# Ensure both are in the same CRS and fix any invalid geometries
target_crs = "EPSG:32633"
CLC_gdf = CLC_gdf.to_crs(target_crs)
Municipality_gdf = Municipality_gdf.to_crs(target_crs)

CLC_gdf['geometry'] = CLC_gdf['geometry'].apply(lambda geom: geom if geom.is_valid else geom.buffer(0))
Municipality_gdf['geometry'] = Municipality_gdf['geometry'].apply(lambda geom: geom if geom.is_valid else geom.buffer(0))
CLC_gdf.to_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\CLC_notprotected_buffer.shp')

#Definition of a function for filtering and merging the different aggregated CLC classes

def filter_merge_save(gdf, attribute, value_prefix, filtered_path, merged_path, tolerance=0.1):
    """
    Filters a GeoDataFrame based on a given attribute and value prefix, merges the geometries,
    regularizes the merged polygon, saves both filtered and merged layers as shapefiles, and 
    returns only the merged GeoDataFrame.

    Parameters:
        gdf (GeoDataFrame): The input GeoDataFrame to filter.
        attribute (str): The name of the attribute to filter on.
        value_prefix (str): The prefix of the attribute value to filter by.
        filtered_path (str): The file path to save the filtered shapefile.
        merged_path (str): The file path to save the merged shapefile.
        tolerance (float): Tolerance for polygon regularization. Default is 0.01.

    Returns:
        GeoDataFrame: The merged GeoDataFrame containing the combined and regularized geometry.
    """
    # Step 1: Filter the GeoDataFrame based on the attribute's value prefix
    filtered_gdf = gdf[gdf[attribute].astype(str).str.startswith(value_prefix)]
    
    # Step 2: Save the filtered GeoDataFrame to the specified path
    filtered_gdf.to_file(filtered_path)
    
    # Step 3: Merge geometries in the filtered GeoDataFrame
    merged_geometry = filtered_gdf.union_all()
    
    # Step 4: Regularize the merged polygon with the specified tolerance
    regularized_geometry = merged_geometry.simplify(tolerance, preserve_topology=True)
    merged_gdf = gpd.GeoDataFrame(geometry=[regularized_geometry], crs=filtered_gdf.crs)
    
    # Step 5: Save the merged and regularized GeoDataFrame to the specified path
    merged_gdf.to_file(merged_path)
    
    # Return the merged GeoDataFrame directly
    return merged_gdf


# Definition of paths to save each CLC aggregation cathegory output
urban_filtered_path = r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\Urban_Areas.shp'
urban_merged_path = r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\Urban_Areas_Merged.shp'

agricultural_filtered_path = r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\Agricultural_Areas.shp'
agricultural_merged_path = r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\Agricultural_Areas_Merged.shp'

forest_filtered_path = r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\Forest_Areas.shp'
forest_merged_path = r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\Forest_Areas_Merged.shp'

on_vegetation_filtered_path = r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\ON_vegetation_Areas.shp'
on_vegetation_merged_path = r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\ON_vegetation_Areas_Merged.shp'

on_no_vegetation_filtered_path = r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\ON_no_vegetation_Areas.shp'
on_no_vegetation_merged_path = r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\ON_no_vegetation_Areas_Merged.shp'

on_water_filtered_path = r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\ON_water_Areas.shp'
on_water_merged_path = r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\ON_water_Areas_Merged.shp'

water_bodies_filtered_path = r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\Water_Bodies_Areas.shp'
water_bodies_merged_path = r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\Water_Bodies_Areas_Merged.shp'

# Run filtering and merging for each category
# Urban Areas
CLC_urban_gdf = filter_merge_save(CLC_gdf, 'Code_18', '1', urban_filtered_path, urban_merged_path)

# Agricultural Areas
CLC_agricultural_gdf = filter_merge_save(CLC_gdf, 'Code_18', '2', agricultural_filtered_path, agricultural_merged_path)

# Forest Areas
CLC_forest_gdf = filter_merge_save(CLC_gdf, 'Code_18', '31', forest_filtered_path, forest_merged_path)

# Other Nature Areas with vegetation
CLC_on_vegetation_gdf = filter_merge_save(CLC_gdf, 'Code_18', '32', on_vegetation_filtered_path, on_vegetation_merged_path)

# Other Nature Areas without vegetation
CLC_on_no_vegetation_gdf = filter_merge_save(CLC_gdf, 'Code_18', '33', on_no_vegetation_filtered_path, on_no_vegetation_merged_path)

# Other Nature Areas with water
CLC_on_water_gdf = filter_merge_save(CLC_gdf, 'Code_18', '4', on_water_filtered_path, on_water_merged_path)

# Water Bodies Areas
CLC_water_bodies_gdf = filter_merge_save(CLC_gdf, 'Code_18', '5', water_bodies_filtered_path, water_bodies_merged_path)

# Dictionary to store total area by region and land cover type
area_by_municipality = Municipality_gdf[['NUTS_ID', 'geometry']].copy()  # Start with a copy of the regions GeoDataFrame
area_by_municipality.set_index('NUTS_ID', inplace=True)  # Use region names as index for easy access

# Define the different land cover types and their corresponding paths
land_cover_data = {
    "urban": CLC_urban_gdf,
    "agricultural": CLC_agricultural_gdf,
    "forest": CLC_forest_gdf,
    "vegetation": CLC_on_vegetation_gdf,
    "no_vegetation": CLC_on_no_vegetation_gdf,
    "water": CLC_on_water_gdf,
    "water_bodies": CLC_water_bodies_gdf
}

# Loop through each land cover type and calculate areas by region
for cover_type, gdf in land_cover_data.items():
    # Create a column to store area for this land cover type
    area_by_municipality[f"{cover_type}_area_he"] = 0

    # Run intersection and area calculation for each region
    for _, region in Municipality_gdf.iterrows():
        region_name = region['LAU_NAME']
        region_geometry = region['geometry']
        
        # Intersect land cover polygons with the region
        intersected_polygons = []
        for _, land_cover_row in gdf.iterrows():
            try:
                intersection = land_cover_row['geometry'].intersection(region_geometry)
                
                # Append valid intersections
                if not intersection.is_empty and intersection.geom_type in ['Polygon', 'MultiPolygon']:
                    intersected_polygons.append(intersection)
            except Exception as e:
                print(f"Intersection error for region {region_name} with {cover_type}: {e}")
        
        # Create GeoDataFrame with intersected polygons
        if intersected_polygons:
            intersected_gdf = gpd.GeoDataFrame(geometry=intersected_polygons, crs=gdf.crs)
            intersected_gdf = intersected_gdf.to_crs("EPSG:32633")  # Ensure the correct CRS
            
            # Calculate area in hectares for this land cover in this region
            total_area_ha = intersected_gdf.geometry.area.sum()/10**4
            area_by_municipality.loc[region_name, f"{cover_type}_area_he"] = total_area_ha

# Reset the index to have 'NUTS_ID' as a column
area_by_municipality.reset_index(inplace=True)

# Ensure that numeric fields are stored with appropriate precision
area_by_municipality = area_by_municipality.astype({
    'urban_area_he': 'float64',
    'agricultural_area_he': 'float64',
    'forest_area_he': 'float64',
    'vegetation_area_he': 'float64',
    'no_vegetation_area_he': 'float64',
    'water_area_he': 'float64',
    'water_bodies_area_he': 'float64'
})

# Save to shapefile with appropriate precision
output_path_area_by_municipality= r"C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\regions_with_land_cover_areas.shp"
area_by_municipality.to_file(output_path_area_by_municipality)
#%%

# Read the shapefile with area data
output_path_area_by_municipality = r"C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\regions_with_land_cover_areas.shp"
area_by_municipality_gdf = gpd.read_file(output_path_area_by_municipality)

# Define columns with area data
land_cover_columns = [
    'urban_area', 'agricultur', 'forest_are', 
    'vegetation', 'no_vegetat', 'water_area', 
    'water_bodi'
]

# Renaming columns for clarity
rename_map = {
    'urban_area': 'Urban Area [ha]',
    'agricultur': 'Agricultural Area [ha]',
    'forest_are': 'Forest Area [ha]',
    'vegetation': 'Vegetation Area [ha]',
    'no_vegetat': 'No Vegetation Area [ha]',
    'water_area': 'Water Area [ha]',
    'water_bodi': 'Water Bodies Area [ha]',
    'NUTS_ID': 'NUTS ID'  
}

# Create a table with NUTS ID and area columns
area_table = area_by_municipality_gdf[land_cover_columns].copy()
area_table['NUTS ID'] = area_by_municipality_gdf.NUTS_ID
area_table = area_table[['NUTS ID'] + land_cover_columns]
area_table.rename(columns=rename_map, inplace=True)

# Define NUTS ID to region name mapping
nuts_name_mapping = {
    "DK01": "Capital Region of Denmark",
    "DK02": "Region Zealand",
    "DK03": "Region of Southern Denmark",
    "DK04": "Central Denmark Region",
    "DK05": "North Denmark Region",
    # Add other mappings as necessary
}

# Map the region names to a new column
area_table['Region Name'] = area_table['NUTS ID'].map(nuts_name_mapping)

# Reorder columns to put Region Name first
area_table = area_table[['Region Name', 'NUTS ID'] + [col for col in area_table.columns if col not in ['Region Name', 'NUTS ID']]]

# Export the table to a CSV file
output_table_path = r"C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\land_cover_area_by_municipality.csv"
area_table.to_csv(output_table_path, index=False)

#%%

#Importing Potected areas from Biodiversity Council (30% of the national area )

Biodiversity_30_gdf = gpd.read_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\Biodiversity Council\scenarie_30_shp.shp')

Bio30_path =r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\Biodiversity Council\Biodiversity30.shp'
filtered_gdf = Biodiversity_30_gdf[Biodiversity_30_gdf['DN'] == 1]

filtered_gdf.to_file(Bio30_path, driver='ESRI Shapefile')

#%%
# Importing the yields from FAO [kg DM/ha]
        
wheat_raster_path=(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\FAO\whea200a_yld.tif') #potential yields, historical rcp
barley_raster_path=(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\FAO\barl200b_yld.tif')
rye_raster_path=(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\FAO\ryes200b_yld.tif')
oat_raster_path=(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\FAO\oats200b_yld.tif')

# Define the shapefile path
Municipality_gdf= gpd.read_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\LAU_RG_01M_2021_3035.shp\Administrative_DK.shp')
output_path = r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\OCHA_Administrative_Boundaries\regions_yields.shp'  # Path to save the output shapefile

# Define the list of rasters and the field names to store the mean values
raster_paths = {
    "wheat_pot": wheat_raster_path, #kg DM/ha
    "barley_pot": barley_raster_path,
    "rye_pot": rye_raster_path,
    "oat_pot": oat_raster_path
}

# Iterate over each raster and calculate the mean within each polygon
for field_name, raster_path in raster_paths.items():
    # Calculate the mean value for the raster within each polygon
    stats = zonal_stats(Municipality_gdf, raster_path, stats="mean", geojson_out=True)
    
    # Extract the mean values from the stats and add to the GeoDataFrame
    mean_values = [feature["properties"]["mean"] for feature in stats]
    Municipality_gdf[field_name] = mean_values

# Define NUTS ID mapping based on region names
nuts_id_mapping = {
    "Albertslund": "DK2",
    "Alleroed": "DK2",
    "Assens": "DK1",
    "Ballerup": "DK2",
    "Billund": "DK1",
    "Bornholm": "DK2",
    "Broendby": "DK2",
    "Broenderslev": "DK1",
    "Christiansoe": "DK2",
    "Dragoer": "DK2",
    "Egedal": "DK2",
    "Esbjerg": "DK1",
    "Fanoe": "DK1",
    "Favrskov": "DK1",
    "Faxe": "DK2",
    "Fredensborg": "DK2",
    "Fredericia": "DK1",
    "Frederiksberg": "DK2",
    "Frederikshavn": "DK1",
    "Frederikssund": "DK2",
    "Furesoe": "DK2",
    "Faaborg-Midtfyn": "DK1",
    "Gentofte": "DK2",
    "Gladsaxe": "DK2",
    "Glostrup": "DK2",
    "Greve": "DK2",
    "Gribskov": "DK2",
    "Guldborgsund": "DK2",
    "Haderslev": "DK1",
    "Halsnaes": "DK2",
    "Hedensted": "DK1",
    "Helsingoer": "DK2",
    "Herlev": "DK2",
    "Herning": "DK1",
    "Hilleroed": "DK2",
    "Hjoerring": "DK1",
    "Holbaek": "DK2",
    "Holstebro": "DK1",
    "Horsens": "DK1",
    "Hvidovre": "DK2",
    "Hoeje_Taastrup": "DK2",
    "Hoersholm": "DK2",
    "Ikast-Brande": "DK1",
    "Ishoej": "DK2",
    "Jammerbugt": "DK1",
    "Kalundborg": "DK2",
    "Kerteminde": "DK1",
    "Kolding": "DK1",
    "Koebenhavn": "DK2",
    "Koege": "DK2",
    "Langeland": "DK1",
    "Lejre": "DK2",
    "Lemvig": "DK1",
    "Lolland": "DK2",
    "Lyngby-Taarbaek": "DK2",
    "Laesoe": "DK1",
    "Mariagerfjord": "DK1",
    "Middelfart": "DK1",
    "Morsoe": "DK1",
    "Norddjurs": "DK1",
    "Nordfyns": "DK1",
    "Nyborg": "DK1",
    "Naestved": "DK2",
    "Odder": "DK1",
    "Odense": "DK1",
    "Odsherred": "DK2",
    "Randers": "DK1",
    "Rebild": "DK1",
    "Ringkoebing-Skjern": "DK1",
    "Ringsted": "DK2",
    "Roskilde": "DK2",
    "Rudersdal": "DK2",
    "Roedovre": "DK2",
    "Samsoe": "DK1",
    "Silkeborg": "DK1",
    "Skanderborg": "DK1",
    "Skive": "DK1",
    "Slagelse": "DK2",
    "Solroed": "DK2",
    "Soroe": "DK2",
    "Stevns": "DK2",
    "Struer": "DK1",
    "Svendborg": "DK1",
    "Syddjurs": "DK1",
    "Soenderborg": "DK1",
    "Thisted": "DK1",
    "Toender": "DK1",
    "Taarnby": "DK2",
    "Vallensbaek": "DK2",
    "Varde": "DK1",
    "Vejen": "DK1",
    "Vejle": "DK1",
    "Vesthimmerlands": "DK1",
    "Viborg": "DK1",
    "Vordingborg": "DK2",
    "Aeroe": "DK1",
    "Aabenraa": "DK1",
    "Aalborg": "DK1",
    "Aarhus": "DK1"
}
# Add NUTS ID column to the regions GeoDataFrame
Municipality_gdf['Market_Area'] = Municipality_gdf['LAU_NAME'].map(nuts_id_mapping)

# Save the output with added zonal statistics fields
Municipality_gdf.to_file(output_path, driver="ESRI Shapefile")

# Create a new DataFrame with region names, crop potentials, and NUTS ID
table_df = Municipality_gdf[["name_en", "wheat_pot", "barley_pot", "rye_pot", "oat_pot", "NUTS_ID"]]

# Rename the columns
rename_map = {
    'wheat_pot': 'Wheat [kgDW/ha]',
    'barley_pot': 'Barley [kgDW/ha]',
    'rye_pot': 'Rye [kgDW/ha]',
    'oat_pot': 'Oat [kgDW/ha]',
    'LAU_NAME': 'Municipality',
    'Market_Area': 'Power Market Area'
}

# Apply the renaming
table_df.rename(columns=rename_map, inplace=True)

# Sort the table by NUTS ID (ascending order)
table_df = table_df.sort_values(by="Power Market Area", ascending=True)

# Reorder the columns to make NUTS ID the second column
table_df = table_df[['Municipality', 'Power Market Area', 'Wheat [kgDW/ha]', 'Barley [kgDW/ha]', 'Rye [kgDW/ha]', 'Oat [kgDW/ha]']]

# Optionally, save to a CSV file
table_df.to_csv(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\region_yields_table.csv', index=False)

# %%
