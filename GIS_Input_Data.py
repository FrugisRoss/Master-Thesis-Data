# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 10:25:22 2024

@author: Rossella Frugis
"""
#%%
#SECTION 1


import geopandas as gpd
import pandas as pd
import rioxarray as rxr
from shapely.geometry import shape
import rasterio
from rasterio.mask import mask
from rasterio.transform import Affine
from rasterstats import zonal_stats
import numpy as np
from IPython.display import display
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid



import os

#%%
#SECTION 2

# Load the DK land Area with administrative boundaries vector layer and merge the administrative boundaries (https://data.humdata.org/dataset/kontur-boundaries-denmark)
DKLand_adm_gdf = gpd.read_file(r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\OCHA_Administrative_Boundaries\Danish_Land.shp')
merged_geometry = DKLand_adm_gdf.union_all()
DKLand_gdf=gpd.GeoDataFrame(geometry=[merged_geometry], crs=DKLand_adm_gdf.crs)
DKLand_gdf.to_file(r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\OCHA_Administrative_Boundaries\Danish_Land_merged.shp')

# Load the WDPA layers and merge them (https://www.protectedplanet.net/en/thematic-areas/wdpa?tab=WDPA)
WDPA0_gdf = gpd.read_file(r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\WDPA_WDOECM_Nov2024_Public_DNK_shp\WDPA_WDOECM_Nov2024_Public_DNK_shp_0\WDPA_WDOECM_Nov2024_Public_DNK_shp-polygons.shp')
WDPA1_gdf = gpd.read_file(r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\WDPA_WDOECM_Nov2024_Public_DNK_shp\WDPA_WDOECM_Nov2024_Public_DNK_shp_1\WDPA_WDOECM_Nov2024_Public_DNK_shp-polygons.shp')
merged_gdf = pd.concat([WDPA0_gdf, WDPA1_gdf], ignore_index=True)
merged_geometry = merged_gdf.union_all()
WDPA_merged_gdf=gpd.GeoDataFrame(geometry=[merged_geometry], crs=DKLand_adm_gdf.crs)
WDPA_merged_gdf.to_file(r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\WDPA_WDOECM_Nov2024_Public_DNK_shp\merged_WDPA.shp')

#Subtracting Protected areas from Danish land 
if DKLand_gdf.crs != WDPA_merged_gdf.crs:
    WDPA_merged_gdf = WDPA_merged_gdf.to_crs(DKLand_gdf.crs)
DKLand_subtracted_gdf= DKLand_gdf['geometry'].difference(WDPA_merged_gdf)
DKLand_notprotected_gdf = gpd.GeoDataFrame(DKLand_gdf.drop(columns='geometry'), geometry=DKLand_subtracted_gdf, crs=DKLand_gdf.crs)
DKLand_notprotected_gdf.to_file(r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\OCHA_Administrative_Boundaries\Danish_Land_notprotected.shp')

# Load the CLC shapefile (https://land.copernicus.eu/en/products/corine-land-cover)
CLC_gdf = gpd.read_file(r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC\Results\U2018_CLC2018_V2020_20u1.shp')

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
output_path_CLC = r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\CLC_notprotected.shp'
CLC_gdf.to_file(output_path)

#%%
#SECTION 3 

DKLand_gdf=gpd.read_file(r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\OCHA_Administrative_Boundaries\Danish_Land_merged.shp')
DKLand_notprotected_gdf=gpd.read_file(r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\OCHA_Administrative_Boundaries\Danish_Land_notprotected.shp')

#Reproject the GeoDataFrames to EPSG:3857
DKLand_gdf = DKLand_gdf.to_crs(epsg=3857)
DKLand_notprotected_gdf = DKLand_notprotected_gdf.to_crs(epsg=3857)

# Compute the areas of the polygons in DKLand_gdf
DKLand_gdf['area_ha'] = DKLand_gdf['geometry'].area / 10**4  # Convert to hectares

# Compute the areas of the polygons in DKLand_notprotected_gdf
DKLand_notprotected_gdf['area_ha'] = DKLand_notprotected_gdf['geometry'].area / 10**4  # Convert to hectares

# Store the area of the polygon in DKLand_gdf in a variable
DKLand_area = DKLand_gdf['area_ha'].iloc[0]

# Store the area of the polygon in DKLand_notprotected_gdf in a variable
DKLand_notprotected_area = DKLand_notprotected_gdf['area_ha'].iloc[0]

# Compute the protected land area
Protected_Land = DKLand_area - DKLand_notprotected_area

# Compute the protected land percentage
Protected_Land_Percentage = (Protected_Land / DKLand_area) * 100

print("Protected Land Area (ha):", Protected_Land)
print("Protected Land Percentage (%):", Protected_Land_Percentage)


#%%
# SECTION 4
# Obtains different files for different Land Cover (aggregated) Classes

# Importing the file with municipalities of Denmark (https://ec.europa.eu/eurostat/web/gisco/geodata/statistical-units/local-administrative-units)
output_path_CLC = r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\CLC_notprotected.shp'
CLC_gdf = gpd.read_file(output_path_CLC)
Municipality_gdf = gpd.read_file(r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\LAU_RG_01M_2021_3035.shp\Administrative_DK.shp')

# Ensure both are in the same CRS and fix any invalid geometries
target_crs = "EPSG:32633"
CLC_gdf = CLC_gdf.to_crs(target_crs)
Municipality_gdf = Municipality_gdf.to_crs(target_crs)

CLC_gdf['geometry'] = CLC_gdf['geometry'].apply(lambda geom: geom if geom.is_valid else geom.buffer(0))
Municipality_gdf['geometry'] = Municipality_gdf['geometry'].apply(lambda geom: geom if geom.is_valid else geom.buffer(0))
CLC_gdf.to_file(r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\CLC_notprotected_buffer.shp')

# Definition of a function for filtering and merging the different aggregated CLC classes
def filter_merge_save(gdf, attribute, values_list, filtered_path, merged_path, tolerance=0.01):
    """
    Filters the GeoDataFrame by a list of attribute values, saves the filtered results, merges the filtered results, and saves to a file.

    Parameters:
    gdf (GeoDataFrame): The input GeoDataFrame.
    attribute (str): The attribute to filter by.
    values_list (list): The list of attribute values to filter by.
    filtered_path (str): The path to save the filtered GeoDataFrame.
    merged_path (str): The path to save the merged GeoDataFrame.
    tolerance (float): The tolerance for simplifying the merged geometry.
    """
    # Step 1: Filter the GeoDataFrame based on the attribute's values list
    filtered_gdf = gdf[gdf[attribute].isin(values_list)]
    
    # Step 2: Save the filtered GeoDataFrame to the specified path
    filtered_gdf.to_file(filtered_path)
    
    # Step 3: Merge geometries in the filtered GeoDataFrame
    merged_geometry = filtered_gdf.unary_union
    
    # Step 4: Regularize the merged polygon with the specified tolerance
    regularized_geometry = merged_geometry.simplify(tolerance, preserve_topology=True)
    merged_gdf = gpd.GeoDataFrame(geometry=[regularized_geometry], crs=filtered_gdf.crs)
    
    # Step 5: Save the merged and regularized GeoDataFrame to the specified path
    merged_gdf.to_file(merged_path)
    
    # Return the merged GeoDataFrame directly
    return merged_gdf

# Definition of paths to save each CLC aggregation category output
urban_filtered_path = r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\Urban_Areas.shp'
urban_merged_path = r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\Urban_Areas_Merged.shp'

agricultural_filtered_path = r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\Agricultural_Areas.shp'
agricultural_merged_path = r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\Agricultural_Areas_Merged.shp'

forest_filtered_path = r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\Forest_Areas.shp'
forest_merged_path = r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\Forest_Areas_Merged.shp'

on_vegetation_filtered_path = r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\ON_vegetation_Areas.shp'
on_vegetation_merged_path = r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\ON_vegetation_Areas_Merged.shp'

on_no_vegetation_filtered_path = r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\ON_no_vegetation_Areas.shp'
on_no_vegetation_merged_path = r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\ON_no_vegetation_Areas_Merged.shp'

on_water_filtered_path = r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\ON_water_Areas.shp'
on_water_merged_path = r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\ON_water_Areas_Merged.shp'

water_bodies_filtered_path = r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\Water_Bodies_Areas.shp'
water_bodies_merged_path = r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\Water_Bodies_Areas_Merged.shp'

# Define the values for each category
urban_values = ['111', '112', '121', '122', '123', '124', '131', '132', '133', '141', '142']
agricultural_values = ['211', '212', '213', '221', '222', '223', '231','243']
forest_values = ['311', '312', '313', '244']
on_vegetation_values = ['321', '322', '323', '324', '241', '242']
on_no_vegetation_values = ['331', '332', '333', '334', '335']
on_water_values = ['411', '412', '421', '422', '423']
water_bodies_values = ['511', '512', '521', '522', '523']


# Run filtering and merging for each category
# Urban Areas
CLC_urban_gdf = filter_merge_save(CLC_gdf, 'Code_18', urban_values, urban_filtered_path, urban_merged_path)

# Agricultural Areas
CLC_agricultural_gdf = filter_merge_save(CLC_gdf, 'Code_18', agricultural_values, agricultural_filtered_path, agricultural_merged_path)

# Forest Areas
CLC_forest_gdf = filter_merge_save(CLC_gdf, 'Code_18', forest_values, forest_filtered_path, forest_merged_path)

# Other Nature Areas with vegetation
CLC_on_vegetation_gdf = filter_merge_save(CLC_gdf, 'Code_18', on_vegetation_values, on_vegetation_filtered_path, on_vegetation_merged_path)

# Other Nature Areas without vegetation
CLC_on_no_vegetation_gdf = filter_merge_save(CLC_gdf, 'Code_18', on_no_vegetation_values, on_no_vegetation_filtered_path, on_no_vegetation_merged_path)

# Other Nature Areas with water
CLC_on_water_gdf = filter_merge_save(CLC_gdf, 'Code_18', on_water_values, on_water_filtered_path, on_water_merged_path)

# Water Bodies Areas
CLC_water_bodies_gdf = filter_merge_save(CLC_gdf, 'Code_18',water_bodies_values, water_bodies_filtered_path, water_bodies_merged_path)

# Dictionary to store total area by region and land cover type
area_by_municipality = Municipality_gdf[['LAU_NAME', 'geometry']].copy()  # Start with a copy of the regions GeoDataFrame
area_by_municipality.set_index('LAU_NAME', inplace=True)  # Use region names as index for easy access

# Define land cover data
land_cover_data = {
    'urban': CLC_urban_gdf,
    'agricultural': CLC_agricultural_gdf,
    'forest': CLC_forest_gdf,
    'vegetation': CLC_on_vegetation_gdf,
    'no_vegetation': CLC_on_no_vegetation_gdf,
    'water': CLC_on_water_gdf,
    'water_bodies': CLC_water_bodies_gdf
}

# Loop through each land cover type and calculate areas by region
for cover_type, gdf in land_cover_data.items():
    column_name = f"{cover_type}_area_ha"

    # Ensure the column exists and is of type float64
    if column_name not in area_by_municipality.columns:
        area_by_municipality[column_name] = 0.0  # Create column with initial value of 0.0

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
            total_area_ha = intersected_gdf.geometry.area.sum() / 10**4
            area_by_municipality.loc[region_name, column_name] = total_area_ha

# Reset the index to have 'LAU_NAME' as a column
area_by_municipality.reset_index(inplace=True)

# Ensure numeric fields are stored with appropriate precision
# Dynamically cast only existing columns to float64
float_columns = [col for col in area_by_municipality.columns if col.endswith('_area_ha')]
area_by_municipality[float_columns] = area_by_municipality[float_columns].astype(float)

# Save to shapefile with appropriate precision
output_path_area_by_municipality = r"C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\municipalities_with_land_cover_areas.shp"
area_by_municipality.to_file(output_path_area_by_municipality)

#%%
#SECTION 5

# Read the shapefile with area data
output_path_area_by_municipality = r"C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\municipalities_with_land_cover_areas.shp"
area_by_municipality_gdf = gpd.read_file(output_path_area_by_municipality, encoding="utf-8")

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
    'LAU_NAME': 'Municipality',
    'Market_Area': 'Power Market Area'   
}

# Create a table with NUTS ID and area columns
area_table = area_by_municipality_gdf[land_cover_columns].copy()
area_table['LAU_NAME'] = area_by_municipality_gdf.LAU_NAME
area_table = area_table[['LAU_NAME'] + land_cover_columns]

# Define NUTS ID to region name mapping
market_area_mapping = {
    "Albertslund": "DK2",
    "Allerød": "DK2",
    "Assens": "DK1",
    "Ballerup": "DK2",
    "Billund": "DK1",
    "Bornholm": "DK2",
    "Brøndby": "DK2",
    "Brønderslev": "DK1",
    "Christiansø": "DK2",
    "Dragør": "DK2",
    "Egedal": "DK2",
    "Esbjerg": "DK1",
    "Fanø": "DK1",
    "Favrskov": "DK1",
    "Faxe": "DK2",
    "Fredensborg": "DK2",
    "Fredericia": "DK1",
    "Frederiksberg": "DK2",
    "Frederikshavn": "DK1",
    "Frederikssund": "DK2",
    "Furesø": "DK2",
    "Faaborg-Midtfyn": "DK1",
    "Gentofte": "DK2",
    "Gladsaxe": "DK2",
    "Glostrup": "DK2",
    "Greve": "DK2",
    "Gribskov": "DK2",
    "Guldborgsund": "DK2",
    "Haderslev": "DK1",
    "Halsnæs": "DK2",
    "Hedensted": "DK1",
    "Helsingør": "DK2",
    "Herlev": "DK2",
    "Herning": "DK1",
    "Hillerød": "DK2",
    "Hjørring": "DK1",
    "Holbæk": "DK2",
    "Holstebro": "DK1",
    "Horsens": "DK1",
    "Hvidovre": "DK2",
    "Høje-Taastrup": "DK2",
    "Hørsholm": "DK2",
    "Ikast-Brande": "DK1",
    "Ishøj": "DK2",
    "Jammerbugt": "DK1",
    "Kalundborg": "DK2",
    "Kerteminde": "DK1",
    "Kolding": "DK1",
    "København": "DK2",
    "Køge": "DK2",
    "Langeland": "DK1",
    "Lejre": "DK2",
    "Lemvig": "DK1",
    "Lolland": "DK2",
    "Lyngby-Taarbæk": "DK2",
    "Læsø": "DK1",
    "Mariagerfjord": "DK1",
    "Middelfart": "DK1",
    "Morsø": "DK1",
    "Norddjurs": "DK1",
    "Nordfyns": "DK1",
    "Nyborg": "DK1",
    "Næstved": "DK2",
    "Odder": "DK1",
    "Odense": "DK1",
    "Odsherred": "DK2",
    "Randers": "DK1",
    "Rebild": "DK1",
    "Ringkøbing-Skjern": "DK1",
    "Ringsted": "DK2",
    "Roskilde": "DK2",
    "Rudersdal": "DK2",
    "Rødovre": "DK2",
    "Samsø": "DK1",
    "Silkeborg": "DK1",
    "Skanderborg": "DK1",
    "Skive": "DK1",
    "Slagelse": "DK2",
    "Solrød": "DK2",
    "Sorø": "DK2",
    "Stevns": "DK2",
    "Struer": "DK1",
    "Svendborg": "DK1",
    "Syddjurs": "DK1",
    "Sønderborg": "DK1",
    "Thisted": "DK1",
    "Tønder": "DK1",
    "Tårnby": "DK2",
    "Vallensbæk": "DK2",
    "Varde": "DK1",
    "Vejen": "DK1",
    "Vejle": "DK1",
    "Vesthimmerlands": "DK1",
    "Viborg": "DK1",
    "Vordingborg": "DK2",
    "Ærø": "DK1",
    "Aabenraa": "DK1",
    "Aalborg": "DK1",
    "Aarhus": "DK1"
}



# Define the desired order of municipalities
desired_order = [
    "Albertslund", "Allerød", "Assens", "Ballerup", "Billund", "Bornholm", "Brøndby", "Brønderslev",
    "Christiansø", "Dragør", "Egedal", "Esbjerg", "Fanø", "Favrskov", "Faxe", "Fredensborg", "Fredericia",
    "Frederiksberg", "Frederikshavn", "Frederikssund", "Furesø", "Faaborg-Midtfyn", "Gentofte", "Gladsaxe",
    "Glostrup", "Greve", "Gribskov", "Guldborgsund", "Haderslev", "Halsnæs", "Hedensted", "Helsingør",
    "Herlev", "Herning", "Hillerød", "Hjørring", "Holbæk", "Holstebro", "Horsens", "Hvidovre", "Høje-Taastrup",
    "Hørsholm", "Ikast-Brande", "Ishøj", "Jammerbugt", "Kalundborg", "Kerteminde", "Kolding", "København",
    "Køge", "Langeland", "Lejre", "Lemvig", "Lolland", "Lyngby-Taarbæk", "Læsø", "Mariagerfjord", "Middelfart",
    "Morsø", "Norddjurs", "Nordfyns", "Nyborg", "Næstved", "Odder", "Odense", "Odsherred", "Randers", "Rebild",
    "Ringkøbing-Skjern", "Ringsted", "Roskilde", "Rudersdal", "Rødovre", "Samsø", "Silkeborg", "Skanderborg",
    "Skive", "Slagelse", "Solrød", "Sorø", "Stevns", "Struer", "Svendborg", "Syddjurs", "Sønderborg", "Thisted",
    "Tønder", "Tårnby", "Vallensbæk", "Varde", "Vejen", "Vejle", "Vesthimmerlands", "Viborg", "Vordingborg",
    "Ærø", "Aabenraa", "Aalborg", "Aarhus"
]

# Ensure the DataFrame is ordered according to the desired order
area_table['LAU_NAME'] = pd.Categorical(area_table['LAU_NAME'], categories=desired_order, ordered=True)
area_table = area_table.sort_values('LAU_NAME')

# Map the region names to a new column
area_table['Market_Area'] = area_table['LAU_NAME'].map(market_area_mapping)

# Reorder columns to put Region Name first
area_table = area_table[['LAU_NAME', 'Market_Area'] + [col for col in area_table.columns if col not in ['LAU_NAME', 'Market_Area']]]
area_table.rename(columns=rename_map, inplace=True)

# Export the table to a CSV file with UTF-8 encoding
output_table_path = r"C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\land_cover_area_by_municipality.csv"
area_table.to_csv(output_table_path, index=False, encoding='utf-8-sig')

#%%
#SECTION 6

#Importing Potected areas from Biodiversity Council (30% of the national area )

Biodiversity_30_gdf = gpd.read_file(r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\Biodiversity Council\scenarie_30_shp.shp')

Bio30_path =r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\Biodiversity Council\Biodiversity30.shp'
filtered_gdf = Biodiversity_30_gdf[Biodiversity_30_gdf['DN'] == 1]

filtered_gdf.to_file(Bio30_path, driver='ESRI Shapefile')
#%%
#SECTION 7
Bio30_path =r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\Biodiversity Council\Biodiversity30.shp'
Bio_30_gdf= gpd.read_file(Bio30_path)

# Definition of paths to save each CLC aggregation cathegory output
urban_merged_path = r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\Urban_Areas_Merged.shp'

agricultural_merged_path = r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\Agricultural_Areas_Merged.shp'

forest_merged_path = r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\Forest_Areas_Merged.shp'

on_vegetation_merged_path = r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\ON_vegetation_Areas_Merged.shp'

on_no_vegetation_merged_path = r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\ON_no_vegetation_Areas_Merged.shp'

on_water_merged_path = r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\ON_water_Areas_Merged.shp'

water_bodies_merged_path = r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\Water_Bodies_Areas_Merged.shp'

urban_gdf = gpd.read_file(urban_merged_path)
agricultural_gdf = gpd.read_file(agricultural_merged_path)
forest_gdf = gpd.read_file(forest_merged_path)
on_vegetation_gdf = gpd.read_file(on_vegetation_merged_path)
on_no_vegetation_gdf = gpd.read_file(on_no_vegetation_merged_path)
on_water_gdf = gpd.read_file(on_water_merged_path)
water_bodies_gdf = gpd.read_file(water_bodies_merged_path)


def check_crs(gdf, expected_crs):
    if gdf.crs.to_epsg() != expected_crs:
        print(f"CRS mismatch: Converting from {gdf.crs.to_epsg()} to {expected_crs}")
        gdf = gdf.to_crs(epsg=expected_crs)
    else:
        print(f"CRS check passed for {gdf}")
    return gdf

# Define the expected CRS
expected_crs = 3035

# Check CRS for each GeoDataFrame
check_crs(urban_gdf, expected_crs)
check_crs(agricultural_gdf, expected_crs)
check_crs(forest_gdf, expected_crs)
check_crs(on_vegetation_gdf, expected_crs)
check_crs(on_no_vegetation_gdf, expected_crs)
check_crs(on_water_gdf, expected_crs)
check_crs(water_bodies_gdf, expected_crs)
check_crs(Bio_30_gdf, expected_crs)

# Intersect Bio_30_gdf with each GeoDataFrame
urban_intersection = gpd.overlay(Bio_30_gdf, urban_gdf, how='intersection')
agricultural_intersection = gpd.overlay(Bio_30_gdf, agricultural_gdf, how='intersection')
forest_intersection = gpd.overlay(Bio_30_gdf, forest_gdf, how='intersection')
on_vegetation_intersection = gpd.overlay(Bio_30_gdf, on_vegetation_gdf, how='intersection')
on_no_vegetation_intersection = gpd.overlay(Bio_30_gdf, on_no_vegetation_gdf, how='intersection')
on_water_intersection = gpd.overlay(Bio_30_gdf, on_water_gdf, how='intersection')
water_bodies_intersection = gpd.overlay(Bio_30_gdf, water_bodies_gdf, how='intersection')

def compute_areas(gdf):
    gdf['area'] = gdf.geometry.area
    return gdf['area'].sum()

# List of intersection GeoDataFrames and their names
intersection_gdfs = [
    ('urban_intersection', urban_intersection),
    ('agricultural_intersection', agricultural_intersection),
    ('forest_intersection', forest_intersection),
    ('on_vegetation_intersection', on_vegetation_intersection),
    ('on_no_vegetation_intersection', on_no_vegetation_intersection),
    ('on_water_intersection', on_water_intersection),
    ('water_bodies_intersection', water_bodies_intersection)
]

agricultural_intersection.to_file(r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\Biodiversity Council\agricultural_intersection.shp')

# Compute areas and create a table
areas = []
for name, gdf in intersection_gdfs:
    area = compute_areas(gdf)
    areas.append((name, area))

# Create a DataFrame to display the results
areas_df = pd.DataFrame(areas, columns=['Intersection_GDF', 'Area'])

# Load the CRS_gdf
CRS_gdf = gpd.read_file(r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\Tekstur2014\Tekstur2014.shp')

# Compute the intersection
intersection_gdf = gpd.overlay(agricultural_intersection, CRS_gdf, how='intersection')

# Compute areas and create a table
areas = []
for name, gdf in intersection_gdfs:
    area = compute_areas(gdf)
    areas.append((name, area))

# Add the computed areas for the new intersection
intersection_area = compute_areas(intersection_gdf)
areas.append(('agricultural_intersection_CRS', intersection_area))

# Create a DataFrame to display the results
areas_df = pd.DataFrame(areas, columns=['Intersection_GDF', 'Area'])
  # %%
#SECTION 8
#Computes the national average of yields and then assigns them to each municipality

wheat_raster_path=(r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\FAO\Wheat_clip.tif') #potential yields, historical rcp
barley_raster_path=(r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\FAO\Barley_clip.tif')
rye_raster_path=(r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\FAO\Rye_clip.tif')
oat_raster_path=(r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\FAO\Oats_clip.tif')

DKLand_gdf=gpd.read_file(r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\OCHA_Administrative_Boundaries\Danish_Land_merged.shp')
Municipality_gdf= gpd.read_file(r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\LAU_RG_01M_2021_3035.shp\Administrative_DK.shp')

# Function to calculate zonal statistics
def calculate_average(raster_path, gdf):
    stats = zonal_stats(gdf, raster_path, stats=['mean'], geojson_out=True)
    # Extract mean values for each polygon
    mean_values = [feature['properties']['mean'] for feature in stats]
    return mean_values

# Calculate averages for the entire DKLand_gdf (national level)
wheat_avg = calculate_average(wheat_raster_path, DKLand_gdf)[0]
barley_avg = calculate_average(barley_raster_path, DKLand_gdf)[0]
rye_avg = calculate_average(rye_raster_path, DKLand_gdf)[0]
oat_avg = calculate_average(oat_raster_path, DKLand_gdf)[0]

# Assign the national average as constant columns in Municipality_gdf
Municipality_gdf['National_Wheat_Avg'] = wheat_avg
Municipality_gdf['National_Barley_Avg'] = barley_avg
Municipality_gdf['National_Rye_Avg'] = rye_avg
Municipality_gdf['National_Oat_Avg'] = oat_avg


# Define Market Area mapping with Municipality and its Respect market area
market_area_mapping = {
    "Albertslund": "DK2",
    "Allerød": "DK2",
    "Assens": "DK1",
    "Ballerup": "DK2",
    "Billund": "DK1",
    "Bornholm": "DK2",
    "Brøndby": "DK2",
    "Brønderslev": "DK1",
    "Christiansø": "DK2",
    "Dragør": "DK2",
    "Egedal": "DK2",
    "Esbjerg": "DK1",
    "Fanø": "DK1",
    "Favrskov": "DK1",
    "Faxe": "DK2",
    "Fredensborg": "DK2",
    "Fredericia": "DK1",
    "Frederiksberg": "DK2",
    "Frederikshavn": "DK1",
    "Frederikssund": "DK2",
    "Furesø": "DK2",
    "Faaborg-Midtfyn": "DK1",
    "Gentofte": "DK2",
    "Gladsaxe": "DK2",
    "Glostrup": "DK2",
    "Greve": "DK2",
    "Gribskov": "DK2",
    "Guldborgsund": "DK2",
    "Haderslev": "DK1",
    "Halsnæs": "DK2",
    "Hedensted": "DK1",
    "Helsingør": "DK2",
    "Herlev": "DK2",
    "Herning": "DK1",
    "Hillerød": "DK2",
    "Hjørring": "DK1",
    "Holbæk": "DK2",
    "Holstebro": "DK1",
    "Horsens": "DK1",
    "Hvidovre": "DK2",
    "Høje-Taastrup": "DK2",
    "Hørsholm": "DK2",
    "Ikast-Brande": "DK1",
    "Ishøj": "DK2",
    "Jammerbugt": "DK1",
    "Kalundborg": "DK2",
    "Kerteminde": "DK1",
    "Kolding": "DK1",
    "København": "DK2",
    "Køge": "DK2",
    "Langeland": "DK1",
    "Lejre": "DK2",
    "Lemvig": "DK1",
    "Lolland": "DK2",
    "Lyngby-Taarbæk": "DK2",
    "Læsø": "DK1",
    "Mariagerfjord": "DK1",
    "Middelfart": "DK1",
    "Morsø": "DK1",
    "Norddjurs": "DK1",
    "Nordfyns": "DK1",
    "Nyborg": "DK1",
    "Næstved": "DK2",
    "Odder": "DK1",
    "Odense": "DK1",
    "Odsherred": "DK2",
    "Randers": "DK1",
    "Rebild": "DK1",
    "Ringkøbing-Skjern": "DK1",
    "Ringsted": "DK2",
    "Roskilde": "DK2",
    "Rudersdal": "DK2",
    "Rødovre": "DK2",
    "Samsø": "DK1",
    "Silkeborg": "DK1",
    "Skanderborg": "DK1",
    "Skive": "DK1",
    "Slagelse": "DK2",
    "Solrød": "DK2",
    "Sorø": "DK2",
    "Stevns": "DK2",
    "Struer": "DK1",
    "Svendborg": "DK1",
    "Syddjurs": "DK1",
    "Sønderborg": "DK1",
    "Thisted": "DK1",
    "Tønder": "DK1",
    "Tårnby": "DK2",
    "Vallensbæk": "DK2",
    "Varde": "DK1",
    "Vejen": "DK1",
    "Vejle": "DK1",
    "Vesthimmerlands": "DK1",
    "Viborg": "DK1",
    "Vordingborg": "DK2",
    "Ærø": "DK1",
    "Aabenraa": "DK1",
    "Aalborg": "DK1",
    "Aarhus": "DK1"
}

Municipality_gdf['Market_Area'] = Municipality_gdf['LAU_NAME'].map(market_area_mapping)

# Rename columns
rename_map = {
    'National_Wheat_Avg': 'Wheat [kgDW/ha]',
    'National_Barley_Avg': 'Barley [kgDW/ha]',
    'National_Rye_Avg': 'Rye [kgDW/ha]',
    'National_Oat_Avg': 'Oat [kgDW/ha]',
    'LAU_NAME': 'Municipality',
    'Market_Area': 'Power Market Area'
}

Municipality_gdf.rename(columns=rename_map, inplace=True)

# Select relevant columns
columns_to_export = ['Municipality', 'Wheat [kgDW/ha]', 'Barley [kgDW/ha]',
                     'Rye [kgDW/ha]', 'Oat [kgDW/ha]', 'Power Market Area']

export_df = Municipality_gdf[columns_to_export]

#Reorder the columns to make NUTS ID the second column
export_df = export_df[['Municipality', 'Power Market Area', 'Wheat [kgDW/ha]', 'Barley [kgDW/ha]', 'Rye [kgDW/ha]', 'Oat [kgDW/ha]']]


# Export to CSV
csv_output_path = r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\region_yields_table.csv'
export_df.to_csv(csv_output_path, index=False, encoding='utf-8-sig')



# %%
#SECTION 9

Municipality_gdf= gpd.read_file(r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\LAU_RG_01M_2021_3035.shp\Administrative_DK.shp')
CRS_gdf = gpd.read_file(r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\Tekstur2014\Tekstur2014.shp')
Agricultural_areas_gdf = gpd.read_file(r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\Agricultural_Areas_Merged.shp')

# Ensure all GeoDataFrames have the same CRS
if Municipality_gdf.crs != CRS_gdf.crs:
    CRS_gdf = CRS_gdf.to_crs(Municipality_gdf.crs)

if Agricultural_areas_gdf.crs != Municipality_gdf.crs:
    Agricultural_areas_gdf = Agricultural_areas_gdf.to_crs(Municipality_gdf.crs)

# If the CRS is not projected, reproject to a suitable projected CRS
if not Municipality_gdf.crs.is_projected:
    Municipality_gdf = Municipality_gdf.to_crs(epsg=3035)
    CRS_gdf = CRS_gdf.to_crs(epsg=3035)
    Agricultural_areas_gdf = Agricultural_areas_gdf.to_crs(epsg=3035)

# Perform intersection to keep only the parts of CRS_gdf that intersect with Agricultural_areas_gdf
intersected_gdf = gpd.overlay(CRS_gdf, Agricultural_areas_gdf, how='intersection')

# Clip the intersected_gdf with the Municipality_gdf
clipped_gdf = gpd.clip(intersected_gdf, Municipality_gdf)

# Compute the area for each municipality
clipped_gdf['area'] = clipped_gdf.geometry.area/10**4

# Group by 'LAU_NAME' and sum the areas
area_per_municipality = clipped_gdf.groupby(Municipality_gdf['LAU_NAME'])['area'].sum().reset_index()

# Reorder the municipalities
desired_order = [
    "Albertslund", "Allerød", "Assens", "Ballerup", "Billund", "Bornholm", "Brøndby", "Brønderslev",
    "Christiansø", "Dragør", "Egedal", "Esbjerg", "Fanø", "Favrskov", "Faxe", "Fredensborg", "Fredericia",
    "Frederiksberg", "Frederikshavn", "Frederikssund", "Furesø", "Faaborg-Midtfyn", "Gentofte", "Gladsaxe",
    "Glostrup", "Greve", "Gribskov", "Guldborgsund", "Haderslev", "Halsnæs", "Hedensted", "Helsingør",
    "Herlev", "Herning", "Hillerød", "Hjørring", "Holbæk", "Holstebro", "Horsens", "Hvidovre", "Høje-Taastrup",
    "Hørsholm", "Ikast-Brande", "Ishøj", "Jammerbugt", "Kalundborg", "Kerteminde", "Kolding", "København",
    "Køge", "Langeland", "Lejre", "Lemvig", "Lolland", "Lyngby-Taarbæk", "Læsø", "Mariagerfjord", "Middelfart",
    "Morsø", "Norddjurs", "Nordfyns", "Nyborg", "Næstved", "Odder", "Odense", "Odsherred", "Randers", "Rebild",
    "Ringkøbing-Skjern", "Ringsted", "Roskilde", "Rudersdal", "Rødovre", "Samsø", "Silkeborg", "Skanderborg",
    "Skive", "Slagelse", "Solrød", "Sorø", "Stevns", "Struer", "Svendborg", "Syddjurs", "Sønderborg", "Thisted",
    "Tønder", "Tårnby", "Vallensbæk", "Varde", "Vejen", "Vejle", "Vesthimmerlands", "Viborg", "Vordingborg",
    "Ærø", "Aabenraa", "Aalborg", "Aarhus"
]

# Create a DataFrame with the desired order
ordered_df = pd.DataFrame({'LAU_NAME': desired_order})

# Merge with the area data
result_df = ordered_df.merge(area_per_municipality, on='LAU_NAME', how='left')

# Export to CSV
csv_output_path = r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\CRS_area.csv'
result_df.to_csv(csv_output_path, index=False, encoding='utf-8-sig')

#%%
# Define file paths
municipality_fp = r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\LAU_RG_01M_2021_3035.shp\Administrative_DK.shp'
biodiversity_fp = r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\Biodiversity Council\scenarie_30_shp.shp'
forest_fp = r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\Biodiversity Council\CLC_forest_merged.shp'
csv_output_path = r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\Forest_area.csv'

# Read only necessary columns to save memory
print("Reading Municipality data...")
Municipality_gdf = gpd.read_file(municipality_fp, usecols=['LAU_NAME', 'geometry'])

print("Reading Biodiversity data...")
Biodiversity_gdf = gpd.read_file(biodiversity_fp, usecols=['geometry'])

print("Reading Forest Areas data...")
Forest_areas_gdf = gpd.read_file(forest_fp, usecols=['geometry'])

# Define target CRS
target_crs = 'EPSG:3035'

# Reproject all GeoDataFrames to target CRS if not already
print("Reprojecting GeoDataFrames to target CRS...")
Municipality_gdf = Municipality_gdf.to_crs(target_crs)
Biodiversity_gdf = Biodiversity_gdf.to_crs(target_crs)
Forest_areas_gdf = Forest_areas_gdf.to_crs(target_crs)

# Simplify geometries to speed up processing (adjust tolerance as needed)
tolerance = 100  # in CRS units (meters for EPSG:3035)
print(f"Simplifying geometries with tolerance={tolerance} meters...")
Municipality_gdf['geometry'] = Municipality_gdf['geometry'].simplify(tolerance, preserve_topology=True)
Biodiversity_gdf['geometry'] = Biodiversity_gdf['geometry'].simplify(tolerance, preserve_topology=True)
Forest_areas_gdf['geometry'] = Forest_areas_gdf['geometry'].simplify(tolerance, preserve_topology=True)

# Perform spatial join between Biodiversity_gdf and Forest_areas_gdf for intersection
print("Performing spatial join between Biodiversity and Forest Areas...")
intersected_gdf = gpd.sjoin(Biodiversity_gdf, Forest_areas_gdf, how='inner', predicate='intersects')

# Drop the index_right column added by sjoin
intersected_gdf = intersected_gdf.drop(columns=['index_right'])

# Clip the intersected_gdf with the Municipality_gdf
print("Clipping intersected geometries with Municipality boundaries...")
clipped_gdf = gpd.clip(intersected_gdf, Municipality_gdf)

# Perform spatial join to add 'LAU_NAME' from Municipality_gdf to clipped_gdf
print("Performing spatial join to attach 'LAU_NAME'...")
clipped_gdf = gpd.sjoin(clipped_gdf, Municipality_gdf[['LAU_NAME', 'geometry']], how='left', predicate='within')

# Drop unnecessary columns resulting from the spatial join
clipped_gdf = clipped_gdf.drop(columns=['index_right'])

# Compute the area for each geometry in hectares (assuming CRS units are meters)
print("Calculating areas...")
clipped_gdf['area'] = clipped_gdf.geometry.area / 10**4  # Converts m² to hectares

# Group by 'LAU_NAME' and sum the areas
print("Aggregating area per municipality...")
area_per_municipality = clipped_gdf.groupby('LAU_NAME')['area'].sum().reset_index()

# Define the desired order of municipalities
desired_order = [
    "Albertslund", "Allerød", "Assens", "Ballerup", "Billund", "Bornholm", "Brøndby", "Brønderslev",
    "Christiansø", "Dragør", "Egedal", "Esbjerg", "Fanø", "Favrskov", "Faxe", "Fredensborg", "Fredericia",
    "Frederiksberg", "Frederikshavn", "Frederikssund", "Furesø", "Faaborg-Midtfyn", "Gentofte", "Gladsaxe",
    "Glostrup", "Greve", "Gribskov", "Guldborgsund", "Haderslev", "Halsnæs", "Hedensted", "Helsingør",
    "Herlev", "Herning", "Hillerød", "Hjørring", "Holbæk", "Holstebro", "Horsens", "Hvidovre", "Høje-Taastrup",
    "Hørsholm", "Ikast-Brande", "Ishøj", "Jammerbugt", "Kalundborg", "Kerteminde", "Kolding", "København",
    "Køge", "Langeland", "Lejre", "Lemvig", "Lolland", "Lyngby-Taarbæk", "Læsø", "Mariagerfjord", "Middelfart",
    "Morsø", "Norddjurs", "Nordfyns", "Nyborg", "Næstved", "Odder", "Odense", "Odsherred", "Randers", "Rebild",
    "Ringkøbing-Skjern", "Ringsted", "Roskilde", "Rudersdal", "Rødovre", "Samsø", "Silkeborg", "Skanderborg",
    "Skive", "Slagelse", "Solrød", "Sorø", "Stevns", "Struer", "Svendborg", "Syddjurs", "Sønderborg", "Thisted",
    "Tønder", "Tårnby", "Vallensbæk", "Varde", "Vejen", "Vejle", "Vesthimmerlands", "Viborg", "Vordingborg",
    "Ærø", "Aabenraa", "Aalborg", "Aarhus"
]

# Create a DataFrame with the desired order
print("Creating ordered DataFrame...")
ordered_df = pd.DataFrame({'LAU_NAME': desired_order})

# Merge with the area data, filling missing values with 0
print("Merging aggregated areas with ordered municipalities...")
result_df = ordered_df.merge(area_per_municipality, on='LAU_NAME', how='left').fillna({'area': 0})

# Ensure the order is maintained as per desired_order
result_df['LAU_NAME'] = pd.Categorical(result_df['LAU_NAME'], categories=desired_order, ordered=True)
result_df = result_df.sort_values('LAU_NAME')

# Export to CSV
print(f"Exporting results to CSV at {csv_output_path}...")
result_df.to_csv(csv_output_path, index=False, encoding='utf-8-sig')

print("Processing complete.")


 # %%

# SECTION 9.1

# Define file paths
municipality_fp = r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\LAU_RG_01M_2021_3035.shp\Administrative_DK.shp'
crs_fp = r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\Tekstur2014\Tekstur2014.shp'
agri_fp = r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\Agricultural_Areas_Merged.shp'
csv_output_path = r'C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\CRS_area.csv'

# Read only necessary columns to save memory
print("Reading Municipality data...")
Municipality_gdf = gpd.read_file(municipality_fp, usecols=['LAU_NAME', 'geometry'])

print("Reading CRS data...")
CRS_gdf = gpd.read_file(crs_fp, usecols=['geometry'])  # Add other necessary columns if needed

print("Reading Agricultural Areas data...")
Agricultural_areas_gdf = gpd.read_file(agri_fp, usecols=['geometry'])  # Add other necessary columns if needed

# Define target CRS
target_crs = 'EPSG:3035'

# Reproject all GeoDataFrames to target CRS if not already
print("Reprojecting GeoDataFrames to target CRS...")
Municipality_gdf = Municipality_gdf.to_crs(target_crs)
CRS_gdf = CRS_gdf.to_crs(target_crs)
Agricultural_areas_gdf = Agricultural_areas_gdf.to_crs(target_crs)

# Simplify geometries to speed up processing (adjust tolerance as needed)
tolerance = 100  # in CRS units (meters for EPSG:3035)
print(f"Simplifying geometries with tolerance={tolerance} meters...")
Municipality_gdf['geometry'] = Municipality_gdf['geometry'].simplify(tolerance, preserve_topology=True)
CRS_gdf['geometry'] = CRS_gdf['geometry'].simplify(tolerance, preserve_topology=True)
Agricultural_areas_gdf['geometry'] = Agricultural_areas_gdf['geometry'].simplify(tolerance, preserve_topology=True)

# Perform spatial join between CRS_gdf and Agricultural_areas_gdf for intersection
print("Performing spatial join between CRS and Agricultural Areas...")
intersected_gdf = gpd.sjoin(CRS_gdf, Agricultural_areas_gdf, how='inner', predicate='intersects')

# Drop the index_right column added by sjoin
intersected_gdf = intersected_gdf.drop(columns=['index_right'])

# Clip the intersected_gdf with the Municipality_gdf
print("Clipping intersected geometries with Municipality boundaries...")
clipped_gdf = gpd.clip(intersected_gdf, Municipality_gdf)

# Perform spatial join to add 'LAU_NAME' from Municipality_gdf to clipped_gdf
print("Performing spatial join to attach 'LAU_NAME'...")
clipped_gdf = gpd.sjoin(clipped_gdf, Municipality_gdf[['LAU_NAME', 'geometry']], how='left', predicate='within')

# Drop unnecessary columns resulting from the spatial join
clipped_gdf = clipped_gdf.drop(columns=['index_right'])

# Compute the area for each geometry in hectares (assuming CRS units are meters)
print("Calculating areas...")
clipped_gdf['area'] = clipped_gdf.geometry.area / 10**4  # Converts m² to hectares

# Group by 'LAU_NAME' and sum the areas
print("Aggregating area per municipality...")
area_per_municipality = clipped_gdf.groupby('LAU_NAME')['area'].sum().reset_index()

# Define the desired order of municipalities
desired_order = [
    "Albertslund", "Allerød", "Assens", "Ballerup", "Billund", "Bornholm", "Brøndby", "Brønderslev",
    "Christiansø", "Dragør", "Egedal", "Esbjerg", "Fanø", "Favrskov", "Faxe", "Fredensborg", "Fredericia",
    "Frederiksberg", "Frederikshavn", "Frederikssund", "Furesø", "Faaborg-Midtfyn", "Gentofte", "Gladsaxe",
    "Glostrup", "Greve", "Gribskov", "Guldborgsund", "Haderslev", "Halsnæs", "Hedensted", "Helsingør",
    "Herlev", "Herning", "Hillerød", "Hjørring", "Holbæk", "Holstebro", "Horsens", "Hvidovre", "Høje-Taastrup",
    "Hørsholm", "Ikast-Brande", "Ishøj", "Jammerbugt", "Kalundborg", "Kerteminde", "Kolding", "København",
    "Køge", "Langeland", "Lejre", "Lemvig", "Lolland", "Lyngby-Taarbæk", "Læsø", "Mariagerfjord", "Middelfart",
    "Morsø", "Norddjurs", "Nordfyns", "Nyborg", "Næstved", "Odder", "Odense", "Odsherred", "Randers", "Rebild",
    "Ringkøbing-Skjern", "Ringsted", "Roskilde", "Rudersdal", "Rødovre", "Samsø", "Silkeborg", "Skanderborg",
    "Skive", "Slagelse", "Solrød", "Sorø", "Stevns", "Struer", "Svendborg", "Syddjurs", "Sønderborg", "Thisted",
    "Tønder", "Tårnby", "Vallensbæk", "Varde", "Vejen", "Vejle", "Vesthimmerlands", "Viborg", "Vordingborg",
    "Ærø", "Aabenraa", "Aalborg", "Aarhus"
]

# Create a DataFrame with the desired order
print("Creating ordered DataFrame...")
ordered_df = pd.DataFrame({'LAU_NAME': desired_order})

# Merge with the area data, filling missing values with 0
print("Merging aggregated areas with ordered municipalities...")
result_df = ordered_df.merge(area_per_municipality, on='LAU_NAME', how='left').fillna({'area': 0})

# Optionally, ensure the order is maintained as per desired_order
result_df['LAU_NAME'] = pd.Categorical(result_df['LAU_NAME'], categories=desired_order, ordered=True)
result_df = result_df.sort_values('LAU_NAME')

# Export to CSV
print(f"Exporting results to CSV at {csv_output_path}...")
result_df.to_csv(csv_output_path, index=False, encoding='utf-8-sig')

print("Processing complete.")
# %%
#SECTION 10

# -----------------------------
# 1. Define the filter_merge_save function
# -----------------------------
def filter_merge_save(
    gdf,
    attribute,
    values_list,
    filtered_path,
    merged_path,
    tolerance=0.01
):
    """
    Filters the GeoDataFrame by a list of attribute values, saves the filtered 
    results, merges the filtered results, and saves to a file.

    Parameters:
    gdf (GeoDataFrame): The input GeoDataFrame.
    attribute (str): The attribute to filter by.
    values_list (list): The list of attribute values to filter by.
    filtered_path (str): The path to save the filtered GeoDataFrame.
    merged_path (str): The path to save the merged GeoDataFrame.
    tolerance (float): The tolerance for simplifying the merged geometry.
    """
    # Step 1: Filter the GeoDataFrame based on the attribute values
    filtered_gdf = gdf[gdf[attribute].isin(values_list)]
    
    # Step 2: Save the filtered GeoDataFrame
    filtered_gdf.to_file(filtered_path)
    
    # Step 3: Merge geometries in the filtered GeoDataFrame
    merged_geometry = filtered_gdf.unary_union
    
    # Step 4: Simplify (regularize) the merged polygon with the specified tolerance
    regularized_geometry = merged_geometry.simplify(tolerance, preserve_topology=True)
    merged_gdf = gpd.GeoDataFrame(geometry=[regularized_geometry], crs=filtered_gdf.crs)
    
    # Step 5: Save the merged and regularized GeoDataFrame
    merged_gdf.to_file(merged_path)
    
    # Return the merged GeoDataFrame
    return merged_gdf

# -----------------------------
# 2. Read input data
# -----------------------------
wdpa_path = r"C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\WDPA_WDOECM_Nov2024_Public_DNK_shp\merged_WDPA.shp"
CLC_path = r"C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\CLC_notprotected.shp"

WDPA_gdf = gpd.read_file(wdpa_path)
CLC_gdf = gpd.read_file(CLC_path)

# -----------------------------
# 3. Reproject to a common CRS
# -----------------------------
expected_crs = 3035
WDPA_gdf = WDPA_gdf.to_crs(epsg=expected_crs)
CLC_gdf = CLC_gdf.to_crs(epsg=expected_crs)

# -----------------------------
# 4. Filter and merge CLC based on Code_18 values
# -----------------------------
forest_values = ['311', '312', '313', '244']

# Define output paths for filtered and merged CLC shapefiles
filtered_forest_path = r"C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\Biodiversity Council\CLC_forest_filtered.shp"
merged_forest_path = r"C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\Biodiversity Council\CLC_forest_merged.shp"

# Use the function to create merged forest geometry
forest_merged_gdf = filter_merge_save(
    CLC_gdf,
    attribute="Code_18",
    values_list=forest_values,
    filtered_path=filtered_forest_path,
    merged_path=merged_forest_path,
    tolerance=0.01
)

# -----------------------------
# 5. Compute the intersection area
# -----------------------------
# Perform overlay (intersection) between merged forest geometry and WDPA
intersection_gdf = gpd.overlay(forest_merged_gdf, WDPA_gdf, how='intersection')

# Calculate total intersection area
intersection_area = intersection_gdf.geometry.area.sum()

print(f"Total intersection area in square meters (EPSG:3035): {intersection_area}")
# %%
