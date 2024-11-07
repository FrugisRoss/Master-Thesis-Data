# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 10:25:22 2024

@author: Rossella Frugis
"""

import geopandas as gpd
import pandas as pd
import rioxarray as rxr
from shapely.geometry import shape
import rasterio.features

#%%
#Obtains different files for different Land Cover (aggregated) Class

# Load the DK land Area with administrative boundaries vector layer and merge the administrative boundaries
DKLand_adm_gdf = gpd.read_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\OCHA_Administrative_Boundaries\Danish_Land.shp')
merged_geometry = DKLand_adm_gdf.unary_union
DKLand_gdf=gpd.GeoDataFrame(geometry=[merged_geometry], crs=DKLand_adm_gdf.crs)
DKLand_gdf.to_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\OCHA_Administrative_Boundaries\Danish_Land_merged.shp')

# Load the WDPA layers and merge them 
WDPA0_gdf = gpd.read_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\WDPA_WDOECM_Nov2024_Public_DNK_shp\WDPA_WDOECM_Nov2024_Public_DNK_shp_0\WDPA_WDOECM_Nov2024_Public_DNK_shp-polygons.shp')
WDPA1_gdf = gpd.read_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\WDPA_WDOECM_Nov2024_Public_DNK_shp\WDPA_WDOECM_Nov2024_Public_DNK_shp_1\WDPA_WDOECM_Nov2024_Public_DNK_shp-polygons.shp')
merged_gdf = pd.concat([WDPA0_gdf, WDPA1_gdf], ignore_index=True)
merged_geometry = merged_gdf.unary_union
WDPA_merged_gdf=gpd.GeoDataFrame(geometry=[merged_geometry], crs=DKLand_adm_gdf.crs)
WDPA_merged_gdf.to_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\WDPA_WDOECM_Nov2024_Public_DNK_shp\merged_WDPA.shp')

#Subtracting Protected areas from Danish land
if DKLand_gdf.crs != WDPA_merged_gdf.crs:
    WDPA_merged_gdf = WDPA_merged_gdf.to_crs(DKLand_gdf.crs)
DKLand_subtracted_gdf= DKLand_gdf['geometry'].difference(WDPA_merged_gdf)
DKLand_notprotected_gdf = gpd.GeoDataFrame(DKLand_gdf.drop(columns='geometry'), geometry=DKLand_subtracted_gdf, crs=DKLand_gdf.crs)
DKLand_notprotected_gdf.to_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\OCHA_Administrative_Boundaries\Danish_Land_notprotected.shp')

# Load the CLC vector layer into a GeoDataFrame
CLC_gdf = gpd.read_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC\Results\U2018_CLC2018_V2020_20u1.shp')

#Keep only the areas that are not protected
if DKLand_notprotected_gdf.crs != CLC_gdf.crs:
    CLC_gdf = CLC_gdf.to_crs(DKLand_notprotected_gdf.crs)
CLC_gdf=gpd.overlay(CLC_gdf,DKLand_notprotected_gdf, how='intersection')
CLC_gdf.to_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\CLC_notprotected.shp')

#Definition of a function for filtering and merging the different aggregated CLC classes

def filter_merge_save(gdf, attribute, value_prefix, filtered_path, merged_path):
    """
    Filters a GeoDataFrame based on a given attribute and value prefix, merges the geometries,
    saves both filtered and merged layers as shapefiles, and returns only the merged GeoDataFrame.

    Parameters:
        gdf (GeoDataFrame): The input GeoDataFrame to filter.
        attribute (str): The name of the attribute to filter on.
        value_prefix (str): The prefix of the attribute value to filter by.
        filtered_path (str): The file path to save the filtered shapefile.
        merged_path (str): The file path to save the merged shapefile.

    Returns:
        GeoDataFrame: The merged GeoDataFrame containing the combined geometry.
    """
    # Step 1: Filter the GeoDataFrame based on the attribute's value prefix
    filtered_gdf = gdf[gdf[attribute].astype(str).str.startswith(value_prefix)]
    
    # Step 2: Save the filtered GeoDataFrame to the specified path
    filtered_gdf.to_file(filtered_path)
    
    # Step 3: Merge geometries in the filtered GeoDataFrame
    merged_geometry = filtered_gdf.unary_union
    merged_gdf = gpd.GeoDataFrame(geometry=[merged_geometry], crs=filtered_gdf.crs)
    
    # Step 4: Save the merged GeoDataFrame to the specified path
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

#%%

# Potected areas from Biodiversity Council

Biodiversity_30_gdf = gpd.read_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\Biodiversity Council\scenarie_030_vector.shp')

Bio30_path =r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\Biodiversity Council\Biodiversity30.shp'
Bio30_merged_path = r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\Biodiversity Council\Biodiversity30_merged.shp'

Bio30_merged_gdf = filter_merge_save(Biodiversity30_gdf, 'DN', '1', Bio30_path, Bio30_merged_path)
