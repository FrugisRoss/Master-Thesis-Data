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

# Load the DK land Area with administrative boundaries vector layer and merge the administrative boundaries
DKLand_adm_gdf = gpd.read_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\GIS-data\QGIS data\OCHA_Administrative_Boundaries\Danish_Land.shp')
merged_geometry = DKLand_adm_gdf.unary_union
DKLand_gdf=gpd.GeoDataFrame(geometry=[merged_geometry], crs=DKLand_adm_gdf.crs)
DKLand_gdf.to_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\GIS-data\QGIS data\OCHA_Administrative_Boundaries\Danish_Land_merged.shp')

# Load the WDPA layers and merge them 
WDPA0_gdf = gpd.read_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\GIS-data\QGIS data\WDPA_WDOECM_Nov2024_Public_DNK_shp\WDPA_WDOECM_Nov2024_Public_DNK_shp_0\WDPA_WDOECM_Nov2024_Public_DNK_shp-polygons.shp')
WDPA1_gdf = gpd.read_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\GIS-data\QGIS data\WDPA_WDOECM_Nov2024_Public_DNK_shp\WDPA_WDOECM_Nov2024_Public_DNK_shp_1\WDPA_WDOECM_Nov2024_Public_DNK_shp-polygons.shp')
merged_gdf = pd.concat([WDPA0_gdf, WDPA1_gdf], ignore_index=True)
merged_geometry = merged_gdf.unary_union
WDPA_merged_gdf=gpd.GeoDataFrame(geometry=[merged_geometry], crs=DKLand_adm_gdf.crs)
WDPA_merged_gdf.to_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\GIS-data\QGIS data\WDPA_WDOECM_Nov2024_Public_DNK_shp\merged_WDPA.shp')

#Subtracting Protected areas from Danish land
if DKLand_gdf.crs != WDPA_merged_gdf.crs:
    WDPA_merged_gdf = WDPA_merged_gdf.to_crs(DKLand_gdf.crs)
DKLand_subtracted_gdf= DKLand_gdf['geometry'].difference(WDPA_merged_gdf)
DKLand_notprotected_gdf = gpd.GeoDataFrame(DKLand_gdf.drop(columns='geometry'), geometry=DKLand_subtracted_gdf, crs=DKLand_gdf.crs)
DKLand_notprotected_gdf.to_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\GIS-data\QGIS data\OCHA_Administrative_Boundaries\Danish_Land_notprotected.shp')

# Load the CLC vector layer into a GeoDataFrame
CLC_gdf = gpd.read_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\GIS-data\QGIS data\CLC\Results\U2018_CLC2018_V2020_20u1.shp')

#Keep only the areas that are not protected
if DKLand_notprotected_gdf.crs != CLC_gdf.crs:
    CLC_gdf = CLC_gdf.to_crs(DKLand_notprotected_gdf.crs)
CLC_gdf=gpd.overlay(CLC_gdf,DKLand_notprotected_gdf, how='intersection')
CLC_gdf.to_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\GIS-data\QGIS data\CLC data extracted\CLC_notprotected.shp')

#Importing the file with the regions for the level of aggregation
regions_gdf = gpd.read_file( r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\GIS-data\QGIS data\OCHA_Administrative_Boundaries\regions.shp')

#%%
# Ensure both are in the same CRS and fix any invalid geometries
target_crs = "EPSG:32633"
CLC_gdf = CLC_gdf.to_crs(target_crs)
regions_gdf = regions_gdf.to_crs(target_crs)

CLC_gdf['geometry'] = CLC_gdf['geometry'].apply(lambda geom: geom if geom.is_valid else geom.buffer(0))
regions_gdf['geometry'] = regions_gdf['geometry'].apply(lambda geom: geom if geom.is_valid else geom.buffer(0))
CLC_gdf.to_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\GIS-data\QGIS data\CLC data extracted\CLC_notprotected_buffer.shp')

#%%
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
    merged_geometry = filtered_gdf.unary_union
    
    # Step 4: Regularize the merged polygon with the specified tolerance
    regularized_geometry = merged_geometry.simplify(tolerance, preserve_topology=True)
    merged_gdf = gpd.GeoDataFrame(geometry=[regularized_geometry], crs=filtered_gdf.crs)
    
    # Step 5: Save the merged and regularized GeoDataFrame to the specified path
    merged_gdf.to_file(merged_path)
    
    # Return the merged GeoDataFrame directly
    return merged_gdf


# Definition of paths to save each CLC aggregation cathegory output
urban_filtered_path = r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\GIS-data\QGIS data\CLC data extracted\Urban_Areas.shp'
urban_merged_path = r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\GIS-data\QGIS data\CLC data extracted\Urban_Areas_Merged.shp'

agricultural_filtered_path = r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\GIS-data\QGIS data\CLC data extracted\Agricultural_Areas.shp'
agricultural_merged_path = r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\GIS-data\QGIS data\CLC data extracted\Agricultural_Areas_Merged.shp'

forest_filtered_path = r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\GIS-data\QGIS data\CLC data extracted\Forest_Areas.shp'
forest_merged_path = r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\GIS-data\QGIS data\CLC data extracted\Forest_Areas_Merged.shp'

on_vegetation_filtered_path = r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\GIS-data\QGIS data\CLC data extracted\ON_vegetation_Areas.shp'
on_vegetation_merged_path = r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\GIS-data\QGIS data\CLC data extracted\ON_vegetation_Areas_Merged.shp'

on_no_vegetation_filtered_path = r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\GIS-data\QGIS data\CLC data extracted\ON_no_vegetation_Areas.shp'
on_no_vegetation_merged_path = r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\GIS-data\QGIS data\CLC data extracted\ON_no_vegetation_Areas_Merged.shp'

on_water_filtered_path = r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\GIS-data\QGIS data\CLC data extracted\ON_water_Areas.shp'
on_water_merged_path = r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\GIS-data\QGIS data\CLC data extracted\ON_water_Areas_Merged.shp'

water_bodies_filtered_path = r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\GIS-data\QGIS data\CLC data extracted\Water_Bodies_Areas.shp'
water_bodies_merged_path = r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\GIS-data\QGIS data\CLC data extracted\Water_Bodies_Areas_Merged.shp'

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
area_by_region = regions_gdf[['name_en', 'geometry']].copy()  # Start with a copy of the regions GeoDataFrame
area_by_region.set_index('name_en', inplace=True)  # Use region names as index for easy access

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
    area_by_region[f"{cover_type}_area_sqm"] = 0
    
    # Run intersection and area calculation for each region
    for _, region in regions_gdf.iterrows():
        region_name = region['name_en']
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
            
            # Calculate area in square meters for this land cover in this region
            total_area_sqm = intersected_gdf.geometry.area.sum()
            area_by_region.loc[region_name, f"{cover_type}_area_sqm"] = total_area_sqm

# Reset the index to have 'name_en' as a column
area_by_region.reset_index(inplace=True)


# Ensure that numeric fields are stored with appropriate precision
area_by_region = area_by_region.astype({
    'urban_area_sqm': 'float64',
    'agricultural_area_sqm': 'float64',
    'forest_area_sqm': 'float64',
    'vegetation_area_sqm': 'float64',
    'no_vegetation_area_sqm': 'float64',
    'water_area_sqm': 'float64',
    'water_bodies_area_sqm': 'float64'
})

# Save to shapefile with appropriate precision
output_path = r"C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\GIS-data\QGIS data\CLC data extracted\regions_with_land_cover_areas.shp"
area_by_region.to_file(output_path)

# Save to shapefile
output_path = r"C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\GIS-data\QGIS data\CLC data extracted\regions_with_land_cover_areas.shp"
area_by_region.to_file(output_path)

# Create a new DataFrame from area_by_region that contains Region Name and Land Cover Area columns
# Assuming `area_by_region` is the DataFrame holding the area information per region

# Extract the columns with area data (one per land cover type) and the region names
land_cover_columns = [
    'urban_area_sqm', 'agricultural_area_sqm', 'forest_area_sqm', 
    'vegetation_area_sqm', 'no_vegetation_area_sqm', 'water_area_sqm', 
    'water_bodies_area_sqm'
]

# Create a new DataFrame with 'region_name' and corresponding area columns
area_table = area_by_region[land_cover_columns].copy()

# Add the region name as a new column
area_table['region_name'] = area_by_region.name_en

# Reorder the columns so 'region_name' is first
area_table = area_table[['region_name'] + land_cover_columns]

# Optionally, you can reset the index if needed (to make 'region_name' a regular column)
area_table.reset_index(drop=True, inplace=True)

#%%
# Export the table to a CSV file
output_table_path = r"C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\land_cover_area_by_region.csv"
area_table.to_csv(output_table_path, index=False)

#%%

#Importing Potected areas from Biodiversity Council (30% of the national area )

Biodiversity_30_gdf = gpd.read_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\GIS-data\QGIS data\Biodiversity Council\scenarie_030_vector.shp')

Bio30_path =r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\GIS-data\QGIS data\Biodiversity Council\Biodiversity30.shp'
filtered_gdf = Biodiversity_30_gdf[Biodiversity_30_gdf['DN'] == 1]

filtered_gdf.to_file(Bio30_path, driver='ESRI Shapefile')

#%%
# Importing the yields from FAO [kg DM/ha]
        
wheat_raster_path=(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\GIS-data\QGIS data\FAO\whea200b_yld.tif') #potential yields
barley_raster_path=(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\GIS-data\QGIS data\FAO\barl200b_yld.tif')
rye_raster_path=(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\GIS-data\QGIS data\FAO\ryes200a_yld.tif')
oat_raster_path=(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\GIS-data\QGIS data\FAO\oats200b_yld.tif')

# Define the shapefile path
shapefile_path = r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\GIS-data\QGIS data\OCHA_Administrative_Boundaries\regions.shp'
output_path = r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\GIS-data\QGIS data\OCHA_Administrative_Boundaries\regions_yields.shp'  # Path to save the output shapefile

# Load the polygon shapefile using geopandas
regions_gdf = gpd.read_file(shapefile_path)

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
    stats = zonal_stats(regions_gdf, raster_path, stats="mean", geojson_out=True)
    
    # Extract the mean values from the stats and add to the GeoDataFrame
    mean_values = [feature["properties"]["mean"] for feature in stats]
    regions_gdf[field_name] = mean_values

# Save the output with added zonal statistics fields
regions_gdf.to_file(output_path, driver="ESRI Shapefile")
display(regions_gdf)

# Create a new DataFrame with region names and crop potentials
table_df = regions_gdf[["name_en", "wheat_pot", "barley_pot", "rye_pot", "oat_pot"]]

# Display the table
print(table_df)

# Optionally, save to a CSV file
table_df.to_csv(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\region_yields_table.csv', index=False)

#%%