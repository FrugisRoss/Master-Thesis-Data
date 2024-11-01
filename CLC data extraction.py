# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 10:25:22 2024

@author: Rossella Frugis
"""

import geopandas as gpd

# Load your layer into a GeoDataFrame
gdf = gpd.read_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC\Results\U2018_CLC2018_V2020_20u1.shp')

# Urban Areas Filtering and Merging
filtered_gdf = gdf[gdf['Code_18'].astype(str).str.startswith('1')]
filtered_gdf.to_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\Urban_Areas.shp')
merged_geometry = filtered_gdf.unary_union
merged_gdf = gpd.GeoDataFrame(geometry=[merged_geometry], crs=filtered_gdf.crs)
merged_gdf.to_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\Urban_Areas_Merged.shp')

# Agricultural Areas Filtering and Merging
filtered_gdf = gdf[gdf['Code_18'].astype(str).str.startswith('2')]
filtered_gdf.to_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\Agricultural_Areas.shp')
merged_geometry = filtered_gdf.unary_union
merged_gdf = gpd.GeoDataFrame(geometry=[merged_geometry], crs=filtered_gdf.crs)
merged_gdf.to_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\Agricultural_Areas_Merged.shp')

# Forest Areas Filtering and Merging
filtered_gdf = gdf[gdf['Code_18'].astype(str).str.startswith('31')]
filtered_gdf.to_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\Forest_Areas.shp')
merged_geometry = filtered_gdf.unary_union
merged_gdf = gpd.GeoDataFrame(geometry=[merged_geometry], crs=filtered_gdf.crs)
merged_gdf.to_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\Forest_Areas_Merged.shp')

# Other Nature Areas with vegetation Filtering and Merging
filtered_gdf = gdf[gdf['Code_18'].astype(str).str.startswith('32')]
filtered_gdf.to_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\ON_vegetation_Areas.shp')
merged_geometry = filtered_gdf.unary_union
merged_gdf = gpd.GeoDataFrame(geometry=[merged_geometry], crs=filtered_gdf.crs)
merged_gdf.to_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\ON_vegetation_Areas_Merged.shp')

# Other Nature Areas without vegetation Filtering and Merging
filtered_gdf = gdf[gdf['Code_18'].astype(str).str.startswith('33')]
filtered_gdf.to_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\ON_no_vegetation_Areas.shp')
merged_geometry = filtered_gdf.unary_union
merged_gdf = gpd.GeoDataFrame(geometry=[merged_geometry], crs=filtered_gdf.crs)
merged_gdf.to_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\ON_no_vegetation_Areas_Merged.shp')

# Other Nature Areas with water Filtering and Merging
filtered_gdf = gdf[gdf['Code_18'].astype(str).str.startswith('4')]
filtered_gdf.to_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\ON_water_Areas.shp')
merged_geometry = filtered_gdf.unary_union
merged_gdf = gpd.GeoDataFrame(geometry=[merged_geometry], crs=filtered_gdf.crs)
merged_gdf.to_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\ON_water_Areas_Merged.shp')

#Water Bodies Areas Filtering and Merging
filtered_gdf = gdf[gdf['Code_18'].astype(str).str.startswith('5')]
filtered_gdf.to_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\Water_Bodies_Areas.shp')
merged_geometry = filtered_gdf.unary_union
merged_gdf = gpd.GeoDataFrame(geometry=[merged_geometry], crs=filtered_gdf.crs)
merged_gdf.to_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\Water_Bodies_Areas_Merged.shp')