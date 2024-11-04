# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 10:25:22 2024

@author: Rossella Frugis
"""

import geopandas as gpd
import pandas as pd

# Load the DK land Area with administrative boundaries vector layer and merge the administrative boundaries
DKLand_adm_gdf = gpd.read_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\OCHA_Administrative_Boundaries\Danish_Land.shp')
merged_geometry = DKLand_adm_gdf.unary_union
DKLand_gdf=gpd.GeoDataFrame(geometry=[merged_geometry], crs=DKLand_adm_gdf.crs)
DKLand_gdf.to_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\OCHA_Administrative_Boundaries\Danish_Land_merged.shp')

# Load the WDPA layers and merge them 
WDPA0_gdf = gpd.read_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\WDPA_WDOECM_Nov2024_Public_DNK_shp\WDPA_WDOECM_Nov2024_Public_DNK_shp_0\WDPA_WDOECM_Nov2024_Public_DNK_shp-polygons.shp')
WDPA1_gdf = gpd.read_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\WDPA_WDOECM_Nov2024_Public_DNK_shp\WDPA_WDOECM_Nov2024_Public_DNK_shp_1\WDPA_WDOECM_Nov2024_Public_DNK_shp-polygons.shp')
WDPA2_gdf = gpd.read_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\WDPA_WDOECM_Nov2024_Public_DNK_shp\WDPA_WDOECM_Nov2024_Public_DNK_shp_2\WDPA_WDOECM_Nov2024_Public_DNK_shp-polygons.shp')
merged_gdf = pd.concat([WDPA0_gdf, WDPA1_gdf, WDPA2_gdf], ignore_index=True)
merged_geometry = merged_gdf.unary_union
WDPA_merged_gdf=gpd.GeoDataFrame(geometry=[merged_geometry], crs=DKLand_adm_gdf.crs)
WDPA_merged_gdf.to_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\WDPA_WDOECM_Nov2024_Public_DNK_shp\merged_WDPA.shp')

# Load the CLC vector layer into a GeoDataFrame
CLC_gdf = gpd.read_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC\Results\U2018_CLC2018_V2020_20u1.shp')

# Urban Areas Filtering and Merging
filtered_CLC_gdf = CLC_gdf[CLC_gdf['Code_18'].astype(str).str.startswith('1')]
filtered_CLC_gdf.to_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\Urban_Areas.shp')
merged_geometry = filtered_CLC_gdf.unary_union
merged_CLC_gdf = gpd.GeoDataFrame(geometry=[merged_geometry], crs=filtered_CLC_gdf.crs)
merged_CLC_gdf.to_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\Urban_Areas_Merged.shp')

# Agricultural Areas Filtering and Merging
filtered_CLC_gdf = CLC_gdf[CLC_gdf['Code_18'].astype(str).str.startswith('2')]
filtered_CLC_gdf.to_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\Agricultural_Areas.shp')
merged_geometry = filtered_CLC_gdf.unary_union
merged_CLC_gdf = gpd.GeoDataFrame(geometry=[merged_geometry], crs=filtered_CLC_gdf.crs)
merged_CLC_gdf.to_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\Agricultural_Areas_Merged.shp')

# Forest Areas Filtering and Merging
filtered_CLC_gdf = CLC_gdf[CLC_gdf['Code_18'].astype(str).str.startswith('31')]
filtered_CLC_gdf.to_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\Forest_Areas.shp')
merged_geometry = filtered_CLC_gdf.unary_union
merged_CLC_gdf = gpd.GeoDataFrame(geometry=[merged_geometry], crs=filtered_CLC_gdf.crs)
merged_CLC_gdf.to_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\Forest_Areas_Merged.shp')

# Other Nature Areas with vegetation Filtering and Merging
filtered_CLC_gdf = CLC_gdf[CLC_gdf['Code_18'].astype(str).str.startswith('32')]
filtered_CLC_gdf.to_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\ON_vegetation_Areas.shp')
merged_geometry = filtered_CLC_gdf.unary_union
merged_CLC_gdf = gpd.GeoDataFrame(geometry=[merged_geometry], crs=filtered_CLC_gdf.crs)
merged_CLC_gdf.to_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\ON_vegetation_Areas_Merged.shp')

# Other Nature Areas without vegetation Filtering and Merging
filtered_CLC_gdf = CLC_gdf[CLC_gdf['Code_18'].astype(str).str.startswith('33')]
filtered_CLC_gdf.to_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\ON_no_vegetation_Areas.shp')
merged_geometry = filtered_CLC_gdf.unary_union
merged_CLC_gdf = gpd.GeoDataFrame(geometry=[merged_geometry], crs=filtered_CLC_gdf.crs)
merged_CLC_gdf.to_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\ON_no_vegetation_Areas_Merged.shp')

# Other Nature Areas with water Filtering and Merging
filtered_CLC_gdf = CLC_gdf[CLC_gdf['Code_18'].astype(str).str.startswith('4')]
filtered_CLC_gdf.to_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\ON_water_Areas.shp')
merged_geometry = filtered_CLC_gdf.unary_union
merged_CLC_gdf = gpd.GeoDataFrame(geometry=[merged_geometry], crs=filtered_CLC_gdf.crs)
merged_CLC_gdf.to_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\ON_water_Areas_Merged.shp')

#Water Bodies Areas Filtering and Merging
filtered_CLC_gdf = CLC_gdf[CLC_gdf['Code_18'].astype(str).str.startswith('5')]
filtered_CLC_gdf.to_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\Water_Bodies_Areas.shp')
merged_geometry = filtered_CLC_gdf.unary_union
merged_CLC_gdf = gpd.GeoDataFrame(geometry=[merged_geometry], crs=filtered_CLC_gdf.crs)
merged_CLC_gdf.to_file(r'C:\Users\Utente\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Input Data\QGIS data\CLC data extracted\Water_Bodies_Areas_Merged.shp')