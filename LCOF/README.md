# LCOF.py – Levelized Cost of Fuel Post-Processing Script

## Overview

`LCOF.py` is a post-processing script designed to analyze results from the Optiflow energy system model. It calculates the **Levelized Cost of Fuel (LCOF)** for different fuel groups and transport sectors, using scenario data exported from Optiflow in GDX format. The script supports both renewable and fossil fuel flows and generates summary tables and visualizations for reporting.

**Author:** Rossella Frugis

---

## What the Script Does

- **Imports Optiflow and Balmorel model results** from `.gdx` files.
- **Calculates LCOF** for each fuel group and process pathway, considering investment, O&M, feedstock, and fuel production data.
- **Handles fossil fuel flows** (Diesel, MDO, Kerosene) with fixed input costs if present.
- **Generates and saves plots** of LCOF by fuel group and by transport sector.
- **Exports results** as CSV files and optionally as PDF images.

---

## Script Structure

### 1. **Imports and Settings**
   - Loads required Python libraries (pandas, numpy, matplotlib, seaborn, plotly, gams.transfer, etc.).
   - Sets display options for pandas DataFrames.

### 2. **User Configuration Section**
   - **Scenario definition:** List of tuples with scenario names and paths to Optiflow result folders.
   - **Output settings:** Toggle for saving plots as PDF and output file paths.

### 3. **Data Import Functions**
   - `Import_allendofmodel()`: Loads end-of-model results from GDX.
   - `Import_OptiflowMR()`: Loads main Optiflow results from GDX.
   - `Import_BalmorelMR()`: Loads Balmorel model results from GDX.

### 4. **Helper Functions**
   - `Avg_yearly_price()`: Calculates average yearly price for electricity, hydrogen, or heat.
   - `Avg_yearly_biomass_price()`: Calculates average yearly biomass price, including transport costs.

### 5. **Main Calculation**
   - `LCOF_calculation()`: 
     - Calculates LCOF for each fuel group.
     - Handles investment, O&M, feedstock, and fuel production data.
     - Exports results as CSV.
   - **Fossil fuel overwrite:** If fossil flows are present, their LCOF is set to the fixed input cost.

### 6. **Plotting Functions**
   - `plot_lcof_bar()`: Plots LCOF by fuel group.
   - `plot_LCOF_bysector()`: Plots LCOF by transport sector.

### 7. **Execution Block**
   - Runs the LCOF calculation for the selected scenario.
   - Generates and saves plots.

---

## How to Use

1. **Install dependencies**  
   Make sure you have the required Python packages:
   ```
   pip install pandas numpy matplotlib seaborn plotly kaleido
   ```
   > Note: `gams.transfer` requires a working GAMS installation.

2. **Configure the script**  
   - Edit the `scenario` list to point to your Optiflow result folders.
   - Set `save_pdf = True` if you want to save plots as PDF.

3. **Run the script**  
   ```
   python LCOF.py
   ```

4. **Outputs**  
   - CSV files with LCOF tables for each fuel group.
   - Plots of LCOF by fuel group and by sector (PDF if enabled).

---

## Notes

- The script expects Optiflow and Balmorel results in GDX format.
- You may need to adapt mappings and scenario paths for your specific model setup.
- For questions, refer to the comments in the code or contact the author.

---