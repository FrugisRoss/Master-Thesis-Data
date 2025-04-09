#%%
### 0.1 Use development scripts or the package installed from pip
# use_development = True
# if use_development:
#     import sys
#     import os
#     # Adjust the sys.path to include the project root directory
#     project_root = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..'))
#     if project_root not in sys.path:
#         sys.path.insert(0, project_root)
#     from src.pybalmorel import MainResults
#     from src.pybalmorel.utils import symbol_to_df
# else:
from pybalmorel import MainResults
from pybalmorel.utils import symbol_to_df
import gams


ws = gams.GamsWorkspace()

res = MainResults(files='MainResults.gdx', paths=  r"C:\Users\sigur\OneDrive - Politecnico di Milano\polimi\magistrale\DTU\Run_on_HPC\Balmorel\Biodiversity_Case_RLC_FOSSIL+50NG\model", scenario_names=['+50 NG'])

df = res.get_result('PRO_YCRAGF')
print(df)



#%%ng
fig, ax = res.plot_profile(scenario='+50 NG', region='Denmark', year=2050,commodity='electricity')
                 
# %%
