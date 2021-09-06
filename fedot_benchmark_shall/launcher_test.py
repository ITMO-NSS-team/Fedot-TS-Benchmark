import pandas as pd
import numpy as np
from launcher import Launcher


short_df = pd.read_csv('../time_series_case_1/data/ts_short.csv')

for label in short_df['series_id'].unique():
    df = short_df[short_df['series_id'] == label]

    launcher = Launcher(framework_name='fedot', data=df,  horizon_size=100, horizon_step=10,
                        timeout=2)
    launcher.run_forecasting(structure_outfolder='output_structure', predict_outfolder='output_predictions')
