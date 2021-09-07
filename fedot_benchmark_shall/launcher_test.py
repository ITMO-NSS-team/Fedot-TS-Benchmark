import os
from launcher import Launcher


for file in os.listdir('input_datasets'):
    data_path = os.path.join('input_datasets', file)

    launcher = Launcher(framework_name='fedot', features=data_path, target='value', horizon_size=100, horizon_step=10,
                        timeout=2)
    launcher.run_forecasting(structure_outfolder='output_structure', predict_outfolder='output_predictions')
