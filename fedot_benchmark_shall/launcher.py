import numpy as np
import os

import pandas as pd
from fedot.api.main import Fedot
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum


class Launcher:
    def __init__(self, framework_name: str, data: pd.DataFrame, horizon_size: int, horizon_step: int = 10,
                 timeout: [float, int] = 2):
        self.framework_name = framework_name
        self.horizons = np.arange(horizon_step, horizon_size, horizon_step)
        self.timeout = timeout

        self.data = data
        self.features = None
        self.target = None

    def run_forecasting(self, predict_outfolder, structure_outfolder):
        if self.framework_name == 'fedot':
            for horizon in self.horizons:
                self._form_input_data(horizon)
                self._run_fedot(horizon, predict_outfolder, structure_outfolder)

        else:
            raise NotImplementedError

    def _run_fedot(self, forecast_len, predict_outfolder, structure_outfolder):
        model = Fedot(problem='ts_forecasting',
                      task_params=TsForecastingParams(forecast_length=forecast_len),
                      timeout=self.timeout,
                      composer_params={'with_tuning': True,
                                       'cv_folds': 2,
                                       'validation_blocks': 2})
        pipeline = model.fit(features=self.features, target=self.target)
        image_name = f'{self.target}_fedot_forcast{forecast_len}_time{self.timeout}.png'
        pipeline_name = f'{self.target}_fedot_forcast{forecast_len}_time{self.timeout}.json'
        pipeline.show(path=os.path.join(structure_outfolder, image_name))
        pipeline.save(path=os.path.join(structure_outfolder, pipeline_name))

        df_name = f'{self.target}_fedot_forcast{forecast_len}_time{self.timeout}.csv'
        forecast = model.predict(features=self.features)
        pd.DataFrame({'Prediction': forecast}).to_csv(os.path.join(predict_outfolder, df_name), index=False)

    def _form_input_data(self, len_forecast):
        train_part = self.data
        task = Task(TaskTypesEnum.ts_forecasting,
                    TsForecastingParams(forecast_length=len_forecast))

        train_input = InputData(idx=np.arange(0, len(train_part)),
                                features=train_part,
                                target=train_part,
                                task=task,
                                data_type=DataTypesEnum.ts)
        self.features = train_input

        predict_input = InputData(idx=np.arange(len(train_part)-len_forecast, len(train_part)),
                                  features=train_part,
                                  target=None,
                                  task=task,
                                  data_type=DataTypesEnum.ts)
        self.target = predict_input