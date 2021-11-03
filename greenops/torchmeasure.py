"""
greenops
--------

Software to measure the footprints of deep learning models at training,
testing and evaluating to reduce energy consumption and carbon footprints.

Copyright rixel 2021
Distributed under the MIT License.
See accompanying file LICENSE.

File: submodule torchmeasure
"""


from time import time

import torch

from .context import Context, DEFAULT_CPU_WATTS
from .measure import DEFAULT_INSTANT_LOG, LogRow, LogSettings, Measure


DEFAULT_CURRENT_STAGE = 'torch_main'


class TorchMeasure(Measure):
    """
    """


    def __init__(self, torch_model : torch.nn.Module,
                 output_file_name : str = None, watch : dict = None,
                 device_data : any = None, csv_delimiter : str = '\t',
                 log_settings : LogSettings = None,
                 instant_log : bool = DEFAULT_INSTANT_LOG,
                 current_stage : str = DEFAULT_CURRENT_STAGE):
        """
        """

        super().__init__(output_file_name, watch, device_data, csv_delimiter,
                         log_settings, instant_log)

        self.change_model(torch_model)
        self.__current_stage = current_stage


    def change_model(self, torch_model : torch.nn.Module):
        """
        """

        self.__torch_model = torch_model
        self.__torch_model.register_forward_hook(self.__update_hook_forward)
        self.__torch_model.register_backward_hook(self.__update_hook_backward)
        self.__last_device = None


    @property
    def current_stage(self) -> str:
        """
        """

        return self.__current_stage


    @current_stage.setter
    def current_stage(self, new_stage : str):
        """
        """

        self.__current_stage = new_stage


    def __perform_update(self, device : torch.device, now : float,
                         prefix : str = '', suffix : str = ''):
        """
        """

        stage_name = prefix + self.__current_stage + suffix
        device_index = 0
        d_name = 'unknown_torch'
        d_consumption = DEFAULT_CPU_WATTS
        try:
            device_index = int(device.index)
        except (TypeError, ValueError):
            pass
        if device.type == 'cuda':
            d_name = Context.get_gpu()[device_index].short_name
            cd_onsumption = Context.get_gpu()[device_index].consumption
        elif device.type == 'cpu':
            d_name = Context.get_cpu()[device_index].short_name
            d_consumption = Context.get_cpu()[device_index].consumption
        if not super().stage_exists(stage_name):
            super().create_stage(stage_name)
        if super()._get_last_update(stage_name) > 0:
            time_delta = now - super()._get_last_update(stage_name)
            super()._add_data_row(LogRow(now, stage_name, d_name,
                            d_consumption, super()._get_epoch(stage_name),
                            time_delta, super()._get_watch_values()))
        super()._increase_epoch(stage_name, 1)
        super()._set_last_update(stage_name, now)


    def __update_hook_backward(self, model, input, output):
        """
        """

        now = time()
        if self.__last_device is not None:
            self.__perform_update(self.__last_device, now, suffix = '_backward')


    def __update_hook_forward(self, model, input, output):
        """
        """

        now = time()
        if isinstance(output, torch.Tensor):
            self.__last_device = output.device
            self.__perform_update(output.device, now, suffix = '_forward')
