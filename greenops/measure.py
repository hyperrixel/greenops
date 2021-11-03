"""
greenops
--------

Software to measure the footprints of deep learning models at training,
testing and evaluating to reduce energy consumption and carbon footprints.

Copyright rixel 2021
Distributed under the MIT License.
See accompanying file LICENSE.

File: submodule measure
"""


from time import asctime, gmtime, localtime, strftime, time

from .context import Context
from .datamodel import DeviceData
from .exceptions import GreenOpsException
from .functions import extensionify_file, join_any, ws_to_kwh


DEFAULT_INCLUDE_DEVICE = True
DEFAULT_INCLUDE_HEADER = True
DEFAULT_INCLUDE_TIME = True
DEFAULT_INCLUDE_STAGE = True
DEFAULT_INSTANT_LOG = True
DEFAULT_STAGE_NAME = 'main'
TIME_FORMAT_GLOBAL = 'utc'
TIME_FORMAT_LOCAL = 'local'
TIME_FORMAT_TIMESTAMP = 'timestamp'


class LogRow:
    """
    """

    def __init__(self, row_time : float, stage : str, device : str,
                 consumption : float, epoch : int, time_delta : float,
                 watch_values : list):
        """
        """

        self.__row_time = row_time
        self.__stage = stage
        self.__device = device
        self.__consumption = consumption
        self.__epoch = epoch
        self.__time_delta = time_delta
        self.__watch_values = watch_values

    @property
    def consumption(self) -> str:
        """
        """

        return self.__consumption


    @property
    def device(self) -> str:
        """
        """

        return self.__device


    @staticmethod
    def device_consumption_from_context(device_str : str) -> tuple:
        """
        """

        device = None
        consumption = None
        if device_str == 'cpu': # Support for torch.device
            device = 'cpu'
            consumption = Context.get_cpu()[0].consumption
        if device_str == 'gpu': # Support for lazy data scientists :)
            device = 'gpu:0'
            consumption = Context.get_gpu()[0].consumption
        else:
            device_parts = device_str.split(':')
            if len(device_parts) == 2:
                pos = None
                try:
                    pos = int(device_parts[1])
                except ValueError:
                    pass
                if pos is not None:
                    if device_parts[0] == 'cpu':
                        if pos < len(Context.get_cpu()):
                            device = device_str
                            consumption = Context.get_cpu()[pos].consumption
                    elif device_parts[0] == 'gpu':
                        if pos < len(Context.get_gpu()):
                            device = device_str
                            consumption = Context.get_gpu()[pos].consumption
            elif len(device_parts) == 1:
                for i, cpu in enumerate(Context.get_cpu()):
                    if cpu.long_name == device_str:
                        device = 'cpu:{}'.format(i)
                        consumption = cpu.consumption
                    elif cpu.short_name == device_str:
                        device = 'cpu:{}'.format(i)
                        consumption = cpu.consumption
                for i, gpu in enumerate(Context.get_gpu()):
                    if gpu.long_name == device_str:
                        device = 'cpu:{}'.format(i)
                        consumption = gpu.consumption
                    elif gpu.short_name == device_str:
                        device = 'gpu:{}'.format(i)
                        consumption = gpu.consumption
        if device is None or consumption is None:
            raise ValueError('greenops.measure.LogRow.' +
                  'device_consumption_from_context: Cannot identify device from'
                  + 'invalid data.')
        return device, consumption


    @staticmethod
    def device_from_parts(device_type : str, device_id : int) -> str:
        """
        """

        return '{}:{}'.format(device_type, device_id)


    @property
    def epoch(self) -> int:
        """
        """

        return self.__epoch


    @property
    def row_time(self) -> str:
        """
        """

        return self.__row_time


    @property
    def stage(self) -> str:
        """
        """

        return self.__stage


    @property
    def time_delta(self) -> float:
        """
        """

        return self.__time_delta


    @property
    def watch_values(self) -> list:
        """
        """

        return self.__watch_values.copy()


class LogSettings:

    def __init__(self, include_header : bool = DEFAULT_INCLUDE_HEADER,
                 include_time : bool = DEFAULT_INCLUDE_TIME,
                 time_format : bool = TIME_FORMAT_TIMESTAMP,
                 include_stage : bool = DEFAULT_INCLUDE_STAGE,
                 include_device : bool = DEFAULT_INCLUDE_DEVICE,
                 watch_keys : list = None):
        """
        """

        self.__include_header = include_header
        self.__include_time = include_time
        self.__time_func = LogSettings.compile_time_function_(time_format)
        self.__time_format = time_format
        self.__include_stage = include_stage
        self.__include_device = include_device
        if watch_keys is None:
            self.__watch_keys = []
        else:
            self.__watch_keys = watch_keys
        self.__line_function = self.compile()
        self.__header = self.create_header()


    def compile(self) -> callable:
        """
        """

        result = lambda r: r # It will be changed anyway
        if self.__include_time:
            if self.__include_stage:
                if self.__include_device:
                    if self.include_watch:
                        result = \
                        lambda r: [self.__time_func(r.row_time), r.stage,
                                   r.device, r.consumption, r.epoch,
                                   r.time_delta] + r.watch_values
                    else:
                        result = \
                        lambda r: [self.__time_func(r.row_time), r.stage,
                                   r.device, r.consumption, r.epoch,
                                   r.time_delta]
                else:
                    if self.include_watch:
                        result = \
                        lambda r: [self.__time_func(r.row_time), r.stage,
                                   r.consumption, r.epoch, r.time_delta
                                   ] + r.watch_values
                    else:
                        result = \
                        lambda r: [self.__time_func(r.row_time), r.stage,
                                   r.consumption, r.epoch, r.time_delta]
            else:
                if self.__include_device:
                    if self.include_watch:
                        result = \
                        lambda r: [self.__time_func(r.row_time), r.device,
                                   r.consumption, r.epoch, r.time_delta
                                   ] + r.watch_values
                    else:
                        result = \
                        lambda r: [self.__time_func(r.row_time), r.device,
                                   r.consumption, r.epoch, r.time_delta]
                else:
                    if self.include_watch:
                        result = \
                        lambda r: [self.__time_func(r.row_time), r.consumption,
                                   r.epoch, r.time_delta
                                   ] + r.watch_values
                    else:
                        result = \
                        lambda r: [self.__time_func(r.row_time), r.consumption,
                                   r.epoch, r.time_delta]
        else:
            if self.__include_stage:
                if self.__include_device:
                    if self.include_watch:
                        result = \
                        lambda r: [r.stage, r.device, r.consumption, r.epoch,
                                   r.time_delta] + r.watch_values
                    else:
                        result = \
                        lambda r: [r.stage, r.device, r.consumption, r.epoch,
                                   r.time_delta]
                else:
                    if self.include_watch:
                        result = \
                        lambda r: [r.stage, r.consumption, r.epoch, r.time_delta
                                   ] + r.watch_values
                    else:
                        result = \
                        lambda r: [r.stage, r.consumption, r.epoch,
                                   r.time_delta]
            else:
                if self.__include_device:
                    if self.include_watch:
                        result = \
                        lambda r: [r.device, r.consumption, r.epoch,
                                   r.time_delta] + r.watch_values
                    else:
                        result = \
                        lambda r: [r.device, r.consumption, r.epoch,
                                   r.time_delta]
                else:
                    if self.include_watch:
                        result = \
                        lambda r: [r.consumption, r.epoch, r.time_delta
                                   ] + r.watch_values
                    else:
                        result = \
                        lambda r: [r.consumption, r.epoch, r.time_delta]
        return result


    def create_header(self) -> str:
        """
        """

        result = [] # It will be changed anyway
        if self.__include_time:
            if self.__include_stage:
                if self.__include_device:
                    if self.include_watch:
                        result = ['time', 'stage', 'device', 'consumption',
                                  'epoch', 'time_delta'
                                  ] + self.__watch_keys
                    else:
                        result = ['time', 'stage', 'device', 'consumption',
                                  'epoch', 'time_delta']
                else:
                    if self.include_watch:
                        result = ['time', 'stage', 'consumption', 'epoch',
                                  'time_delta'] + self.__watch_keys
                    else:
                        result = ['time', 'stage','consumption', 'epoch',
                                  'time_delta']
            else:
                if self.__include_device:
                    if self.include_watch:
                        result = ['time', 'device','consumption', 'epoch',
                                  'time_delta'] + self.__watch_keys
                    else:
                        result = ['time', 'device', 'consumption', 'epoch',
                                  'time_delta']
                else:
                    if self.include_watch:
                        result = ['time', 'consumption', 'epoch', 'time_delta'
                                  ] + self.__watch_keys
                    else:
                        result = ['time', 'consumption', 'epoch', 'time_delta']
        else:
            if self.__include_stage:
                if self.__include_device:
                    if self.include_watch:
                        result = ['stage', 'device', 'consumption', 'epoch',
                                  'time_delta'] + self.__watch_keys
                    else:
                        result = ['stage', 'device', 'consumption', 'epoch',
                                  'time_delta']
                else:
                    if self.include_watch:
                        result = ['stage', 'consumption', 'epoch', 'time_delta'
                                  ] + self.__watch_keys
                    else:
                        result = ['stage', 'consumption', 'epoch', 'time_delta']
            else:
                if self.__include_device:
                    if self.include_watch:
                        result = ['device', 'consumption', 'epoch', 'time_delta'
                                  ] + self.__watch_keys
                    else:
                        result = ['device', 'consumption', 'epoch',
                                  'time_delta']
                else:
                    if self.include_watch:
                        result = ['consumption', 'epoch', 'time_delta'
                                   ] + self.__watch_keys
                    else:
                        result = ['consumption', 'epoch', 'time_delta']
        return result


    @property
    def header_as_list(self) -> list:
        """
        """

        return self.__header


    @property
    def include_device(self) -> bool:
        """
        """

        return self.__include_device


    @property
    def include_header(self) -> bool:
        """
        """

        return self.__include_header


    @property
    def include_stage(self) -> bool:
        """
        """

        return self.__include_stage


    @property
    def include_time(self) -> bool:
        """
        """

        return self.__include_time


    @property
    def include_watch(self) -> bool:
        """
        """

        return len(self.__watch_keys) == 0


    def line_as_list(self, row : LogRow) -> list:
        """
        """

        return self.__line_function(row)


    @property
    def time_format(self) -> str:
        """
        """

        return self.__time_format


    @property
    def watch_keys(self) -> str:
        """
        """

        return self.__watch_keys.copy()


    @staticmethod
    def compile_time_function_(time_format : str) -> callable:
        """
        """

        result = lambda t: int(round(t)) # Default and ultimate fallback
        if time_format == TIME_FORMAT_GLOBAL:
            result = lambda t: asctime(gmtime(t))
        elif time_format == TIME_FORMAT_LOCAL:
            result = lambda t: strftime('%x %X', localtime())
        elif time_format.startswith('global:'):
            time_str = time_format.replace('global:', '')
            result = lambda t: strftime(time_str, gmtime())
        elif time_format.startswith('local:'):
            time_str = time_format.replace('local:', '')
            result = lambda t: strftime(time_str, localtime())
        return result


class Measure:
    """
    """

    def __init__(self, output_file_name : str = None, watch : dict = None,
                 device_data : any = None, csv_delimiter : str = '\t',
                 log_settings : LogSettings = None,
                 instant_log : bool = DEFAULT_INSTANT_LOG):
        """
        """

        if output_file_name is None:
            self.__output_file_name = 'greenops_{}.csv'.format(int(time()))
        else:
            self.__output_file_name = extensionify_file(output_file_name, 'csv')
        if watch is None:
            self.__watch = {}
        else:
            self.__watch = watch
        self.__device_data = None
        if isinstance(device_data, DeviceData):
            self.__device_data = device_data
        elif isinstance(device_data, str):
            device, consumption = LogSettings.device_consumption_from_context(
                                                                    device_data)
            self.__device_data = DeviceData(device, device, consumption)
        Context.init()
        if self.__device_data is None: # Default and ultimate fallback
            self.__device_data = Context.get_cpu()[0]
        self.__csv_delimiter = csv_delimiter
        if log_settings is None:
            self.__log_settings = LogSettings(watch_keys=list(
                                                        self.__watch.keys()))
        else:
            self.__log_settings = log_settings
        self.__instant_log = instant_log
        self.reset()


    def create_stage(self, stage_name : str):
        """
        """

        self.__data[stage_name] = []
        self.__data_pointers[stage_name] = 0
        self.__last_updates[stage_name] = 0
        self.__last_starts[stage_name] = 0
        self.__epochs[stage_name] = 0


    @property
    def csv_delimiter(self) -> str:
        """
        """

        return self.__csv_delimiter


    @property
    def instant_log(self) -> bool:
        """
        """

        return self.__instant_log


    @property
    def log_settings(self) -> LogSettings:
        """
        """

        return self.__log_settings


    @property
    def output_file_name(self) -> str:
        """
        """

        return self.__output_file_name


    def reset(self, stage_name : str = None):
        """
        """

        if stage_name is None:
            self.__first_row = True
            self.__data = {}
            self.__data_pointers = {}
            self.__last_updates = {}
            self.__last_starts = {}
            self.__epochs = {}
        else:
            self.create_stage(stage_name)


    def save_new_rows(self, stage_name : str):
        """
        """

        with open(self.__output_file_name, 'a', encoding='utf8') as outstream:
            if self.__first_row:
                self.__first_row = False
                if self.__log_settings.include_header:
                    outstream.write('{}\n'.format(join_any(self.__csv_delimiter,
                                        self.__log_settings.header_as_list)))
            while self.__data_pointers[stage_name] <\
                                                len(self.__data[stage_name]):
                outstream.write('{}\n'.format(join_any(self.__csv_delimiter,
                        self.__log_settings.line_as_list(self.__data[stage_name]
                        [self.__data_pointers[stage_name]]))))
                self.__data_pointers[stage_name] += 1


    def save_stats(self, file_name : str = None):
        """
        """

        if file_name is None:
            the_file = self.__output_file_name
        else:
            the_file = extensionify_file(file_name, 'csv')
        with open(the_file, 'w', encoding='utf8') as outstream:
            if self.__log_settings.include_header:
                outstream.write('{}\n'.format(join_any(self.__csv_delimiter,
                                        self.__log_settings.header_as_list)))
            for stage in self.__data.keys():
                for row in self.__data[stage]:
                    outstream.write('{}\n'.format(join_any(self.__csv_delimiter,
                                        self.__log_settings.line_as_list(row))))


    def stage_exists(self, stage_name : str) -> bool:
        """
        """

        return all([stage_name in self.__data.keys(),
                    stage_name in self.__data_pointers.keys(),
                    stage_name in self.__last_updates.keys(),
                    stage_name in self.__last_starts.keys(),
                    stage_name in self.__epochs.keys()])


    def start(self, stage_name : str = DEFAULT_STAGE_NAME):
        """
        """

        if not self.stage_exists(stage_name):
            self.create_stage(stage_name)
        self.__last_starts[stage_name] = time()


    @property
    def stats_summary(self) -> str:
        """
        """

        total_times = 0.0
        total_consumption = 0.0
        total_cost = 0.0
        stage_times = {}
        stage_consumptions = {}
        for stage in self.__data.keys():
            stage_times[stage] = 0.0
            stage_consumptions[stage] = 0.0
            for row in self.__data[stage]:
                total_times += row.time_delta
                stage_times[stage] += row.time_delta
                consumption = row.time_delta * row.consumption
                stage_consumptions[stage] += consumption

        result = 'greenops.measure.Measure stats summary:\n\nStages:\n'
        for stage in stage_times.keys():
            kwh = ws_to_kwh(stage_consumptions[stage])
            total_consumption += kwh
            usd = Context.get_power_price().price * kwh
            total_cost += usd
            result += ' --- Stage "{}" lasted {:.4f} seconds and '.format(
                      stage, stage_times[stage]
                      ) + 'consumed {:.2f} kWh. Cost: {:.2f} USD\n'.format(kwh,
                      usd)
        result += '\n\nTotal time: {:.4f} seconds\nTotal consumption '.format(
                  total_times) + '{:.2f} kWh\nTotal cost {:.2f} '.format(
                  total_consumption, total_cost) + 'USD\n\n{}\n\n{}'.format(
                  Context.get_power_price(), Context.get_power_sources())
        return result


    def stop(self, stage_name : str = DEFAULT_STAGE_NAME):
        """
        """

        now = time()
        try:
            time_delta = now - self.__last_starts[stage_name]
        except KeyError:
            raise GreenOpsException('greenops.measure.Measure.stop(): Cannot ' +
                                    'a non-running epoch period.')
        if self.__last_starts[stage_name] == 0:
            raise GreenOpsException('greenops.measure.Measure.stop(): Cannot ' +
                                    'a non-running epoch period.')
        self._add_data_row(LogRow(now, stage_name,
                                  self.__device_data.short_name,
                                  self.__device_data.consumption,
                                  self.__epochs[stage_name], time_delta,
                                  list(self.__watch.values())))
        self.__epochs[stage_name] += 1


    def update(self, stage_name : str = DEFAULT_STAGE_NAME):
        """
        """

        now = time()
        if not self.stage_exists(stage_name):
            self.create_stage(stage_name)
        if self.__last_updates[stage_name] > 0:
            time_delta = now - self.__last_updates[stage_name]
            self._add_data_row(LogRow(now, stage_name,
                                      self.__device_data.short_name,
                                      self.__device_data.consumption,
                                      self.__epochs[stage_name], time_delta,
                                      list(self.__watch.values())))
        self.__epochs[stage_name] += 1
        self.__last_updates[stage_name] = now


    @property
    def watch_keys(self) -> list:
        """
        """

        return list(self.__watch.keys())


    def _add_data_row(self, row : LogRow):
        """
        """

        self.__data[row.stage].append(row)
        if self.__instant_log:
            self.save_new_rows(row.stage)


    def _get_last_update(self, stage_name : str) -> float:
        """
        """

        return self.__last_updates[stage_name]


    def _get_epoch(self, stage_name : str) -> int:
        """
        """

        return self.__epochs[stage_name]


    def _get_watch_values(self) -> list:
        """
        """

        return list(self.__watch.values())


    def _increase_epoch(self, stage_name : str, value : int):
        """
        """

        self.__epochs[stage_name] += value


    def _set_last_update(self, stage_name : str, new_value : float):
        """
        """

        self.__last_updates[stage_name] = new_value


    def __str__(self) -> str:
        """
        """

        return self.stats_summary
