"""
greenops
--------

Software to measure the footprints of deep learning models at training,
testing and evaluating to reduce energy consumption and carbon footprints.

Copyright rixel 2021
Distributed under the MIT License.
See accompanying file LICENSE.

File: submodule context
"""


from json import load as json_load, dump as json_dump
from locale import getdefaultlocale
from os.path import dirname, join, isfile
from platform import processor
from threading import Thread

from .api import Api
from .datamodel import DeviceData, PowerPriceData, PowerSourcesData, Response


DEFAULT_CPU_WATTS = 125.0
DEFAULT_GPU_WATTS = 175.0
DEFAULT_POWER_PRICE = 1.0
DEFAULT_SOURCE_RATE = 0.125


class Context:
    """
    Maintains greenops's context
    ============================

    Notes
    -----
        This class is a singleton.
    """

    __cpu = []
    __country_code = ''
    __gpu = []
    __init_done = False
    __power_price = None
    __power_sources = None


    @classmethod
    def config(cls, config_data : dict = None, save_to_file : str = None):
        """
        Configures the context
        ======================

        Parameters
        ----------
        config_data : dict, optioal (None if omitted)
            Data to get the configuration from, if omitted context detects the
            needed information.
        save_to_file : str, optioal (None if omitted)
            File to save to, if omitted, default location is used.
        """

        if config_data is None:
            config_data = {}
        if 'CountryCode' in config_data.keys():
            Context.set_country_code(config_data['CountryCode'])
        else:
            Context.detect_country_code()
        if 'CPU' in config_data.keys():
            Context.set_cpu(config_data['CPU'])
        else:
            Context.detect_cpu()
        if 'GPU' in config_data.keys():
            Context.set_gpu(config_data['GPU'])
        else:
            Context.detect_gpu()
        if 'PowerPrice' in config_data.keys():
            Context.set_power_price(config_data['PowerPrice'])
        else:
            Context.detect_power_price()
        if 'PowerSources' in config_data.keys():
            Context.set_power_sources(config_data['PowerSources'])
        else:
            Context.detect_power_sources()
        if save_to_file is not None:
            Context.save(save_to_file)


    @classmethod
    def detect_country_code(cls) -> str:
        """
        Detect country code
        ===================
        """

        country_code = 'US'
        language_code, codepage = getdefaultlocale()
        if language_code is not None:
            country_code = language_code.split('_')[1]
        cls.__country_code = country_code


    @classmethod
    def detect_cpu(cls):
        """
        Detect CPUs
        ===========
        """

        final_devices = []
        cpu_name = processor() # FUTURE: a much more robust way is required.
        helper_data = Api.get_consumption_by_long_name(cpu_name)
        if Context.check_response_(helper_data):
            if len(helper_data.strict_match) == 1:
                final_devices = [helper_data.strict_match[0]]
        if len(final_devices) == 0:
            final_devices = [DeviceData('Default', 'DefaultBrand',
                                        DEFAULT_CPU_WATTS)]
        cls.__cpu = final_devices


    @classmethod
    def detect_gpu(cls):
        """
        Detect GPUs
        ===========
        """

        devices = []
        try:
            from torch import cuda
            if cuda.is_available():
                device_names = [cuda.get_device_name(d)
                                for d in range(cuda.device_count())]
        except ModuleNotFoundError:
            pass
        if len(devices) == 0: # FUTURE: filter if pynvml finds exact consumption
            try:
                import pynvml
                pynvml.nvmlInit()
                for i in range(pynvml.nvmlDeviceGetCount()):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle).decode('utf8')
                try:
                    consumption = pynvml.nvmlDeviceGetPowerUsage(handle
                                                                    ) / 1000.0
                    devices.append(DeviceData(name, name, consumption))
                except pynvml.nvml.NVMLError_NotSupported:
                    devices.append(name)
                pynvml.nvmlShutdown()
            except ModuleNotFoundError:
                pass
        final_devices = []
        for device in devices:
            if isinstance(device, DeviceData):
                final_devices.append(device)
            else:
                helper_data = Api.get_consumption_by_short_name(device)
                if len(helper_data.strict_match) > 0:
                    final_devices.append(helper_data.strict_match[0])
                elif len(helper_data.partial_match) > 0:
                    final_devices.append(helper_data.partial_match[0])
        if len(final_devices) == 0:
            final_devices = [DeviceData('Default', 'DefaultBrand',
                                        DEFAULT_GPU_WATTS)]
        cls.__gpu = final_devices


    @classmethod
    def detect_power_price(cls):
        """
        Detect local power price
        ========================
        """

        helper_data = Api.get_cost_by_country_code(cls.__country_code)
        if Context.check_response_(helper_data):
            if len(helper_data.strict_match) > 0:
                cls.__power_price = helper_data.strict_match[0]
            elif len(helper_data.partial_match) > 0:
                cls.__power_price = helper_data.partial_match[0]
        if cls.__power_price is None:
            cls.__power_price = PowerPriceData('AA', 'DefaultCountry',
                                              DEFAULT_POWER_PRICE)

    @classmethod
    def detect_power_sources(cls):
        """
        Detect local electricity source rates
        =====================================
        """

        helper_data = Api.get_source_rates_by_country_code(cls.__country_code)
        if Context.check_response_(helper_data):
            if len(helper_data.strict_match) > 0:
                cls.__power_sources = helper_data.strict_match[0]
            elif len(helper_data.partial_match) > 0:
                cls.__power_sources = helper_data.partial_match[0]
        if cls.__power_sources is None:
            cls.__power_sources = PowerSourcesData('AA', 'DefaultCountry',
                                    DEFAULT_SOURCE_RATE, DEFAULT_SOURCE_RATE,
                                    DEFAULT_SOURCE_RATE, DEFAULT_SOURCE_RATE,
                                    DEFAULT_SOURCE_RATE, DEFAULT_SOURCE_RATE,
                                    DEFAULT_SOURCE_RATE, DEFAULT_SOURCE_RATE)


    @classmethod
    def get_country_code(cls) -> str:
        """
        Get country code
        ================

        Returns
        -------
        str
            Two letter country code.
        """

        if not cls.__init_done:
            Context.init()
        return cls.__country_code


    @classmethod
    def get_cpu(cls) -> list:
        """
        Get CPUs
        ========

        Returns
        -------
        list
            List of detected CPUs.
        """

        if not cls.__init_done:
            Context.init()
        return cls.__cpu.copy()


    @classmethod
    def get_gpu(cls) -> list:
        """
        Get GPUs
        ========

        Returns
        -------
        list
            List of detected GPUs.
        """

        if not cls.__init_done:
            Context.init()
        return cls.__gpu.copy()


    @classmethod
    def get_power_price(cls) -> PowerPriceData:
        """
        Get local power price
        =====================

        Returns
        -------
        PowerPriceData
            Local power prices information.
        """

        if not cls.__init_done:
            Context.init()
        return cls.__power_price


    @classmethod
    def get_power_sources(cls) -> PowerSourcesData:
        """
        Get local electricity source rates
        ==================================

        Returns
        -------
        PowerSourcesData
            Local source rates.
        """

        if not cls.__init_done:
            Context.init()
        return cls.__power_sources


    @classmethod
    def init(cls):
        """
        Initializes the context if needed
        =================================
        """

        if not cls.__init_done:
            if isfile(join(dirname(__file__), 'greenops_context.json')):
                Context.load(join(dirname(__file__), 'greenops_context.json'))
            else:
                Context.config()
                Context.save(join(dirname(__file__), 'greenops_context.json'))
            cls.__init_done = True


    @classmethod
    def load(cls, file_name : str):
        """
        Load context from file
        ======================

        Parameters
        ----------
        file_name : str
            The name of the file.

        Notes
        -----
            Instead of raise an error it prints the reason if loading was not
            successful.
        """

        data = {}
        try:
            with open(file_name, 'r', encoding='utf8') as instream:
                data = json_load(instream)
        except Exception as error:
            print('greenops.context.Context.load(): Cannot load configuration,'
                  + ' {} exception occured.'.format(error.__class__.__name__))
        Context.config(data)


    @classmethod
    def save(cls, file_name : str):
        """
        Save context to file
        ====================

        Parameters
        ----------
        file_name : str
            The name of the file.
        """

        data = {'CountryCode' : cls.__country_code,
                'CPU' : [e.as_dict for e in cls.__cpu],
                'GPU' : [e.as_dict for e in cls.__gpu],
                'PowerPrice' : cls.__power_price.as_dict,
                'PowerSources' : cls.__power_sources.as_dict}
        with open(file_name, 'w', encoding='utf8') as outstream:
            json_dump(data, outstream, indent='\t')


    @classmethod
    def set_country_code(cls, country_code : str):
        """
        Set country code
        ================

        Parameters
        ----------
        country_code : str
            Two letter country code.

        Raises
        ------
        ValueError
            If country code is more or less than 2 letters.
        """

        if len(country_code) != 2:
            raise ValueError('greenops.context.Context.set_country_code():'
                             + ' Country code is exactly 2 letters')
        cls.__country_code = country_code.upper()


    @classmethod
    def set_cpu(cls, *args):
        """
        Set CPUs
        ========

        Parameters
        ----------
        args
            Data to restore CPU list from.

        Raises
        ------
        ValueError
            If data is not appropriate.
        """

        data = Context.restore_devices_(*args)
        if data is None:
            raise ValueError('greenops.context.Context.set_cpu(): Cannot' +
                             'set CPU data from invalid data.')
        cls.__cpu = data


    @classmethod
    def set_gpu(cls, *args):
        """
        Set GPUs
        ========

        Parameters
        ----------
        args
            Data to restore GPU list from.

        Raises
        ------
        ValueError
            If data is not appropriate.
        """

        data = Context.restore_devices_(*args)
        if data is None:
            raise ValueError('greenops.context.Context.set_gpu(): Cannot' +
                             'set CPU data from invalid data.')
        cls.__gpu = data


    @classmethod
    def set_power_price(cls, *args, **kwargs):
        """
        Set local power price
        =====================

        Parameters
        ----------
        args, kwargs
            Data to restore a PowerPriceData object from.

        Raises
        ------
        ValueError
            If data is not appropriate.
        """

        new_data = None
        if len(args) == 1:
            if isinstance(args[0], dict):
                new_data = PowerPriceData.from_dict(args[0])
        else:
            new_data = PowerPriceData.from_dict(kwargs)
        if new_data is None:
            raise ValueError('greenops.context.Context.set_power_price(): ' +
                             'Cannot set power price data from invalid data.')
        cls.__power_price = new_data


    @classmethod
    def set_power_sources(cls, *args, **kwargs):
        """
        Set local electricity source rates
        ==================================

        Parameters
        ----------
        args, kwargs
            Data to restore a PowerSourcesData object from.

        Raises
        ------
        ValueError
            If data is not appropriate.
        """

        new_data = None
        if len(args) == 1:
            if isinstance(args[0], dict):
                new_data = PowerSourcesData.from_dict(args[0])
        else:
            new_data = PowerSourcesData.from_dict(kwargs)
        if new_data is None:
            raise ValueError('greenops.context.Context.set_power_sources(): ' +
                             'Cannot set power sources data from invalid data.')
        cls.__power_sources = new_data


    @staticmethod
    def check_response_(response : Response) -> bool:
        """
        """

        result = False
        if isinstance(response, Response):
            if response.is_good:
                result = True
        return result


    @staticmethod
    def restore_devices_(*args) -> any:
        """
        Restore device data
        ===================

        Parameters
        ----------
        args
            Data to restore DeviceData objects from.

        Returns
        -------
        list(DeviceData) | None
            List of DeviceData object in case of success, else None.

        Notes
        -----
            This is a helper function. Its use from outside has risks.
        """

        def transform_list(list_data : list) -> any:
            """
            Transform list to device data if possible
            =========================================

            Parameters
            ----------
            list
                List of data to restore DeviceData objects from.

            Returns
            -------
            list(DeviceData) | None
                List of DeviceData object in case of success, else None.

            Notes
            -----
                This is an inner function to optimize code (readability).
            """

            transformed = None
            data = []
            for element in list_data:
                if isinstance(element, dict):
                    data.append(DeviceData.from_dict(element))
                elif isinstance(element, DeviceData):
                    data.append(element)
                else:
                    data.append(None)
            if all([e is not None for e in data]):
                transformed = data
            return transformed

        result = None
        if len(args) == 1:
            if isinstance(args[0], dict):
                data = DeviceData.from_dict(args[0])
                if data is not None:
                    result = [data]
            elif isinstance(args[0], DeviceData):
                result = [args[0]]
            elif isinstance(args[0], (list, tuple)):
                result = transform_list(args[0])
        elif len(args) > 1:
            result = transform_list(args)
        return result
