"""
greenops
--------

Software to measure the footprints of deep learning models at training,
testing and evaluating to reduce energy consumption and carbon footprints.

Copyright rixel 2021
Distributed under the MIT License.
See accompanying file LICENSE.

File: submodule datamodel
"""


from abc import ABC, abstractmethod

from .functions import percentify


class DictInterface(ABC):
    """
    Interface to serialize, deserialize
    ===================================
    """

    @property
    @abstractmethod
    def as_dict(self) -> dict:
        """
        Serializes the object to dict
        =============================

        Returns
        -------
        dict
            Object's data.
        """


    @staticmethod
    @abstractmethod
    def from_dict(data : dict, throw_error : bool = False) -> any:
        """
        Deserialize the object from a dict
        ==================================

        Parameters
        ----------
        ata : dict
            Data to deserialize the object from.
        throw_error : bool, optional (False if omitted)
            Whether to throw error or not.
        """


class DeviceData(DictInterface):
    """
    """

    def __init__(self, short_name : str, long_name : str,
                 consumption : float):
        """
        """

        self.__short_name = short_name
        self.__long_name = long_name
        self.__consumption = float(consumption)


    @property
    def as_dict(self) -> dict:
        """
        Serializes the object to dict
        =============================

        Returns
        -------
        dict
            Object's data.
        """

        return {'ShortName' : self.__short_name,
                'LongName' : self.__long_name,
                'Consumption' : self.__consumption}


    @property
    def consumption(self) -> float:
        """
        """

        return self.__consumption


    @staticmethod
    def from_dict(data : dict, throw_error : bool = False) -> any:
        """
        """

        result = None
        if all(['Consumption' in data.keys(), 'LongName' in data.keys(),
                'ShortName' in data.keys()]):
            result = DeviceData(data['ShortName'], data['LongName'],
                                data['Consumption'])
        elif throw_error:
            raise ValueError('greenops.datamodel.DeviceData.from_dict(): ' +
                             'Cannot restore device from invalid data.')
        return result


    @property
    def long_name(self) -> str:
        """
        """

        return self.__long_name


    @property
    def short_name(self) -> str:
        """
        """

        return self.__short_name


    def __repr__(self) -> str:
        """
        """

        return 'DeviceData({}, {}, {})'.format(self.__short_name,
                self.__long_name, self.__consumption)


    def __str__(self) -> str:
        """
        """

        return 'Device "{}" ("{}") consumes {} Watts.'.format(self.__long_name,
                self.__short_name, self.__consumption)


class PowerPriceData(DictInterface):
    """
    """

    def __init__(self, country_code : str, country_name : str,
                 price : float):
        """
        """

        self.__country_code = country_code
        self.__country_name = country_name
        self.__price = float(price)


    @property
    def as_dict(self) -> dict:
        """
        Serializes the object to dict
        =============================

        Returns
        -------
        dict
            Object's data.
        """

        return {'CountryCode' : self.__country_code,
                'CountryName' : self.__country_name,
                'Price' : self.__price}


    @property
    def country_code(self) -> str:
        """
        """

        return self.__country_code


    @property
    def country_name(self) -> str:
        """
        """

        return self.__country_name


    @staticmethod
    def from_dict(data : dict, throw_error : bool = False) -> any:
        """
        Deserialize the object from a dict
        ==================================

        Parameters
        ----------
        ata : dict
            Data to deserialize the object from.
        throw_error : bool, optional (False if omitted)
            Whether to throw error or not.
        """

        result = None
        if all(['CountryCode' in data.keys(), 'CountryName' in data.keys(),
                'Price' in data.keys()]):
            result = PowerPriceData(data['CountryCode'], data['CountryName'],
                                   data['Price'])
        elif throw_error:
            raise ValueError('greenops.datamodel.PowerPriceData.from_dict(): ' +
                             'Cannot restore device from invalid data.')
        return result


    @property
    def price(self) -> float:
        """
        """

        return self.__price


    def __repr__(self) -> str:
        """
        """

        return 'PowerPrice({}, {}, {})'.format(self.__country_code,
                self.__country_name, self.__price)

    def __str__(self) -> str:
        """
        """

        return 'Electricity in {} [{}] costs {} '.format(self.__country_name,
                self.__country_code, self.__price
                ) + 'in USD per kilowatt-hour (kWh)'


class PowerSourcesData(DictInterface):
    """
    """

    def __init__(self, country_code : str, country_name : str, coal : float,
                 gas : float, hydro : float, other_renewables : float,
                 solar : float, oil : float, wind : float, nuclear : float):
        """
        """

        self.__country_code = country_code
        self.__country_name = country_name
        self.__coal = float(coal)
        self.__gas = float(gas)
        self.__hydro = float(hydro)
        self.__other_renewables = float(other_renewables)
        self.__solar = float(solar)
        self.__oil = float(oil)
        self.__wind = float(wind)
        self.__nuclear = float(nuclear)


    @property
    def as_dict(self) -> dict:
        """
        Serializes the object to dict
        =============================

        Returns
        -------
        dict
            Object's data.
        """

        return {'CountryCode' : self.__country_code,
                'CountryName' : self.__country_name,
                'Coal' : self.__coal, 'Gas' : self.__gas,
                'Hydro' : self.__hydro,
                'OtherRenewables' : self.__other_renewables,
                'Solar' : self.__solar, 'Oil' : self.__oil,
                'Wind' : self.__wind, 'Nuclear' : self.__nuclear}


    @property
    def coal(self) -> float:
        """
        """

        return self.__coal


    @property
    def country_code(self) -> str:
        """
        """

        return self.__country_code


    @property
    def country_name(self) -> str:
        """
        """

        return self.__country_name


    @staticmethod
    def from_dict(data : dict, throw_error : bool = False) -> any:
        """
        Deserialize the object from a dict
        ==================================

        Parameters
        ----------
        ata : dict
            Data to deserialize the object from.
        throw_error : bool, optional (False if omitted)
            Whether to throw error or not.
        """

        result = None
        if all(['CountryCode' in data.keys(), 'CountryName' in data.keys(),
                'Coal' in data.keys(), 'Gas' in data.keys(),
                'Hydro' in data.keys(), 'OtherRenewables' in data.keys(),
                'Solar' in data.keys(), 'Oil' in data.keys(),
                'Wind' in data.keys(), 'Nuclear' in data.keys()]):
            result = PowerSourcesData(data['CountryCode'], data['CountryName'],
                                      data['Coal'], data['Gas'], data['Hydro'],
                                      data['OtherRenewables'], data['Solar'],
                                      data['Oil'], data['Wind'],
                                      data['Nuclear'])
        elif throw_error:
            raise ValueError('greenops.datamodel.PowerSourcesData.from_dict():'
                             + ' Cannot restore device from invalid data.')
        return result


    @property
    def gas(self) -> float:
        """
        """

        return self.__gas


    @property
    def hydro(self) -> float:
        """
        """

        return self.__hydro


    @property
    def nuclear(self) -> float:
        """
        """

        return self.__nuclear


    @property
    def oil(self) -> float:
        """
        """

        return self.__oil


    @property
    def other_renewables(self) -> float:
        """
        """

        return self.__other_renewables


    @property
    def solar(self) -> float:
        """
        """

        return self.__solar


    @property
    def wind(self) -> float:
        """
        """

        return self.__wind


    def __repr__(self) -> str:
        """
        """

        return 'PowerSources({}, {}, {}, {}, {}, {}, {}, {}, {}, {})'.format(
                self.__country_code, self.__country_name, self.__coal,
                self.__gas, self.__hydro, self.__other_renewables,
                self.__solar, self.__oil, self.__wind, self.__nuclear)

    def __str__(self) -> str:
        """
        """

        return 'Electricity in {} [{}] is produced '.format(self.__country_name,
                self.__country_code) + 'from coal {}, oil {}, gas {}, '.format(
                percentify(self.__coal), percentify(self.__oil), percentify(
                self.__gas)) + 'nuclear power {}, hydroelectric power '.format(
                percentify(self.__nuclear)) + '{}, solar power {}, wind'.format(
                percentify(self.__hydro), percentify(self.__solar)
                ) + ' power {}, other renewable sources {}'.format(percentify(
                self.__wind), percentify(self.__other_renewables))


class Response:
    """
    """

    def __init__(self, query_str : str, error_str : str, result_equals : list,
                 result_contains : list):
        """
        """

        self.__query = query_str
        self.__error = error_str
        self.__strict_match = result_equals
        self.__partial_match = result_contains


    @property
    def error(self) -> str:
        """
        """

        return self.__error


    @staticmethod
    def from_dict(data : dict, result_type : any) -> any:
        """
        Deserialize the object from a dict
        ==================================

        Parameters
        ----------
        ata : dict
            Data to deserialize the object from.
        throw_error : bool, optional (False if omitted)
            Whether to throw error or not.
        """

        result = None
        if isinstance(data, dict):
            if all(['query' in data.keys(), 'error' in data.keys(),
                    'result' in data.keys()]):
                if isinstance(data['result'], dict):
                    if all(['equals' in data['result'].keys(),
                            'contains' in data['result'].keys(),]):
                        result = Response(data['query'], data['error'],
                                          [result_type.from_dict(e)
                                          for e in data['result']['equals']],
                                          [result_type.from_dict(e)
                                          for e in data['result']['contains']])
        return result


    @property
    def is_good(self) -> bool:
        """
        """

        return len(self.__error) == 0


    @property
    def partial_match(self) -> str:
        """
        """

        return self.__partial_match


    @property
    def query(self) -> str:
        """
        """

        return self.__query


    @property
    def strict_match(self) -> str:
        """
        """

        return self.__strict_match
