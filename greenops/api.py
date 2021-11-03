"""
greenops
--------

Software to measure the footprints of deep learning models at training,
testing and evaluating to reduce energy consumption and carbon footprints.

Copyright rixel 2021
Distributed under the MIT License.
See accompanying file LICENSE.

File: submodule api
"""


from requests import get as requests_get

from .datamodel import DeviceData, PowerPriceData, PowerSourcesData, Response


class Api:
    """
    Maintains API connection with the greenops API service
    ======================================================
    """


    __api_url = 'greenops.hyperrixel.com/api'
    __maximum_connection_tries = 2
    __protocol = 'https'


    @classmethod
    def decrease__maximum_connection_tries(cls):
        """
        Decrease maximum connection of tries by one
        ===========================================
        """

        if cls.__maximum_connection_tries > 1:
            cls.__maximum_connection_tries -= 1


    @classmethod
    def get_api_url(cls) -> str:
        """
        Get the URL of the API
        ======================

        Returns
        -------
        str
            The base URL of the API.
        """

        return cls.__api_url


    @classmethod
    def get_consumption_by_long_name(cls, name : str) -> any:
        """
        Acces to endpoint getConsumptionByLongName
        ==========================================

        Parameters
        ----------
        name : str
            Name to search for.

        Returns
        -------
        dict | None
            Dict in case of any level of success, else None.
        """

        return cls.download_from_endpoint_('getConsumptionByLongName', name,
                                           DeviceData)


    @classmethod
    def get_consumption_by_short_name(cls, name : str) -> any:
        """
        Acces to endpoint getConsumptionByShortName
        ===========================================

        Parameters
        ----------
        name : str
            Name to search for.

        Returns
        -------
        dict | None
            Dict in case of any level of success, else None.
        """

        return cls.download_from_endpoint_('getConsumptionByShortName', name,
                                           DeviceData)


    @classmethod
    def get_cost_by_country_code(cls, code : str) -> any:
        """
        Acces to endpoint getCostByCountryCode
        ======================================

        Parameters
        ----------
        code : str
            Code to search for.

        Returns
        -------
        dict | None
            Dict in case of any level of success, else None.
        """

        return cls.download_from_endpoint_('getCostByCountryCode', code,
                                           PowerPriceData)


    @classmethod
    def get_cost_by_country_name(cls, name : str) -> any:
        """
        Acces to endpoint getCostByCountryName
        ======================================

        Parameters
        ----------
        name : str
            Name to search for.

        Returns
        -------
        dict | None
            Dict in case of any level of success, else None.
        """

        return cls.download_from_endpoint_('getCostByCountryName', name,
                                           PowerPriceData)


    @classmethod
    def get_maximum_connection_tries(cls) -> int:
        """
        Get the number of maximum connection tries
        ==========================================

        Returns
        -------
        int
            The number of maximum connection tries.
        """

        return cls.__maximum_connection_tries


    @classmethod
    def get_source_rates_by_country_code(cls, code : str) -> any:
        """
        Acces to endpoint getSourceRatesByCountryCode
        =============================================

        Parameters
        ----------
        code : str
            Code to search for.

        Returns
        -------
        dict | None
            Dict in case of any level of success, else None.
        """

        return cls.download_from_endpoint_('getSourceRatesByCountryCode', code,
                                           PowerSourcesData)


    @classmethod
    def get_source_rates_by_country_name(cls, name : str) -> any:
        """
        Acces to endpoint getSourceRatesByCountryName
        =============================================

        Parameters
        ----------
        name : str
            Name to search for.

        Returns
        -------
        dict | None
            Dict in case of any level of success, else None.
        """

        return cls.download_from_endpoint_('getSourceRatesByCountryName', name,
                                           PowerSourcesData)


    @classmethod
    def get_protocol(cls) -> str:
        """
        Get connection protocol
        =======================

        Returns
        -------
        str
            The name of the protocol.
        """

        return cls.__protocol


    @classmethod
    def increase__maximum_connection_tries(cls):
        """
        Increase maximum connection of tries by one
        ===========================================
        """

        cls.__maximum_connection_tries += 1


    @classmethod
    def use_http(cls):
        """
        Set protocoel to http
        =====================
        """

        cls.__protocol = 'http'


    @classmethod
    def use_https(cls):
        """
        Set protocoel to https
        ======================
        """

        cls.__protocol = 'https'


    @classmethod
    def download_from_endpoint_(cls, endpoint_name : str, query : str,
                                result_type : any) -> any:
        """
        Download from endpoint
        ======================

        Parameters
        ----------
        endpoint_name : str
            Name of the endpoint to connect.
        query : str
            Query to search for.

        Returns
        -------
        dict | None
            Dict in case of any level of success, else None.

        Notes
        -----
            This is a helper function. Its use from outside has risks.
        """

        result = None
        final_url = '{}://{}/{}/{}'.format(cls.__protocol, cls.__api_url,
                                           endpoint_name, query)
        tries_count = 0
        while result is None and tries_count < cls.__maximum_connection_tries:
            handler = requests_get(final_url)
            if handler.status_code == 200:
                try:
                    result = handler.json()
                except Exception:
                    pass
            tries_count += 1
        return Response.from_dict(result, result_type)
