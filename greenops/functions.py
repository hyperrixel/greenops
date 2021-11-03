"""
greenops
--------

Software to measure the footprints of deep learning models at training,
testing and evaluating to reduce energy consumption and carbon footprints.

Copyright rixel 2021
Distributed under the MIT License.
See accompanying file LICENSE.

File: submodule functions
"""


def extensionify_file(file_name : str, extension : str) -> str:
    """
    """

    if file_name.split('.')[-1] != extension:
        result = '{}.{}'.format(output_file_name, extension)
    else:
        result = file_name
    return result


def join_any(delimiter : str, some_iterable : any) -> str:
    """
    """

    result = ''
    len_some_iterable = len(some_iterable)
    if len_some_iterable > 0:
        result += str(some_iterable[0])
        pointer = 1
        while pointer < len_some_iterable:
            result += '{}{}'.format(delimiter, some_iterable[pointer])
            pointer += 1
    return result


def percentify(value : float) -> str:
    """
    """

    return '{:.2f} %'.format(value * 100.0)


def ws_to_kwh(watt_seconds : float) -> float:
    """
    """

    return watt_seconds / 3600000
