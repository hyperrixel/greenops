"""
greenops
--------

Software to measure the footprints of deep learning models at training,
testing and evaluating to reduce energy consumption and carbon footprints.

Copyright rixel 2021
Distributed under the MIT License.
See accompanying file LICENSE.

File: submodule advanced
"""

from threading import Thread
from time import asctime, gmtime, localtime, strftime, time

from .measure import DEFAULT_STAGE_NAME, Measure


class ThreadMeasure(Measure):
    """
    """

    def create_stage(self, stage_name : str):
        """
        """

        Thread(target=super().create_stage, args=(stage_name, )).start()


    def reset(self, stage_name : str = None):
        """
        """

        Thread(target=super().reset, args=(stage_name, )).start()


    def save_new_rows(self, stage_name : str = None):
        """
        """

        Thread(target=super().save_new_rows, args=(stage_name, )).start()


    def save_stats(self, file_name : str = None):
        """
        """

        Thread(target=super().save_stats, args=(file_name, )).start()


    def start(self, stage_name : str = DEFAULT_STAGE_NAME):
        """
        """

        Thread(target=super().start, args=(stage_name, )).start()


    def stop(self, stage_name : str = DEFAULT_STAGE_NAME):
        """
        """

        Thread(target=super().stop, args=(stage_name, )).start()


    def update(self, stage_name : str = DEFAULT_STAGE_NAME):
        """
        """

        Thread(target=super().update, args=(stage_name, )).start()
