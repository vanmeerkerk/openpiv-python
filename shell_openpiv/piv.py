""" Class structure for PIV analysis

author: Mike van Meerkerk (Deltares)
date: 28 June 2023
"""
# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

from dataclasses import dataclass
from pathlib import Path
from typing import List, Union


@dataclass
class PIV_Config(object):
    """Class structure for PIV configuration

    Standard PIV configuration settings
    """

    # TODO: Implement pydantic for type checking
    # TODO: Add configuration settings. Evaluate if a YAML configuration file can return a PIV_Config object
    # with a multipass argument.
    window_size: Union[int, List[tuple]] = 32
    overlap: Union[int, List[tuple]] = 16
    multipass: bool = False

    # class PIV_settings(object):
    #     def __init__(self, annual_volatility_target):
    #         self.__setstate__({annual_volatility_target: annual_volatility_target})

    #     def __setstate__(self, kw):
    #         self.annual_volatility_target = kw.get('annual_volatility_target')
    #         self.daily = self.annual_volatility_target/np.sqrt(252)


class PIV(object):
    """Class structure for PIV analysis

    The PIV class read the configuration file and sets up the analysis.
    """

    def __init__(
        self,
        config_file: Union(str, Path),
    ) -> None:
        """Initialize the PIV class

        Parameters:
        ----------
        config_file : str or Path
            Path to the configuration file

        Returns:
        -------

        """
        # Check if config_file is a string or a Path
        if isinstance(config_file, str):
            config_file = Path(config_file)
        # Assign the config_file to the class
        self.config_file = config_file


class Image_Pair(object):
    """Class structure for an image pair"""

    def __init__(self) -> None:
        pass
