# -*- coding: utf-8 -*-
"""heater_ctrl.py - Control the cryostat temp using the lakeshore.

Several functions used to control the lakeshore temperature controller.
Functionality to set the lakeshore to a particular heater, set a
temperature, and check the temperature is stable.
"""

# region Import modules.
from __future__ import annotations
from typing import Union
import time

import pyvisa as pv

import util as ut
# endregion


def heater_setup(tc_rm: pv.Resource, channel: Union[int, str],
                 sample_or_warmup: str) -> None:
    """Function sets up lakeshore to "sample" or "warmup" mode."""

    # region Convert str argument to string.
    if sample_or_warmup == 'sample':
        s_or_w = 0
    elif sample_or_warmup == 'warmup':
        s_or_w = 1
    else:
        raise Exception('Neither sample nor warmup chosen.')
    # endregion

    # region Output setup of heater to PID (5).
    ut.safe_write(f' OUTMODE {s_or_w}  5 {channel}', 0.5, tc_rm)
    # endregion


def set_temp(tc_rm: pv.Resource, temp: float, lna_or_load: str) -> None:
    """Sets the temp of either the lna or load in the cryostat."""
    # region Send sample heater cmd if LNA or warmup cmd if Load.
    if lna_or_load == 'lna':
        ut.safe_write(f'SETP 0 {temp}', 0.5, tc_rm)  # Sample heater to 20k
        ut.safe_write('RANGE 0 8', 0.5, tc_rm)
    elif lna_or_load == 'load':
        ut.safe_write(f'SETP 1 {temp}', 0.5, tc_rm)
        ut.safe_write('RANGE 1 1', 0.5, tc_rm)
    time.sleep(2)  # This time could possibly be modified.
    # endregion


def _check_temp(tc_rm: pv.Resource,  channel: int, target_temp: float) -> int:
    """Checks temperature over 10s to ascertain stability."""
    # region Set function variables.
    check = 0
    error_target = 1  # Error is 2X this variable, usually 0.4.
    # endregion.

    # region Check temperature at 0 and 10s if stable check is 1
    temp = ut.safe_query(f'KRDG? {channel}', 0.5, tc_rm, 'lakeshore', True)
    if temp < target_temp - error_target or temp > target_temp + error_target:
        check = 0
    if target_temp - error_target < temp < target_temp + error_target:
        check = 0
        time.sleep(10)
        temp = ut.safe_query(f'KRDG? {channel}', 0.5, tc_rm, 'lakeshore', True)
        if target_temp - error_target < temp < target_temp + error_target:
            check = 1
    return check
    # endregion


def load_temp_stabilisation(tc_rm, channel, temp):
    """Stabilises cryostat at given temperature."""
    # region Check temperature until within range for 10 seconds.
    ut.safe_write(f'SCAN{channel},0', 4, tc_rm)
    while _check_temp(tc_rm, channel, temp) == 0:
        time.sleep(0.5)
    ut.safe_write(f'SCAN{channel},0', 0.5, tc_rm)
    return 1
    # endregion
