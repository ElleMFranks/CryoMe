# -*- coding: utf-8 -*-
"""heater_ctrl.py - Control the cryostat temp using the lakeshore.

Several functions used to control the lakeshore temperature controller.
Functionality to set the lakeshore to a particular heater, set a
temperature, and check the temperature is stable.
"""

# region Import modules.
from __future__ import annotations
import logging
from typing import Union
from time import sleep, perf_counter

from pyvisa import Resource

import instruments
import util
# endregion


def heater_setup(tc_rm: Resource, channel: Union[int, str],
                 sample_or_warmup: str) -> None:
    """Function sets up lakeshore to "sample" or "warmup" mode."""

    # region Convert str argument to string.
    if sample_or_warmup == 'sample':
        s_or_w = 0
        util.safe_write(f'HTRSET {s_or_w} 100 0.2 ')
    elif sample_or_warmup == 'warmup':
        s_or_w = 1
        util.safe_write(f'HTRSET {s_or_w} 2 0 0.3', 0.5, tc_rm)
    else:
        raise Exception('Neither sample nor warmup chosen.')
    # endregion

    # region Output setup of heater to PID (5).
    util.safe_write(f' OUTMODE {s_or_w}  5 {channel}', 0.5, tc_rm)
    # endregion

    


def set_temp(tc_rm: Resource, temp: float, lna_or_load: str) -> None:
    """Sets the temp of either the lna or load in the cryostat."""
    # region Send sample heater cmd if LNA or warmup cmd if Load.
    if lna_or_load == 'lna':
        util.safe_write(f'SETP 0 {temp}', 0.5, tc_rm)  # Sample heater to 20k
        util.safe_write('RANGE 0 8', 0.5, tc_rm)
    elif lna_or_load == 'load':
        util.safe_write(f'SETP 1 {temp}', 0.5, tc_rm)
        util.safe_write('RANGE 1 1', 0.5, tc_rm)
    sleep(2)  # This time could possibly be modified.
    # endregion


def _check_temp(tc_rm: Resource,  channel: int, target_temp: float,
                st_time: float = 10) -> int:
    """Checks temperature over 10s to ascertain stability.
    
    Args:
        tc_rm: The temperature controller resource manager.
        channel: The channel on the temperature controller.
        temp: Temperature to stabilise at.
        st_time: Time to check stabilisation over.
     """
    # region Set function variables.
    check = False
    error_target = 1  # Error is 2X this variable, usually 0.4.
    # endregion.

    # region Check temperature at 0 and 10s if stable check is 1
    temp = util.safe_query(f'KRDG? {channel}', 0.5, tc_rm, 'lakeshore', True)
    print(f'Channel {channel}, Target: {target_temp:+.2f} K, Currently: {temp:+.2f} K', end='\r')
    if target_temp - error_target < temp < target_temp + error_target:
        sleep(st_time)
        temp = util.safe_query(
            f'KRDG? {channel}', 0.5, tc_rm, 'lakeshore', True)
        print(f'Channel {channel}, Target: {target_temp:+.2f}, Currently: {temp:+.2f} K', end='\r')
        if target_temp - error_target < temp < target_temp + error_target:
            print(f'Channel {channel}, Target: {target_temp:+.2f}, Currently: {temp:+.2f} K')
            return True
    # endregion


def temp_stabilisation(tc_rm: Resource, channel: int, target_temp: float,
                            st_time: float = 10) -> bool:
    """Stabilises cryostat at given temperature.
    
    Args:
        tc_rm: The temperature controller resource manager.
        channel: The channel on the temperature controller.
        temp: Temperature to stabilise at.
        st_time: Time to check stabilisation over.
    """
    # region Check temperature until within range for st_time seconds.
    start = perf_counter()
    util.safe_write(f'SCAN{channel},0', 4, tc_rm)
    while not _check_temp(tc_rm, channel, target_temp, st_time):
        sleep(0.5)
        time = perf_counter() - start
        if time > 420:
            user_response = input(
            f'Continue waiting to stabilise y/n?')
            if user_response == 'y':
                return True
            else:
                start = perf_counter() - 360
    util.safe_write(f'SCAN{channel},0', 0.5, tc_rm)
    # endregion


def get_load_temp_target(
        hot_or_cold: str, 
        tc_settings: instruments.TempControllerSettings) -> float:
    """Returns the hot or cold temperature target."""
    log = logging.getLogger(__name__)
    if hot_or_cold == 'Cold':
        temp_target = float(tc_settings.cold_target)
    elif hot_or_cold == 'Hot':
        temp_target = float(tc_settings.hot_target)
    else:
        raise Exception('Hot or cold not passed correctly')
    log.info(f'Initiating {hot_or_cold.lower()} measurement.')
    log.info(f'Setting load to {temp_target} K.')
    return temp_target


def get_loop_temps(
        tc_rm: Resource,
        instr_settings: instruments.InstrumentSettings) -> list[float]:
    """Returns the measured temperatures on the requested channels."""
    # region Get and return temperatures for requested temp sensors.
    buffer_time = instr_settings.buffer_time
    tc_settings = instr_settings.temp_ctrl_settings
    if tc_rm is not None:
        # region Measure and store load, lna, and extra sensor temps.
        tc_rm.write(f'SCAN {tc_settings.load_lsch},0')
        sleep(5)
        load_temp = util.safe_query(
            f'KRDG? {tc_settings.load_lsch}', buffer_time, tc_rm,
            'lakeshore', True)
        tc_rm.write(f'SCAN {tc_settings.lna_lsch},0')
        sleep(5)
        lna_temp = util.safe_query(
            f'KRDG? {tc_settings.lna_lsch}', buffer_time, tc_rm,
            'lakeshore', True)
        if tc_settings.extra_sensors_en:
            tc_rm.write(f'SCAN {tc_settings.extra_1_lsch},0')
            sleep(5)
            extra_1_temp = util.safe_query(
                f'KRDG? {tc_settings.extra_1_lsch}', buffer_time, tc_rm,
                'lakeshore', True)
            tc_rm.write(f'SCAN {tc_settings.extra_2_lsch},0')
            sleep(5)
            extra_2_temp = util.safe_query(
                f'KRDG? {tc_settings.extra_2_lsch}', buffer_time, tc_rm,
                'lakeshore', True)
        else:
            extra_1_temp = 'NA'
            extra_2_temp = 'NA'
        tc_rm.write(f'SCAN {tc_settings.load_lsch},0')
        sleep(5)
        # endregion

    # region Provide dummy temperature measurement values for debugging.
    else:
        load_temp = 35
        lna_temp = 21
        if tc_settings.extra_sensors_en:
            extra_1_temp = 40
            extra_2_temp = 30
        else:
            extra_1_temp = 'NA'
            extra_2_temp = 'NA'
    # endregion

    return [load_temp, lna_temp, extra_1_temp, extra_2_temp]
    # endregion


def set_loop_temps(tc_rm: Resource, load_temp_target: float,
                  instr_settings: instruments.InstrumentSettings) -> list:
    """Sets/gets temperatures, ensure stability/status of heater."""

    # region Unpack objects/set up logger/set initial variables.
    log = logging.getLogger(__name__)
    load_temp_set = False
    lna_temp_set = False
    tc_settings = instr_settings.temp_ctrl_settings
    lna_temp_target = tc_settings.lna_target
    # endregion

    if tc_rm is not None:
        util.safe_write(f'SCAN {tc_settings.lna_lsch},0', 
                    instr_settings.buffer_time, tc_rm)
    while not lna_temp_set:
        if tc_rm is not None:
            # region Check for heater errors and wait for stabilisation.
            pre_heater_status = util.safe_query(
                'HTRST? 0', instr_settings.buffer_time, tc_rm, 'lakeshore')
            set_temp(tc_rm, lna_temp_target, 'lna')
            temp_stabilisation(
                tc_rm, tc_settings.lna_lsch, tc_settings.lna_target, 5)
            lna_temp = util.safe_query(
                f'KRDG? {tc_settings.lna_lsch}', instr_settings.buffer_time, 
                tc_rm, 'lakeshore')
            post_heater_status = util.safe_query(
                'HTRST? 0', instr_settings.buffer_time, tc_rm, 'lakeshore')
            # endregion

            # region If LNA set without problem then exit loop.
            if (lna_temp_target - 1 < float(lna_temp) < lna_temp_target + 1) \
                    and pre_heater_status == '0\r' \
                    and post_heater_status == '0\r':
                lna_temp_set = True
            else:
                log.warning(f'Failed heating loop, trying again.'
                            f'Pre-set status: {pre_heater_status}.  '
                            f'Post-set status: {post_heater_status}.  '
                            f'Temperature: {lna_temp}K.  '
                            f'Target: {lna_temp_target}K.')
            # endregion
        else:
            lna_temp_set = True

    if tc_rm is not None:
        util.safe_write(f'SCAN {tc_settings.load_lsch},0', 
                    instr_settings.buffer_time, tc_rm)
    while not load_temp_set:
        if tc_rm is not None:
            # region Check heater errors and wait for stabilisation.
            pre_heater_status = util.safe_query(
                'HTRST? 1', instr_settings.buffer_time, tc_rm, 'lakeshore')
            set_temp(tc_rm, load_temp_target, 'load')
            temp_stabilisation(
                tc_rm, tc_settings.load_lsch, load_temp_target, 5)
            load_temp = util.safe_query(
                f'KRDG? {tc_settings.load_lsch}', instr_settings.buffer_time, 
                tc_rm, 'lakeshore')
            post_heater_status = util.safe_query(
                'HTRST? 1', instr_settings.buffer_time, tc_rm, 'lakeshore')
            # endregion

            if pre_heater_status != '0\r' or post_heater_status != '0\r':
                set_temp(tc_rm, load_temp_target, 'load')

            # region If load set without problem, exit loop, else warning.
            if (load_temp_target - 1 < float(load_temp) < load_temp_target + 1) \
                    and pre_heater_status == '0\r' \
                    and post_heater_status == '0\r':
                load_temp_set = True
            else:
                log.warning(f'Failed heating loop, trying again.'
                            f'Pre-set status: {pre_heater_status}.  '
                            f'Post-set status: {post_heater_status}.  '
                            f'Temperature: {load_temp}K.  '
                            f'Target: {load_temp_target}K.')
            # endregion
        else:
            load_temp_set = True