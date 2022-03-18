# -*- coding: utf-8 -*-
"""measurement.py - Methods to carry out a single measurement.

This module contains the method to take a single measurement set.
External modules should call the measurement() function, which
depending on the passed variables, will call the _meas_loop()
function in specific ways. The measurement loop will record the
temperatures/powers required for the y factor measurement at each
frequency. They are stored as hot/cold results objects as found
in outputs.py and returned.
"""

# region Import modules.
from __future__ import annotations
from time import sleep
from typing import Union, Optional
import logging
import random

import numpy as np
import tqdm

import heater_ctrl
import instruments
import outputs
import config_handling
import util
# endregion

def _meas_loop(
        settings: config_handling.Settings, hot_or_cold: str,
        res_managers: instruments.ResourceManagers
        ) -> outputs.LoopInstanceResult:
    """The measurement algorithm for each hot or cold measurement.

    Checks if hot or cold, sets load to target temp, then moves through
    intermediate frequencies taking power and temperature at each one.
    Returns an object containing the measured power, and load and lna
    temperatures.

    Args:
        settings: The settings for the measurement instance.
        hot_or_cold: Either 'Hot' or 'Cold', decides which target
            temperature to set the lakeshore to.
        res_managers: The resource managers for all the instruments
            being used in this measurement.

    Returns:
        Object containing measured power, and load/lna/extra
        temperatures for either hot/cold temperature measurement.
    """

    # region Unpack from passed objects and initialise arrays.
    log = logging.getLogger(__name__)
    tc_settings = settings.instr_settings.temp_ctrl_settings
    tc_rm = res_managers.tc_rm
    spec_an_rm = res_managers.sa_rm
    sig_gen_rm = res_managers.sg_rm
    sig_gen_settings = settings.instr_settings.sig_gen_settings
    inter_freqs_array = sig_gen_settings.if_freq_array
    powers, load_temps, lna_temps, pre_loop_lna_temps = ([] for _ in range(4))
    pre_loop_extra_1_temps, pre_loop_extra_2_temps = ([] for _ in range(2))
    post_loop_lna_temps, post_loop_extra_1_temps = ([] for _ in range(2))
    post_loop_extra_2_temps = []
    buffer_time = settings.instr_settings.buffer_time
    # endregion

    # region Decide target temperature.
    temp_target = heater_ctrl.get_load_temp_target(hot_or_cold, tc_settings)
    # endregion

    if tc_rm is not None:
        heater_ctrl.set_loop_temps(
            tc_rm, temp_target, settings.instr_settings)
        pre_loop_temps = heater_ctrl.get_loop_temps(
            tc_rm, settings.instr_settings)
    else:
        pre_loop_temps = [temp_target + 0.1, 18, 22, 10]
    pre_loop_lna_temp = pre_loop_temps[1]
    pre_loop_extra_1_temp = pre_loop_temps[2]
    pre_loop_extra_2_temp = pre_loop_temps[3]
    # endregion

    # region Set spec an for single measurement mode.
    if spec_an_rm is not None:
        spec_an_rm.write('INIT:CONT 0')
        sleep(buffer_time)
    # endregion

    # region Sweep requested frequencies measuring power and load temp.
    log.info(f'Temperature stable at {pre_loop_temps[0]} K, starting sweep.', )
    pbar = tqdm.tqdm(
        total=len(inter_freqs_array), ncols=110,
        desc="Loop Prog", leave=True, position=0,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} '
                   '[Elapsed: {elapsed}, To Go: {remaining}]{postfix}')

    i = 0
    while i < len(inter_freqs_array):
        try:
            # region Set signal generator to intermediate frequency.
            if sig_gen_rm is not None \
                    and sig_gen_settings.vna_or_sig_gen == 'vna':
                sig_gen_rm.write(f':SENSE:FREQ:CW {inter_freqs_array[i]} GHz')
            elif sig_gen_rm is not None and \
                    sig_gen_settings.vna_or_sig_gen == 'sig gen':
                sig_gen_rm.write(
                    f'PL {sig_gen_settings.sig_gen_pwr_lvls[i]} DM')
                sleep(buffer_time)
                sig_gen_rm.write(f'CW {inter_freqs_array[i]} GZ')
            # endregion

            # region Store pre-loop temperatures, and during loop load temp.
            if tc_rm is not None:
                load_temp = util.safe_query(
                    f'KRDG? {tc_settings.load_lsch}', buffer_time, tc_rm,
                    'lakeshore', True)
                if temp_target - 1 > load_temp > temp_target + 1:
                    log.warning('Fallen out of temp range during measurement.')
                    pre_loop_temps = heater_ctrl.set_loop_temps(
                        tc_rm, temp_target, settings.instr_settings)
            else:
                load_temp = temp_target + (round(random.uniform(-0.1, 0.1), 2))

            # region Send command for sig an to take a measurement sweep.
            if spec_an_rm is not None:
                spec_an_rm.write('INIT:IMM')
                sleep(buffer_time)
            # endregion

            # region Measure and store marker power at requested frequency.
            if spec_an_rm is not None:
                util.safe_query('*OPC?', buffer_time, spec_an_rm, 'spec an')
                marker_power = util.safe_query(
                    ':CALC:MARK1:Y?', buffer_time, spec_an_rm, 'spec an')
                powers.append(float(marker_power.strip()))
            else:
                sleep(0.2)
                if hot_or_cold == 'Hot':
                    marker_power = -50 + round(random.uniform(1.2, 2), 2)
                else:
                    marker_power = -50 + round(random.uniform(0.3, 0.6), 2)
                powers.append(marker_power)
            # endregion

            load_temps.append(load_temp)
            pre_loop_lna_temps.append(pre_loop_lna_temp)
            pre_loop_extra_1_temps.append(pre_loop_extra_1_temp)
            pre_loop_extra_2_temps.append(pre_loop_extra_2_temp)
            # endregion

            i += 1
            pbar.update()
        except Exception as _e:
            log.error(f'Error at {inter_freqs_array[i]}: {_e}')
            continue
    # endregion
    pbar.close()
    log.info('Frequency sweep completed.')

    # region Put spec an in continuous measurement mode.
    if spec_an_rm is not None:
        spec_an_rm.write('INIT:CONT 1')
    # endregion

    # region Measure post loop lna temperature.
    post_loop_temps = heater_ctrl.get_loop_temps(
        tc_rm, temp_target, settings.instr_settings)
    post_loop_lna_temp = post_loop_temps[1]
    post_loop_extra_1_temp = post_loop_temps[2]
    post_loop_extra_2_temp = post_loop_temps[3]

    # region Store post loop lna/extra temperatures.
    for _ in powers:
        post_loop_lna_temps.append(post_loop_lna_temp)
        post_loop_extra_1_temps.append(post_loop_extra_1_temp)
        post_loop_extra_2_temps.append(post_loop_extra_2_temp)

    pre_loop_t_lna = np.array(pre_loop_lna_temps)
    post_loop_t_lna = np.array(post_loop_lna_temps)
    lna_temps = ((pre_loop_t_lna + post_loop_t_lna) / 2).tolist()
    # endregion

    # region Return loop instance result.
    log.info(f'{hot_or_cold} measurement complete.')
    return outputs.LoopInstanceResult(
        hot_or_cold, powers, load_temps, lna_temps,
        outputs.PrePostTemps(pre_loop_lna_temps, post_loop_lna_temps,
                             pre_loop_extra_1_temps, post_loop_extra_1_temps,
                             pre_loop_extra_2_temps, post_loop_extra_2_temps))
    # endregion


def _closest_temp_then_other(
        init_temp: float, res_managers: instruments.ResourceManagers,
        settings: config_handling.Settings
        ) -> list[outputs.LoopInstanceResult]:
    """Triggers measurement loops, first closest temp, then other."""

    tc_settings = settings.instr_settings.temp_ctrl_settings
    distance_to_hot = abs(init_temp - float(tc_settings.hot_target))
    distance_to_cold = abs(init_temp - float(tc_settings.cold_target))
    closer_to_hot = bool(distance_to_hot < distance_to_cold)
    hot = None
    cold = None

    for _ in range(2):
        if closer_to_hot:
            hot = _meas_loop(settings, 'Hot', res_managers)
            closer_to_hot = not closer_to_hot
        else:
            cold = _meas_loop(settings, 'Cold', res_managers)
            closer_to_hot = not closer_to_hot

    return [hot, cold]
    # endregion


def measurement(
        settings: config_handling.Settings,
        res_managers: instruments.ResourceManagers,
        trimmed_input_data: config_handling.TrimmedInputs,
        hot_cold_count: Optional[int] = None
        ) -> Union[outputs.Results, outputs.LoopInstanceResult]:
    """Conducts a full measurement using the chosen algorithm.

    Function works differently depending on the measurement method
    the user has entered.

    For alternating temperatures or manual entry method the settings
    log gets configured and updated, the measurement and bias id is
    found, and then the hot and cold measurements are taken in
    whichever order is quicker. The results are returned as a results
    object.

    For all cold then all hot measurements, if the measurement is cold
    the measurement and bias ids are found. A hot or cold measurement
    is made, and that hot or cold measurement is returned.

    Args:
        settings: Settings for the measurement session.
        res_managers: The resource managers for the instruments in use.
        trimmed_input_data: The trimmed loss/calibration input data.
        hot_cold_count: Either 'Hot' or 'Cold', decides whether the
            measurement returns a Hot or Cold object from the
            measurement loop. Only relevant in all cold then all hot
            mode.

    Returns:
        An object containing the results from the measurement ready to
        be saved. Returned for both alternating temperatures and manual
        entry measurement algorithms. Or an object containing the
        measured power, load, LNA, and extra temperatures. Relevant to
        all cold to all hot measurement algorithm.
    """

    # region Unpack from passed variables
    log = logging.getLogger(__name__)
    meas_settings = settings.meas_settings
    tc_settings = settings.instr_settings.temp_ctrl_settings
    sig_gen_settings = settings.instr_settings.sig_gen_settings
    instr_settings = settings.instr_settings
    # endregion

    # region Handle Alt Temperatures / Manual Entry Measurements.
    if meas_settings.measure_method in ['AT', 'MEM']:
        # region Get current temperature of the load
        if res_managers.tc_rm is not None:
            init_temp = util.safe_query(
                f'KRDG? {tc_settings.load_lsch}', instr_settings.buffer_time,
                res_managers.tc_rm, 'lakeshore', True)
        else:
            init_temp = 20
        # endregion

        log.info('Deciding whether to start hot or cold...')
        log.info(f'Initial temperature = {init_temp}K')

        # region Carry out measurement closest to initial temperature.
        # Heat up or cool down and do next one
        loop_res = _closest_temp_then_other(init_temp, res_managers, settings)
        # endregion

        # region Save and return results.
        standard_results = outputs.Results(
            outputs.LoopPair(loop_res[1], loop_res[0]),
            outputs.ResultsMetaInfo(
                meas_settings.comment, sig_gen_settings.freq_array,
                meas_settings.order, meas_settings.is_calibration,
                meas_settings.analysis_bws,
                trimmed_input_data.trimmed_loss,
                trimmed_input_data.trimmed_cal_data))

        return standard_results
        # endregion
    # endregion

    # region Handle All Cold Then All Hot.
    if meas_settings.measure_method == 'ACTAH':
        # region If all cold then all hot measurement, start cold.
        if hot_cold_count == 0:
            cold = _meas_loop(settings, 'Cold', res_managers)
            return cold
        # endregion

        # region Once cold measurements done, start on hot measurements.
        if hot_cold_count == 1:
            hot = _meas_loop(settings, 'Hot', res_managers)
            return hot
        # endregion

        # region Handle variable error.
        raise Exception('')
        # endregion
        # endregion
    # endregion

    # region Handle Calibration Measurement.
    if meas_settings.measure_method == 'Calibration':
        # region Get current temperature of the load
        if res_managers.tc_rm is not None:
            init_temp = util.safe_query(
                f'KRDG? {tc_settings.load_lsch}', instr_settings.buffer_time,
                res_managers.tc_rm, 'lakeshore', True)
        else:
            init_temp = 20.0
        log.info('Deciding whether to start hot or cold...')
        log.info(f'Initial temperature = {init_temp}K')
        # endregion

        # region Carry out measurement closest to initial temperature.
        # Heat up or cool down and do next one
        loop_res = _closest_temp_then_other(init_temp, res_managers, settings)
        # endregion

        # region Save and return results.
        calibration_results = outputs.Results(
            outputs.LoopPair(loop_res[1], loop_res[0]),
            outputs.ResultsMetaInfo(
                meas_settings.comment, sig_gen_settings.freq_array,
                meas_settings.order, meas_settings.is_calibration,
                meas_settings.analysis_bws,
                trimmed_input_data.trimmed_loss))

        return calibration_results
        # endregion
    # endregion
    raise Exception('')
