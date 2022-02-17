# -*- coding: utf-8 -*-
"""measurement.py - Methods to carry out a single measurement.

This module contains the method to take a single measurement set.
External modules should call the measurement() function, which
depending on the passed variables, will call the _meas_loop()
function in specific ways. The measurement loop will record the
temperatures/powers required for the y factor measurement at each
frequency. They are stored as hot/cold results objects as found
in output_classes.py and returned.
"""

# region Import modules.
from __future__ import annotations
from typing import (Union, Optional)
import random as rd
import time

import numpy as np
import progressbar as pb

import heater_ctrl as hc
import instr_classes as ic
import output_classes as oc
import settings_classes as sc
import util as ut
# endregion


def _get_temp_target(
        hot_or_cold: str, ls_settings: ic.TempControllerSettings) -> float:
    if hot_or_cold == 'Cold':
        temp_target = float(ls_settings.cold_target)
    elif hot_or_cold == 'Hot':
        temp_target = float(ls_settings.hot_target)
    else:
        raise Exception('Hot or cold not passed correctly')
    print(f'Initiating {hot_or_cold.lower()} measurement '
          f'- setting load to {temp_target}K')
    return temp_target


def _get_temps(tc_rm, temp_target: float,
               instr_settings: ic.InstrumentSettings) -> list[float]:
    # region Get and return temperatures for requested temp sensors.
    buffer_time = instr_settings.buffer_time
    ls_settings = instr_settings.temp_ctrl_settings
    if tc_rm is not None:
        # region Measure and store load, lna, and extra sensor temps.
        tc_rm.write(f'SCAN {ls_settings.load_lsch},0')
        time.sleep(5)
        load_temp = ut.safe_query(
            f'KRDG? {ls_settings.load_lsch}', buffer_time, tc_rm,
            'lakeshore', True)
        tc_rm.write(f'SCAN {ls_settings.lna_lsch},0')
        time.sleep(5)
        lna_temp = ut.safe_query(
            f'KRDG? {ls_settings.lna_lsch}', buffer_time, tc_rm,
            'lakeshore', True)
        if ls_settings.extra_sensors_en:
            tc_rm.write(f'SCAN {ls_settings.extra_1_lsch},0')
            time.sleep(5)
            extra_1_temp = ut.safe_query(
                f'KRDG? {ls_settings.extra_1_lsch}', buffer_time, tc_rm,
                'lakeshore', True)
            tc_rm.write(f'SCAN {ls_settings.extra_2_lsch},0')
            time.sleep(5)
            extra_2_temp = ut.safe_query(
                f'KRDG? {ls_settings.extra_2_lsch}', buffer_time, tc_rm,
                'lakeshore', True)
        else:
            extra_1_temp = 'NA'
            extra_2_temp = 'NA'
        tc_rm.write(f'SCAN {ls_settings.load_lsch},0')
        time.sleep(5)
        # endregion

    # region Provide dummy temperature measurement values for debugging.
    else:
        load_temp = temp_target
        lna_temp = 21
        if ls_settings.extra_sensors_en:
            extra_1_temp = 40
            extra_2_temp = 30
        else:
            extra_1_temp = 'NA'
            extra_2_temp = 'NA'
    # endregion

    return [load_temp, lna_temp, extra_1_temp, extra_2_temp]
    # endregion


def _meas_loop(
        settings: sc.Settings, hot_or_cold: str,
        res_managers: ic.ResourceManagers, prev_meas_same_temp: bool = False
        ) -> oc.LoopInstanceResult:
    """The measurement algorithm for each hot or cold measurement.

    Checks if hot or cold, sets load to target temp, then moves through
    intermediate frequencies taking power and temperature at each one.
    Returns an object containing the measured power, and load and lna
    temperatures.

    Args:
        hot_or_cold: Either 'Hot' or 'Cold', decides which target
            temperature to set the lakeshore to.
        res_managers: The resource managers for all the instruments
            being used in this measurement.
        prev_meas_same_temp: Determines whether the stability check is
            needed. Always false except for during ACTAH measurements
            when the previous measurement was at the requested
            temperature.

    Returns:
        Object containing measured power, and load/lna/extra
        temperatures for either hot/cold temperature measurement.
    """

    # region Unpack from passed objects and initialise arrays.
    ls_settings = settings.instr_settings.temp_ctrl_settings
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
    temp_target = _get_temp_target(hot_or_cold, ls_settings)
    # endregion

    if tc_rm is not None and not prev_meas_same_temp:
        # region Set Lakeshore to target temp, wait for stabilisation.
        hc.set_temp(tc_rm, temp_target, 'load')
        hc.load_temp_stabilisation(
            tc_rm, ls_settings.load_lsch, temp_target)
        # endregion

    pre_loop_temps = _get_temps(tc_rm, temp_target, settings.instr_settings)
    pre_loop_lna_temp = pre_loop_temps[1]
    pre_loop_extra_1_temp = pre_loop_temps[2]
    pre_loop_extra_2_temp = pre_loop_temps[3]
    # endregion

    # region Set spec an for single measurement mode.
    if spec_an_rm is not None:
        spec_an_rm.write('INIT:CONT 0')
        time.sleep(buffer_time)
    # endregion

    # region Sweep requested frequencies measuring power and load temp.
    print(f'Temperature stable at {pre_loop_temps[0]}K')
    print('Beginning frequency sweep:')
    pwr_lvl_cnt = 0
    pbar = pb.ProgressBar(max_value=len(inter_freqs_array)).start()
    for i, inter_frequency in enumerate(inter_freqs_array):

        # region Prep spec an for next measurement by resetting.
        if spec_an_rm is not None:
            spec_an_rm.write('INIT:IMM')
            time.sleep(buffer_time)
        # endregion

        # region Set signal generator to intermediate frequency.
        if sig_gen_rm is not None and sig_gen_settings.vna_or_sig_gen == 'vna':
            sig_gen_rm.write(f':SENSE:FREQ:CW {inter_frequency} GHz')
        elif sig_gen_rm is not None and \
                sig_gen_settings.vna_or_sig_gen == 'sig gen':
            sig_gen_rm.write(
                f'PL {sig_gen_settings.sig_gen_pwr_lvls[pwr_lvl_cnt]} DM')
            pwr_lvl_cnt += 1
            time.sleep(buffer_time)
            sig_gen_rm.write(f'CW {inter_frequency} GZ')
        # endregion

        # region Measure and store marker power at requested frequency.
        if spec_an_rm is not None:
            ut.safe_query('*OPC?', buffer_time, spec_an_rm, 'spec an')
            marker_power = ut.safe_query(
                ':CALC:MARK1:Y?', buffer_time, spec_an_rm, 'spec an')
            powers.append(float(marker_power.strip()))
        else:
            if hot_or_cold == 'Hot':
                marker_power = -50 + round(rd.uniform(1.2, 2), 2)
            else:
                marker_power = -50 + round(rd.uniform(0.3, 0.6), 2)
            powers.append(marker_power)
        # endregion

        # region Store pre-loop temperatures, and during loop load temp.
        if tc_rm is not None:
            load_temp = ut.safe_query(
                f'KRDG? {ls_settings.load_lsch}', buffer_time, tc_rm,
                'lakeshore', True)
        else:
            load_temp = temp_target + (round(rd.uniform(-0.1, 0.1), 2))

        load_temps.append(load_temp)
        pre_loop_lna_temps.append(pre_loop_lna_temp)
        pre_loop_extra_1_temps.append(pre_loop_extra_1_temp)
        pre_loop_extra_2_temps.append(pre_loop_extra_2_temp)

        # endregion

        # region Update progress bar
        pbar.update(i)
        # endregion
    pbar.update(len(inter_freqs_array))
    pbar.finish()
    # endregion

    # region Put spec an in continuous measurement mode.
    if spec_an_rm is not None:
        spec_an_rm.write('INIT:CONT 1')
    # endregion

    # region Measure post loop lna temperature.
    post_loop_temps = _get_temps(tc_rm, temp_target, settings.instr_settings)
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
    print(f'{hot_or_cold} power measurement complete.')
    return oc.LoopInstanceResult(
        hot_or_cold, powers, load_temps, lna_temps,
        oc.PrePostTemps(pre_loop_lna_temps, post_loop_lna_temps,
                         pre_loop_extra_1_temps, post_loop_extra_1_temps,
                         pre_loop_extra_2_temps, post_loop_extra_2_temps))
    # endregion


def _closest_temp_then_other(
        init_temp: float, res_managers: ic.ResourceManagers,
        settings: sc.Settings) -> list[oc.LoopInstanceResult]:

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
        settings: sc.Settings,
        res_managers: ic.ResourceManagers,
        trimmed_input_data: sc.TrimmedInputs,
        hot_cold_count: Optional[int] = None,
        prev_meas_same_temp: Optional[bool] = False
        ) -> Union[oc.Results, oc.LoopInstanceResult]:
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
        trimmed_input_data:
        hot_cold_count: Either 'Hot' or 'Cold', decides whether the
            measurement returns a Hot or Cold object from the
            measurement loop. Only relevant in all cold then all hot
            mode.
        prev_meas_same_temp: Default false, but for ACTAH if set to true
            will skip the stabilisation step in the measurement loop.

    Returns:
        An object containing the results from the measurement ready to
        be saved. Returned for both alternating temperatures and manual
        entry measurement algorithms. Or an object containing the
        measured power, load, LNA, and extra temperatures. Relevant to
        all cold to all hot measurement algorithm.
    """

    # region Unpack from passed variables
    meas_settings = settings.meas_settings
    tc_settings = settings.instr_settings.temp_ctrl_settings
    sig_gen_settings = settings.instr_settings.sig_gen_settings
    instr_settings = settings.instr_settings
    # endregion

    # region Handle Alt Temperatures / Manual Entry Measurements.
    if meas_settings.measure_method in ['AT', 'MEM']:
        # region Get current temperature of the load
        if res_managers.tc_rm is not None:
            init_temp = ut.safe_query(
                'KRDG? {ls_settings.load_lsch', instr_settings.buffer_time,
                res_managers.tc_rm, 'lakeshore', True)
        else:
            init_temp = 20
        # endregion

        print('Deciding whether to start hot or cold...')
        print(f'Initial temperature = {init_temp}K')

        # region Carry out measurement closest to initial temperature.
        # Heat up or cool down and do next one
        loop_res = _closest_temp_then_other(init_temp, res_managers, settings)
        # endregion

        # region Save and return results.
        standard_results = oc.Results(
            oc.LoopPair(loop_res[1], loop_res[0]),
            oc.ResultsMetaInfo(
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
            cold = _meas_loop(
                settings, 'Cold', res_managers, prev_meas_same_temp)
            print('Cold measurement finished')
            return cold
        # endregion

        # region Once cold measurements done, start on hot measurements.
        if hot_cold_count == 1:
            hot = _meas_loop(
                settings, 'Hot', res_managers, prev_meas_same_temp)
            print('Hot measurement finished')
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
            init_temp = ut.safe_query(
                f'KRDG? {tc_settings.load_lsch}', instr_settings.buffer_time,
                res_managers.tc_rm, 'lakeshore', True)
        else:
            init_temp = 20.0
        print('Deciding whether to start hot or cold...')
        print(f'Initial temperature = {init_temp}K')
        # endregion

        # region Carry out measurement closest to initial temperature.
        # Heat up or cool down and do next one
        loop_res = _closest_temp_then_other(init_temp, res_managers, settings)
        # endregion

        # region Save and return results.
        calibration_results = oc.Results(
            oc.LoopPair(loop_res[1], loop_res[0]),
            oc.ResultsMetaInfo(
                meas_settings.comment, sig_gen_settings.freq_array,
                meas_settings.order, meas_settings.is_calibration,
                meas_settings.analysis_bws,
                trimmed_input_data.trimmed_loss))

        return calibration_results
        # endregion
    # endregion

    raise Exception('')
