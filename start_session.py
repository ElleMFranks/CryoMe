# -*- coding: utf-8 -*-
"""start_session.py - Trigger a measurement with given settings.
"""

# region Import modules
from __future__ import annotations

import numpy as np
import pandas as pd
import pyvisa as pv

import bias_ctrl as bc
import chain_select as cs
import instr_classes as ic
import lna_classes as lc
import meas_algorithms as ma
import settings_classes as scl
import socket_communication as sc
import util as ut
# endregion


def _trigger_algorithm(settings, lna_biases, lna_nominals, res_managers,
                       trimmed_input_data):
    meas_settings = settings.meas_settings

    # region Alternating Temperatures.
    if meas_settings.measure_method == 'AT':
        ma.alternating_temps(
            settings, lna_biases, lna_nominals, res_managers,
            trimmed_input_data)
    # endregion

    # region All Cold Then All Hot.
    elif meas_settings.measure_method == 'ACTAH':
        ma.all_cold_to_all_hot(
            settings, lna_biases, lna_nominals, res_managers,
            trimmed_input_data)
    # endregion

    # region Manual Entry Measurement.
    elif meas_settings.measure_method == 'MEM':
        # region Set and measure manual LNA biases.
        # region Set up LNA1.
        lna_1_man = meas_settings.direct_lnas.manual_lna_settings.lna_1_man
        if settings.instr_settings.bias_psu_settings.bias_psu_en:
            bc.bias_set(res_managers.psu_rm, lna_1_man,
                        settings.instr_settings.bias_psu_settings,
                        settings.instr_settings.buffer_time)
            lna_1_man.lna_measured_column_data(res_managers.psu_rm)
        # endregion
        
        # region Set up LNA2.
        if meas_settings.lna_cryo_layout.lnas_per_chain == 2:
            lna_2_man = meas_settings.direct_lnas.manual_lna_settings.lna_2_man
            if settings.instr_settings.bias_psu_settings.bias_psu_en:
                bc.bias_set(res_managers.psu_rm, lna_2_man,
                            settings.instr_settings.bias_psu_settings,
                            settings.instr_settings.buffer_time)
                lna_2_man.lna_measured_column_data(res_managers.psu_rm)
        else:
            lna_2_man = None
        # endregion

        lna_man_biases = [lna_1_man, lna_2_man]

        ma.manual_entry_measurement(
            settings, lna_man_biases, res_managers, trimmed_input_data)
        # endregion
    # endregion

    # region Calibration.
    elif meas_settings.measure_method == 'Calibration':
        ma.calibration_measurement(
            settings, res_managers, trimmed_input_data.trimmed_loss)
    # endregion


def _input_trim(
        untrimmed_data: np.ndarray, freq_array: list,
        is_cal_data: bool = False) -> list:
    """Trims data to the requested frequency points."""
    trimmed_data = []
    for freq in freq_array:
        freq_diff = []
        for j, _ in enumerate(untrimmed_data):
            diff = abs(untrimmed_data[j, 0] - freq)
            freq_diff.append(diff)
        if is_cal_data:
            trimmed_data.append(untrimmed_data[np.argmin(freq_diff), :])
        else:
            trimmed_data.append(untrimmed_data[np.argmin(freq_diff), 1])
    return trimmed_data


def _comment_handling(comment_en: bool) -> str:
    """Handles optional user entry comment."""
    # region Measurement comments handling
    if comment_en:
        comment = input("Please input measurement comment: ")
    else:
        print('Measurement comment not enabled')
        comment = ''
    return comment
    # endregion


def _res_manager_setup(
        instr_settings: ic.InstrumentSettings) -> ic.ResourceManagers:
    """Sets up resource managers for instrumentation."""
    # region Unpack settings
    sig_an_settings = instr_settings.sig_an_settings
    sig_gen_settings = instr_settings.sig_gen_settings
    temp_ctrl_settings = instr_settings.temp_ctrl_settings
    bias_psu_settings = instr_settings.bias_psu_settings
    # endregion

    # region Initialise resource managers.
    res_manager = pv.ResourceManager()
    res_manager.list_resources()
    # endregion

    sig_an_rm = None
    sig_gen_rm = None
    temp_ctrl_rm = None
    psu_rm = None

    # region Set up spectrum analyser.
    if sig_an_settings.sig_an_en:
        sig_an_rm = res_manager.open_resource('GPIB1::18::INSTR')
    # endregion

    # region Set up signal generator.
    if sig_gen_settings.sig_gen_en and \
            sig_gen_settings.vna_or_sig_gen == 'vna':
        sig_gen_rm = res_manager.open_resource('GPIB1::16::INSTR')
    elif sig_gen_settings.sig_gen_en and \
            sig_gen_settings.vna_or_sig_gen == 'sig gen':
        sig_gen_rm = res_manager.open_resource('GPIB1::8::INSTR')
        print(ut.safe_query(
            'OI', instr_settings.buffer_time, sig_gen_rm, 'sig gen'))
    # endregion

    # region Set up lakeshore temperature controller.
    if temp_ctrl_settings.temp_ctrl_en:
        temp_ctrl_rm = res_manager.open_resource('GPIB1::12::INSTR')
    # endregion

    # region Set up PSX bias supply.
    if bias_psu_settings.bias_psu_en:
        psu_rm = sc.InstrumentSocket("10.99.9.58", 5025, timeout=10000)
        # Ensure psx is initialised safely
        if bias_psu_settings.psu_safe_init:
            with psu_rm.open_instrument():
                bc.psu_safe_init(
                    psu_rm, instr_settings.buffer_time,
                    ic.PSULimits(bias_psu_settings.d_i_lim,
                                 bias_psu_settings.v_step_lim),
                    bias_psu_settings.g_v_lower_lim)
    # endregion

    # region Initialise each instrument with defined settings.
    if sig_an_settings.sig_an_en:
        sig_an_settings.spec_an_init(
            sig_an_rm, instr_settings.buffer_time)
    if sig_gen_settings.sig_gen_en:
        if sig_gen_settings.vna_or_sig_gen == 'vna':
            sig_gen_settings.vna_init(sig_gen_rm, instr_settings.buffer_time)
    if temp_ctrl_settings.temp_ctrl_en:
        temp_ctrl_settings.lakeshore_init(
            temp_ctrl_rm, instr_settings.buffer_time, 'warm up')
    if bias_psu_settings.bias_psu_en:
        with psu_rm.open_instrument():
            bias_psu_settings.psx_init(
                psu_rm, instr_settings.buffer_time, 0,
                bias_psu_settings.g_v_lower_lim)
    # endregion
    print('Instrumentation initialised')
    # endregion

    # region Create class instance to keep ResourceManagers together.
    return ic.ResourceManagers(sig_an_rm, sig_gen_rm, temp_ctrl_rm, psu_rm)
    # endregion


def start_session(settings: scl.Settings) -> None:
    """Begins a session using the settings passed from Cryome.

    Args:
        settings: Contains all the settings for the session.
    """
    # region Unpack classes.
    instr_settings = settings.instr_settings
    bias_psu_settings = instr_settings.bias_psu_settings
    meas_settings = settings.meas_settings
    # endregion

    # region Measurement comments handling
    meas_settings.comment = _comment_handling(meas_settings.comment_en)
    # endregion

    # region Initialise instrumentation
    print('Initialising instrumentation...')

    # region Ensure correct cryostat channel is set
    if instr_settings.switch_settings.switch_en:
        cs.cryo_chain_switch(0.5, meas_settings)
    meas_settings.lna_ids = meas_settings.lna_cryo_layout.cryo_chain
    # endregion

    # region Initialise resource managers.
    res_managers = _res_manager_setup(instr_settings)
    # endregion

    # region Set back end (cryostat and room-temperature) LNAs.
    cs.back_end_lna_setup(settings, res_managers.psu_rm)
    # endregion

    # region Set up nominal LNA bias points.
    # Initialise LNA biases to nominal points before sweeping stages and
    # set up back end LNAs. Defined nominal stages as two separate LNAs
    # in case of different voltage drops across wires.
    if not meas_settings.is_calibration:
        lna_nominals = lc.NominalLNASettings(settings)
        lna_biases = [lna_nominals.lna_1_nom_bias, lna_nominals.lna_2_nom_bias]

    # region Trim loss/cal/sig gen power arrays for requested frequency.
    # Ensure loss, calibration, and sig gen power arrays are correct for
    # number of frequency points.

    # region Trim input calibration data.
    freq_array = instr_settings.sig_gen_settings.freq_array
    if not meas_settings.is_calibration:
        untrimmed_cal_data = np.array(
            pd.read_csv(settings.file_struc.in_cal_file_path, header=3))
        trimmed_cal_data = _input_trim(untrimmed_cal_data, freq_array, True)
    else:
        trimmed_cal_data = None
    # endregion

    # region Trim loss.
    untrimmed_loss = np.array(pd.read_csv(settings.file_struc.loss_path))
    trimmed_loss = _input_trim(untrimmed_loss, freq_array)

    # region Save trimmed loss and calibration data as an object.
    trimmed_input_data = scl.TrimmedInputs(trimmed_loss, trimmed_cal_data)
    # endregion
    # endregion

    # region Trim sig gen powers.
    if instr_settings.sig_gen_settings.vna_or_sig_gen == 'sig gen':
        untrimmed_pwr = np.array(pd.read_csv(settings.file_struc.pwr_lvls))
        trimmed_pwr = _input_trim(untrimmed_pwr, freq_array)
        instr_settings.sig_gen_settings.set_sig_gen_pwr_lvls(trimmed_pwr)
    # endregion
    # endregion

    # region Get/set session ID.
    meas_settings.config_session_id(settings.file_struc)
    # endregion

    # region Trigger measurement with power supply enabled.
    if bias_psu_settings.bias_psu_en:
        with res_managers.psu_rm.open_instrument():
            _trigger_algorithm(settings, lna_biases, lna_nominals,
                               res_managers, trimmed_input_data)
    # endregion
    # region If PSU not enabled, trigger measurement without it.
    elif not meas_settings.is_calibration:
        _trigger_algorithm(settings, lna_biases, lna_nominals,
                           res_managers, trimmed_input_data)
    else:
        _trigger_algorithm(settings, None, None, res_managers, trimmed_input_data)
    # endregion

    # region Turn PSX off safely
    if bias_psu_settings.bias_psu_en:
        with res_managers.psu_rm.open_instrument():
            bc.psu_safe_init(
                res_managers.psu_rm, instr_settings.buffer_time,
                ic.PSULimits(bias_psu_settings.d_i_lim,
                             bias_psu_settings.v_step_lim),
                bias_psu_settings.g_v_lower_lim)
    # endregion

    # region Close resource managers.
    del res_managers
    # endregion
