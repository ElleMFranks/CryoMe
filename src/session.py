# -*- coding: utf-8 -*-
"""start_session.py - Trigger a measurement with given settings.

The measurement system is initialised from here, inputs are handled, and
the requested algorithm is called.
"""

# region Import modules
from __future__ import annotations
import logging

import numpy as np
import pandas as pd
import pyvisa

import bias_ctrl
import chain_select
import instruments
import lnas
import meas_algorithms
import config_handling
# endregion


def _trigger_algorithm(settings, lna_biases, res_managers,
                       trimmed_input_data) -> None:
    """Triggers the requested measurement algorithm."""

    # region Unpack objects and set up logging.
    log = logging.getLogger(__name__)
    meas_settings = settings.meas_settings
    # endregion

    # region Alternating Temperatures.
    if meas_settings.measure_method == 'AT':
        log.info('Triggered alternating temperature measurement.')
        meas_algorithms.alternating_temps(
            settings, lna_biases, res_managers, trimmed_input_data)
    # endregion

    # region All Cold Then All Hot.
    elif meas_settings.measure_method == 'ACTAH':
        log.info('Triggered all cold then all hot measurement.')
        meas_algorithms.all_cold_to_all_hot(
            settings, lna_biases, res_managers, trimmed_input_data)
    # endregion

    # region Manual Entry Measurement.
    elif meas_settings.measure_method == 'MEM':
        log.info('Triggered manual entry measurement.')
        # region Set and measure manual LNA biases.
        # region Set up LNA1.
        lna_1_man = meas_settings.direct_lnas.manual_lna_settings.lna_1_man
        if settings.instr_settings.bias_psu_settings.bias_psu_en:
            bias_ctrl.adaptive_bias_set(res_managers.psu_rm, lna_1_man,
                               settings.instr_settings.bias_psu_settings,
                               settings.instr_settings.buffer_time)
            lna_1_man.lna_measured_column_data(res_managers.psu_rm)
        # endregion

        # region Set up LNA2.
        if meas_settings.lna_cryo_layout.lnas_per_chain == 2:
            lna_2_man = meas_settings.direct_lnas.manual_lna_settings.lna_2_man
            if settings.instr_settings.bias_psu_settings.bias_psu_en:
                bias_ctrl.adaptive_bias_set(res_managers.psu_rm, lna_2_man,
                                   settings.instr_settings.bias_psu_settings,
                                   settings.instr_settings.buffer_time)
                lna_2_man.lna_measured_column_data(res_managers.psu_rm)
        else:
            lna_2_man = None
        # endregion

        lna_man_biases = [lna_1_man, lna_2_man]

        meas_algorithms.manual_entry_measurement(
            settings, lna_man_biases, res_managers, trimmed_input_data)
        # endregion
    # endregion

    # region Calibration.
    elif meas_settings.measure_method == 'Calibration':
        log.info('Triggered calibration measurement.')
        meas_algorithms.calibration_measurement(
            settings, res_managers, trimmed_input_data.trimmed_loss)
    # endregion


def _input_trim(
        untrimmed_data: np.ndarray, freq_array: list,
        is_cal_data: bool = False) -> list:
    """Trims data to the requested frequency points."""

    # region Trim and return input data.
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
    # endregion


def _comment_handling(comment_en: bool) -> str:
    """Handles optional user entry comment."""

    # region Measurement comments handling
    if comment_en:
        comment = input("Please input measurement comment: ")
    else:
        comment = 'NA'
    return comment
    # endregion


def _res_manager_setup(instr_settings: instruments.InstrumentSettings
                       ) -> instruments.ResourceManagers:
    """Sets up resource managers for instrumentation."""

    # region Unpack settings
    sig_an_settings = instr_settings.sig_an_settings
    sig_gen_settings = instr_settings.sig_gen_settings
    temp_ctrl_settings = instr_settings.temp_ctrl_settings
    bias_psu_settings = instr_settings.bias_psu_settings
    # endregion

    # region Initialise resource managers.
    res_manager = pyvisa.ResourceManager()
    res_manager.list_resources()
    # endregion

    # region Default resource managers to None.
    sig_an_rm = None
    sig_gen_rm = None
    temp_ctrl_rm = None
    psu_rm = None
    # endregion

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
    # endregion

    # region Set up lakeshore temperature controller.
    if temp_ctrl_settings.temp_ctrl_en:
        temp_ctrl_rm = res_manager.open_resource('GPIB1::12::INSTR')
    # endregion

    # region Set up PSX bias supply.
    if bias_psu_settings.bias_psu_en:
        psu_rm = res_manager.open_resource('TCPIP0::10.99.9.58::5025::SOCKET')
        psu_rm.read_termination = '\n'
        psu_rm.write_termination = '\n'
        # Ensure psx is initialised safely
        if bias_psu_settings.psu_safe_init and not bias_psu_settings.skip_psu_init:
            bias_ctrl.psu_safe_init(
                psu_rm, instr_settings.buffer_time,
                instruments.PSULimits(bias_psu_settings.v_step_lim,
                             bias_psu_settings.d_i_lim),
                bias_psu_settings.g_v_lower_lim)
    # endregion

    # region Initialise each instrument with defined settings.
    if sig_an_settings.sig_an_en:
        sig_an_settings.spec_an_init(
            sig_an_rm, instr_settings.buffer_time)
    if sig_gen_settings.sig_gen_en:
        if sig_gen_settings.vna_or_sig_gen == 'vna':
            sig_gen_settings.vna_init(sig_gen_rm, instr_settings.buffer_time)
        elif sig_gen_settings.vna_or_sig_gen == 'sig gen':
            sig_gen_settings.sig_gen_init(
                sig_gen_rm, instr_settings.buffer_time)
    if temp_ctrl_settings.temp_ctrl_en:
        temp_ctrl_settings.lakeshore_init(
            temp_ctrl_rm, instr_settings.buffer_time)
    if bias_psu_settings.bias_psu_en:
        bias_psu_settings.psx_init(
            psu_rm, instr_settings.buffer_time, 0,
            bias_psu_settings.g_v_lower_lim)
    # endregion

    # region Create class instance to keep ResourceManagers together.
    return instruments.ResourceManagers(
        sig_an_rm, sig_gen_rm, temp_ctrl_rm, psu_rm)
    # endregion


def start_session(settings: config_handling.Settings) -> None:
    """Begins a session using the settings passed from Cryome.

    Args:
        settings: Contains all the settings for the session.
    """

    # region Unpack classes and set up logging.
    log = logging.getLogger(__name__)
    instr_settings = settings.instr_settings
    bias_psu_settings = instr_settings.bias_psu_settings
    meas_settings = settings.meas_settings
    # endregion

    # region Measurement comments handling.
    meas_settings.comment = _comment_handling(meas_settings.comment_en)
    # endregion

    # region Trim input data.
    log.info('Trimming input data...')

    # region Trim loss/cal/sig gen power arrays for requested frequency.
    # Ensure loss, calibration, and sig gen power arrays are correct for
    # number of frequency points.
    # region Trim input calibration data.
    freq_array = instr_settings.sig_gen_settings.freq_array
    if not meas_settings.is_calibration:
        untrimmed_cal_data = np.array(
            pd.read_csv(settings.file_struc.in_cal_file_path, header=3))
        trimmed_cal_data = _input_trim(untrimmed_cal_data, freq_array, True)
        log.cdebug('Calibration data trimmed.')
    else:
        trimmed_cal_data = None
    # endregion

    # region Trim loss.
    trimmed_loss = _input_trim(
        np.array(pd.read_csv(settings.file_struc.loss_path)), freq_array)
    log.cdebug('Loss trimmed.')

    # region Save trimmed loss and calibration data as an object.
    trimmed_input_data = config_handling.TrimmedInputs(
        trimmed_loss, trimmed_cal_data)
    # endregion
    # endregion

    # endregion

    # region Trim sig gen powers.
    if instr_settings.sig_gen_settings.vna_or_sig_gen == 'sig gen':
        trimmed_pwr = _input_trim(
            np.array(pd.read_csv(settings.file_struc.pwr_lvls)), freq_array)
        instr_settings.sig_gen_settings.set_sig_gen_pwr_lvls(trimmed_pwr)
        log.cdebug('Signal generator input powers trimmed.')
    # endregion

    log.info('Input data trimmed.')
    # endregion

    # region Initialise instrumentation.
    log.info('Initialising instrumentation...')

    # region Ensure correct cryostat channel is set
    if instr_settings.switch_settings.switch_en:
        chain_select.cryo_chain_switch(0.5, meas_settings)
        log.info('Switch set.')
    # endregion

    # region Set LNA IDs.
    meas_settings.lna_ids = meas_settings.lna_cryo_layout.cryo_chain
    # endregion

    # region Initialise resource managers.
    res_managers = _res_manager_setup(instr_settings)
    # endregion

    log.info('Instrumentation initialised.')
    # endregion

    # region Set back end (cryostat and room-temperature) LNAs.
    if not bias_psu_settings.skip_psu_init:
        check_skip_be_setup = False
        while not check_skip_be_setup:
            user_check = input(
                'Are you sure you want to skip back end LNA biasing setup? (y/n): ')
            if user_check == 'n':
                chain_select.back_end_lna_setup(settings, res_managers.psu_rm)
                check_skip_be_setup = True
            if user_check == 'y':
                check_skip_be_setup = True
    # endregion

    # region Set up nominal LNA bias points.
    log.cdebug('Setting up nominal LNA bias objects...')
    # Initialise LNA biases to nominal points before sweeping stages and
    # set up back end LNAs. Defined nominal stages as two separate LNAs
    # in case of different voltage drops across wires.
    if not meas_settings.is_calibration:
        lna_nominals = lnas.NominalLNASettings(settings)
        lna_biases = [lna_nominals.lna_1_nom_bias, lna_nominals.lna_2_nom_bias]
    else:
        lna_nominals = None
        lna_biases = None
    log.cdebug('Nominal LNA bias objects set up.')
    # endregion

    # region Trigger measurement.
    log.info('Triggering measurement...')
    # region Trigger measurement with power supply enabled.
    if bias_psu_settings.bias_psu_en:
        _trigger_algorithm(settings, lna_biases,
                           res_managers, trimmed_input_data)
    # endregion

    # region If PSU not enabled, trigger measurement without it.
    elif not meas_settings.is_calibration:
        _trigger_algorithm(settings, lna_biases,
                           res_managers, trimmed_input_data)
    else:
        _trigger_algorithm(settings, None, res_managers, trimmed_input_data)
    # endregion
    # endregion

    if not settings.meas_settings.is_calibration:
        config_handling.FileStructure.write_to_file(settings.file_struc.res_log_path, '', 'a', 'row')
        config_handling.FileStructure.write_to_file(settings.file_struc.settings_path, '', 'a', 'row')
    else:
        config_handling.FileStructure.write_to_file(settings.file_struc.cal_settings_path, '', 'a', 'row')
    # region Turn PSX off safely.
    if bias_psu_settings.bias_psu_en:
        log.info('Turning off PSU...')
        bias_ctrl.psu_safe_init(
            res_managers.psu_rm, instr_settings.buffer_time,
            instruments.PSULimits(bias_psu_settings.v_step_lim,
                                  bias_psu_settings.d_i_lim),
            bias_psu_settings.g_v_lower_lim)
        log.info('PSU turned off.')
    # endregion

    # region Close resource managers.
    del res_managers
    # endregion
