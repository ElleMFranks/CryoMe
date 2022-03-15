# -*- coding: utf-8 -*-
"""meas_algorithm.py - Decides how each full measurement happens.

Contains the different measurement algorithms which can be used for the
y factor measurement:

    * All Cold To All Hot (ACTAH)
    * Alternating Temperatures (AT)
    * Manual Entry Measurement (MEM)
    * Calibration

ACTAH and AT are bias sweeping algorithms, MEM is a single measurement
with given parameters, and Calibration is a single measurement with only
the back end LNAs present.
"""

# region Import modules.
from __future__ import annotations
from itertools import product
import copy as cp
import logging

import tqdm

import bias_ctrl
import instruments
import lnas
import measurement
import outputs
import output_saving
import config_handling
# endregion


def all_cold_to_all_hot(
        settings: config_handling.Settings,
        lna_biases: list[lnas.LNABiasSet],
        res_managers: instruments.ResourceManagers,
        trimmed_input_data: config_handling.TrimmedInputs) -> None:
    """Parallel sweep where cold measurements are taken, then hot.

    This  method loops through each drain current for each drain
    voltage for each stage for each lna for cold and then hot
    temperatures. A set of cold measurements is taken for each
    point in the bias sweep, the cryostat temperature is then
    taken up to the hot temperature, and the set of hot for each
    bias point is taken. The results are then processed all at once.

    Args:
        settings: The settings for the measurement instance.
        lna_biases: The biases for the LNAs ut.
        res_managers: An object containing the resource managers for the
            instruments used in the measurement.
        trimmed_input_data: The trimmed loss/calibration data.
    """

    # region Unpack objects, instantiate arrays, and set up logging.
    log = logging.getLogger(__name__)
    sweep_settings = settings.sweep_settings
    meas_settings = settings.meas_settings
    lna_1_bias = lna_biases[0]
    lna_2_bias = lna_biases[1]
    d_v_nom = settings.sweep_settings.d_v_nominal
    d_i_nom = settings.sweep_settings.d_i_nominal
    hot_array = []
    cold_array = []
    hot_cold = [0, 1]
    lna_1_array = []
    lna_2_array = []
    prev_lna_ut = 0
    states = list(product(
        hot_cold, sweep_settings.lna_sequence, sweep_settings.stage_sequence,
        sweep_settings.d_v_sweep, sweep_settings.d_i_sweep))
    # endregion

    # region Loop through states measuring at each.
    print('')
    for i, state in enumerate(tqdm.tqdm(
        states, ncols=110, leave=False, desc="Sweep Prog",
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} '
                   '[Elapsed: {elapsed}, To Go: {remaining}]{postfix}\n')):

        # region Unpack state object and set additional variable.
        temp_ut = state[0]
        lna_ut = state[1]
        stage_ut = state[2]
        d_v_ut = state[3]
        d_i_ut = state[4]
        # endregion

        # region Log measurement state.
        if temp_ut == 0:
            log.info(f'Measurement: {i + 1} - HotOrCold: Cold - LNA: {lna_ut} '
                     f'- DV: {d_v_ut:.2f} V - DI: {d_i_ut:.2f} mA.')
        else:
            log.info(f'Measurement: {i + 1} - HotOrCold: Hot - LNA:{lna_ut} '
                     f'- DV:{d_v_ut:.2f} V - DI:{d_i_ut:.2f} mA.')
        # endregion

        # region Configure LNA biasing settings to send to bias control.

        # region LNA 1.
        if lna_ut == 1:
            lna_1_bias.sweep_setup(
                stage_ut, d_v_ut, d_i_ut, d_v_nom, d_i_nom)

            if meas_settings.lna_cryo_layout.lnas_per_chain == 2:
                lna_2_bias.nominalise(d_v_nom, d_i_nom)
        # endregion

        # region LNA 2.
        elif lna_ut == 2 and \
                meas_settings.lna_cryo_layout.lnas_per_chain == 2:
            lna_1_bias.nominalise(d_v_nom, d_i_nom)

            lna_2_bias.sweep_setup(
                stage_ut, d_v_ut, d_i_ut, d_v_nom, d_i_nom)
        # endregion

        # endregion

        # region Set and store LNA 1 PSU settings.
        if res_managers.psu_rm is not None and (
                lna_ut == 1 or i == 0 or lna_ut != prev_lna_ut):
            bias_ctrl.bias_set(
                res_managers.psu_rm, lna_1_bias,
                settings.instr_settings.bias_psu_settings,
                settings.instr_settings.buffer_time)

        lna_1_bias.lna_measured_column_data(res_managers.psu_rm)

        if temp_ut == 0:
            lna_1_array.append(cp.deepcopy(lna_1_bias))
        # endregion

        # region Set and store LNA 2 PSU settings.
        if meas_settings.lna_cryo_layout.lnas_per_chain == 2:
            if res_managers.psu_rm is not None and (
                    lna_ut == 2 or i == 0 or lna_ut != prev_lna_ut):
                bias_ctrl.bias_set(
                    res_managers.psu_rm, lna_2_bias,
                    settings.instr_settings.bias_psu_settings,
                    settings.instr_settings.buffer_time)

            lna_2_bias.lna_measured_column_data(res_managers.psu_rm)
            lna_2_array.append(cp.deepcopy(lna_2_bias))

        else:
            if temp_ut == 0:
                lna_2_array.append(cp.deepcopy(lna_2_bias))
        # endregion

        # region Trigger measurement
        if temp_ut == 0:
            cold_array.append(measurement.measurement(
                settings, res_managers, trimmed_input_data, temp_ut))

        else:
            hot_array.append(measurement.measurement(
                settings, res_managers, trimmed_input_data, temp_ut))

        print('\n')

        prev_lna_ut = lna_ut
        # endregion
    print('')
    # endregion

    # region Analyse and save each set of hot and cold results.
    log.info('Starting results saving.')
    freq_array = settings.instr_settings.sig_gen_settings.freq_array
    for i, _ in enumerate(hot_array):
        result = outputs.Results(
                     outputs.LoopPair(cold_array[i], hot_array[i]),
                     outputs.ResultsMetaInfo(
                         meas_settings.comment, freq_array,
                         meas_settings.order, meas_settings.is_calibration,
                         meas_settings.analysis_bws,
                         trimmed_input_data.trimmed_loss,
                         trimmed_input_data.trimmed_cal_data))

        output_saving.save_standard_results(
            settings, result, i + 1, lna_1_array[i], lna_2_array[i])
    log.info('All results saved.')
    # endregion


def alternating_temps(
        settings: config_handling.Settings,
        lna_biases: list[lnas.LNABiasSet],
        res_managers: instruments.ResourceManagers,
        trimmed_input_data: config_handling.TrimmedInputs) -> None:
    """Series sweep where temp is alternated between measurements.

    For each LNA, for each stage, for each drain voltage, for each
    drain current a hot or cold temperature measurement is made, the
    temperature is then taken to the alternative temperature and
    another measurement is made.  Each individual measurement is saved
    as the measurement progresses.  This sequential method is less at
    risk of going wrong as should the measurement be interrupted only
    the measurement being done at that instant is lost, instead of all
    the results.

    Args:
        settings: The settings for the measurement session.
        lna_biases: The target bias values for the LNAs in the cryostat
            chain.
        res_managers: An object containing the resource managers for the
            instruments used in the measurement.
        trimmed_input_data: The trimmed loss/calibration input data.
    """

    # region Unpack classes and set up logging.
    log = logging.getLogger(__name__)
    sweep_settings = settings.sweep_settings
    meas_settings = settings.meas_settings
    lna_1_bias = lna_biases[0]
    lna_2_bias = lna_biases[1]
    d_v_nom = settings.sweep_settings.d_v_nominal
    d_i_nom = settings.sweep_settings.d_i_nominal
    prev_lna_ut = None
    states = list(product(
            sweep_settings.lna_sequence, sweep_settings.stage_sequence,
            sweep_settings.d_v_sweep, sweep_settings.d_i_sweep))
    # endregion

    # region Iterate measuring and saving lna/stage/bias value states.
    print('')
    for i, state in enumerate(tqdm.tqdm(
        states, ncols=110, leave=False, desc="Sweep Prog",
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} '
                   '[Elapsed: {elapsed}, To Go: {remaining}]{postfix}\n')):

        if i >= sweep_settings.alt_temp_sweep_skips:
            # region Get iterables from state object.
            lna_ut = state[0]
            stage_ut = state[1]
            d_v_ut = state[2]
            d_i_ut = state[3]
            # endregion

            # region Configure LNA Biasing.
            # region LNA 1.
            if lna_ut == 1:

                lna_1_bias.sweep_setup(
                    stage_ut, d_v_ut, d_i_ut, d_v_nom, d_i_nom)

                if meas_settings.lna_cryo_layout.lnas_per_chain == 2:
                    lna_2_bias.nominalise(d_v_nom, d_i_nom)
            # endregion

            # region LNA 2.
            elif lna_ut == 2 and \
                    meas_settings.lna_cryo_layout.lnas_per_chain == 2:
                lna_1_bias.nominalise(d_v_nom, d_i_nom)

                lna_2_bias.sweep_setup(
                    stage_ut, d_v_ut, d_i_ut, d_v_nom, d_i_nom)
            # endregion
            # endregion

            # region Set and enable power supply.
            # region LNA 1.
            if res_managers.psu_rm is not None and (
                    lna_ut == 1 or i == 0 or lna_ut != prev_lna_ut):
                bias_ctrl.bias_set(
                    res_managers.psu_rm, lna_1_bias,
                    settings.instr_settings.bias_psu_settings,
                    settings.instr_settings.buffer_time)

            lna_1_bias.lna_measured_column_data(res_managers.psu_rm)
            # endregion

            # region LNA 2.
            if meas_settings.lna_cryo_layout.lnas_per_chain == 2:
                if res_managers.psu_rm is not None and (
                        lna_ut == 2 or i == 0 or lna_ut != prev_lna_ut):
                    bias_ctrl.bias_set(
                        res_managers.psu_rm, lna_2_bias,
                        settings.instr_settings.bias_psu_settings,
                        settings.instr_settings.buffer_time)
                lna_2_bias.lna_measured_column_data(res_managers.psu_rm)
            # endregion
            # endregion

            # region Trigger measurement.
            standard_results = measurement.measurement(
                settings, res_managers, trimmed_input_data)
            # endregion

            # region Analyse and save results.
            output_saving.save_standard_results(
                settings, standard_results, i + 1, lna_1_bias, lna_2_bias)
            # endregion

            # region Update status and continue sweep.
            log.info('Measurement finished, incrementing bias sweep')
            # endregion

        log.info('All results saved.')
        prev_lna_ut = lna_ut
        # endregion


def calibration_measurement(settings: config_handling.Settings,
                            res_managers: instruments.ResourceManagers,
                            trimmed_loss: list[float]) -> None:
    """Triggers and saves a calibration measurement.

    Calibration measurements are output into the calibration folder into
    a folder for whichever specific chain the calibration is done with.
    Other measurements are corrected with the output of this type of
    measurement. Only the backend LNAs should be present during this
    measurement.

    Args:
        settings:
        res_managers: An object containing the resource managers for the
            instruments used in the measurement.
        trimmed_loss: The loss to be accounted for at each frequency
            point as obtained by interpolation/decimation of the
            measured loss over frequency.
    """
    # region Trigger measurement and save results.
    meas_settings = settings.meas_settings
    be_lna_settings = meas_settings.direct_lnas.be_lna_settings

    rtbe_lna_bias = be_lna_settings.rtbe_chain_a_lna
    rtbe_stg = be_lna_settings.rtbe_chain_a_lna.stage_1

    if settings.instr_settings.switch_settings.cryo_chain == 1:
        crbe_lna_bias = be_lna_settings.crbe_chain_1_lna
        crbe_stg = be_lna_settings.crbe_chain_1_lna.stage_1

    elif settings.instr_settings.switch_settings.cryo_chain == 2:
        crbe_lna_bias = be_lna_settings.crbe_chain_2_lna
        crbe_stg = be_lna_settings.crbe_chain_2_lna.stage_1

    elif settings.instr_settings.switch_settings.cryo_chain == 3:
        crbe_lna_bias = be_lna_settings.crbe_chain_3_lna
        crbe_stg = be_lna_settings.crbe_chain_3_lna.stage_1
    else:
        raise Exception('Cryostat chain not set.')

    calibration_result = measurement.measurement(
        settings, res_managers, config_handling.TrimmedInputs(trimmed_loss))

    be_biases = [crbe_lna_bias, rtbe_lna_bias]
    be_stages = [crbe_stg, rtbe_stg]
    output_saving.save_calibration_results(
        be_biases, be_stages, settings, calibration_result)
    # endregion


def manual_entry_measurement(
        settings: config_handling.Settings,
        lna_biases: list[lnas.LNABiasSet],
        res_managers: instruments.ResourceManagers,
        trimmed_input_data: config_handling.TrimmedInputs) -> None:
    """Single measurement point with user input bias conditions.

    User inputs bias array for a noise temperature measurement, this
    function then applies that bias condition, conducts the test, and
    saves the result.

    Args:
        settings: Contains settings for measurement session.
        res_managers: An object containing the resource managers for the
            instruments used in the measurement.
        lna_biases:
        trimmed_input_data:
    """
    # region Set bias ID, trigger measurement, save results.
    bias_id = 1
    lna_1_bias = lna_biases[0]
    lna_2_bias = lna_biases[1]

    standard_results = measurement.measurement(
        settings, res_managers, trimmed_input_data)

    output_saving.save_standard_results(
        settings, standard_results, bias_id, lna_1_bias, lna_2_bias)
    # endregion
