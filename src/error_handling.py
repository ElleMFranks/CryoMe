# -*- coding: utf-8 -*-
"""error_handling.py - Input checking functions and error types.

This module provides functions to check the user input settings to
ensure the program can run successfully. Adding additional functions is
easy, and they are all called in the classes modules.
"""

# region Import modules.
from __future__ import annotations
from re import L

from pathvalidate import validate_filename

import instruments
import util
import config_handling
# endregion


# region Exception types.
class InstrumentError(Exception):
    def __init__(self, instrument: str, error_msg: str):
        error_msg = f'Instrument error from {instrument}. ' \
            f'Message: {error_msg}'
        Exception.__init__(self, error_msg)

class BiasingError(InstrumentError):
    def __init__(self, problem):
        InstrumentError.__init__(self, 'LNA/Power Supply', problem)

class InternalVariableError(Exception):
    pass

class ConfigError(Exception):
    pass

class YAMLError(ConfigError):
    def __init__(self, parameter: str, problem: str = None) -> None:
        error_msg = f'Incorrect {parameter}. {problem}'
        ConfigError.__init__(self, error_msg)

class AdvancedYAMLError(ConfigError):
    def __init__(self, parameter: str, problem: str = None) -> None:
        error_msg = f'Incorrect {parameter} in advanced_config.yml. {problem}'
        ConfigError.__init__(self, error_msg)
# endregion


# region Measurement settings.
def validate_lna_sequence(lna_sequence: list, lnas_per_chain: int) -> None:
    """Ensures you can't put a non existent LNA in LNA Sequence."""
    if lnas_per_chain == 1 and 2 in lna_sequence:
        raise YAMLError(
            'lnas_per_chain or lna_sequence',                
            'If 2 is in lna_sequence, lnas_per_chain must = 2.')


def validate_lna_info(lna_info: config_handling.LNAInfo) -> None:
    """LNA info error handling."""
    if lna_info.lna_cryo_layout.cryo_chain not in [1, 2, 3]:
        raise YAMLError('cryo_chain', 'Chain must be 1, 2, or 3.')

    if 1 > lna_info.lna_cryo_layout.stages_per_lna > 3:
        raise YAMLError('stages_per_lna', 'Maximum 3 stages per chain.')


def validate_misc(misc: config_handling.Misc) -> None:
    """Misc error handling"""
    if not isinstance(misc.dark_mode_plot, bool):
        raise AdvancedYAMLError('dark_mode_plot', 
                                'Dark mode plot must be true or false.')

    if not isinstance(misc.comment, str):
        raise YAMLError('comment', 
                        'Comment must be a string, check quotation marks.')

    if misc.order != 1:
        util.yes_no('Order', misc.order, '')


def validate_cal_info(cal_info: config_handling.CalInfo,
                      session_info: config_handling.SessionInfo) -> None:
    """Cal info error handling."""
    is_calibration = cal_info.is_calibration
    measure_method = session_info.measure_method

    incongruency_one = bool(
        is_calibration and measure_method != 'Calibration')
    incongruency_two = bool(
        not is_calibration and measure_method == 'Calibration')

    if incongruency_one or incongruency_two:
        raise YAMLError(
            'is_calibration or measure_method', 
            'is_calibration and measure_method currently incompatible.')

    if 1 > cal_info.in_cal_file_id:
        raise YAMLError('in_cal_file_id', 
                        'Input cal file ID must be 1 or greater.')


def validate_session_info(session_info: config_handling.SessionInfo) -> None:
    """Session info error handling."""
    validate_filename(session_info.project_title)
    if session_info.measure_method not in ['ACTAH', 'AT',
                                           'MEM', 'Calibration']:
        raise YAMLError('measure_method', 'Invalid measurement method.')
# endregion


# region Sweep settings.
def validate_meas_sequence(meas_sequence: config_handling.MeasSequence) -> None:
    """Measurement sequence error handling."""
    if (any(meas_sequence.stage_sequence) not in [1, 2, 3]) \
            or (0 >= len(meas_sequence.stage_sequence) > 3):
        if meas_sequence.stage_sequence:
            raise YAMLError('stage_sequence', 'Invalid stage sequence.')

    if (any(meas_sequence.lna_sequence) not in [1, 2]) \
            or (0 > len(meas_sequence.lna_sequence) > 2):
        if meas_sequence.lna_sequence:
            raise YAMLError('lna_sequence', 'Invalid LNA sequence.')


def check_nominals(nominal_bias: config_handling.NominalBias) -> None:
    """Nominal bias error handling."""
    if nominal_bias is not None:
        if 3 > nominal_bias.d_i_nominal > 9 and nominal_bias is not None:
            util.yes_no('Nominal Drain Current', nominal_bias.d_i_nominal,
                      'mA')

        if 0.5 > nominal_bias.d_v_nominal > 1.3 and nominal_bias is not None:
            util.yes_no('Nominal Drain Voltage', nominal_bias.d_v_nominal,
                      'V')


def validate_sweep_setup_vars(sweep_setup_vars: config_handling.SweepSetupVars) -> None:
    """Sweep setup variable error handling."""
    if sweep_setup_vars.num_of_d_v > 6:
        util.yes_no('Number of Drain Voltage Points',
                  sweep_setup_vars.num_of_d_v, '')

    if sweep_setup_vars.num_of_d_i > 6:
        util.yes_no('Number of Drain Current Points',
                  sweep_setup_vars.num_of_d_i, '')

    if 0.3 > sweep_setup_vars.d_v_min > 1.3:
        util.yes_no('Minimum Drain Voltage', sweep_setup_vars.d_v_min, 'V')

    if 0.3 > sweep_setup_vars.d_v_max > 1.3:
        util.yes_no('Maximum Drain Voltage', sweep_setup_vars.d_v_max, 'V')

    if 3 > sweep_setup_vars.d_i_min > 9:
        util.yes_no('Minimum Drain Current', sweep_setup_vars.d_i_min, 'mA')

    if 3 > sweep_setup_vars.d_i_max > 9:
        util.yes_no('Maximum Drain Current', sweep_setup_vars.d_i_max, 'mA')

    if sweep_setup_vars.d_i_min > sweep_setup_vars.d_i_max:
        raise YAMLError('d_i_min and d_i_max', 
                        'Minimum current greater than maximum.')

    if sweep_setup_vars.d_v_min > sweep_setup_vars.d_v_max:
        raise YAMLError('d_v_min and d_v_max',
                        'Minimum voltage greater than maximum.')

    if sweep_setup_vars.alt_temp_sweep_skips < 0 or not isinstance(
            sweep_setup_vars.alt_temp_sweep_skips, int):
        raise YAMLError('alt_temp_sweep_skips', 
                        'Incorrectly specified alt_temp_sweep_skips.')
# endregion


# region Instrumentation settings.
# region Signal analyser settings.
def validate_sa_freq_settings(sa_freq_settings: instruments.SpecAnFreqSettings) -> None:
    """Check signal analyser frequency settings."""
    if 0 > sa_freq_settings.center_freq > 10:
        raise YAMLError('center_freq', 'Invalid center frequency.')

    if 0.01 > sa_freq_settings.center_freq > 1:
        util.yes_no('Center Frequency', sa_freq_settings.center_freq, 'GHz')

    if sa_freq_settings.marker_freq != sa_freq_settings.center_freq:
        util.yes_no('Marker Frequency', sa_freq_settings.marker_freq, 'GHz')

    if 0.01 > sa_freq_settings.marker_freq > 1:
        util.yes_no('Marker Frequency', sa_freq_settings.marker_freq, 'GHz')

    if 1 > sa_freq_settings.freq_span > sa_freq_settings.marker_freq:
        util.yes_no('Frequency Span', sa_freq_settings.freq_span, 'MHz')


def check_sa_bw_settings(sa_bw_settings: instruments.SpecAnBWSettings,
                         sa_freq_settings: instruments.SpecAnFreqSettings) -> None:
    """Check signal analyser bandwidth settings."""
    if 1 > sa_bw_settings.res_bw > 100:
        util.yes_no('Resolution Bandwidth', sa_bw_settings.res_bw, 'MHz')

    if 1 > sa_bw_settings.vid_bw > 99:
        util.yes_no('Video Bandwidth', sa_bw_settings.vid_bw, 'Hz')

    if (sa_freq_settings.freq_span / 2) > sa_bw_settings.power_bw > \
            (sa_freq_settings.freq_span * 2):
        util.yes_no('Power Bandwidth', sa_bw_settings.power_bw, 'MHz')


def check_sa_ampl_settings(sa_ampl_settings: instruments.SpecAnAmplSettings) -> None:
    """Check signal analyser amplitude settings."""
    if 0 > sa_ampl_settings.atten > 30:
        util.yes_no('Attenuation', sa_ampl_settings.atten, 'dB')

    if -50 > sa_ampl_settings.ref_lvl > 0:
        util.yes_no('Reference Level', sa_ampl_settings.ref_lvl, 'dBm')
# endregion


# region Signal generator settings.
def check_freq_sweep_settings(
        freq_sweep_settings: instruments.FreqSweepSettings) -> None:
    """Check frequency sweep settings."""
    if 60 > freq_sweep_settings.min_freq > 350:
        util.yes_no('Minimum Frequency', freq_sweep_settings.min_freq, 'GHz')

    if 80 > freq_sweep_settings.max_freq > 400:
        util.yes_no('Maximum Frequency', freq_sweep_settings.max_freq, 'GHz')

    if freq_sweep_settings.freq_step_size not in [0.2, 0.4, 1, 4, 8]:
        util.yes_no('Frequency Step Size',
                  freq_sweep_settings.freq_step_size, 'GHz')

    if freq_sweep_settings.inter_freq_factor != 8:
        util.yes_no('Frequency Multi Factor',
                  freq_sweep_settings.inter_freq_factor, '')
# endregion


# region Temperature controller settings.
def validate_temp_ctrl_channels(temp_ctrl_channels: instruments.TempCtrlChannels) -> None:
    """Check temperature controller channel settings."""

    if not isinstance(temp_ctrl_channels.extra_sensors_en, bool):
        raise YAMLError('extra_sensors_en', 
                        'extra_sensors_en must be True or False.')

    temp_ctrl_channels_list = [
        temp_ctrl_channels.load_channels.chain_1,
        temp_ctrl_channels.load_channels.chain_2,
        temp_ctrl_channels.load_channels.chain_3,
        temp_ctrl_channels.lna_channels.chain_1,
        temp_ctrl_channels.lna_channels.chain_2,
        temp_ctrl_channels.lna_channels.chain_3]
        
    for channel in temp_ctrl_channels_list:
        if int(channel) not in range(1, 11):
            raise YAMLError('temp_ctrl_channels', 'Check lakeshore channels.')

    

def validate_temp_stabilisation_times(
    temp_stabilisation_times: instruments.TempStabilisationTimes) -> None:
    """Ensure temperature stabilisation times are valid."""
    first_on_lna = temp_stabilisation_times.first_on_lna
    second_on_load = temp_stabilisation_times.second_on_load
    third_on_lna = temp_stabilisation_times.third_on_lna
    final_on_load = temp_stabilisation_times.final_on_load
    args = [first_on_lna, second_on_load, third_on_lna, final_on_load]

    for time in args:
        if not isinstance(time, float) and not isinstance(time, int):
            raise YAMLError('temp_stabilisation_times',
                            'Check temperature stabilisation times.')
        if time < 0:
            raise YAMLError('temp_stabilisation_times', 
                            'Check temperature stabilisation times.')
        if time > 100:
            util.yes_no('Stabilisation Time', time, 's')


def check_temp_targets(temp_targets: instruments.TempTargets) -> None:
    """Check temperature controller target temperatures."""
    if temp_targets.cold_target > temp_targets.hot_target:
        raise YAMLError('cold_target or hot_target', 
                        'Hot temperature must be greater than cold.')

    if 30 > temp_targets.hot_target > 80:
        util.yes_no('Hot Target', temp_targets.hot_target, 'K')

    if 5 > temp_targets.cold_target > 40:
        util.yes_no('Cold Target', temp_targets.cold_target, 'K')
# endregion


# region Bias PSU settings.
def check_g_v_search_settings(
        g_v_search_settings: instruments.GVSearchSettings) -> None:
    """Check gate voltage drain current search settings."""
    g_v_lower_lim = g_v_search_settings.g_v_lower_lim
    g_v_upper_lim = g_v_search_settings.g_v_upper_lim
    num_of_g_v_brd_steps = g_v_search_settings.num_of_g_v_brd_steps
    num_of_g_v_mid_steps = g_v_search_settings.num_of_g_v_mid_steps
    num_of_g_v_nrw_steps = g_v_search_settings.num_of_g_v_nrw_steps

    if g_v_lower_lim > g_v_upper_lim:
        raise AdvancedYAMLError('g_v_low_lim or g_v_up_lim', 
                                'Check gate voltage limits')

    if g_v_upper_lim - g_v_lower_lim < 0.1:
        input('Gate voltage limits very close, check before continuing.')

    if -0.5 > g_v_lower_lim > 0:
        util.yes_no('Lower Gate Voltage Limit', g_v_lower_lim, 'V')

    if 0 > g_v_upper_lim > 0.5:
        util.yes_no('Upper Gate Voltage Limit', g_v_upper_lim, 'V')

    if num_of_g_v_brd_steps not in range(3, 7):
        util.yes_no('Number of Gate Voltage Broad Steps',
                  num_of_g_v_brd_steps, 'V')

    if num_of_g_v_mid_steps not in range(3, 7):
        util.yes_no('Number of Gate Voltage Middle Steps',
                  num_of_g_v_mid_steps, 'V')

    if num_of_g_v_nrw_steps not in range(3, 7):
        util.yes_no('Number of Gate Voltage Narrow Steps',
                  num_of_g_v_nrw_steps, 'V')


def check_psu_limits(psu_limits: instruments.PSULimits) -> None:
    """Check power supply limits."""
    if 3 > psu_limits.d_i_lim > 10:
        util.yes_no('Drain Current Limit', psu_limits.d_i_lim, 'mA')

    if 0.1 > psu_limits.v_step_lim > 0.5:
        util.yes_no('Voltage Step Limit', psu_limits.v_step_lim, 'V')


def validate_psu_meta_settings(psu_meta_settings: instruments.PSUMetaSettings) -> None:
    """Check power supply meta settings."""
    if not isinstance(psu_meta_settings.bias_psu_en, bool):
        raise AdvancedYAMLError('bias_psu_en', 'Must be True or False.')

    if not isinstance(psu_meta_settings.psu_safe_init, bool):
        raise AdvancedYAMLError('psu_safe_init', 'Must be True or False.')
# endregion
# endregion
