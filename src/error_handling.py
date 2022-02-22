# -*- coding: utf-8 -*-
"""error_handling.py - Input checking functions and error types.
"""

# region Import modules.
from __future__ import annotations

from pathvalidate import validate_filename

import instr_classes as ic
import util as ut
import settings_classes as sc
# endregion


# region Measurement settings.
def check_lna_sequence(lna_sequence: list, lnas_per_chain: int) -> None:
    """Ensures you can't put a non existent LNA in LNA Sequence."""
    if lnas_per_chain == 1 and 2 in lna_sequence:
        raise Exception('If 2 is in lna_sequence, lnas_per_chain must = 2.')


def check_lna_info(lna_info: sc.LNAInfo) -> None:
    """LNA info error handling."""
    if lna_info.lna_cryo_layout.cryo_chain not in [1, 2, 3]:
        raise Exception('Chain must be 1, 2, or 3.')

    if 1 > lna_info.lna_cryo_layout.stages_per_lna > 3:
        raise Exception('Maximum 3 stages per chain.')


def check_misc(misc: sc.Misc) -> None:
    """Misc error handling"""
    if not isinstance(misc.dark_mode_plot, bool):
        raise Exception('Dark mode plot must be true or false.')

    if not isinstance(misc.comment_en, bool):
        raise Exception('Comment_en must be true or false.')

    if misc.order != 1:
        ut.yes_no('Order', misc.order, '')


def check_cal_info(cal_info: sc.CalInfo, session_info: sc.SessionInfo) -> None:
    """Cal info error handling."""
    if cal_info.is_calibration \
            and session_info.measure_method != 'Calibration':
        raise Exception('is_calibration and measure_method cal error.')

    if session_info.measure_method == 'Calibration' \
            and not cal_info.is_calibration:
        raise Exception('is_calibration and measure_method cal error.')

    if 1 > cal_info.in_cal_file_id:
        raise Exception('Input cal file ID must be 1 or greater.')


def check_session_info(session_info: sc.SessionInfo) -> None:
    """Session info error handling."""
    validate_filename(session_info.project_title)
    if session_info.measure_method not in ['ACTAH', 'AT',
                                           'MEM', 'Calibration']:
        raise Exception('Invalid measurement method.')
# endregion


# region Sweep settings.
def check_meas_sequence(meas_sequence: sc.MeasSequence) -> None:
    """Measurement sequence error handling."""
    if (any(meas_sequence.stage_sequence) not in [1, 2, 3]) \
            or (0 >= len(meas_sequence.stage_sequence) > 3):
        if meas_sequence.stage_sequence:
            raise Exception('Invalid stage sequence.')

    if (any(meas_sequence.lna_sequence) not in [1, 2]) \
            or (0 > len(meas_sequence.lna_sequence) > 2):
        if meas_sequence.lna_sequence:
            raise Exception('Invalid LNA sequence.')


def check_nominals(nominal_bias: sc.NominalBias) -> None:
    """Nominal bias error handling."""
    if nominal_bias is not None:
        if 3 > nominal_bias.d_i_nominal > 9 and nominal_bias is not None:
            ut.yes_no('Nominal Drain Current', nominal_bias.d_i_nominal,
                      'mA')

        if 0.5 > nominal_bias.d_v_nominal > 1.3 and nominal_bias is not None:
            ut.yes_no('Nominal Drain Voltage', nominal_bias.d_v_nominal,
                      'V')


def check_sweep_setup_vars(sweep_setup_vars: sc.SweepSetupVars) -> None:
    """Sweep setup variable error handling."""
    if sweep_setup_vars.num_of_d_v > 6:
        ut.yes_no('Number of Drain Voltage Points',
                  sweep_setup_vars.num_of_d_v, '')

    if sweep_setup_vars.num_of_d_i > 6:
        ut.yes_no('Number of Drain Current Points',
                  sweep_setup_vars.num_of_d_i, '')

    if 0.3 > sweep_setup_vars.d_v_min > 1.3:
        ut.yes_no('Minimum Drain Voltage', sweep_setup_vars.d_v_min, 'V')

    if 0.3 > sweep_setup_vars.d_v_max > 1.3:
        ut.yes_no('Maximum Drain Voltage', sweep_setup_vars.d_v_max, 'V')

    if 3 > sweep_setup_vars.d_i_min > 9:
        ut.yes_no('Minimum Drain Current', sweep_setup_vars.d_i_min, 'mA')

    if 3 > sweep_setup_vars.d_i_max > 9:
        ut.yes_no('Maximum Drain Current', sweep_setup_vars.d_i_max, 'mA')

    if sweep_setup_vars.d_i_min > sweep_setup_vars.d_i_max:
        raise Exception('Minimum current greater than maximum.')

    if sweep_setup_vars.d_v_min > sweep_setup_vars.d_v_max:
        raise Exception('Minimum voltage greater than maximum.')

    if sweep_setup_vars.alt_temp_sweep_skips < 0 or not isinstance(
            sweep_setup_vars.alt_temp_sweep_skips, int):
        raise Exception('Incorrectly specified alt_temp_sweep_skips.')
# endregion


# region Instrumentation settings.
# region Signal analyser settings.
def check_sa_freq_settings(sa_freq_settings: ic.SpecAnFreqSettings) -> None:
    """Check signal analyser frequency settings."""
    if 0 > sa_freq_settings.center_freq > 10:
        raise Exception('Invalid center frequency.')

    if 0.01 > sa_freq_settings.center_freq > 1:
        ut.yes_no('Center Frequency', sa_freq_settings.center_freq, 'GHz')

    if sa_freq_settings.marker_freq != sa_freq_settings.center_freq:
        ut.yes_no('Marker Frequency', sa_freq_settings.marker_freq, 'GHz')

    if 0.01 > sa_freq_settings.marker_freq > 1:
        ut.yes_no('Marker Frequency', sa_freq_settings.marker_freq, 'GHz')

    if 1 > sa_freq_settings.freq_span > sa_freq_settings.marker_freq:
        ut.yes_no('Frequency Span', sa_freq_settings.freq_span, 'MHz')


def check_sa_bw_settings(sa_bw_settings: ic.SpecAnBWSettings,
                         sa_freq_settings: ic.SpecAnFreqSettings) -> None:
    """Check signal analyser bandwidth settings."""
    if 1 > sa_bw_settings.res_bw > 100:
        ut.yes_no('Resolution Bandwidth', sa_bw_settings.res_bw, 'MHz')

    if 1 > sa_bw_settings.vid_bw > 99:
        ut.yes_no('Video Bandwidth', sa_bw_settings.vid_bw, 'Hz')

    if (sa_freq_settings.freq_span / 2) > sa_bw_settings.power_bw > \
            (sa_freq_settings.freq_span * 2):
        ut.yes_no('Power Bandwidth', sa_bw_settings.power_bw, 'MHz')


def check_sa_ampl_settings(sa_ampl_settings: ic.SpecAnAmplSettings) -> None:
    """Check signal analyser amplitude settings."""
    if 0 > sa_ampl_settings.atten > 30:
        ut.yes_no('Attenuation', sa_ampl_settings.atten, 'dB')

    if -50 > sa_ampl_settings.ref_lvl > 0:
        ut.yes_no('Reference Level', sa_ampl_settings.ref_lvl, 'dBm')
# endregion


# region Signal generator settings.
def check_freq_sweep_settings(
        freq_sweep_settings: ic.FreqSweepSettings) -> None:
    """Check frequency sweep settings."""
    if 60 > freq_sweep_settings.min_freq > 350:
        ut.yes_no('Minimum Frequency', freq_sweep_settings.min_freq, 'GHz')

    if 80 > freq_sweep_settings.max_freq > 400:
        ut.yes_no('Maximum Frequency', freq_sweep_settings.max_freq, 'GHz')

    if freq_sweep_settings.freq_step_size not in [0.2, 0.4, 1, 4, 8]:
        ut.yes_no('Frequency Step Size',
                  freq_sweep_settings.freq_step_size, 'GHz')

    if freq_sweep_settings.inter_freq_factor != 8:
        ut.yes_no('Frequency Multi Factor',
                  freq_sweep_settings.inter_freq_factor, '')
# endregion


# region Temperature controller settings.
def check_temp_ctrl_channels(temp_ctrl_channels: ic.TempCtrlChannels) -> None:
    """Check temperature controller channel settings."""
    if int(temp_ctrl_channels.chn1_lna_lsch) not in range(1, 11):
        raise Exception('Check lakeshore channels.')

    if int(temp_ctrl_channels.chn2_lna_lsch) not in range(1, 11):
        raise Exception('Check lakeshore channels.')

    if int(temp_ctrl_channels.chn3_lna_lsch) not in range(1, 11):
        raise Exception('Check lakeshore channels.')

    if int(temp_ctrl_channels.load_lsch) not in range(1, 11):
        raise Exception('Check lakeshore channels.')

    if not isinstance(temp_ctrl_channels.extra_sensors_en, bool):
        raise Exception('extra_sensors_en must be True or False.')


def check_temp_targets(temp_targets: ic.TempTargets) -> None:
    """Check temperature controller target temperatures."""
    if temp_targets.cold_target > temp_targets.hot_target:
        raise Exception('Hot temperature must be greater than cold.')

    if 30 > temp_targets.hot_target > 80:
        ut.yes_no('Hot Target', temp_targets.hot_target, 'K')

    if 5 > temp_targets.cold_target > 40:
        ut.yes_no('Cold Target', temp_targets.cold_target, 'K')
# endregion


# region Bias PSU settings.
def check_g_v_search_settings(
        g_v_search_settings: ic.GVSearchSettings) -> None:
    """Check gate voltage drain current search settings."""
    g_v_lower_lim = g_v_search_settings.g_v_lower_lim
    g_v_upper_lim = g_v_search_settings.g_v_upper_lim
    num_of_g_v_brd_steps = g_v_search_settings.num_of_g_v_brd_steps
    num_of_g_v_mid_steps = g_v_search_settings.num_of_g_v_mid_steps
    num_of_g_v_nrw_steps = g_v_search_settings.num_of_g_v_nrw_steps

    if g_v_lower_lim > g_v_upper_lim:
        raise Exception('Check Gate Voltage Limits')

    if g_v_upper_lim - g_v_lower_lim < 0.1:
        input('Gate voltage limits very close, check before continuing.')

    if -0.5 > g_v_lower_lim > 0:
        ut.yes_no('Lower Gate Voltage Limit', g_v_lower_lim, 'V')

    if 0 > g_v_upper_lim > 0.5:
        ut.yes_no('Upper Gate Voltage Limit', g_v_upper_lim, 'V')

    if num_of_g_v_brd_steps not in range(3, 7):
        ut.yes_no('Number of Gate Voltage Broad Steps',
                  num_of_g_v_brd_steps, 'V')

    if num_of_g_v_mid_steps not in range(3, 7):
        ut.yes_no('Number of Gate Voltage Middle Steps',
                  num_of_g_v_mid_steps, 'V')

    if num_of_g_v_nrw_steps not in range(3, 7):
        ut.yes_no('Number of Gate Voltage Narrow Steps',
                  num_of_g_v_nrw_steps, 'V')


def check_psu_limits(psu_limits: ic.PSULimits) -> None:
    """Check power supply limits."""
    if 3 > psu_limits.d_i_lim > 10:
        ut.yes_no('Drain Current Limit', psu_limits.d_i_lim, 'mA')

    if 0.1 > psu_limits.v_step_lim > 0.5:
        ut.yes_no('Voltage Step Limit', psu_limits.v_step_lim, 'V')


def check_psu_meta_settings(psu_meta_settings: ic.PSUMetaSettings) -> None:
    """Check power supply meta settings."""
    if not isinstance(psu_meta_settings.bias_psu_en, bool):
        raise Exception('bias_psu_en must be True or False.')

    if not isinstance(psu_meta_settings.psu_safe_init, bool):
        raise Exception('psu_safe_init must be True or False.')
# endregion
# endregion
