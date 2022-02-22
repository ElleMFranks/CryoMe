# -*- coding: utf-8 -*-
"""CryoMe Automated Y Factor Noise Temperature Measurement Package.

This is the top level module of the package where the user variables are
set, and the required measurement algorithm is called. For more
information on the package as a whole, see the Documentation folder in
the code directory.

Use of __future__ import to avoid problems with type hinting and
circular importing mean that this code will only work with Python 3.7 or
higher.

Todo:
    * Documentation
        * Write detail paragraph for module docstrings.
        * Check project docstrings.
    * Features
"""

# region Import modules
from __future__ import annotations
import logging
import os
import pathlib as pl
import sys

import yaml as ym

import settings_classes as sc
import lna_classes as lc
import instr_classes as ic
import output_classes as oc
import start_session as ss
# endregion


def main():
    """Main for CryoMe."""
    os.chdir(os.path.dirname(os.path.dirname(sys.argv[0])))

    # region Set up logging.
    stream_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        '%m-%d %H:%M:%S')
    file_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s')

    logging.addLevelName(15, "CDEBUG")

    def cdebug(self, message, *args, **kws):
        """Logging level setup."""
        self._log(15, message, args, **kws)

    logging.Logger.cdebug = cdebug
    logstream = logging.StreamHandler()
    logstream.setLevel(logging.INFO)
    logstream.setFormatter(stream_format)
    log = logging.getLogger()
    log.addHandler(logstream)
    log.setLevel(logging.DEBUG)
    # endregion

    # region Measure each chain as requested.
    with open(pl.Path(str(os.getcwd()) + '\\settings.yml'),
              encoding='utf-8') as _f:
        config = ym.safe_load(_f)

    for i, cryo_chain in enumerate(
            config['bias_sweep_settings']['chain_sequence']):

        # region Reset all class instances.
        if i > 0:
            del sig_an_settings
            del sig_gen_settings
            del temp_ctrl_settings
            del bias_psu_settings
            del switch_settings
            del be_lna_settings
            del manual_lna_settings
            del meas_settings
            del sweep_settings
            del instr_settings
            del file_struc
        # endregion

        # region Put user settings into class instances.
        # region Base level.
        # region Signal Analyser Settings
        sa_ampl_settings = ic.SpecAnAmplSettings(
            config['signal_analyser_settings']['atten'],
            config['signal_analyser_settings']['ref_level'])
        sa_bw_settings = ic.SpecAnBWSettings(
            config['signal_analyser_settings']['vid_bw'],
            config['signal_analyser_settings']['res_bw'],
            config['signal_analyser_settings']['power_bw'])
        sa_freq_settings = ic.SpecAnFreqSettings(
            config['signal_analyser_settings']['center_freq'],
            config['signal_analyser_settings']['marker_freq'],
            config['signal_analyser_settings']['freq_span'])
        sig_an_settings = ic.SignalAnalyserSettings(
            sa_freq_settings, sa_bw_settings, sa_ampl_settings,
            config['available_instruments']['sig_an_en'])
        # endregion

        # region Signal Generator Settings.
        freq_sweep_settings = ic.FreqSweepSettings(
            config['signal_generator_settings']['min_freq'],
            config['signal_generator_settings']['max_freq'],
            config['signal_generator_settings']['freq_step_size'],
            config['measurement_settings']['inter_freq_factor'])

        sig_gen_settings = ic.SignalGeneratorSettings(
            freq_sweep_settings,
            config['signal_generator_settings']['vna_or_sig_gen'],
            config['available_instruments']['sig_gen_en'])
        # endregion

        # region Temperature Controller Settings
        temp_ctrl_channels = ic.TempCtrlChannels(
            config['temperature_controller_settings']['allchn_load_lsch'],
            config['temperature_controller_settings']['chn1_lna_lsch'],
            config['temperature_controller_settings']['chn2_lna_lsch'],
            config['temperature_controller_settings']['chn3_lna_lsch'],
            config['temperature_controller_settings']['extra_sensors_en'])
        temp_targets = ic.TempTargets(
            config['temperature_controller_settings']['t_hot_target'],
            config['temperature_controller_settings']['t_cold_target'])
        temp_ctrl_settings = ic.TempControllerSettings(
            temp_ctrl_channels, temp_targets, cryo_chain,
            config['available_instruments']['temp_ctrl_en'])
        # endregion

        # region Bias Power Supply Settings
        g_v_search_settings = ic.GVSearchSettings(
            config['bias_psu_settings']['g_v_low_lim'],
            config['bias_psu_settings']['g_v_up_lim'],
            config['bias_psu_settings']['num_g_v_brd_steps'],
            config['bias_psu_settings']['num_g_v_mid_steps'],
            config['bias_psu_settings']['num_g_v_nrw_steps'])
        psu_limits = ic.PSULimits(
            config['bias_psu_settings']['v_step_lim'],
            config['bias_psu_settings']['d_i_lim'])
        psu_meta_settings = ic.PSUMetaSettings(
            config['measurement_settings']['psu_safe_init'],
            config['available_instruments']['bias_psu_en'],
            config['measurement_settings']['instr_buffer_time'])
        bias_psu_settings = ic.BiasPSUSettings(
            g_v_search_settings, psu_limits, psu_meta_settings)
        # endregion

        # region Switch Settings.
        switch_settings = ic.SwitchSettings(
            cryo_chain,
            config['available_instruments']['switch_en'])
        # endregion
        # endregion

        # region Mid level.
        # region Instrument Settings.
        instr_settings = ic.InstrumentSettings(
            sig_an_settings, sig_gen_settings, temp_ctrl_settings,
            bias_psu_settings, switch_settings,
            config['measurement_settings']['instr_buffer_time'])
        # endregion

        # region Measurement Settings.
        lna_ids = sc.LNAIDs(
            config['lna_ids']['chain_1_lna_1_id'],
            config['lna_ids']['chain_1_lna_2_id'],
            config['lna_ids']['chain_2_lna_1_id'],
            config['lna_ids']['chain_2_lna_2_id'],
            config['lna_ids']['chain_3_lna_1_id'],
            config['lna_ids']['chain_3_lna_2_id'])

        cal_info = sc.CalInfo(
            config['measurement_settings']['is_calibration'],
            config['measurement_settings']['in_cal_file_id'])

        lna_cryo_layout = sc.LNACryoLayout(
            cryo_chain,
            config['measurement_settings']['lnas_per_chain'],
            config['measurement_settings']['stages_per_lna'],
            config['measurement_settings']['stage_1_2_same'],
            config['measurement_settings']['stage_2_3_same'])

        misc = sc.Misc(
            config['measurement_settings']['comment_en'],
            config['measurement_settings']['dark_mode_plot'],
            config['measurement_settings']['order'])

        analysis_sub_bws = oc.AnalysisBandwidths(
            config['analysis_bandwidths']['ana_bw_1_min_max'],
            config['analysis_bandwidths']['ana_bw_2_min_max'],
            config['analysis_bandwidths']['ana_bw_3_min_max'],
            config['analysis_bandwidths']['ana_bw_4_min_max'],
            config['analysis_bandwidths']['ana_bw_5_min_max'])

        session_info = sc.SessionInfo(
            config['measurement_settings']['project_title'],
            config['measurement_settings']['measure_method'],
            analysis_sub_bws)

        # region Back End LNA Settings.
        be_lna_settings = lc.BackEndLNASettings(
            config['back_end_lna_settings'],
            config['back_end_lna_settings']['be_d_i_limit'])
        # endregion

        # region Manual LNA Settings.
        if config['measurement_settings']['measure_method'] == 'MEM':
            manual_lna_settings = lc.ManualLNASettings(
                config['manual_entry_lna_settings'],
                lna_cryo_layout,
                config['bias_psu_settings']['d_i_lim'])
        else:
            manual_lna_settings = None
        # endregion

        direct_lnas = sc.DirectLNAs(be_lna_settings, manual_lna_settings)

        lna_info = sc.LNAInfo(direct_lnas, lna_ids, lna_cryo_layout)

        meas_settings = sc.MeasurementSettings(
            session_info, lna_info, cal_info, misc)
        # endregion

        # region Sweep Settings.
        meas_sequence = sc.MeasSequence(
            config['bias_sweep_settings']['stage_sequence'],
            config['bias_sweep_settings']['lna_sequence'])

        sweep_setup_vars = sc.SweepSetupVars(
            config['bias_sweep_settings']['num_of_d_v'],
            config['bias_sweep_settings']['num_of_d_i'],
            config['bias_sweep_settings']['d_v_min'],
            config['bias_sweep_settings']['d_v_max'],
            config['bias_sweep_settings']['d_i_min'],
            config['bias_sweep_settings']['d_i_max'],
            config['bias_sweep_settings']['alt_temp_sweep_skips'])

        if not config['measurement_settings']['is_calibration']:
            nominal_bias = sc.NominalBias(
                config['bias_sweep_settings']['d_v_nominal'],
                config['bias_sweep_settings']['d_i_nominal'])
        else:
            nominal_bias = None

        sweep_settings = sc.SweepSettings(
            meas_sequence, sweep_setup_vars, nominal_bias, lna_cryo_layout)
        # endregion

        # region File Structure.
        file_struc = sc.FileStructure(
            config['measurement_settings']['project_title'],
            config['measurement_settings']['in_cal_file_id'],
            cryo_chain)
        # endregion
        # endregion

        # region Top level.
        settings = sc.Settings(meas_settings, sweep_settings,
                               instr_settings, file_struc)
        # endregion
        # endregion

        # region Get/set session ID and config log file writer.
        meas_settings.config_session_id(file_struc)
        log_path = file_struc.get_log_path(meas_settings.session_id)
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_format)
        log.addHandler(file_handler)
        # endregion

        # region Trigger measurement
        try:
            ss.start_session(settings)
        except:
            log.exception(sys.exc_info()[0])
        # endregion

    input('Press Enter to exit...')
    # endregion


if __name__ == '__main__':
    main()
