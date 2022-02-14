# -*- coding: utf-8 -*-
"""cryome.py - Y Factor Noise Temperature Method Automation Package.

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
        * Document socket_communication.py if pyvisa impossible.
    * Features
        * See if way of replacing socket_comms with pyvisa for psx.
        * Add extra results analysis log.
"""

# region Import modules
from __future__ import annotations

import settings_classes as scl
import lna_classes as lc
import instr_classes as ic
import output_classes as oc
import start_session as ss
# endregion


def main():
    """Main for CryoMe."""
    # region User defined settings
    # region Measurement settings
    comment_en = False       # True prompts user input session comment.
    dark_mode_plot = True    # True saves plots in dark mode palette.
    in_cal_file_id = 1       # Decides which calibration file to use.
    instr_buffer_time = 0.3  # Seconds to wait between isntr commands.
    inter_freq_factor = 8    # Multiplier factor from sig gen.
    is_calibration = True    # Enables calibration handling.
    lnas_per_chain = 1       # How many LNAs per chain?
    order = 1                # Leave as 1 unless explicitly known why.
    project_title = 'ESO LNAs'   # Becomes folder title.
    psu_safe_init = True
    stage_1_2_same = False   # If true L1S1 = L1S2 and L2S1 = L2S2
    stage_2_3_same = False   # If true L1S2 = L1S3 and L2S2 = L2S3
    stages_per_lna = 3       # How many stages per LNA?
    # For measure_method:
    # AllColdToAllHot = 'ACTAH', ManualEntryMeasure = 'MEM',
    # AlternatingTemps = 'AT',  Calibration = 'Calibration'
    measure_method = 'Calibration'  # Chooses the measurement algorithm.
    # endregion

    # region LNA IDs
    chain_1_lna_1_id = 50     # LNA ID number for 1st LNA in chain 1.
    chain_1_lna_2_id = 51     # LNA ID number for 2nd LNA in chain 1.
    chain_2_lna_1_id = 52     # LNA ID number for 1st LNA in chain 2.
    chain_2_lna_2_id = 53     # LNA ID number for 2nd LNA in chain 2.
    chain_3_lna_1_id = 54     # LNA ID number for 1st LNA in chain 3.
    chain_3_lna_2_id = 55     # LNA ID number for 2nd LNA in chain 3.
    # endregion

    # region Bias Sweep Settings
    alt_temp_sweep_skips = 0  # How many meas to skip on 'AT' alg sweep.
    chain_sequence = [1]      # Order of cryostat chains to measure.
    lna_sequence = [1]        # [FirstLNAToSweep, SecondLNAToSweep].
    stage_sequence = [1, 2, 3]   # Which order to sweep stages in.
    d_i_max = 6               # mA
    d_i_min = 4               # mA
    d_i_nominal = 4.0         # mA
    d_v_max = 1.1             # V
    d_v_min = 0.7             # V
    d_v_nominal = 1.0         # V
    num_of_d_i = 2            # V
    num_of_d_v = 2            # V
    # endregion

    # region Bias PSU Settings
    d_i_lim = 10            # mA
    g_v_low_lim = -0.25     # V
    g_v_up_lim = +0.25      # V
    num_g_v_brd_steps = 5   # Broad steps
    num_g_v_mid_steps = 5   # Mid steps
    num_g_v_nrw_steps = 5   # Narrow steps
    v_step_lim = 0.3        # V
    # endregion

    # region Signal Analyser Settings
    atten = 10           # dB
    center_freq = 0.075  # GHz
    freq_span = 25       # MHz
    marker_freq = 0.075  # GHz
    power_bw = 24        # MHz
    ref_level = -20      # dBm
    res_bw = 8           # MHz
    vid_bw = 10          # Hz
    # endregion

    # region Signal Generator Settings
    freq_step_size = 0.2      # GHz
    max_freq = 116          # GHz
    min_freq = 67           # GHz
    vna_or_sig_gen = 'vna'  # Either 'vna' or 'sig gen'
    # endregion

    # region Temperature Controller Settings
    # Sensor channel variables named in format:
    # lakeshore_chain1/2/3/all_lna/load_channel
    allchn_load_lsch = '08'
    chn1_lna_lsch = '07'
    chn2_lna_lsch = '09'
    chn3_lna_lsch = '10'
    extra_sensors_en = False
    t_cold_target = 20.2    # Kelvin - To give 20K
    t_hot_target = 50.7     # 60.9K - To give 60K
    # endregion

    # region Back End LNA Settings
    # These variables given in format:
    # cryo/room-temp_backend_all/chain#_drain_voltage/current
    be_d_i_limit = 20       # mA

    crbe_chn1_d_v = 1.26    # V
    crbe_chn1_d_i = 13.9    # mA
    crbe_chn1_g_v = 0.133   # V

    crbe_chn2_d_v = 1       # V
    crbe_chn2_d_i = 14      # mA
    crbe_chn2_g_v = 0.15    # V

    crbe_chn3_d_v = 1.25    # V
    crbe_chn3_d_i = 16.5    # mA
    crbe_chn3_g_v = 0.15    # V

    rtbe_chna_d_v = 1.22    # V
    rtbe_chna_d_i = 16      # mA
    rtbe_chna_g_v = -0.178  # V
    # endregion

    # region Manual Entry LNA Settings
    # These variables named in the format:
    # manual_lna#_stage#_drain_voltage/current
    man_l1_s1_d_v = 1    # V
    man_l1_s1_d_i = 6    # mA
    man_l1_s2_d_v = 1    # V
    man_l1_s2_d_i = 6    # mA
    man_l1_s3_d_v = 1.1  # V
    man_l1_s3_d_i = 6    # mA

    man_l2_s1_d_v = 1.2  # V
    man_l2_s1_d_i = 8    # mA
    man_l2_s2_d_v = 1.2  # V
    man_l2_s2_d_i = 8    # mA
    man_l2_s3_d_v = 1.1  # V
    man_l2_s3_d_i = 6    # mA
    # endregion

    # region Analysis bandwidths
    ana_bw_1_min_max = [67, 90]   # Min GHz to max GHz
    ana_bw_2_min_max = [90, 116]  # Min GHz to max GHz
    ana_bw_3_min_max = []         # Min GHz to max GHz
    ana_bw_4_min_max = []         # Min GHz to max GHz
    ana_bw_5_min_max = []         # Min GHz to max GHz
    # endregion

    # region Available Instruments
    bias_psu_en = True
    sig_an_en = True
    sig_gen_en = True
    switch_en = True
    temp_ctrl_en = True
    # endregion
    # endregion

    # region Measure each chain as requested.
    for i, cryo_chain in enumerate(chain_sequence):

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

        # region Put user variables into class instances.
        # region Base level.
        # region Signal Analyser Settings
        sa_ampl_settings = ic.SpecAnAmplSettings(atten, ref_level)
        sa_bw_settings = ic.SpecAnBWSettings(vid_bw, res_bw, power_bw)
        sa_freq_settings = ic.SpecAnFreqSettings(
            center_freq, marker_freq, freq_span)
        sig_an_settings = ic.SignalAnalyserSettings(
            sa_freq_settings, sa_bw_settings, sa_ampl_settings, sig_an_en)
        # endregion

        # region Signal Generator Settings.
        freq_sweep_settings = ic.FreqSweepSettings(
            min_freq, max_freq, freq_step_size, inter_freq_factor)
        sig_gen_settings = ic.SignalGeneratorSettings(
            freq_sweep_settings, vna_or_sig_gen, sig_gen_en)
        # endregion

        # region Temperature Controller Settings
        temp_ctrl_channels = ic.TempCtrlChannels(
            allchn_load_lsch, chn1_lna_lsch, chn2_lna_lsch,
            chn3_lna_lsch, extra_sensors_en)
        temp_targets = ic.TempTargets(t_hot_target, t_cold_target)
        temp_ctrl_settings = ic.TempControllerSettings(
            temp_ctrl_channels, temp_targets, cryo_chain, temp_ctrl_en)
        # endregion

        # region Bias Power Supply Settings
        g_v_search_settings = ic.GVSearchSettings(
            g_v_low_lim, g_v_up_lim, num_g_v_brd_steps,
            num_g_v_mid_steps, num_g_v_nrw_steps)
        psu_limits = ic.PSULimits(v_step_lim, d_i_lim)
        psu_meta_settings = ic.PSUMetaSettings(
            psu_safe_init, bias_psu_en, instr_buffer_time)
        bias_psu_settings = ic.BiasPSUSettings(
            g_v_search_settings, psu_limits, psu_meta_settings)
        # endregion

        # region Switch Settings.
        switch_settings = ic.SwitchSettings(cryo_chain, switch_en)
        # endregion
        # endregion

        # region Mid level.
        # region Instrument Settings.
        instr_settings = ic.InstrumentSettings(
            sig_an_settings, sig_gen_settings, temp_ctrl_settings,
            bias_psu_settings, switch_settings, instr_buffer_time)
        # endregion

        # region Measurement Settings.
        lna_ids = scl.LNAIDs(chain_1_lna_1_id, chain_1_lna_2_id,
                             chain_2_lna_1_id, chain_2_lna_2_id,
                             chain_3_lna_1_id, chain_3_lna_2_id)

        cal_info = scl.CalInfo(is_calibration, in_cal_file_id)

        lna_cryo_layout = scl.LNACryoLayout(
            cryo_chain, lnas_per_chain, stages_per_lna, stage_1_2_same,
            stage_2_3_same)

        misc = scl.Misc(comment_en, dark_mode_plot, order)

        analysis_sub_bws = oc.AnalysisBandwidths(
            ana_bw_1_min_max, ana_bw_2_min_max, ana_bw_3_min_max,
            ana_bw_4_min_max, ana_bw_5_min_max)

        session_info = scl.SessionInfo(
            project_title, measure_method, analysis_sub_bws)

        # region Back End LNA Settings.
        be_lna_bias_vars = [
            'crbe_chn1_d_v', 'crbe_chn2_d_v', 'crbe_chn3_d_v',
            'crbe_chn1_d_i', 'crbe_chn2_d_i', 'crbe_chn3_d_i',
            'crbe_chn1_g_v', 'crbe_chn2_g_v', 'crbe_chn3_g_v',
            'rtbe_chna_d_v', 'rtbe_chna_d_i', 'rtbe_chna_g_v']
        be_lna_biases = {}
        for variable in be_lna_bias_vars:
            be_lna_biases[variable] = eval(variable)
        be_lna_settings = lc.BackEndLNASettings(be_lna_biases, be_d_i_limit)
        # endregion

        # region Manual LNA Settings.
        if measure_method == 'MEM':
            manual_lna_bias_vars = [
                'man_l1_s1_d_v', 'man_l1_s2_d_v', 'man_l1_s3_d_v',
                'man_l1_s1_d_i', 'man_l1_s2_d_i', 'man_l1_s3_d_i',
                'man_l2_s1_d_v', 'man_l2_s2_d_v', 'man_l2_s3_d_v',
                'man_l2_s1_d_i', 'man_l2_s2_d_i', 'man_l2_s3_d_i']
            manual_lna_biases = {}
            for variable in manual_lna_bias_vars:
                manual_lna_biases[variable] = eval(variable)
            manual_lna_settings = lc.ManualLNASettings(
                manual_lna_biases, lna_cryo_layout, d_i_lim)
        else:
            manual_lna_settings = None
        # endregion

        direct_lnas = scl.DirectLNAs(be_lna_settings, manual_lna_settings)

        lna_info = scl.LNAInfo(direct_lnas, lna_ids, lna_cryo_layout)

        meas_settings = scl.MeasurementSettings(
            session_info, lna_info, cal_info, misc)
        # endregion

        # region Sweep Settings.
        meas_sequence = scl.MeasSequence(stage_sequence, lna_sequence)

        sweep_setup_vars = scl.SweepSetupVars(
            num_of_d_v, num_of_d_i, d_v_min, d_v_max, d_i_min, d_i_max,
            alt_temp_sweep_skips)

        if not is_calibration:
            nominal_bias = scl.NominalBias(d_v_nominal, d_i_nominal)
        else:
            nominal_bias = None

        sweep_settings = scl.SweepSettings(
            meas_sequence, sweep_setup_vars, nominal_bias, lna_cryo_layout)
        # endregion

        # region File Structure.
        file_struc = scl.FileStructure(
            project_title, in_cal_file_id, cryo_chain)
        # endregion
        # endregion

        # region Top level.
        settings = scl.Settings(meas_settings, sweep_settings,
                                instr_settings, file_struc)
        # endregion
        # endregion

        # region Trigger measurement
        ss.start_session(settings)
        # endregion
    # endregion


if __name__ == '__main__':
    main()
