# -*- coding: utf-8 -*-
"""output_saving.py - Handles saving results/plots into correct places.

Results are plotted in 1080p and output in a csv. Standard results are
saved in the Results>Project>Session folder in the cryome directory.
Calibration results are handled differently and saved in the
Calibration>Chain folder.
"""

# region Import modules.
from __future__ import annotations
from time import sleep
from typing import Optional
import logging
import os
import pathlib

from matplotlib import offsetbox
from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tabulate

import config_handling
import lnas
import outputs
# endregion


def _save_plot(
        results: outputs.Results,
        meas_settings: config_handling.MeasurementSettings,
        lna_biases: list[lnas.LNABiasSet], save_path: pathlib.Path,
        bias_id: Optional[int] = None,
        calibration_id: Optional[int] = None) -> None:
    """Creates and saves plot for given results set.

    For standard results plot the noise and gain, and save them as
    png images in a separate file. For calibration plot the noise
    only.

    Args:
        results: The results to plot.
        meas_settings: The measurement settings for the session.
        lna_biases: The LNA settings for the measured LNAs.
        save_path: Where to save the plot.
        bias_id: The BiasID, or where these results are in a bias sweep.
        calibration_id: The calibration ID if a calibration measurement.
    """

    # region Unpack objects/ensure logger isn't dominated by matplotlib.
    logging.getLogger('matplotlib.font_manager').disabled = True
    lna_1_bias = lna_biases[0]
    lna_2_bias = lna_biases[1]
    # endregion

    # region Set plot variables.
    lab_font = 12
    pixel_line_width = 0.7
    dark = '#181818'
    light = '#eeeeee'
    blue = '#030bfc'
    green = '#057510'
    orange = '#f74600'
    # endregion

    # region Set figure.
    fig, axis = plt.subplots()
    fig.set_size_inches(20.92, 11.77)
    fig.set_dpi(91.79)
    fig.tight_layout(pad=5)
    # endregion

    # region Set up figure to 1080p either light or dark mode.
    if meas_settings.dark_mode_plot:
        font_color = light
        fig.set_facecolor(dark)
    else:
        font_color = dark
        fig.set_facecolor(light)
    # endregion

    # region Create bias data text box to go on plot.
    tabulate.PRESERVE_WHITESPACE = True

    if lna_2_bias is None or meas_settings.is_calibration:
        lna_2_g_v_data = ['NA', 'NA', 'NA']
        lna_2_d_v_data = ['NA', 'NA', 'NA']
        lna_2_d_i_data = ['NA', 'NA', 'NA']

    else:
        lna_2_g_v_data = lna_2_bias.lna_g_v_strs()
        lna_2_d_v_data = lna_2_bias.lna_d_v_strs()
        lna_2_d_i_data = lna_2_bias.lna_d_i_strs()

    if not meas_settings.is_calibration:
        bias_header = ['Bias', 'L1S1', 'L1S2', 'L1S3', 'L2S1', 'L2S2', 'L2S3']
        bias_row_1 = ['VG / V', *lna_1_bias.lna_g_v_strs(), *lna_2_g_v_data]
        bias_row_2 = ['VD / V', *lna_1_bias.lna_d_v_strs(), *lna_2_d_v_data]
        bias_row_3 = ['ID / mA', *lna_1_bias.lna_d_i_strs(), *lna_2_d_i_data]
        bias_table = tabulate.tabulate(
            [bias_header, bias_row_1, bias_row_2, bias_row_3],
            headers="firstrow", disable_numparse=True, tablefmt="plain")
        plot_bias_box = offsetbox.AnchoredText(bias_table, loc='lower right')
    # endregion

    # region Set up basic plot parameters
    axis.set_xlabel('Frequency / GHz', color=font_color, fontsize=lab_font)
    axis.set_ylabel(
        'Noise Temperature / Kelvin', color=font_color, fontsize=lab_font)
    axis.set_xlim(
        results.freq_array[0], 
        results.freq_array[len(results.freq_array) - 1])
    axis.set_ylim(0, +160)
    axis.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    axis.yaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    axis.grid(linestyle='--', which='major', linewidth=pixel_line_width)
    axis.grid(linestyle='--', which='minor', linewidth=pixel_line_width - 0.3)
    axis.tick_params('both', colors=font_color, labelsize=lab_font)
    # endregion

    # region Handle calibration results.
    if meas_settings.is_calibration:
        axis.set_ylim(0, +500)
        plot_title = 'Noise Temperature / Frequency'
        plot_details = (
                f'Cryostat Chain: {meas_settings.lna_cryo_layout.cryo_chain}  '
                + f'Calibration: {calibration_id}')
        plot_detail_box = offsetbox.AnchoredText(
            plot_details, loc='upper right')
        axis.add_artist(plot_detail_box)
        axis.set_title(plot_title, color=font_color, fontsize=lab_font + 2)
        axis.plot(results.freq_array, results.loss_cor_noise_temp,
                  color=orange, linewidth=pixel_line_width)
    # endregion

    # region Handle standard results.
    else:
        # region Set title, and plot details text box.
        plot_title = 'Noise Temperature & Gain / Frequency'
        plot_details = f'Project: {meas_settings.project_title}' \
                       + f' - LNA ID/s: {meas_settings.lna_id_str}' \
                       + f' - Session ID: {meas_settings.session_id}' \
                       + f' - Bias ID: {bias_id}'
        plot_detail_box = offsetbox.AnchoredText(
            plot_details, loc='upper right')
        axis.add_artist(plot_detail_box)
        # endregion

        # region Plot the noise
        axis.set_title(plot_title, color=font_color, fontsize=lab_font + 2)
        noise_temp_cor_plot = axis.plot(
            results.freq_array, results.noise_temp.cal_loss_cor, color=blue,
            linewidth=pixel_line_width, label='Calibrated & Loss Corrected')
        t_corrected_plot = axis.plot(
            results.freq_array, results.noise_temp.uncal_loss_cor,
            color=orange, linewidth=pixel_line_width,
            label='Uncalibrated & Loss Corrected')
        t_uncal_plot = axis.plot(
            results.freq_array, results.noise_temp.uncal_loss_uncor,
            color=green, linewidth=pixel_line_width,
            label='Uncalibrated & Not Loss Corrected')
        ax2 = axis.twinx()
        gain_plot = ax2.plot(
            results.freq_array, results.gain.gain_db, color=dark,
            linewidth=pixel_line_width, label='Gain')
        # endregion

        # region Create legend, and set up Gain axis.
        plots = noise_temp_cor_plot + t_corrected_plot + \
            t_uncal_plot + gain_plot
        labels = (plot.get_label() for plot in plots)
        axis.legend(
            plots, labels, loc='lower left', numpoints=1, fontsize=lab_font,
            ncol=4)
        ax2.set_ylabel('Gain / dB', color=font_color, fontsize=lab_font)
        ax2.tick_params(labelcolor=font_color, labelsize=lab_font)
        ax2.set_ylim(0, +40)
        axis.add_artist(plot_bias_box)
        # endregion
    # endregion

    # region Save and close plots.
    fig.savefig(save_path)
    # fig.show()
    sleep(0.5)  # Pause before/after close - box glitch.
    plt.close('all')
    sleep(0.5)
    # endregion


def save_standard_results(
        settings: config_handling.Settings, results: outputs.Results,
        bias_id: int, lna_1_bias: lnas.LNABiasSet,
        lna_2_bias: Optional[lnas.LNABiasSet] = None) -> None:
    """Update settings log and create raw results CSV.

    Args:
        settings: The settings for the measurement session.
        results: The results to save.
        bias_id: The BiasID, or where these results are in a bias sweep.
        lna_1_bias: The LNA settings for the first measured LNA.
        lna_2_bias: The LNA settings for the second measured LNA.
    """

    # region Unpack objects and set up logging.
    log = logging.getLogger(__name__)
    meas_settings = settings.meas_settings
    instr_settings = settings.instr_settings
    file_struc = settings.file_struc
    if meas_settings.lna_cryo_layout.cryo_chain == 1:
        crbe_lna = meas_settings.direct_lnas.be_lna_settings.crbe_chain_1_lna
    if meas_settings.lna_cryo_layout.cryo_chain == 2:
        crbe_lna = meas_settings.direct_lnas.be_lna_settings.crbe_chain_2_lna
    if meas_settings.lna_cryo_layout.cryo_chain == 3:
        crbe_lna = meas_settings.direct_lnas.be_lna_settings.crbe_chain_3_lna
    rtbe_lna = meas_settings.direct_lnas.be_lna_settings.rtbe_chain_a_lna
    # endregion

    crbe_lna.lna_measured_column_data()
    rtbe_lna.lna_measured_column_data()

    chain_cal_id = f'{meas_settings.lna_cryo_layout.cryo_chain} x {meas_settings.in_cal_file_id}'

    # region Update settings log.
    log.info('Updating settings log...')
    # region Set up data to input to settings log.
    if lna_2_bias is not None:
        set_col_data = [
            meas_settings.project_title, meas_settings.lna_id_str,
            str(meas_settings.session_id), str(bias_id), chain_cal_id,
            results.date_str, results.time_str, meas_settings.comment, None,
            meas_settings.lna_cryo_layout.lnas_per_chain, 
            results.config_ut.lna, 
            meas_settings.lna_cryo_layout.stages_per_lna, 
            results.config_ut.stage, None,
            *instr_settings.sig_an_settings.spec_an_col_data(), None,
            *lna_1_bias.lna_set_column_data(), None,
            *lna_2_bias.lna_set_column_data(), None,
            *lna_1_bias.lna_meas_column_data, None,
            *lna_2_bias.lna_meas_column_data, None,
            *crbe_lna.lna_set_column_data(),
            *rtbe_lna.lna_set_column_data(), None,
            *crbe_lna.lna_meas_column_data,
            *rtbe_lna.lna_meas_column_data, None,
            instr_settings.bias_psu_settings.lna_1_d_r,
            instr_settings.bias_psu_settings.lna_2_d_r,
            instr_settings.bias_psu_settings.crbe_d_r,
            instr_settings.bias_psu_settings.rtbe_d_r]
    else:
        set_col_data = [
            meas_settings.project_title, meas_settings.lna_id_str,
            str(meas_settings.session_id), str(bias_id), chain_cal_id,
            results.date_str, results.time_str, meas_settings.comment, None,
            meas_settings.lna_cryo_layout.lnas_per_chain,
            results.config_ut.lna,
            meas_settings.lna_cryo_layout.stages_per_lna, 
            results.config_ut.stage, None,
            *instr_settings.sig_an_settings.spec_an_col_data(), None,
            *lna_1_bias.lna_set_column_data(), None,
            'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', None,
            *lna_1_bias.lna_meas_column_data, None,
            'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', None,
            *crbe_lna.lna_set_column_data(),
            *rtbe_lna.lna_set_column_data(), None,
            *crbe_lna.lna_meas_column_data,
            *rtbe_lna.lna_meas_column_data, None,
            instr_settings.bias_psu_settings.lna_1_d_r,
            instr_settings.bias_psu_settings.lna_2_d_r,
            instr_settings.bias_psu_settings.crbe_d_r,
            instr_settings.bias_psu_settings.rtbe_d_r]
    # endregion

    # region Add row to settings log.
    file_struc.write_to_file(
        file_struc.settings_path, set_col_data, 'a', 'row')
    log.info('Settings log updated. Updating results log...')
    # endregion

    # region Update results log.
    res_log_data = results.results_ana_log_data(meas_settings, bias_id)
    file_struc.write_to_file(
        file_struc.res_log_path, res_log_data, 'a', 'row')
    log.info('Results log updated. Saving plot...')
    # endregion

    # region Setup and save plot.
    lna_biases = [lna_1_bias, lna_2_bias]
    _save_plot(results, meas_settings, lna_biases,
               file_struc.output_file_path(file_struc.results_directory,
                                        meas_settings, bias_id, 'png'),
               bias_id)
    log.info('Plot saved. Saving raw results to be processed.')
    # endregion

    # region Save results as csv.

    # region Figure out output file name.
    # Create file name in format: Raw Meas# LNA# Bias#.csv/png.
    results_header_1 = [
        'Project:', f'{meas_settings.project_title}',
        'LNA ID/s:', f'{meas_settings.lna_id_str}',
        'Session ID:', f'{meas_settings.session_id}',
        'Bias ID:', f'{bias_id}',
        'Date/Time', f'{results.date_str} {results.time_str}',
        'Comment:' f'{meas_settings.comment}']
    # endregion

    # region Set up header for output csv.
    # Header to contain measurement and settings details
    lna_1_header = lna_1_bias.lna_header()
    if lna_2_bias is not None:
        lna_1_header = lna_1_bias.lna_header()
        lna_2_header = lna_2_bias.lna_header()
        results_header_2 = [
            'L1S1:', *lna_1_header[0:6], 
            'L1S2:', *lna_1_header[7:13],
            'L1S3:', *lna_1_header[14:20]]
        results_header_3 = [    
            'L2S1:', *lna_2_header[0:6],
            'L2S2:', *lna_2_header[7:13], 
            'L2S3:', *lna_2_header[14:20]]
    else:
        results_header_2 = ['L1S1:', *lna_1_header[0:6],
                            'L1S2:', *lna_1_header[7:13],
                            'L1S3:', *lna_1_header[14:20]]
        results_header_3 = [
            'L2S1:', 'GV(V)', 'NA', 'DV(V)', 'NA', 'DI(mA)', 'NA',
            'L2S2:', 'GV(V)', 'NA', 'DV(V)', 'NA', 'DI(mA)', 'NA',
            'L2S3:', 'GV(V)', 'NA', 'DV(V)', 'NA', 'DI(mA)', 'NA']

    results_header_4 = [
        'CRBE:', *crbe_lna.lna_header()[0:6], 
        'RTBE:', *rtbe_lna.lna_header()[0:6]]

    results_header_5 = [*instr_settings.sig_an_settings.header()]
    # endregion

    # region Set up column titles and data for csv output.
    results_col_titles = results.std_output_column_titles()

    results_col_data = results.std_output_data()
    results_csv_data = [
        results_header_1, results_header_2, results_header_3,
        results_header_4, results_header_5, results_col_titles]

    # Add data for each frequency point measured
    for i, _ in enumerate(results.freq_array):
        results_csv_data.append(results_col_data[i, :])
    # endregion

    # region Save to results output csv
    file_struc.write_to_file(
        file_struc.output_file_path(file_struc.results_directory, meas_settings, 
                                 bias_id, 'csv'),
        results_csv_data, 'w', 'rows')
    log.info('Results pre-analysed, plotted, and saved.\n')
    # endregion


def save_calibration_results(
        be_biases: list[lnas.LNABiasSet], be_stages: list[lnas.StageBiasSet],
        settings: config_handling.Settings, results: outputs.Results) -> None:
    """Save calibration results and update calibration settings log.

    Args:
        be_biases: The backend LNA bias settings.
        be_stages: The backend LNA stage settings.
        settings: The measurement settings for the session.
        results: The results to save.
    """
    # region Unpack objects and set up logging.
    log = logging.getLogger(__name__)
    meas_settings = settings.meas_settings
    cryo_chain = settings.meas_settings.lna_cryo_layout.cryo_chain
    crbe_lna_bias = be_biases[0]
    rtbe_lna_bias = be_biases[1]
    crbe_stg = be_stages[0]
    rtbe_stg = be_stages[1]
    # endregion

    # region Read settings log, increment from max cal ID for chain.
    log.cdebug('Getting calibration ID.')
    cal_settings_log = pd.read_csv(settings.file_struc.cal_settings_path)
    cal_settings_log = np.array(cal_settings_log)
    chain_list = cal_settings_log[:, 1]
    cal_id_list = cal_settings_log[:, 2]
    trim_cal_id_list = []
    for entry, _ in enumerate(chain_list):
        if cryo_chain == chain_list[entry]:
            trim_cal_id_list.append(cal_id_list[entry])
    if len(trim_cal_id_list) != 0:
        cal_id = int(1 + max(trim_cal_id_list))
    else:
        cal_id = 1
    log.cdebug(f'Calibration ID = {cal_id}')
    # endregion

    # region Set file names for csv and plot.
    log.cdebug('Setting file names for csv and plot.')
    cal_output_path = pathlib.Path(
        str(settings.file_struc.cal_directory) + f'\\Chain {cryo_chain}')
    os.makedirs(cal_output_path, exist_ok=True)
    cal_csv_path = pathlib.Path(
        str(cal_output_path) + f'\\Chain {cryo_chain} '
        f'Calibration {cal_id}.csv')
    cal_png_path = pathlib.Path(
        str(cal_output_path) + f'\\Chain {cryo_chain} '
        f'Calibration {cal_id}.png')
    # endregion

    # region Send results to be plotted and saved.
    log.info('Saving plot...')
    _save_plot(
        results, settings.meas_settings, be_biases, cal_png_path,
        calibration_id=cal_id)
    log.info('Plot saved.')
    # endregion

    # region Add cal measurement settings to the cal settings log.
    log.info('Saving calibration settings...')
    cal_settings_col_data = [
        meas_settings.project_title, cryo_chain, str(cal_id),
        results.date_str, results.time_str, meas_settings.comment, None,
        *settings.instr_settings.sig_an_settings.spec_an_col_data(), None,
        *crbe_lna_bias.lna_set_column_data(True),
        *rtbe_lna_bias.lna_set_column_data(True), None,
        *crbe_lna_bias.lna_meas_column_data,
        *rtbe_lna_bias.lna_meas_column_data]

    settings.file_struc.write_to_file(
        settings.file_struc.cal_settings_path, cal_settings_col_data, 
        'a', 'row')
    log.info('Calibration settings saved.')
    # endregion

    # region Save calibration results.
    log.info('Saving results...')
    # region Set up cal results csv header.
    cal_results_header_1 = [
        'Chain', f'{cryo_chain}', 
        'Calibration:', f'{cal_id}',
        'Project:', f'{meas_settings.project_title}',
        'Date/Time:', f'{results.date_str} {results.time_str}',
        'Comment:', f'{meas_settings.comment}']

    cal_results_header_2 = [
        'CRBE:', *crbe_stg.header()[0:6], 'RTBE:', *rtbe_stg.header()[0:6]]

    cal_results_header_3 = [settings.instr_settings.sig_an_settings.header()]

    cal_results_col_titles = results.cal_output_column_titles()
    # endregion

    # region Add results to calibration result output csv.
    cal_results_col_data = results.cal_output_data()
    cal_results_csv_data = [cal_results_header_1, cal_results_header_2,
                            cal_results_header_3, cal_results_col_titles]

    for i, _ in enumerate(results.freq_array):
        cal_results_csv_data.append(cal_results_col_data[i, :])

    settings.file_struc.write_to_file(
        cal_csv_path, cal_results_csv_data, 'w', 'rows')

    log.info('Results plotted and saved.\n')
    # endregion
    # endregion
