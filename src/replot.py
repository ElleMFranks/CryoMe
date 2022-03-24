"""replot.py - Re plots an imported result csv with new limits.

Todo:
    * Figure out how to import data for bias table.
"""

# region Import modules
from __future__ import annotations
from dataclasses import dataclass
from time import sleep
from typing import Union
import os
import pathlib
import re
import sys

from matplotlib import offsetbox, ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tabulate
from instruments import FreqSweepSettings

import output_saving
# endregion


@dataclass
class TrimmedInCalData:
    """Set of the trimmed inputs to plot for a calibration plot."""
    freqs: np.ndarray
    loss_cor_noise_temp: np.ndarray


@dataclass
class TrimmedInStdData:
    """Set of the trimmed inputs to plot for a standard plot."""
    freqs: np.ndarray
    uncal_loss_uncor: np.ndarray 
    uncal_loss_cor: np.ndarray
    cal_loss_cor: np.ndarray
    gain: np.ndarray


@dataclass
class UserInputs:
    dark_mode_plot: bool
    replot_all_in_session_folder: bool
    new_noise_y_limit: int
    new_gain_y_limit: int
    input_path: pathlib.Path


@dataclass
class CalMetaData:
    """Meta data for the input calibration data."""
    cal_id: int
    cryo_chain: int
    user_inputs: UserInputs


@dataclass
class StdMetaData:
    """Meta data for the standard input data."""
    bias_id: int
    session_id: int
    lna_id_str: str
    measure_method: str
    project_title: str
    user_inputs: UserInputs


def replot_data(
        trimmed_data: Union[TrimmedInCalData, TrimmedInStdData], 
        meta_data: Union[StdMetaData, CalMetaData]) -> None:
    """Re-plots the trimmed data."""
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
    if meta_data.dark_mode_plot:
        font_color = light
        fig.set_facecolor(dark)
    else:
        font_color = dark
        fig.set_facecolor(light)
    # endregion

    # region Create bias data text box to go on plot.
    tabulate.PRESERVE_WHITESPACE = True

    if lna_2_bias is None or isinstance(meta_data, CalMetaData):
        lna_2_g_v_data = ['NA', 'NA', 'NA']
        lna_2_d_v_data = ['NA', 'NA', 'NA']
        lna_2_d_i_data = ['NA', 'NA', 'NA']

    else:
        lna_2_g_v_data = lna_2_bias.lna_g_v_strs()
        lna_2_d_v_data = lna_2_bias.lna_d_v_strs()
        lna_2_d_i_data = lna_2_bias.lna_d_i_strs()

    if isinstance(meta_data, StdMetaData):
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
    axis.set_xlim(trimmed_data.freqs[0], 
                  trimmed_data.freqs[len(trimmed_data.freqs) - 1])
    axis.set_ylim(0, new_y_limit)
    axis.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    axis.yaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    axis.grid(linestyle='--', which='major', linewidth=pixel_line_width)
    axis.grid(linestyle='--', which='minor', linewidth=pixel_line_width - 0.3)
    axis.tick_params('both', colors=font_color, labelsize=lab_font)
    # endregion

    # region Handle calibration results.
    if isinstance(meta_data, CalMetaData):
        axis.set_ylim(0, new_y_limit)
        plot_title = 'Noise Temperature / Frequency'
        plot_details = (
                f'Cryostat Chain: {meta_data.cryo_chain}  '
                + f'Calibration: {meta_data.cal_id}')
        plot_detail_box = offsetbox.AnchoredText(
            plot_details, loc='upper right')
        axis.add_artist(plot_detail_box)
        axis.set_title(plot_title, color=font_color, fontsize=lab_font + 2)
        axis.plot(trimmed_data.freqs, trimmed_data.loss_cor_noise_temp,
                  color=orange, linewidth=pixel_line_width)
    # endregion

    # region Handle standard results.
    else:
        # region Set title, and plot details text box.
        plot_title = 'Noise Temperature & Gain / Frequency'
        plot_details = f'Project: {meta_data.project_title}' \
                       + f' - LNA ID/s: {meta_data.lna_id_str}' \
                       + f' - Session ID: {meta_data.session_id}' \
                       + f' - Bias ID: {meta_data.bias_id}'
        plot_detail_box = offsetbox.AnchoredText(
            plot_details, loc='upper right')
        axis.add_artist(plot_detail_box)
        # endregion

        # region Plot the noise
        axis.set_title(plot_title, color=font_color, fontsize=lab_font + 2)
        noise_temp_cor_plot = axis.plot(
            trimmed_data.freqs, trimmed_data.cal_loss_cor, color=blue,
            linewidth=pixel_line_width, label='Calibrated & Loss Corrected')
        t_corrected_plot = axis.plot(
            trimmed_data.freqs, trimmed_data.uncal_loss_cor,
            color=orange, linewidth=pixel_line_width,
            label='Uncalibrated & Loss Corrected')
        t_uncal_plot = axis.plot(
            trimmed_data.freqs, trimmed_data.uncal_loss_uncor,
            color=green, linewidth=pixel_line_width,
            label='Uncalibrated & Not Loss Corrected')
        ax2 = axis.twinx()
        gain_plot = ax2.plot(
            trimmed_data.freqs, trimmed_data.gain, color=dark,
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
    fig.savefig(png_path)
    # fig.show()
    sleep(0.5)  # Pause before/after close - box glitch.
    plt.close('all')
    sleep(0.5)
    # endregion

def trim_imported_data(
        imported_data:np.ndarray, 
        meta_data: Union[CalMetaData, StdMetaData]) -> np.ndarray:
    """Trims the imported data to the relevant data."""
    freqs = imported_data[:, 0]

    if isinstance(meta_data, StdMetaData):
        uncal_loss_uncor = imported_data[:, 11]
        uncal_loss_cor = imported_data[:, 12]
        cal_loss_cor = imported_data[:, 13]
        gain = imported_data[:, 14]
        return TrimmedInStdData(freqs, uncal_loss_uncor, uncal_loss_cor,
                                cal_loss_cor, gain)
    elif isinstance(meta_data, CalMetaData):
        loss_cor_noise_temp = imported_data[:, 8]
        return TrimmedInCalData(freqs, loss_cor_noise_temp)
    else:
        raise Exception('Invalid meta data.')


def get_user_inputs() -> UserInputs:
    """Returns validated user inputs."""
    dark_mode_plot = None
    replot_all_in_session_folder = None
    new_y_noise_limit = None
    new_y_gain_limit = None
    results_folder = None
    results_csv_title = None
    
    while not isinstance(dark_mode_plot, bool):
        dark_mode_plot = input('New plot in dark mode? (y/n):')
            if dark_mode_plot == 'y':
                dark_mode_plot = True
            if dark_mode_plot == 'n':
                dark_mode_plot = False

    while not isinstance(replot_all_in_session_folder, bool):
        replot_all_in_session_folder = input(
            'Replot all in session folder? (y/n):')
            if replot_all_in_session_folder = 'y':
                replot_all_in_session_folder = True
            if replot_all_in_session_folder = 'n':
                replot_all_in_session_folder = False

    while not isinstance(new_noise_y_limit, int):
        try:
            new_noise_y_limit = int(
                input('Please enter new integer noise y limit:'))
        except:
            continue

    while not isinstance(new_gain_y_limit, int):
        try:
            new_gain_y_limit = int(
                input('Please enter new integer gain y limit:'))
        except:
            continue
    
    while not isinstance(results_folder, pathlib.Path):
        try:
            results_folder = pathlib.Path(input(
                'Copy and paste full folder path here: '))
        except:
            continue

    while not isinstance(result_csv_title, pathlib.Path):
        try:
            results_csv_title = pathlib.Path(input(
                'Copy and paste csv name to be replotted: '))
        except:
            continue

    input_path = pathlib.Path(f'{results_folder}\\{results_csv_title}')
    
    return UserInputs(dark_mode_plot, replot_all_in_session_folder, 
                      new_noise_y_limit, new_gain_y_limit, input_path)

def pull_data(meta_data: Union[CalMetaData, StdMetaData]) -> np.ndarray:
    """Imports the user requested CSV data."""
    data_imported = False
    while not data_imported:
        try:
            if isinstance(meta_data, StdMetaData):
                input_data = pd.read_csv(pathlib.Path(
                    f'{meta_data.user_inputs.input_path}.csv', header=5)
                data_imported = True
            elif isinstance(meta_data, CalMetaData):
                input_data = pd.read_csv(pathlib.Path(
                    f'{meta_data.user_inputs.input_path}.csv', header=4)
                data_imported = True
        except:
            try:
                results_folder = input(
                    'Copy and paste full folder path here:')
                results_csv_title = input(
                    'Copy and paste csv name to be replotted:')
                meta_data.user_inputs.input_path = pathlib.Path(
                    f'{results_folder}\\{results_csv_title}')
            except:
                continue

def pull_lna_data(meta_data: Union[CalMetaData, StdMetaData]) -> np.ndarray:
    """Pulls LNA data from results csv."""
    input_data = pd.read_csv(pathlib.Path(
        f'{meta_data.user_inputs.input_path}.csv'))

    if isinstance(meta_data, StdMetaData):
        lna_1_biases = {
            'l1_s1_gv': input_data[1].2,
            'l1_s1_dv': input_data[1].4,
            'l1_s1_di': input_data[1].6,
            'l1_s2_gv': input_data[1].9,
            'l1_s2_dv': input_data[1].11,
            'l1_s2_di': input_data[1].13,
            'l1_s3_gv': input_data[1].16,
            'l1_s3_dv': input_data[1].17,
            'l1_s3_di': input_data[1].18}
        
        lna_2_biases = {
            'l2_s1_gv': input_data[2].2,
            'l2_s1_dv': input_data[2].4,
            'l2_s1_di': input_data[2].6,
            'l2_s2_gv': input_data[2].9,
            'l2_s2_dv': input_data[2].11,
            'l2_s2_di': input_data[2].13,
            'l2_s3_gv': input_data[2].16,
            'l2_s3_dv': input_data[2].17,
            'l2_s3_di': input_data[2].18}

    if isinstance(meta_data, CalMetaData):
        rtbe_biases = {
            'rtbe_gv': input_data[1].2,
            'rtbe_dv': input_data[1].4,
            'rtbe_di': input_data[1].6}
        crbe_biases = {
            'crbe_gv': input_data[1].9,
            'crbe_dv': input_data[1].11,
            'crbe_di': input_data[1].13}

def get_meta_data(user_inputs: UserInputs) -> Union[CalMetaData, StdMetaData]:
    """Returns the meta data for the new plot."""
    

    # region Get useful information from the path/file names.
    path_data = re.findall(
        '\d+', f'{user_inputs.input_path.split("\\")[-1]}')

    # region Calibration data.
    if 'calibrations' in f'{user_inputs.input_path}':
        cryo_chain = path_data[0]
        cal_id = path_data[1]
        return CalMetaData(cal_id, cryo_chain, user_inputs)
    # endregion
    # region Standard data
    else:
        session_id = str_data[0]
        bias_id = path_data[len(path_data-1)]
        project_title = input_csv_path.split('\\')[-3]
        path_data = input_csv_path.split('\\')[-2]
        lna_id_str = path_data[16:22]
        measure_method = path_data[27:]
        return StdMetaData(bias_id, session_id, lna_id_str, 
                           measure_method, project_title, user_inputs)
    # endregion
    # endregion

def main():
    """Main for replot."""

    # region Get user inputs.
    user_inputs = get_user_inputs()
    # endregion

    # region Get meta data.
    meta_data = get_meta_data(user_inputs)
    # endregion

    # region Import noise temp/gain data.
    imported_data = pull_data(meta_data)
    # endregion

    # region Import LNA data.
    imported_lna_data = pull_lna_data(meta_data)

    # region Trim imported data.
    trimmed_data = trim_imported_data(imported_data, meta_data)
    # endregion

    # region Replot and save imported data.
    replot_data(trimmed_data, meta_data)
    # endregion

if __name__ == '__main__':
    main()
