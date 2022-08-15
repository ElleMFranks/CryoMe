"""replot.py - Re plots an imported result csv with new limits.

Todo:
    * Figure out how to import data for bias table.
"""

# region Import modules
from __future__ import annotations
from dataclasses import dataclass, replace
from time import sleep
from typing import Union
import csv
import multiprocessing
import os
import pathlib
import re

from matplotlib import offsetbox, ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tabulate
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
    input_paths: Union[pathlib.Path, list[pathlib.Path]]


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


@dataclass
class PlotColours:
    dark: str
    light: str
    blue: str
    green: str
    orange: str


@dataclass
class PlotVars:
    label_font_size: int
    pixel_line_width: float
    font_colour: str
    colours: PlotColours


def config_plot(meta_data: Union[StdMetaData, CalMetaData],
                trimmed_data: Union[TrimmedInStdData, 
                                    TrimmedInCalData]) -> tuple[plt.Figure, 
                                                                plt.Axes, 
                                                                PlotVars]:
    """Configures fig/axis/plot_vars objects."""
    # region Configure objects.
    # dark, light, blue, green, orange
    colours = PlotColours(
        '#181818', '#ffffff', '#030bfc', '#057510', '#f74600')
    pixel_line_width = 1
    label_font_size = 14
    x_label = 'Frequency / GHz'
    y_label = 'Noise Temperature / Kelvin'
    fig, axis = plt.subplots()

    # region Configure fig object to 1080p either light or dark mode.
    fig.set_size_inches(20.92, 11.77)
    fig.set_dpi(91.79)
    fig.tight_layout(pad=5)
    if meta_data.user_inputs.dark_mode_plot:
        font_colour = colours.light
        fig.set_facecolor(colours.dark)
    else:
        font_colour = colours.dark
        fig.set_facecolor(colours.light)
    # endregion

    # region Configure axis object.
    axis.set_xlim(trimmed_data.freqs.min(), trimmed_data.freqs.max())
    axis.set_ylim(0, meta_data.user_inputs.new_noise_y_limit)
    axis.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    axis.yaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    axis.grid(linestyle='--', which='major', linewidth=pixel_line_width)
    axis.grid(linestyle='--', which='minor', linewidth=pixel_line_width - 0.3)
    axis.tick_params('both', colors=font_colour, labelsize=label_font_size)
    axis.set_xlabel(x_label, color=font_colour, fontsize=label_font_size)
    axis.set_ylabel(y_label, color=font_colour, fontsize=label_font_size)
    # endregion

    # region Create plot_vars object as first output.
    plot_vars = [label_font_size, pixel_line_width, font_colour, colours]
    plot_vars = PlotVars(*plot_vars)
    # endregion

    return fig, axis, plot_vars
    # endregion

def get_bias_box(imported_lna_data: list[dict]) -> offsetbox:
    """Returns an offsetbox with lna biasing data."""
    # region Create bias data text box to go on plot.
    tabulate.PRESERVE_WHITESPACE = True

    lna_1 = imported_lna_data[2]
    lna_2 = imported_lna_data[3]
    
    # region Unpack dictionaries.
    lna_1_g_v_data = [lna_1['s1_gv'], lna_1['s2_gv'], lna_1['s3_gv']]
    lna_1_d_v_data = [lna_1['s1_dv'], lna_1['s2_dv'], lna_1['s3_dv']]
    lna_1_d_i_data = [lna_1['s1_di'], lna_1['s2_di'], lna_1['s3_di']]
    lna_2_g_v_data = [lna_2['s1_gv'], lna_2['s2_gv'], lna_2['s3_gv']]
    lna_2_d_v_data = [lna_2['s1_dv'], lna_2['s2_dv'], lna_2['s3_dv']]
    lna_2_d_i_data = [lna_2['s1_di'], lna_2['s2_di'], lna_2['s3_di']]
    # endregion

    # region Put bias data into table.
    bias_header = ['Bias', 'L1S1', 'L1S2', 'L1S3', 'L2S1', 'L2S2', 'L2S3']
    bias_row_1 = ['VG / V', *lna_1_g_v_data, *lna_2_g_v_data]
    bias_row_2 = ['VD / V', *lna_1_d_v_data, *lna_2_d_v_data]
    bias_row_3 = ['ID / mA', *lna_1_d_i_data, *lna_2_d_i_data]

    bias_table = tabulate.tabulate(
        [bias_header, bias_row_1, bias_row_2, bias_row_3],
        headers="firstrow", disable_numparse=True, tablefmt="plain")
    plot_bias_box = offsetbox.AnchoredText(bias_table, loc='lower right')
    # endregion

    return plot_bias_box
    # endregion

def make_cal_plot(
        axis: plt.Axes, meta_data: CalMetaData, plot_vars: PlotVars,
        trimmed_data: TrimmedInCalData) -> plt.Axes:
    """Plots calibration plot."""
    # region Plot calibration.
    plot_title = 'Noise Temperature / Frequency'
    plot_details = (
            f'Cryostat Chain: {meta_data.cryo_chain}  '
            + f'Calibration: {meta_data.cal_id}')
    plot_detail_box = offsetbox.AnchoredText(
        plot_details, loc='upper right')
    axis.add_artist(plot_detail_box)
    axis.set_title(plot_title, color=plot_vars.font_colour, 
                   fontsize=plot_vars.label_font_size + 2)
    axis.plot(trimmed_data.freqs, trimmed_data.loss_cor_noise_temp,
              color=plot_vars.colours.orange, 
              linewidth=plot_vars.pixel_line_width)
    return axis
    # endregion

def make_std_plot(
        axis: plt.Axes, meta_data: StdMetaData, plot_vars: PlotVars,
        trimmed_data: TrimmedInStdData) -> plt.Axes:
    """Plots standard plot."""
    # region Plot standard results.
    # region Set title, and plot details text box.
    plot_title = 'Noise Temperature & Gain / Frequency'

    plot_details = f'Project: {meta_data.project_title}' \
                   + f' - LNA ID/s: {meta_data.lna_id_str}' \
                   + f' - Session ID: {meta_data.session_id}' \
                   + f' - Bias ID: {meta_data.bias_id}'

    plot_labels = ['Noise','Gain']

    #plot_detail_box = offsetbox.AnchoredText(plot_details, loc='upper right')
    # endregion

    # region Plot the noise
    ax2 = axis.twinx()

    #axis.add_artist(plot_detail_box)

    axis.set_title(plot_title, 
                   color=plot_vars.font_colour, 
                   fontsize=plot_vars.label_font_size + 2)

    noise_temp_cor_plot = axis.plot(trimmed_data.freqs, 
                                    trimmed_data.cal_loss_cor, 
                                    color=plot_vars.colours.blue,
                                    linewidth=plot_vars.pixel_line_width, 
                                    label=plot_labels[0])
    
    gain_plot = ax2.plot(trimmed_data.freqs, 
                         trimmed_data.gain, 
                         color=plot_vars.colours.dark,
                         linewidth=plot_vars.pixel_line_width, 
                         label=plot_labels[1])
    # endregion

    # region Create legend, and set up Gain axis.
    plots = noise_temp_cor_plot + gain_plot
    labels = (plot.get_label() for plot in plots)

    axis.legend(plots, 
                labels, 
                loc='lower left',
                numpoints=1, 
                fontsize=plot_vars.label_font_size,
                ncol=4)

    ax2.set_ylabel('Gain / dB', 
                   color=plot_vars.font_colour, 
                   fontsize=plot_vars.label_font_size)

    ax2.tick_params(labelcolor=plot_vars.font_colour, 
                    labelsize=plot_vars.label_font_size)

    ax2.set_ylim(0, meta_data.user_inputs.new_gain_y_limit)
    #axis.add_artist(get_bias_box(imported_lna_data))
    # endregion

    return axis
    # endregion

def replot_data(
        trimmed_data: Union[TrimmedInCalData, TrimmedInStdData], 
        meta_data: Union[StdMetaData, CalMetaData]) -> None:
    """Re-plots the trimmed data."""
    # region Replot data.
    # region Set plot variables.
    fig, axis, plot_vars = config_plot(meta_data, trimmed_data)
    # endregion

    # region Handle results.
    # region Handle calibration results.
    if isinstance(meta_data, CalMetaData):
        axis = make_cal_plot(
            axis, meta_data, plot_vars, trimmed_data)
    # endregion

    # region Handle standard results.
    elif isinstance(meta_data, StdMetaData):
        axis = make_std_plot(
            axis, meta_data, plot_vars, trimmed_data)
    # endregion

    else:
        raise Exception('Invalid meta data.')
    # endregion

    # region Save and close plots.
    unsaved = True
    while unsaved:
        try:
            fig.savefig(f'{meta_data.user_inputs.input_paths}_replot.png')
            unsaved = False
        except:
            print(f'Please close {meta_data.user_inputs.input_path}.png')
            continue
    sleep(0.5)  # Pause before/after close - box glitch.
    plt.close('all')
    sleep(0.5)
    # endregion
    # endregion

def trim_imported_data(
        imported_data:np.ndarray, 
        meta_data: Union[CalMetaData, StdMetaData]) -> np.ndarray:
    """Trims the imported data to the relevant data."""
    # region Trim data.
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
    # endregion

def get_user_inputs() -> UserInputs:
    """Returns validated user inputs."""
    dark_mode_plot = None
    replot_all_in_session_folder = None
    new_noise_y_limit = None
    new_gain_y_limit = None
    results_folder = None
    results_csv_title = None
    
    # while not isinstance(dark_mode_plot, bool):
    #     dark_mode_plot = input('New plot in dark mode? (y/n): ')
    #     if dark_mode_plot == 'y':
    #         dark_mode_plot = True
    #     if dark_mode_plot == 'n':
    #         dark_mode_plot = False

    # while not isinstance(replot_all_in_session_folder, bool):
    #     replot_all_in_session_folder = input(
    #         'Replot all in session folder? (y/n): ')
    #     if replot_all_in_session_folder == 'y':
    #         replot_all_in_session_folder = True
    #     if replot_all_in_session_folder == 'n':
    #         replot_all_in_session_folder = False

    # while not isinstance(new_noise_y_limit, int):
    #     try:
    #         new_noise_y_limit = int(
    #             input('Please enter new integer noise y limit: '))
    #     except:
    #         continue

    # while not isinstance(new_gain_y_limit, int):
    #     try:
    #         new_gain_y_limit = int(
    #             input('Please enter new integer gain y limit: '))
    #     except:
    #         continue
  
    dark_mode_plot = False
    replot_all_in_session_folder = True
    new_noise_y_limit = 100
    new_gain_y_limit = 25
  
    
    while not isinstance(results_folder, pathlib.Path):
        try:
            results_folder = pathlib.Path(input(
                'Copy and paste full folder path here: '))
        except:
            continue
    
    if not replot_all_in_session_folder:
        while not isinstance(results_csv_title, pathlib.Path):
            try:
                results_csv_title = pathlib.Path(input(
                    'Copy and paste csv name to be replotted: '))
                input_paths = pathlib.Path(
                    f'{results_folder}\\{results_csv_title}')
            except:
                continue
    else:
        input_paths = []
        for root,dirs,files in os.walk(results_folder):
            for file in files:
                if file.endswith('.csv'):
                    input_paths.append(
                        pathlib.Path(f'{results_folder}\\{file[:-4]}'))

    return UserInputs(dark_mode_plot, replot_all_in_session_folder, 
                      new_noise_y_limit, new_gain_y_limit, input_paths)

def pull_data(meta_data: Union[CalMetaData, StdMetaData]) -> np.ndarray:
    """Imports the user requested CSV data."""
    data_imported = False
    while not data_imported:
        try:
            if isinstance(meta_data, StdMetaData):
                input_data = np.array(pd.read_csv(pathlib.Path(
                    f'{meta_data.user_inputs.input_paths}.csv'), header=6))
                return input_data
            elif isinstance(meta_data, CalMetaData):
                input_data = np.array(pd.read_csv(pathlib.Path(
                    f'{meta_data.user_inputs.input_paths}.csv'), header=5))
                return input_data
        except Exception as _e:
            print(f'{_e}')
            try:
                results_folder = input(
                    'Copy and paste full folder path here:')
                results_csv_title = input(
                    'Copy and paste csv name to be replotted: ')
                meta_data.user_inputs_ = replace(
                    meta_data.user_inputs, 
                    input_paths = pathlib.Path(
                    f'{results_folder}\\{results_csv_title}'))
            except:
                continue


def get_meta_data(user_inputs: UserInputs) -> Union[CalMetaData, StdMetaData]:
    """Returns the meta data for the new plot."""
    

    # region Get useful information from the path/file names.
    input_csv_path = f'{user_inputs.input_paths}.csv'
    split_path = str(user_inputs.input_paths).split("\\")[-1]
    path_data = re.findall('\d+', f'{split_path}')

    # region Calibration data.
    if 'calibrations' in f'{user_inputs.input_paths}':
        cryo_chain = path_data[0]
        cal_id = path_data[1]
        return CalMetaData(cal_id, cryo_chain, user_inputs)
    # endregion
    # region Standard data
    else:
        session_id = path_data[0]
        bias_id = path_data[len(path_data)-1]
        project_title = input_csv_path.split('\\')[-3]
        path_data = input_csv_path.split('\\')[-2]
        lna_id_str = path_data[16:22]
        measure_method = path_data[27:]
        return StdMetaData(bias_id, session_id, lna_id_str, 
                           measure_method, project_title, user_inputs)
    # endregion
    # endregion

def replot(process_number: int, user_inputs: UserInputs) -> None:

    # region Setup instance user inputs.
    if user_inputs.replot_all_in_session_folder:
        instance_user_inputs = UserInputs(
            user_inputs.dark_mode_plot, 
            user_inputs.replot_all_in_session_folder,
            user_inputs.new_noise_y_limit, 
            user_inputs.new_gain_y_limit,
            user_inputs.input_paths[process_number])
    else:
        instance_user_inputs = user_inputs
    # endregion

    # region Get meta data.
    meta_data = get_meta_data(instance_user_inputs)
    # endregion

    # region Import noise temp/gain data.
    imported_data = pull_data(meta_data)
    # endregion

    # region Import LNA data.
    #imported_lna_data = pull_lna_data(meta_data)
    # endregion

    # region Trim imported data.
    trimmed_data = trim_imported_data(imported_data, meta_data)
    # endregion

    # region Replot and save imported data.
    #replot_data(trimmed_data, meta_data, imported_lna_data)
    replot_data(trimmed_data, meta_data)
    
    print(f'File {process_number} replotted.')
    # endregion

def main():
    """Main for replot."""
    # region Get user inputs.
    user_inputs = get_user_inputs()
    # endregion

    # region Trigger replots using multiprocessing.
    if isinstance(user_inputs.input_paths, list):
        processes = []
        for i, _ in enumerate(user_inputs.input_paths):
            p = multiprocessing.Process(target=replot, args=[i, user_inputs])
            p.start()
            processes.append(p)

        for process in processes:
            process.join()
    else:
        replot(0, user_inputs)
    # endregion

if __name__ == '__main__':
    main()
