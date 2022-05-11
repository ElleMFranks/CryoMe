"""metaproc.py - Completes data analysis on sets of results."""

# region Import modules.
from __future__ import annotations
import pathlib
import itertools
import math
import os
import re

import numpy as np

import pandas as pd

import plotting
import data_structs
# endregion

def _plot_data(user_settings: data_structs.SessionSettings, 
               file_struc: data_structs.MetaprocFileStructure,
               chain_data: data_structs.ChainData) -> None:
    """Plot the user input data."""
    
    bias_accuracy_plots = []

    lna_stages = itertools.product(
        np.array(range(chain_data.num_of_lnas)) + 1, 
        np.array(range(chain_data.num_of_stages)) + 1)

    for lna, stage in lna_stages:
        bias_accuracy_plots.append(
            plotting.BiasAccuracyPlot(
                *chain_data.get_bias_acc_data(lna, stage), 
                user_settings, lna, stage))

    plotting.BiasAccuracyPlot.show_bias_acc_plot()

    set_gain_map = plotting.GainMapData(
        chain_data.lna_1_stages.stage_1_data.set_biases)
    
    # region Plot heat maps.
    gain_heat_map = plotting.GainMapData()
    noise_temp_heat_map = plotting.NoiseMapData()
    # endregion
    
def _get_data(file_struc: data_structs.MetaprocFileStructure, 
              session_settings: data_structs.SessionSettings
              ) -> data_structs.ChainData:
    """Get the input data from the input files."""

    # region Load input data into arrays.
    res_log_data = np.array(
        pd.read_csv(file_struc.input_res_log_path, header=2))
    set_log_data = np.array(
        pd.read_csv(file_struc.input_set_log_path, header=1))
    # endregion

    # region Extract session data from input data.
    res_session_ids = res_log_data[:,2]
    set_session_ids = set_log_data[:,2]
    input_res_log_data = []
    input_set_log_data = []

    for i, session_id in enumerate(res_session_ids):
        if not math.isnan(session_id):
            if int(session_id) == int(session_settings.session_id):
                input_res_log_data.append(res_log_data[i,:])

    for i, session_id in enumerate(set_session_ids):
        if not math.isnan(session_id):
            if int(session_id) == int(session_settings.session_id):
                input_set_log_data.append(set_log_data[i,:])

    input_res_log_data = np.array(input_res_log_data)
    input_set_log_data = np.array(input_set_log_data)
    # endregion

    input_data = data_structs.ChainData(
        session_settings.session_id,
        data_structs.InputLogData(input_set_log_data, input_res_log_data))

    input_data.lna_1_id = session_settings.lna_1_id
    if input_data.num_of_lnas == 2:
        input_data.lna_2_id = session_settings.lna_2_id

    session_settings.num_of_lnas = input_data.num_of_lnas
    session_settings.num_of_stages = input_data.num_of_stages

    return input_data

def _get_user_inputs() -> tuple[data_structs.SessionSettings, 
                                data_structs.MetaprocFileStructure]:
    """Return the settings for the script to configure outputs."""

    # region Initialise variables for isinstance loops.
    results_path = None
    settings_path = None
    results_file = None
    settings_file = None
    session_directory = None
    # endregion

    # region Set up the file structure.
    while not isinstance(session_directory, pathlib.Path):
        try:
            session_directory = pathlib.Path(input(
                'Copy and paste the session folder path here: '))
            project_directory = os.path.dirname(
                os.path.dirname(session_directory))
            project_dir_files = os.listdir(project_directory)
            for log in project_dir_files:
                if 'Settings Log.csv' in log:
                    settings_path = pathlib.Path(f'{project_directory}\\{log}')
                if 'Results Log.csv' in log:
                    results_path = pathlib.Path(f'{project_directory}\\{log}')
            if settings_path == None or results_path == None:
                raise Exception('Cannot find logs.')
            proj_info = re.findall(
                r'\d+', os.path.basename(session_directory))
            session_id = proj_info[0]
            if len(proj_info) == 2:
                lna_1_id = f'{proj_info[1]}'
                lna_2_id = None
            elif len(proj_info) == 3:
                lna_2_id = f'{proj_info[2]}'
            else:
                raise Exception('Invalid session folder.')
            session_settings = data_structs.SessionSettings(
                session_id, lna_1_id, lna_2_id)
        except:
            session_directory = None
            continue

    file_struc = data_structs.MetaprocFileStructure(
        settings_path, results_path)
    # endregion

    return session_settings, file_struc

def main():
    """Main for metaproc."""
    # region Get user inputs.
    session_settings, file_struc = _get_user_inputs()
    # endregion

    # region Load in data.
    chain_data = _get_data(file_struc, session_settings)
    # endregion

    # region Plot and save input data.
    _plot_data(session_settings, file_struc, chain_data) 
    # endregion

if __name__ == '__main__':
    main()
# endregion
