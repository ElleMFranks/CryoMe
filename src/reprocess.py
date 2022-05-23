# region Import modules.
from __future__ import annotations
from dataclasses import dataclass
from typing import Union, Optional
import csv
import multiprocessing
import os
import pathlib
import sys

import numpy as np

import config_handling
import outputs
import replot
# endregion


@dataclass
class InputFiles:
    result_folder_path: pathlib.Path
    result_csv_title: Union[str, list[str]]
    calibration_folder_path: pathlib.Path
    calibration_title: str
    loss_folder_path: pathlib.Path
    loss_title: str


@dataclass
class UserInputs:
    input_files: InputFiles
    cal_id: int
    chain: int
    reprocess_all_in_session_folder: bool
    new_loss: bool


class ReprocFolderStructure:

    def __init__(self, user_inputs: UserInputs):
        self.input_files = user_inputs.input_files
        self.reproc_folder = pathlib.Path(
            f'{self.input_files.result_folder_path}\\'
            + f'Reprocess Ch{user_inputs.chain} Cal{user_inputs.cal_id}')
        self.reproc_csv_path = pathlib.Path(
            f'{self.reproc_folder}\\'
            + f'{str(self.input_files.result_csv_title)[:-4]} Reproc.csv')
        os.makedirs(self.reproc_folder, exist_ok=True)


def _trim_input_cal_data(
        input_cal_data: list, input_data: list,
        input_loss_data: Optional[list]) -> tuple(list, list):
    """Trims results to frequency points in new calibration."""
    input_cal_data = np.array(input_cal_data)
    input_data = np.array(input_cal_data)

    # region Return calibration rows for each frequency in input data.
    trimmed_cal_data = []
    for i, _ in enumerate(input_data):
        in_freq = input_data[i, 0]
        in_cal_freq = input_cal_data[i, 0]
        if in_freq == in_cal_freq:
            trimmed_cal_data.append(input_cal_data[i,:])

    trimmed_loss_data = []
    if isinstance(input_loss_data, list):
        for i, _ in enumerate(input_data):
            if input_data[i, 0] == input_loss_data[i, 0]:
                trimmed_loss_data = input_loss_data[i, 1]
    else:
        loss_data = np.array(input_cal_data)
        trimmed_loss_data = loss_data[:, 1]
        trimmed_loss_data = trimmed_loss_data.astype(float)

    if len(trimmed_cal_data) == len(input_cal_data) and \
            len(trimmed_loss_data) == len(input_cal_data):
        return list(trimmed_cal_data), list(trimmed_loss_data)

    raise Exception('New calibration not compatible with ' 
                    + 'requested results, check frequencies.')
    # endregion

def _get_input_data(
        file_structure: ReprocFolderStructure) -> tuple[list, list]:
    """Return data from individual results file."""

    # region Import results data and return header & non-header rows.
    input_files = file_structure.input_files
    with open(pathlib.Path(
            f'{input_files.result_folder_path}\\'
            + f'{input_files.result_csv_title}')) as csv_file:
        csv_reader = csv.reader(csv_file)
        rows = list(csv_reader)
    header = rows[0:6]
    data = rows[6:]
    return header, data
    # endregion

def _get_cal_info(file_structure: ReprocFolderStructure) -> list:
    """Return the new calibration data to reprocess with."""

    # region Import calibration csv and return data.
    input_files = file_structure.input_files
    with open(pathlib.Path(
            f'{input_files.calibration_folder_path}\\'
            + f'{input_files.calibration_title}')) as csv_file:
        csv_reader = csv.reader(csv_file)
        rows = list(csv_reader)
    return rows[4:]
    # endregion

def _get_loss(file_structure: ReprocFolderStructure) -> list:
    """Return the new loss data."""

    # region Open the loss file, read out and return non=header rows.
    input_files = file_structure.input_files
    with open(pathlib.Path(
        f'{input_files.loss_folder_path}\\{input_files.loss_title}.csv')
              ) as csv_file:
        csv_reader = csv.reader(csv_file)
        rows = list(csv_reader)
    return rows[1:]
    # endregion

def _get_results(trimmed_input_cal_data: list,
                 input_loss: list, input_file_data: list) -> outputs.Results:
    """Returns reprocessed results."""

    # region Convert to numpy arrays for ease of manipulation
    trimmed_input_cal_data = np.array(trimmed_input_cal_data)
    input_file_data = np.array(input_file_data)
    # endregion

    # region Build up results object from input results file.
    freqs = list(trimmed_input_cal_data[:,0].astype(float))

    cold_pre_post_temps = outputs.PrePostTemps(
        list(input_file_data[:,15].astype(float)),
        list(input_file_data[:,16].astype(float)),
        list(input_file_data[:,19]),
        list(input_file_data[:,20]),
        list(input_file_data[:,23]),
        list(input_file_data[:,24]))

    hot_pre_post_temps = outputs.PrePostTemps(
        list(input_file_data[:,17].astype(float)), 
        list(input_file_data[:,18].astype(float)),
        list(input_file_data[:,21]), 
        list(input_file_data[:,22]),
        list(input_file_data[:,25]), 
        list(input_file_data[:,26]))

    cold = outputs.LoopInstanceResult(
        'cold', list(input_file_data[:,1].astype(float)), 
        list(input_file_data[:,5].astype(float)), 
        list(input_file_data[:,6].astype(float)), cold_pre_post_temps)

    hot = outputs.LoopInstanceResult(
        'hot', list(input_file_data[:,2].astype(float)), 
        list(input_file_data[:,3].astype(float)), 
        list(input_file_data[:,4].astype(float)), hot_pre_post_temps)

    loop_pair = outputs.LoopPair(cold, hot)
    ana_bws = config_handling.AnalysisBandwidths(None, None, None, None, None)

    results_meta_info = outputs.ResultsMetaInfo(
        None, freqs, 1, False, ana_bws, input_loss, trimmed_input_cal_data)
    # endregion 

    # region Return fully built results object.
    return outputs.Results(loop_pair, results_meta_info)
    # endregion

def _resave_results(
    file_structure: ReprocFolderStructure, results: outputs.Results,
    output_header: list[str]) -> None:
    """Plots and saves reprocessed results."""

    # region Join header and results together, write to file.
    data = []
    data.extend(output_header)
    data.extend(results.std_output_data())
    out_path = file_structure.reproc_csv_path
    config_handling.FileStructure.write_to_file(out_path, data, 'w', 'rows')
    # endregion

    # region Plot output using replot.
    out_path = str(file_structure.reproc_csv_path)[:-4]
    out_path = pathlib.Path(out_path)
    replot_user_inputs = replot.UserInputs(True, False, 160, 40, out_path)
    replot.replot(0, replot_user_inputs)
    # endregion

def reprocess(process_number: int, user_inputs: UserInputs):
    """Reprocess input results with given calibration and loss inputs.
    """
    # region Remake User inputs for specific process.
    if not isinstance(user_inputs.input_files.result_csv_title, str):
        input_files = InputFiles(
            user_inputs.input_files.result_folder_path,
            user_inputs.input_files.result_csv_title[process_number],
            user_inputs.input_files.calibration_folder_path,
            user_inputs.input_files.calibration_title,
            user_inputs.input_files.loss_folder_path,
            user_inputs.input_files.loss_title)

        user_inputs = UserInputs(
            input_files, user_inputs.cal_id, 
            user_inputs.chain, user_inputs.reprocess_all_in_session_folder, 
            user_inputs.new_loss)
    # endregion

    # region Configure file structure to save new results to.
    file_struc = ReprocFolderStructure(user_inputs)
    # endregion

    # region Get input calibration data.
    input_cal_data = _get_cal_info(file_struc)
    # endregion

    # region Get loss array if required.
    if user_inputs.new_loss:
        input_loss = _get_loss(file_struc)
    else:
        input_loss = None
    # endregion

    # region Get arrays for reprocessing from data to be reprocessed.
    output_header, input_file_data = _get_input_data(file_struc)
    # endregion

    # region Trim calibration data.
    trimmed_input_cal_data, trimmed_input_loss = _trim_input_cal_data(
        input_cal_data, input_file_data, input_loss)
    # endregion

    # region Send individual arrays to be made into new Results object.
    reproc_results = _get_results(
        trimmed_input_cal_data, trimmed_input_loss, input_file_data)
    # endregion

    # region Send results object to be plotted and saved.
    _resave_results(file_struc, reproc_results, output_header)
    # endregion

def get_user_inputs() -> UserInputs:
    """Returns an object containing user input variables for instance.
    """
    # region Initialise variables.
    input_files = None
    cal_id = None
    chain = None
    reprocess_all_in_session_folder = None
    new_loss = None
    new_loss_path = None
    result_path = None
    result_csv_title = None
    # endregion

    # region replot_all_in_session_folder
    while not isinstance(reprocess_all_in_session_folder, bool):
        reprocess_all_in_session_folder = input(
            'Reprocess all in session folder? (y/n): ')
        if reprocess_all_in_session_folder == 'y':
            reprocess_all_in_session_folder = True
        if reprocess_all_in_session_folder == 'n':
            reprocess_all_in_session_folder = False
    # endregion
    
    # region Results inputs.
    result_check = False
    while not result_check:
        try:
            result_folder = pathlib.Path(input(
                'Copy and paste folder of results file: '))
            if not reprocess_all_in_session_folder:
                result_csv_title = input(
                    'Copy and paste title of result file here: ')
                result_csv_title += '.csv'
                result_path = pathlib.Path(
                    f'{result_folder}\\{result_csv_title}')
                result_check = open(result_path)
                result_check.close()
                result_check = True
            else: 
                result_csv_title = []
                for root,dirs,files in os.walk(result_folder):
                    for file in files:
                        if file.endswith('.csv'):
                            result_csv_title.append(file)
                for csv in result_csv_title:
                    result_path = pathlib.Path(f'{result_folder}\\{csv}')
                    result_check = open(result_path)
                    result_check.close()
                    result_check = True
        except Exception as _e:
            print(f'{_e}')
            input('Please check results file(s).')
    # endregion

    cal_check = False
    # region Calibration inputs.
    while not cal_check:
        try:
            chain = int(input(
                'Please enter the cryostat chain of the new calibration: '))
            cal_id = int(input(
                'Please enter the new calibration ID: '))
            new_cal_csv_title = f'Chain {chain} Calibration {cal_id}.csv'
            new_cal_folder = f'{os.getcwd()}\\calibrations\\Chain {chain}'
            new_cal_folder = pathlib.Path(new_cal_folder)
            new_cal_csv_path = pathlib.Path(
                f'{new_cal_folder}\\{new_cal_csv_title}')
            cal_check = open(new_cal_csv_path)
            cal_check.close()
            cal_check = True
        except Exception as _e:
            print(f'{_e}')
            print('Please enter a valid chain and calibration ID.')
    # endregion

    # region Loss inputs.
    while not isinstance(new_loss, bool):
        new_loss = input('Use a new loss file? (y/n): ')
        if new_loss.lower() == 'y':
            new_loss = True
        if new_loss.lower() == 'n':
            new_loss = False

    if new_loss:
        loss_check = False
        while not loss_check:
            try:
                new_loss_folder = pathlib.Path(input(
                    'Copy and paste folder of new loss file: '))
                new_loss_title = input(
                    'Copy and paste title of loss file here: ')
                new_loss_path = pathlib.Path(
                    f'{new_loss_folder}\\{new_loss_title}.csv')
                loss_check = open(new_loss_path)
                loss_check.close()
                loss_check = True
            except Exception as _e:
                print(f'{_e}')
                print('Please enter a valid loss file.')
    else:
        new_loss_folder = None
        new_loss_title = None
    # endregion

    # region Construct and return function outputs.
    input_files = InputFiles(
        result_folder, result_csv_title, new_cal_folder, 
        new_cal_csv_title, new_loss_folder, new_loss_title)

    return UserInputs(
        input_files, cal_id, chain, reprocess_all_in_session_folder, new_loss)
    # endregion

def main():
    """Main for reprocess script."""
    # region Fix system path to make it consistent wherever linked from.
    os.chdir(os.path.dirname(os.path.dirname(sys.argv[0])))
    # endregion

    # region Get user inputs.
    user_inputs = get_user_inputs()
    # endregion
    
    # region Trigger reprocessing using multiprocessing.
    if isinstance(user_inputs.input_files.result_csv_title, list):
        processes = []
        for i, _ in enumerate(user_inputs.input_files.result_csv_title):
            p = multiprocessing.Process(
                target=reprocess, args=[i, user_inputs])
            p.start()
            processes.append(p)

        for process in processes:
            process.join()
    else:
        reprocess(0, user_inputs)
    # endregion

if __name__ == '__main__':
    main()
