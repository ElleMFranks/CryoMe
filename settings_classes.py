# -*- coding: utf-8 -*-
"""settings_classes.py - Stores/handles measurement instance settings.
"""

# region Import modules.
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import csv
import os
import pathlib as pl

import numpy as np
import pandas as pd

import error_handling as eh
import instr_classes as ic
import lna_classes as lc
import output_classes as oc
import util as ut
# endregion


# region Base Level Classes.
# region Measurement Settings.
@dataclass()
class CalInfo:
    """Object containing user inputs regarding calibration.

    Constructor Attributes:
        is_calibration (bool): Decides whether the measurement is
            handled as a calibration, if it is then there are different
            ways to process and store the results.
        in_cal_file_id (int): The calibration id number to take data
            from in the calibration directory.
    """
    is_calibration: bool
    in_cal_file_id: int


@dataclass()
class Misc:
    """Contains miscellaneous variables for the measurement session.

    Constructor Attributes:
        comment_en (bool): If true the user is prompted for a comment.
        dark_mode_plot (bool): If true, output plots are saved with
            dark mode palette.
        order (int): Should always be 1 unless you explicitly know what
            you're doing. This variable is used in the results' data
            processing methods.
    """
    comment_en: bool = False
    dark_mode_plot: bool = True
    order: int = 1


@dataclass()
class AnalysisBandwidths:
    """Sub bandwidths to analyse data over defined by min-max GHz freqs.

    Constructor Attributes:
        bw_1_min_max (Optional[list[float]]): Sub bandwidth min max GHz
            freq 1.
        bw_2_min_max (Optional[list[float]]): Sub bandwidth min max GHz
            freq 2.
        bw_3_min_max (Optional[list[float]]): Sub bandwidth min max GHz
            freq 3.
        bw_4_min_max (Optional[list[float]]): Sub bandwidth min max GHz
            freq 4.
        bw_5_min_max (Optional[list[float]]): Sub bandwidth min max GHz
            freq 5.
    """
    bw_1_min_max: Optional[list[float]]
    bw_2_min_max: Optional[list[float]]
    bw_3_min_max: Optional[list[float]]
    bw_4_min_max: Optional[list[float]]
    bw_5_min_max: Optional[list[float]]


# region Session Settings.
@dataclass()
class SessionInfo:
    """Object containing the session title and measurement method.

    Constructor Attributes:
        project_title (str): The title of the project for the current
            measurement session. No special characters  and keep it
            short, becomes results folder/file titles.

        measure_method (str): Decides which measurement algorithm to use
            for the current measurement.
            AllColdToAllHot = 'ACTAH' - Does all cold then all hot
            measurements. Parallel measurements.
            AlternatingTemps: 'AT' - Does one cold, then the
            corresponding hot measurement. Sequential measurements.
            ManualEntryMeasure: 'MEM' - Measures the bias point which is
            manually entered.
    """
    project_title: str
    measure_method: str
    analysis_bws: AnalysisBandwidths
# endregion


# region LNA Settings.
@dataclass()
class LNACryoLayout:
    """Settings describing layout of LNAs under test in the cryostat.

    Constructor Attributes:
        cryo_chain (int): This is the chain under test for the current
            measurement session, either 1, 2, or 3.
        lnas_per_chain (int): How many LNAs per cryostat chain, 1 or 2.
        stages_per_lna (int): The technical number of stages in the lnas
            under test. If an amplifier is 3 stage, but the second and
            third are the same, this number is 3, and the stage_2_3_same
            variable should be set true.
        stage_2_3_same (bool): If the second and third stage of the lnas
            under test are the same psx channel, this needs to be set to
            true. Used to set the bias psx settings. Similar to
            stage_1_2_same.
    """
    cryo_chain: int
    lnas_per_chain: int
    stages_per_lna: int
    stage_1_2_same: bool
    stage_2_3_same: bool


@dataclass()
class DirectLNAs:
    """Object containing user input manual/backend LNA biases/settings.

    Constructor Attributes:
        be_lna_settings (BackEndLNASettings): Object containing the LNA
            biases and settings for the room temperature and cryostat
            back end LNAs.
        manual_lna_settings (ManualLNASettings): Object containing the
            LNA biases and settings for the LNAs made from the manual
            bias inputs.
    """
    be_lna_settings: lc.BackEndLNASettings
    manual_lna_settings: lc.ManualLNASettings


@dataclass()
class LNAIDs:
    """Object containing the LNA ID for each of the LNAs under test.

    Constructor Attributes:
        chain_1_lna_1_id (Optional[int]): LNA ID for chain 1 lna 1.
        chain_1_lna_2_id (Optional[int]): LNA ID for chain 1 lna 2.
        chain_2_lna_1_id (Optional[int]): LNA ID for chain 2 lna 1.
        chain_2_lna_2_id (Optional[int]): LNA ID for chain 2 lna 2.
        chain_3_lna_1_id (Optional[int]): LNA ID for chain 3 lna 1.
        chain_3_lna_2_id (Optional[int]): LNA ID for chain 3 lna 2.
    """
    chain_1_lna_1_id: Optional[int]
    chain_1_lna_2_id: Optional[int]
    chain_2_lna_1_id: Optional[int]
    chain_2_lna_2_id: Optional[int]
    chain_3_lna_1_id: Optional[int]
    chain_3_lna_2_id: Optional[int]
# endregion
# endregion


# region Sweep Settings.
@dataclass
class MeasSequence:
    """Stage and LNA sequences to decide overall measurement sequence.

    Constructor Attributes:
        stage_sequence (list[int]): List of which LNA stages are to have
            their bias swept.
        lna_sequence (list[int]): List of which LNAs are to have their
            stages bias swept.
    """
    stage_sequence: list[int]
    lna_sequence: list[int]


@dataclass()
class SweepSetupVars:
    """Variables describing drain voltages/currents to sweep through.

    Constructor Attributes:
        num_of_d_v (int): The number of drain voltage points in the bias
            sweep. Should not be set much higher than six.
        num_of_d_i (int): The number of drain current points in the bias
            sweep. Should not be set much higher than six.
        d_v_min (float): The minimum drain voltage bias to measure (V).
        d_v_max (float): The maximum drain voltage bias to measure (V).
        d_i_min (float): The minimum drain current bias to measure (mA).
        d_i_max (float): The minimum drain current bias to measure (mA).
        alt_temp_sweep_skips (int): For the Alternating Temp algorithm
            this chooses how many conditions to skip. This is useful if
            an interruption took place. Can figure out the number for
            this using the results log.
    """
    num_of_d_v: int
    num_of_d_i: int
    d_v_min: float
    d_v_max: float
    d_i_min: float
    d_i_max: float
    alt_temp_sweep_skips: int = 0


@dataclass()
class NominalBias:
    """The nominal drain voltage/current for the measurement session.

    Constructor Attributes:
        d_v_nominal (float): The nominal drain voltage in volts, all
            stages not being swept will be set to this.
        d_i_nominal (float): This is the nominal drain current in
            milli-amps, all stages not being swept will be set to this.
    """
    d_v_nominal: float
    d_i_nominal: float
# endregion


# region File Inputs.
@dataclass()
class TrimmedInputs:
    """ Object containing the trimmed loss and calibration data.

    Constructor Attributes:
        trimmed_loss (list[float]): Cryostat losses at each requested
            freq point in dB.
        trimmed_cal_data (list[float]): The input calibration data for
            each requested frequency points.
    """
    trimmed_loss: list[float]
    trimmed_cal_data: Optional[list[float]] = None
# endregion
# endregion


# region Mid Level Classes.
# region Collected LNA Subclasses.
@dataclass()
class LNAInfo:
    """Object containing all the session LNA settings.

    Constructor Attributes:
        direct_lnas (DirectLNAs): User input manual/backend LNA
            biases/settings.
        lna_ids (LNAIDs): The LNA ID for each of the LNAs under test.
        lna_cryo_layout (LNACryoLayout): Settings describing layout of
            LNAs under test in the cryostat.
    """
    direct_lnas: DirectLNAs
    lna_ids: LNAIDs
    lna_cryo_layout: LNACryoLayout


@dataclass
class LNAsUTIDs:
    """Object containing the LNA IDs of the LNAs under test.

    Constructor Attributes:
        lna_1_id (int): ID of the first LNA in the chain under test.
        lna_2_id (int): ID of the second LNA in the chain under test.
    """
    lna_1_id: int
    lna_2_id: int
# endregion
# endregion


# region Upper Level Classes.
class MeasurementSettings(SessionInfo, LNAInfo, CalInfo, Misc):
    """Non-instrument settings for a measurement session.

    Attributes:
        lnas_ut_ids (LNAsUTIDs): The LNA IDs of the LNAs under test.
        lna_id_str (str): String of the LNAs ut IDs for saving results.
        session_id (int): The ID of the measurement session.
        comment (str): User defined string for additional information
            required for the measurement session.
    """

    def __init__(self, session_info: SessionInfo, lna_info: LNAInfo,
                 cal_info: CalInfo, misc: Misc) -> None:
        """Constructor for the MeasurementSettings class.

        Args:
            session_info:
            lna_info:
            cal_info:
            misc:
        """
        # region Variable error handling.
        eh.check_session_info(session_info)
        eh.check_misc(misc)
        eh.check_cal_info(cal_info, session_info)
        eh.check_lna_info(lna_info)
        # endregion

        # region Initialise subclasses.
        SessionInfo.__init__(self, *ut.get_dataclass_args(session_info))
        Misc.__init__(self, *ut.get_dataclass_args(misc))
        CalInfo.__init__(self, *ut.get_dataclass_args(cal_info))
        LNAInfo.__init__(self, *ut.get_dataclass_args(lna_info))
        # endregion

        # region Initialise attributes to set later.
        if self.lna_cryo_layout.cryo_chain == 1:
            self.lnas_ut_ids = LNAsUTIDs(
                self.lna_ids.chain_1_lna_1_id,
                self.lna_ids.chain_1_lna_2_id)
        elif self.lna_cryo_layout.cryo_chain == 2:
            self.lna_ut_ids = LNAsUTIDs(
                self.lna_ids.chain_2_lna_1_id,
                self.lna_ids.chain_2_lna_2_id)
        elif self.lna_cryo_layout.cryo_chain == 3:
            self.lna_ut_ids = LNAsUTIDs(
                self.lna_ids.chain_3_lna_1_id,
                self.lna_ids.chain_3_lna_2_id)
        self.lna_id_str = f'{self.lna_ut_ids.lna_1_id}x{self.lna_ut_ids.lna_2_id}'
        self.session_id = None
        self.comment = None
        # endregion

    @property
    def lnas_ut_ids(self) -> LNAsUTIDs:
        return self._lnas_ut_ids

    @lnas_ut_ids.setter
    def lnas_ut_ids(self, value: LNAsUTIDs) -> None:
        self._lnas_ut_ids = value

    @property
    def lna_id_str(self) -> str:
        return self._lna_id_str

    @lna_id_str.setter
    def lna_id_str(self, value: str) -> None:
        self._lna_id_str = value

    @property
    def comment(self) -> str:
        """Gets comment for the measurement session."""
        return self._comment

    @comment.setter
    def comment(self, value: str) -> None:
        self._comment = value

    @property
    def session_id(self) -> int:
        return self._session_id

    @session_id.setter
    def session_id(self, value: int) -> None:
        self._session_id = value

    def config_lna_ids(self, cryo_chain: int) -> None:
        """Ensures the LNA IDs are set up for the current chain."""

        # region Chain 1.
        if cryo_chain == 1:
            self.lnas_ut_ids = LNAsUTIDs(
                self.lna_ids.chain_1_lna_1_id,
                self.lna_ids.chain_1_lna_2_id)
        # endregion

        # region Chain 2.
        elif cryo_chain == 2:
            self.lnas_ut_ids = LNAsUTIDs(
                self.lna_ids.chain_2_lna_1_id,
                self.lna_ids.chain_2_lna_2_id)
        # endregion

        # region Chain 3.
        elif cryo_chain == 3:
            self.lnas_ut_ids = LNAsUTIDs(
                self.lna_ids.chain_3_lna_1_id,
                self.lna_ids.chain_3_lna_2_id)
        # endregion

        # region Handle variable error.
        else:
            raise Exception('Cryo chain must be 1, 2, or 3.')
        # endregion

        if self.lnas_ut_ids.lna_2_id is not None:
            lna_id_str = f'{self.lnas_ut_ids.lna_1_id}x' \
                         f'{self.lnas_ut_ids.lna_2_id}'
        else:
            lna_id_str = f'{self.lnas_ut_ids.lna_1_id}'

        self.lna_id_str = lna_id_str

    def config_session_id(self, file_struc: FileStructure) -> None:
        """Set session ID for this measurement session."""
        settings_log_exists = os.path.isfile(file_struc.settings_path)
        # region Scan settings log, session ID = highest existing + 1.
        if settings_log_exists:
            settings_log = np.array(pd.read_csv(
                file_struc.settings_path, dtype=str))
            if len(settings_log) != 0:
                session_id = int(settings_log[len(settings_log) - 1, 2]) + 1
            else:
                session_id = 1
        else:
            session_id = 1
        # endregion
        self.session_id = session_id


class SweepSettings(MeasSequence, SweepSetupVars, NominalBias, LNACryoLayout):
    """Settings for the bias sweep.

    Attributes:
        d_v_sweep (list): The list of drain voltage values to sweep.
        d_i_sweep (list): The list of drain current values to sweep.
    """

    def __init__(
            self, meas_sequence: MeasSequence,
            sweep_setup_vars: SweepSetupVars,
            nominal_bias: NominalBias, lna_cryo_layout: LNACryoLayout) -> None:
        """Constructor for the  SweepSettings class.

        Args:
            meas_sequence:
            sweep_setup_vars:
            nominal_bias:
            lna_cryo_layout:
        """
        # region Check variables for errors.
        eh.check_meas_sequence(meas_sequence)
        eh.check_nominals(nominal_bias)
        eh.check_sweep_setup_vars(sweep_setup_vars)
        # endregion

        # region Set args to attributes.
        MeasSequence.__init__(self, *ut.get_dataclass_args(meas_sequence))
        if nominal_bias is not None:
            NominalBias.__init__(self, *ut.get_dataclass_args(nominal_bias))
        SweepSetupVars.__init__(self, *ut.get_dataclass_args(sweep_setup_vars))
        LNACryoLayout.__init__(self, *ut.get_dataclass_args(lna_cryo_layout))
        # endregion

        # region Calculate additional attributes from args.
        self.d_v_sweep = np.linspace(
            self.d_v_min, self.d_v_max, self.num_of_d_v, dtype=np.float16)
        self.d_i_sweep = np.linspace(
            self.d_i_min, self.d_i_max, self.num_of_d_i, dtype=np.float16)
        # endregion

        # region Handle duplicate staging.
        if self.stage_1_2_same:
            if 2 in self.stage_sequence:
                while 2 in self.stage_sequence:
                    self.stage_sequence.remove(2)
        if self.stage_2_3_same:
            if 3 in self.stage_sequence:
                while 3 in self.stage_sequence:
                    self.stage_sequence.remove(3)

        if self.lnas_per_chain == 1:
            if 2 in self.lna_sequence:
                while 2 in self.lna_sequence:
                    self.lna_sequence.remove(2)
        # endregion


class FileStructure:
    """Cryome file path structure.

    Attributes:
        results_directory (Path): The path for the results' folder. If
            this doesn't exist, it is created when an instance of this
            class is created.
        cal_directory (Path): The path for the calibration data folder.
            If this doesn't exist, it is created when an instance of
            this class is created.
        cal_settings_path (Path): The path for the calibration settings
            log csv.
        loss_path (Path): The path of the untrimmed loss data.
        pwr_lvls (Path): The path for the signal generator power
            calibration values file.
        in_cal_file_path (Path): The path for the input calibration file
            for this measurement.
        settings_path (Path): The path for the settings log file.
        res_log_path (Path): The path for the results log file.
    """

    def __init__(self, project_title: str, in_cal_file_id: int,
                 cryo_chain: int) -> None:
        """Constructor for FileStructure class.

        Args:
            project_title: The title of the project for the current
                measurement session. No special characters and keep it
                short, becomes results folder/file titles.
            in_cal_file_id: The calibration id number to take data from
                in the calibration directory.
            cryo_chain: This is the chain under test for the current
                measurement session, either 1, 2, or 3.
        """

        # region Set attributes from args.
        cwd = os.getcwd()
        self.results_directory = pl.Path(
            str(cwd) + f'\\Results\\{project_title}')
        self.cal_directory = pl.Path(
            str(cwd) + '\\Calibrations')
        self.cal_settings_path = pl.Path(
            str(self.cal_directory) + '\\Calibration Settings Log.csv')
        self.loss_path = pl.Path(
            str(self.cal_directory) + '\\Loss.csv')
        self.pwr_lvls = pl.Path(
            str(self.cal_directory) + '\\Sig Gen Power Levels.csv')
        self.in_cal_file_path = pl.Path(
            str(self.cal_directory) +
            f'\\Chain {cryo_chain}' +
            f'\\Chain {cryo_chain} Calibration {in_cal_file_id}.csv')
        self.settings_path = pl.Path(
            str(self.results_directory) +
            f'\\{project_title} Settings Log.csv')
        self.res_log_path = pl.Path(
            str(self.results_directory) +
            f'\\{project_title} Results Log.csv')
        # endregion

        # region Check results/calibration folder exist, if not create.
        os.makedirs(self.cal_directory, exist_ok=True)
        os.makedirs(self.results_directory, exist_ok=True)
        # endregion

        # region If normal and cal settings logs don't exist make them.
        self._settings_log_setup()
        self._results_log_setup()
        # endregion

    def _settings_log_setup(self):
        """Sets up the normal and calibration settings logs."""
        # region Setup standard settings log.
        if not os.path.isfile(self.settings_path):
            settings_col_titles = oc.Results.std_settings_column_titles()
            with open(self.settings_path, 'w',
                      newline='', encoding='utf-8') as file:
                writer = csv.writer(
                    file, quoting=csv.QUOTE_NONE, escapechar='\\')
                writer.writerow(settings_col_titles)
        # endregion

        # region Setup calibration settings log.
        if not os.path.isfile(self.cal_settings_path):
            cal_settings_col_titles = oc.Results.cal_settings_column_titles()
            with open(self.cal_settings_path,
                      'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(
                    file, quoting=csv.QUOTE_NONE, escapechar='\\')
                writer.writerows([cal_settings_col_titles])
        # endregion

    def _results_log_setup(self):
        """Sets up the results log."""
        # region Setup results log
        if not os.path.isfile(self.res_log_path):
            res_log_col_header = oc.Results.results_ana_log_header()
            res_log_col_titles = oc.Results.results_ana_log_column_titles()
            with open(self.res_log_path,
                      'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(
                    file, quoting=csv.QUOTE_NONE, escapechar='\\')
                writer.writerows([res_log_col_header, res_log_col_titles])
        # endregion
# endregion


# region Top Level Class.
@dataclass
class Settings:
    """Top level class containing all program settings.

    Constructor Attributes:
        meas_settings (MeasurementSettings): Non-instrument settings for
            a measurement session.
        sweep_settings (SweepSettings): Settings for the bias sweep.
        instr_settings (InstrumentSettings): Instrumentation settings
            classes.
        file_struc (FileStructure): Cryome file path structure.
    """
    meas_settings: MeasurementSettings
    sweep_settings: SweepSettings
    instr_settings: ic.InstrumentSettings
    file_struc: FileStructure
# endregion
