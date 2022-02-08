# -*- coding: utf-8 -*-
"""output_classes.py - Neatly stores/processes results sets.

Two types of results are within this module. LoopInstanceResult is the
result of a single measurement loop. Results contain a set of two
LoopInstanceResult instances, one for a Cold measurement and another for
a Hot measurement.
"""

# region Import modules.
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Union
import datetime as dt
import math as mt
import os
import pathlib as pl

import numpy as np

import settings_classes as sc
import util as ut
# endregion


# region Base level classes.
@dataclass()
class PrePostTemps:
    """Set of pre and post measurement loop temperature sensor readings.

    Constructor arguments:
        pre_loop_lna_temps (list): Measured LNA temp pre meas loop (K).
        post_loop_lna_temps (list): Measured LNA temp post meas loop
            (K).
        pre_loop_extra_1_temps (list): Measured temp from first extra
            sensor pre meas loop.
        post_loop_extra_1_temps (list): Measured temp from first extra
            sensor post meas loop.
        pre_loop_extra_2_temps (list): Measured temp from second extra
            sensor pre meas loop.
        post_loop_extra_2_temps (list): Measured temp from second extra
            sensor post meas loop.
    """
    pre_loop_lna_temps: list
    post_loop_lna_temps: list
    pre_loop_extra_1_temps: list
    post_loop_extra_1_temps: list
    pre_loop_extra_2_temps: list
    post_loop_extra_2_temps: list


@dataclass()
class LoopInstanceResult:
    """Results from one measurement loop.

    Constructor Attributes:
        hot_or_cold (str): Either 'Hot' or 'Cold'.
        powers (list): Measured power over the power bandwidth (dBm)
        load_temps (list): Measured load temp during measurement (K).
        lna_temps (list): The average of the pre and post LNA temps.
        pre_post_temps (PrePostTemps): Temp sensor channel readings
            before and after measurement loop.
    """
    hot_or_cold: str
    powers: list
    load_temps: list
    lna_temps: list
    pre_post_temps: PrePostTemps
# endregion


# region Mid level classes.
@dataclass()
class LoopPair:
    """Set of hot & cold measurements, i.e. one raw measurement result.

    Constructor Attributes:
        cold (LoopInstanceResult): Results from the cold meas loop.
        hot (LoopInstanceResult): Results from the hot meas loop.
    """
    cold: LoopInstanceResult
    hot: LoopInstanceResult


@dataclass()
class InputCalData:
    """Input calibration data variables for standard results analysis.

    Constructor Attributes:
        cold_powers (list[float]): Cold powers from the cal measurement.
        hot_powers (list[float]): Hot powers from the cal measurement.
        loss_cor_noise_temp (list[float]): Loss corrected noise temp
            measurements from the cal measurement.
    """
    cold_powers: list[float]
    hot_powers: list[float]
    loss_cor_noise_temp: list[float]


@dataclass()
class CorrectedTemps:
    """Loss corrected temperatures.

    Constructor Attributes:
        cold_temps (list[float]): Loss corrected cold temperatures.
        hot_temps (list[float]): Loss corrected hot temperatures.
    """
    cold_temps: list[float]
    hot_temps: list[float]


@dataclass()
class Gain:
    """Gain in dB and non-dB formats.

    Constructor Attributes:
        gain (list[float]): Calculated gain in non-dB.
        gain_db (list[float]): Calculated gain in dB.
    """
    gain: list[float]
    gain_db: list[float]


@dataclass()
class NoiseTemp:
    """Calculated noise temperatures.

    Constructor Attributes:
        uncal_loss_uncor (list[float]): Uncalibrated noise temperature
            without loss correction.
        uncal_loss_cor (list[float]): Uncalibrated noise temperature
            with loss correction.
        cal_loss_cor (list[float]): Calibrated noise temperature with
            loss correction.
    """
    uncal_loss_uncor: list[float]
    uncal_loss_cor: list[float]
    cal_loss_cor: list[float]


@dataclass()
class StandardAnalysedResults:
    """Calculated standard analysed results.

    Constructor Attributes:
        input_cal_data (InputCalData): Input calibration data variables
            for standard results analysis.
        corrected_temps (CorrectedTemps): Loss corrected temperatures.
        y_factor (list[float]): Calculated y factor for each req freq.
        gain (Gain): Gain in dB and non-dB formats.
        noise_temp (NoiseTemp): Calculated noise temperatures.
    """
    input_cal_data: InputCalData
    corrected_temps: CorrectedTemps
    y_factor: list[float]
    gain: Gain
    noise_temp: NoiseTemp


@dataclass()
class CalibrationAnalysedResults:
    """Calculated calibration analysed results.

    Constructor Attributes:
        corrected_temps (CorrectedTemps): Loss corrected temperatures.
        y_factor (list[float]): Calculated y factor for each req freq.
        loss_cor_noise_temp (list[float]): Noise temperature with loss
            correction.
    """
    corrected_temps: CorrectedTemps
    y_factor: list[float]
    loss_cor_noise_temp: list[float]


@dataclass
class ResultsMetaInfo:
    """Non measurement output results information.

    Constructor Attributes:
        comment (str): User defined string for additional information
            required for the measurement session.
        freq_array (list[float]): Requested array of frequencies to
            obtain the noise temperature and/or gain results at.
        order (int): Should always be 1 unless you explicitly know what
            you're doing. This variable is used in the results' data
            processing methods.
        is_calibration (bool): Decides whether the measurement is
            handled as a calibration, if it is then there are different
            ways to process and store the results.
        trimmed_loss (list[float]): Cryostat losses at each requested
            freq point in dB.
        trimmed_in_cal_data (Optional[list[float]]): The input
            calibration data for each requested frequency points.
    """
    comment: str
    freq_array: list[float]
    order: int
    is_calibration: bool
    trimmed_loss: list[float]
    trimmed_in_cal_data: Optional[list[float]] = None
# endregion


# region Results processing for top level class.
def process(loop_pair: LoopPair, results_meta_info: ResultsMetaInfo
            ) -> Union[StandardAnalysedResults, CalibrationAnalysedResults]:
    """Processes a pair of hot/cold standard/cal measurement results.

    Args:
        loop_pair (LoopPair): Set of hot & cold measurements, i.e. one
            raw measurement result.
        results_meta_info (ResultsMetaInfo): Non measurement output
            results information.
    """
    freq_array = results_meta_info.freq_array
    loss = results_meta_info.trimmed_loss
    order = results_meta_info.order

    # region Input Calibration Data.
    if not results_meta_info.is_calibration:
        in_cal_data_np = np.array(results_meta_info.trimmed_in_cal_data)
        in_cal_noise_temps = in_cal_data_np[:, 8]
        hot_cal_powers = in_cal_data_np[:, 3]
        cold_cal_powers = in_cal_data_np[:, 2]
        in_cal_data = InputCalData(cold_cal_powers, hot_cal_powers,
                                   in_cal_noise_temps)
    # endregion

    # region Initialise arrays.
    cor_hot_temps = []
    cor_cold_temps = []
    y_factor = []
    uncal_loss_uncor_noise_temp = []
    gain_cal = []
    gain_cal_db = []
    uncal_loss_uncor_noise_temp = []
    uncal_loss_cor_noise_temp = []
    cal_loss_cor_noise_temp = []
    # endregion

    # region Temperature correction internal function.
    def _temp_correction(load_t, lna_t, index):
        """Corrects a temperature measurement for system loss.
        """

        # region Calculate corrected temperature
        _a = ((1 - (1 / loss[index])) / (0.23036 * 10 * mt.log10(loss[index])))

        ts1 = (load_t / loss[index])
        ts2 = 0
        if order != 1:
            ts2 = ((1 - order) * _a * (lna_t + load_t)) / 2
        ts3 = (order * (lna_t * (1 - _a)) - (
                load_t * ((1 / loss[index]) - _a)))

        return ts1 + ts2 + ts3
        # endregion
    # endregion

    for i, _ in enumerate(freq_array):

        # region Corrected Temperatures
        cor_hot_temps.append(_temp_correction(loop_pair.hot.load_temps[i],
                                              loop_pair.hot.lna_temps[i], i))
        cor_cold_temps.append(_temp_correction(loop_pair.cold.load_temps[i],
                                               loop_pair.cold.lna_temps[i], i))
        # endregion

        # region Y Factor
        y_factor.append((10 ** (loop_pair.hot.powers[i] / 10))
                        / (10 ** (loop_pair.cold.powers[i] / 10)))
        # endregion

        # region Uncal Loss Uncor Noise Temp
        if 0.999 < y_factor[i] < 1.001:
            uncal_loss_uncor_noise_temp.append(10000)
        else:
            uncal_loss_uncor_noise_temp.append(
                (loop_pair.hot.load_temps[i] - (y_factor[i] *
                                                loop_pair.cold.load_temps[i]))
                / (y_factor[i] - 1))
        # endregion

        # region Gain
        if not results_meta_info.is_calibration:
            gain_s1 = ((10 ** (loop_pair.hot.powers[i] / 10))
                       - (10 ** (loop_pair.cold.powers[i] / 10)))
            gain_s2 = ((10 ** (hot_cal_powers[i] / 10))
                       - (10 ** (cold_cal_powers[i] / 10)))
            gain_cal.append(gain_s1 / gain_s2)

            if gain_cal[i] > 0:
                gain_cal_db.append(10 * mt.log10(gain_cal[i]))
            elif gain_cal[i] <= 0:
                gain_cal_db.append(0)
        # endregion

        uncal_loss_cor_noise_temp.append(
            (cor_hot_temps[i] - (y_factor[i] * cor_cold_temps[i]))
            / (y_factor[i] - 1))

        # region Calibrated loss corrected noise temperature.
        if not results_meta_info.is_calibration:
            cal_loss_cor_noise_temp.append(
                uncal_loss_cor_noise_temp[i] -
                (in_cal_noise_temps[i] / gain_cal[i]))
        # endregion

    corrected_temps = CorrectedTemps(cor_cold_temps, cor_hot_temps)

    if not results_meta_info.is_calibration:
        gain = Gain(gain_cal, gain_cal_db)

        noise_temp = NoiseTemp(uncal_loss_uncor_noise_temp,
                               uncal_loss_cor_noise_temp,
                               cal_loss_cor_noise_temp)

        return StandardAnalysedResults(in_cal_data, corrected_temps, y_factor,
                                       gain, noise_temp)

    return CalibrationAnalysedResults(corrected_temps, y_factor,
                                      uncal_loss_cor_noise_temp)
# endregion


# region Top level class.
class Results(LoopPair, StandardAnalysedResults, CalibrationAnalysedResults,
              ResultsMetaInfo):
    """Overall results incorporating hot and cold measurements.

    Attributes:
    """

    def __init__(self, loop_pair: LoopPair,
                 results_meta_info: ResultsMetaInfo) -> None:
        """Constructor for the Results class.

        Args:
            loop_pair:
            results_meta_info:
        """

        # region Initialise subclasses and process results.
        ResultsMetaInfo.__init__(
            self, *ut.get_dataclass_args(results_meta_info))

        LoopPair.__init__(self, *ut.get_dataclass_args(loop_pair))

        if self.is_calibration:
            CalibrationAnalysedResults.__init__(
                self, *ut.get_dataclass_args(
                    process(loop_pair, results_meta_info)))
        else:
            StandardAnalysedResults.__init__(
                self, *ut.get_dataclass_args(
                    process(loop_pair, results_meta_info)))
        # endregion

        # region Set time and date of creation as attributes.
        self.present = dt.datetime.now()
        self.date_str = (str(self.present.year)
                         + str(self.present.month)
                         + str(self.present.day))
        self.time_str = (str(self.present.hour) + ' '
                         + str(self.present.minute))
        # endregion

    @staticmethod
    def std_settings_column_titles() -> list[str]:
        """Returns the standard settings column titles."""
        settings_col_titles = [
            'Project Title', 'LNA ID/s (axb)', 'Session ID', 'BiasID',
            'Date', 'Time', 'Comment','', 'Center Frequency (GHz)', 
            'Marker Frequency (GHz)', 'Resolution Bandwidth (MHz)', 
            'Video Bandwidth (Hz)', 'Frequency Span (MHz)', 
            'Power Bandwidth (MHz)', 'Attenuation (dB)','',
            'L1S1GV Set (V)', 'L1S1DV Set (V)', 'L1S1DI Set (mA)',
            'L1S2GV Set (V)', 'L1S2DV Set (V)', 'L1S2DI Set (mA)',
            'L1S3GV Set (V)', 'L1S3DV Set (V)', 'L1S3DI Set (mA)','',
            'L2S1GV Set (V)', 'L2S1DV Set (V)', 'L2S1DI Set (mA)',
            'L2S2GV Set (V)', 'L2S2DV Set (V)', 'L2S2DI Set (mA)',
            'L2S3GV Set (V)', 'L2S3DV Set (V)', 'L2S3DI Set (mA)','',
            'L1S1GV Meas (V)', 'L1S1DV Meas (V)', 'L1S1DI Meas (mA)',
            'L1S2GV Meas (V)', 'L1S2DV Meas (V)', 'L1S2DI Meas (mA)',
            'L1S3GV Meas (V)', 'L1S3DV Meas (V)', 'L1S3DI Meas (mA)','',
            'L2S1GV Meas (V)', 'L2S1DV Meas (V)', 'L2S1DI Meas (mA)',
            'L2S2GV Meas (V)', 'L2S2DV Meas (V)', 'L2S2DI Meas (mA)',
            'L2S3GV Meas (V)', 'L2S3DV Meas (V)', 'L2S3DI Meas (mA)','',
            'CRBE GV Set (V)', 'CRBE DV Set (V)', 'CRBE DI Set (mA)', 
            'RTBE GV Set (V)', 'RTBE DV Set (V)', 'RTBE DI Set (mA)','', 
            'CRBE GV Meas (V)', 'CRBE DV Meas (V)', 'CRBE DI Meas (mA)',
            'RTBE GV Meas (V)', 'RTBE DV Meas (V)', 'RTBE DI Meas (mA)']
        return settings_col_titles

    @staticmethod
    def std_output_column_titles() -> list[str]:
        """Returns the standard output column titles."""
        results_col_titles = [
            'Frequency (GHz)', 'Cold Power (dBm)', 'Hot Power (dBm)',
            'Hot Load Temps (K)', 'Hot LNA Temps (K)',
            'Cold Load Temps (K)', 'Cold LNA Temps (K)',
            'Loss (dB)', 'Y Factor',
            'Corrected Hot Load Temps (K)', 'Corrected Cold Load Temps',
            'Uncalibrated Noise Temperature (K)',
            'Corrected Noise Temperature (K)',
            'Calibrated & Corrected Noise Temperature (K)', 'Gain (dB)',
            'Pre Cold Measurement LNA Temp (K)',
            'Post Cold Measurement LNA Temp (K)',
            'Pre Hot Measurement LNA Temp (K)',
            'Post Hot Measurement LNA Temp (K)',
            'Pre Cold Measurement 1st Extra Temp (K)',
            'Post Cold Measurement 1st Extra Temp (K)',
            'Pre Hot Measurement 1st Extra Temp (K)',
            'Post Hot Measurement 1st Extra Temp (K)',
            'Pre Cold Measurement 2nd Extra Temp (K)',
            'Post Cold Measurement 2nd Extra Temp (K)',
            'Pre Hot Measurement 2nd Extra Temp (K)',
            'Post Hot Measurement 2nd Extra Temp (K)']
        return results_col_titles

    def std_output_data(self) -> np.ndarray:
        """Return the data for the standard results output csv."""
        return np.column_stack((
            self.freq_array, self.cold.powers, self.hot.powers,
            self.hot.load_temps, self.hot.lna_temps, self.cold.load_temps,
            self.cold.lna_temps, self.trimmed_loss, self.y_factor,
            self.corrected_temps.hot_temps, self.corrected_temps.cold_temps,
            self.noise_temp.uncal_loss_uncor, self.noise_temp.uncal_loss_cor,
            self.noise_temp.cal_loss_cor, self.gain.gain_db,
            self.cold.pre_post_temps.pre_loop_lna_temps,
            self.cold.pre_post_temps.post_loop_lna_temps,
            self.hot.pre_post_temps.pre_loop_lna_temps,
            self.hot.pre_post_temps.post_loop_lna_temps,
            self.cold.pre_post_temps.pre_loop_extra_1_temps,
            self.cold.pre_post_temps.post_loop_extra_1_temps,
            self.hot.pre_post_temps.pre_loop_extra_1_temps,
            self.hot.pre_post_temps.post_loop_extra_1_temps,
            self.cold.pre_post_temps.pre_loop_extra_2_temps,
            self.cold.pre_post_temps.post_loop_extra_2_temps,
            self.hot.pre_post_temps.pre_loop_extra_2_temps,
            self.hot.pre_post_temps.post_loop_extra_2_temps))

    @staticmethod
    def cal_settings_column_titles() -> list[str]:
        """Returns the calibration settings file column titles."""
        cal_settings_col_titles = [
            'Project Title', 'Cryostat Chain', 'Calibration ID', 
            'Date', 'Time', 'Comment', '',
            'Center Frequency (GHz)', 'Marker Frequency (GHz)',
            'Resolution Bandwidth (MHz)', 'Video Bandwidth (Hz)',
            'Frequency Span (MHz)', 'Power Bandwidth (MHz)',
            'Attenuation (dB)', '', 'CRBE GV Set (V)', 'CRBE DV Set (V)',
            'CRBE DI Set (mA)', 'RTBE GV Set (V)', 'RTBE DV Set (V)',
            'RTBE DI Set (mA)', '', 'CRBE GV Meas (V)',
            'CRBE DV Meas (V)', 'CRBE DI Meas (mA)', 'RTBE GV Meas (V)',
            'RTBE DV Meas (V)', 'RTBE DI Meas (mA)']
        return cal_settings_col_titles

    @staticmethod
    def cal_output_column_titles() -> list[str]:
        """Returns the calibration output column titles."""
        cal_results_col_titles = [
            'Frequency (GHz)', 'Loss (dB)',
            'Cold Powers (dBm)', 'Hot Powers (dBm)',
            'Hot Load Temps (K)', 'Hot LNA Temps (K)',
            'Cold Load Temps (K)', 'Cold LNA Temps (K)',
            'Calibrated Noise Temp (K)',
            'Pre Cold Measurement LNA Temp (K)',
            'Post Cold Measurement LNA Temp (K)',
            'Pre Hot Measurement LNA Temp (K)',
            'Post Hot Measurement LNA Temp (K)',
            'Pre Cold Measurement 1st Extra Temp (K)',
            'Post Cold Measurement 1st Extra Temp (K)',
            'Pre Hot Measurement 1st Extra Temp (K)',
            'Post Hot Measurement 1st Extra Temp (K)',
            'Pre Cold Measurement 2nd Extra Temp (K)',
            'Post Cold Measurement 2nd Extra Temp (K)',
            'Pre Hot Measurement 2nd Extra Temp (K)',
            'Post Hot Measurement 2nd Extra Temp (K)']
        return cal_results_col_titles

    def cal_output_data(self) -> np.ndarray:
        """Return the data for the calibration results output csv."""
        return np.column_stack((
            self.freq_array, self.trimmed_loss,
            self.cold.powers, self.hot.powers,
            self.hot.load_temps, self.hot.lna_temps,
            self.cold.load_temps, self.cold.lna_temps,
            self.loss_cor_noise_temp,
            self.cold.pre_post_temps.pre_loop_lna_temps,
            self.cold.pre_post_temps.post_loop_lna_temps,
            self.hot.pre_post_temps.pre_loop_lna_temps,
            self.hot.pre_post_temps.post_loop_lna_temps,
            self.cold.pre_post_temps.pre_loop_extra_1_temps,
            self.cold.pre_post_temps.post_loop_extra_1_temps,
            self.hot.pre_post_temps.pre_loop_extra_1_temps,
            self.hot.pre_post_temps.post_loop_extra_1_temps,
            self.cold.pre_post_temps.pre_loop_extra_2_temps,
            self.cold.pre_post_temps.post_loop_extra_2_temps,
            self.hot.pre_post_temps.pre_loop_extra_2_temps,
            self.hot.pre_post_temps.post_loop_extra_2_temps))

    @staticmethod
    def output_file_path(
            directory: pl.Path, meas_settings: sc.MeasurementSettings,
            bias_id: int, csv_or_png: str) -> pl.Path:
        """Returns the output csv or png path."""

        l_str = f'LNA {meas_settings.lna_id_str} '
        s_str = f'Session {meas_settings.session_id} '
        b_str = f'Bias {bias_id}'
        a_str = f'Alg {meas_settings.measure_method}'

        while len(l_str) < 11:
            l_str += ' '
        while len(s_str) < 12:
            s_str += ' '

        session_folder = f'\\{s_str}{l_str}{a_str}'
        session_folder_dir = str(directory) + session_folder
        os.makedirs(session_folder_dir, exist_ok=True)

        results_csv_title = f'\\{s_str}{l_str}{b_str}.csv'
        results_png_title = f'\\{s_str}{l_str}{b_str}.png'

        results_csv_path = pl.Path(
            str(directory) + session_folder + results_csv_title)
        results_png_path = pl.Path(
            str(directory) + session_folder + results_png_title)
        if csv_or_png == 'csv':
            return results_csv_path
        if csv_or_png == 'png':
            return results_png_path
# endregion
