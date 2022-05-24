# -*- coding: utf-8 -*-
"""outputs.py - Neatly stores/processes results sets.

Two types of results are within this module. LoopInstanceResult is the
result of a single measurement loop. Results contain a set of two
LoopInstanceResult instances, one for a cold measurement and another 
for a hot measurement.
"""

# region Import modules.
from __future__ import annotations
from _typeshed import NoneType
from dataclasses import dataclass
from typing import Optional, Union
import datetime
import math
import statistics as stats
import time

import numpy as np

import config_handling
import error_handling
import util
# endregion


# region Base level classes.
@dataclass()
class ConfigUT:
    """Object to store the LNA and Stage under test.
    
    Constructor arguments:
        temp (int): Temperature under test, cold is 0, hot is 1.
        lna (int): LNA under test, 1/2.
        stage (int): Stage under test, 1/2/3.
        d_v (float): Drain voltage under test.
        d_i (float): Drain current under test.
    """
    temp: int
    lna: int
    stage: int
    d_v: float
    d_i: float


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
        times (list): Time for each measurement in the loop (s).
    """
    hot_or_cold: str
    powers: list
    load_temps: list
    lna_temps: list
    pre_post_temps: PrePostTemps
    times: list[float]


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
# endregion


@dataclass()
class GainPostProc:
    """Post-processing gain outputs.

    Constructor Attributes:
        avg_gain_full (float): Full bandwidth average gain (dBm)
        gain_std_dev_full (float): Full bandwidth gain standard
            deviation.
        gain_range_full (float): Full bandwidth gain range (dB)
        avg_gain_bws (list[Optional[float]]): Sub bandwidth average
            gain (dBm).
        gain_std_dev_bws (list[Optional[float]]): Sub bandwidth 
            standard deviations.
        gain_range_bws (list[Optional[float]]): Sub bandwidth ranges.
    """
    avgs: list[Optional[float]]
    std_devs: list[Union[str, float]]
    mins: list[Union[str, float]]
    maxs: list[Union[str, float]]
    ranges: list[Union[str, float]]

    def as_tuple(self, index: int) -> tuple:
        """Returns instance as tuple of attributes."""
        return self.avgs[index], self.std_devs[index], self.mins[index], \
            self.maxs[index], self.ranges[index]


@dataclass()
class NoiseTempPostProc:
    """Post-processing noise temperature outputs.

    Constructor Attributes:
        avg_noise_temp_full (float):
        noise_temp_std_dev_full (float):
        avg_noise_temp_bws (list[Optional[float]]):
        noise_temp_std_dev_bws (float[Optional[float]]):
    """
    avgs: list[Optional[float]]
    std_devs: list[Optional[float]]
    mins: list[Optional[float]]
    maxs: list[Optional[float]]
    ranges: list[Optional[float]]

    def as_tuple(self, index: int) -> tuple:
        """Returns instance as tuple of attributes."""
        return self.avgs[index], self.std_devs[index], self.mins[index], \
            self.maxs[index], self.ranges[index]


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


@dataclass()
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
        sub_bws (list[Optional[list[float]]]): The minimum and maximum
            GHz frequencies of the sub-bandwidths for post-processing
            analysis.
        trimmed_loss (list[float]): Cryostat losses at each requested
            freq point in dB.
        trimmed_in_cal_data (Optional[list[float]]): The input
            calibration data for each requested frequency points.
    """
    comment: str
    freq_array: list[float]
    order: int
    is_calibration: bool
    sub_bws: AnalysisBandwidths
    trimmed_loss: list[float]
    trimmed_in_cal_data: Optional[list[float]] = None


@dataclass()
class PostProcResults:
    """Contains gain and noise temperature results analysis.

    Constructor Attributes:
        gain_post_proc (GainPostProc): Post processed gain.
        noise_temp_post_proc (NoiseTempPostProc): Post processed noise
            temperature.
    """
    gain_post_proc: GainPostProc
    noise_temp_post_proc: NoiseTempPostProc
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
        in_cal_noise_temps = in_cal_data_np[:, 8].astype(float)
        hot_cal_powers = in_cal_data_np[:, 3].astype(float)
        cold_cal_powers = in_cal_data_np[:, 2].astype(float)
        in_cal_data = InputCalData(cold_cal_powers, hot_cal_powers,
                                   in_cal_noise_temps)
    # endregion

    # region Initialise arrays.
    cor_hot_temps = []
    cor_cold_temps = []
    y_factor = []
    gain_cal = []
    gain_cal_db = []
    uncal_loss_uncor_noise_temp = []
    uncal_loss_cor_noise_temp = []
    cal_loss_cor_noise_temp = []
    # endregion

    # region Temperature correction internal function.
    def _temp_correction(load_t: float, lna_t: float, index: int) -> float:
        """Corrects a temperature measurement for system loss.
        """
        load_t = float(load_t)
        lna_t = float(lna_t)

        # region Calculate corrected temperature
        _a = ((1 - (1 / loss[index]))
              / (0.23036 * 10 * math.log10(loss[index])))

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
        cor_hot_temps.append(
            _temp_correction(loop_pair.hot.load_temps[i],
                             loop_pair.hot.lna_temps[i], i))
        cor_cold_temps.append(
            _temp_correction(loop_pair.cold.load_temps[i],
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
                gain_cal_db.append(10 * math.log10(gain_cal[i]))
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

# region Top level class post-processing
def _post_process(freqs, gain: Gain, noise_temperature,
                  bws: AnalysisBandwidths) -> PostProcResults:
    """Carries out post-processing on results set."""
    untrimmed_freqs = np.array(freqs)
    untrimmed_non_db_gain = np.array(gain.gain)
    untrimmed_db_gain = np.array(gain.gain_db)
    untrimmed_noise_temperature = np.array(noise_temperature)
    freqs = np.array(freqs)
    non_db_gain = np.array(gain.gain)
    db_gain = np.array(gain.gain_db)
    noise_temperature = np.array(noise_temperature)



    f_gain = np.column_stack((freqs, non_db_gain, db_gain))
    f_nt = np.column_stack((freqs, noise_temperature))

    

    

    bandwidths = [bws.bw_1_min_max, bws.bw_2_min_max, bws.bw_3_min_max,
                  bws.bw_4_min_max, bws.bw_5_min_max]

    def _bw_analyse(freq_results: np.ndarray, is_gain: bool) -> list:
        """Returns [mean, standard deviations] for sub-bandwidths.

        Args:
            freq_results: An array [frequency (GHz), result].
            is_gain: If true analyses as gain.

        Returns:
            [Bandwidth[mean, standard deviation, (if gain then) range]]
        """

        avgs = []
        std_devs = []
        mins = []
        maxs = []
        rngs = []
        if is_gain:
            avgs.append(10 * math.log10(abs(stats.mean(non_db_gain))))
            std_devs.append(stats.stdev(non_db_gain))
            mins.append(min(db_gain))
            maxs.append(max(db_gain))
            rngs.append(max(db_gain) - min(db_gain))
        else:
            avgs.append(stats.mean(noise_temperature))
            std_devs.append(stats.stdev(noise_temperature))
            mins.append(min(noise_temperature))
            maxs.append(max(noise_temperature))
            rngs.append(max(noise_temperature) - min(noise_temperature))
        for bandwidth in bandwidths:
            if bandwidth:
                trimmed_res = []
                trimmed_res_db = []
                for freq_res in freq_results:
                    if bandwidth[0] <= freq_res[0] <= bandwidth[1]:
                        trimmed_res.append(freq_res[1])
                        if is_gain:
                            trimmed_res_db.append(freq_res[2])
                if is_gain:
                    avgs.append(10 * math.log10(abs(stats.mean(trimmed_res))))
                    mins.append(min(trimmed_res_db))
                    maxs.append(max(trimmed_res_db))
                    rngs.append(max(trimmed_res_db) - min(trimmed_res_db))
                else:
                    avgs.append(stats.mean(trimmed_res))
                    mins.append(min(trimmed_res))
                    maxs.append(max(trimmed_res))
                    rngs.append(max(trimmed_res) - min(trimmed_res))
                std_devs.append(stats.stdev(trimmed_res))
            else:
                avgs.append('NA')
                std_devs.append('NA')
                mins.append('NA')
                maxs.append('NA')
                rngs.append('NA')
        return [avgs, std_devs, mins, maxs, rngs]

    return PostProcResults(GainPostProc(*_bw_analyse(f_gain, True)),
                           NoiseTempPostProc(*_bw_analyse(f_nt, False)))


# region Top level class.
class Results(LoopPair, StandardAnalysedResults, CalibrationAnalysedResults,
              ResultsMetaInfo, AnalysisBandwidths, PostProcResults):
    """Overall results incorporating hot and cold measurements.

    Attributes:
        present (_S): Present time from datetime.
        time_str (str): Present time as a string.
        date_str (str): Present date as a string.
    """
    __doc__ += f'\n    LoopPair: {LoopPair.__doc__}\n'
    __doc__ += f'    StandardAnalysedResults: ' \
               f'{StandardAnalysedResults.__doc__}\n'
    __doc__ += f'    CalibrationAnalysedResults: ' \
               f'{CalibrationAnalysedResults.__doc__}\n'
    __doc__ += f'    ResultsMetaInfo: {ResultsMetaInfo.__doc__}\n'
    __doc__ += f'    AnalysisBandwidths: {AnalysisBandwidths.__doc__}\n'
    __doc__ += f'    PostProcResults: {PostProcResults.__doc__}'

    def __init__(self, loop_pair: LoopPair,
                 results_meta_info: ResultsMetaInfo) -> None:
        """Constructor for the Results class.

        Args:
            loop_pair: The pair of hot and cold loop measurements.
            results_meta_info: The meta information for the results set.
        """

        # region Initialise subclasses and process results.
        ResultsMetaInfo.__init__(
            self, *util.get_dataclass_args(results_meta_info))

        LoopPair.__init__(self, *util.get_dataclass_args(loop_pair))

        if self.is_calibration:
            CalibrationAnalysedResults.__init__(
                self, *util.get_dataclass_args(
                    process(loop_pair, results_meta_info)))
        else:
            AnalysisBandwidths.__init__(
                self, *util.get_dataclass_args(results_meta_info.sub_bws))

            StandardAnalysedResults.__init__(
                self, *util.get_dataclass_args(
                    process(loop_pair, results_meta_info)))

            PostProcResults.__init__(self, *util.get_dataclass_args(
                _post_process(self.freq_array, self.gain,
                              self.noise_temp.cal_loss_cor,
                              results_meta_info.sub_bws)))
        # endregion

        # region Set time and date of creation as attributes.
        self.present = datetime.datetime.now()
        self.date_str = (str(self.present.year) + ' '
                         + str(self.present.month)+ ' '
                         + str(self.present.day))
        self.time_str = (str(self.present.hour) + ' '
                         + str(self.present.minute))
        # endregion

    @property
    def session_timings(self) -> SessionTimings:
        return self._session_timings

    @session_timings.setter
    def session_timings(self, value: SessionTimings) -> None:
        if isinstance(value, SessionTimings):
            self._session_timings = value
        else:
            raise error_handling.InternalVariableError(
                'session_timings must be instance of SessionTimings class.')

    @property
    def config_ut(self) -> ConfigUT:
        """The configuration of the measurement (temp/lna/stage/dv/di).
        """
        return self._config_ut

    @config_ut.setter
    def config_ut(self, value: ConfigUT) -> None:
        if isinstance(value, ConfigUT):
            if value.lna in [1, 2] and value.stage in [1, 2, 3]:
                self._config_ut = value
            else:
                raise error_handling.InternalVariableError(
                    'Invalid config passed.')
        else:
            raise error_handling.InternalVariableError(
                'Invalid config passed.')

    @staticmethod
    def std_settings_column_titles() -> list[str]:
        """Returns the standard settings column titles."""
        settings_col_titles = [
            'Project Title', 'LNA ID/s (axb)', 'Session ID', 'BiasID', 
            'Chain/CalID', 'Date YMD', 'Time', 'Comment', ' ', 'LNAs/Chain', 
            'LNA UT', 'Stages/LNA', 'Stage UT', ' ', 'Center Frequency (GHz)',
            'Marker Frequency (GHz)', 'Resolution Bandwidth (MHz)',
            'Video Bandwidth (Hz)', 'Frequency Span (MHz)',
            'Power Bandwidth (MHz)', 'Attenuation (dB)', ' ',
            'L1S1GV Set (V)', 'L1S1DV Set (V)', 'L1S1DI Set (mA)',
            'L1S2GV Set (V)', 'L1S2DV Set (V)', 'L1S2DI Set (mA)',
            'L1S3GV Set (V)', 'L1S3DV Set (V)', 'L1S3DI Set (mA)', ' ',
            'L2S1GV Set (V)', 'L2S1DV Set (V)', 'L2S1DI Set (mA)',
            'L2S2GV Set (V)', 'L2S2DV Set (V)', 'L2S2DI Set (mA)',
            'L2S3GV Set (V)', 'L2S3DV Set (V)', 'L2S3DI Set (mA)', ' ',
            'L1S1GV Meas (V)', 'L1S1DV Meas (V)', 'L1S1DI Meas (mA)',
            'L1S2GV Meas (V)', 'L1S2DV Meas (V)', 'L1S2DI Meas (mA)',
            'L1S3GV Meas (V)', 'L1S3DV Meas (V)', 'L1S3DI Meas (mA)', ' ',
            'L2S1GV Meas (V)', 'L2S1DV Meas (V)', 'L2S1DI Meas (mA)',
            'L2S2GV Meas (V)', 'L2S2DV Meas (V)', 'L2S2DI Meas (mA)',
            'L2S3GV Meas (V)', 'L2S3DV Meas (V)', 'L2S3DI Meas (mA)', ' ',
            'CRBE GV Set (V)', 'CRBE DV Set (V)', 'CRBE DI Set (mA)',
            'RTBE GV Set (V)', 'RTBE DV Set (V)', 'RTBE DI Set (mA)', ' ',
            'CRBE GV Meas (V)', 'CRBE DV Meas (V)', 'CRBE DI Meas (mA)',
            'RTBE GV Meas (V)', 'RTBE DV Meas (V)', 'RTBE DI Meas (mA)', ' ',
            'LNA 1 DR (Ohms)', 'LNA 2 DR (Ohms)', 
            'CRBE DR (Ohms)', 'RTBE DR (Ohms)']
        return settings_col_titles

    @staticmethod
    def results_ana_log_header() -> list[str]:
        """Returns part of the header for the results analysis log."""
        res_ana_log_header = [
            '', '', '', '',
            '', '', '', '',
            '', '', '', '', '', '', '',
            'FBW', 'FBW', 'FBW', 'FBW', 'FBW',
            'FBW', 'FBW', 'FBW', 'FBW', 'FBW', '',
            'BW1', 'BW1', 'BW1', 'BW1', 'BW1',
            'BW1', 'BW1', 'BW1', 'BW1', 'BW1', '',
            'BW2', 'BW2', 'BW2', 'BW2', 'BW2',
            'BW2', 'BW2', 'BW2', 'BW2', 'BW2', '',
            'BW3', 'BW3', 'BW3', 'BW3', 'BW3',
            'BW3', 'BW3', 'BW3', 'BW3', 'BW3', '',
            'BW4', 'BW4', 'BW4', 'BW4', 'BW4',
            'BW4', 'BW4', 'BW4', 'BW4', 'BW4', '',
            'BW5', 'BW5', 'BW5', 'BW5', 'BW5',
            'BW5', 'BW5', 'BW5', 'BW5', 'BW5']
        return res_ana_log_header

    @staticmethod
    def results_ana_log_column_titles() -> list[str]:
        """Returns the column titles for the results analysis log."""
        res_ana_log_col_titles = [
            'Project Title', 'LNA ID/s (axb)', 'Session ID', 'BiasID',
            'Date', 'Time', 'Comment', ' ',
            'FBW', 'BW 1', 'BW 2', 'BW 3', 'BW 4', 'BW 5', ' ',
            'Gain Avg (dB)', 'Gain Std Dev', 'Gain Min (dB)',
            'Gain Max (dB)', 'Gain Range (dB)',
            'Noise Temp Avg (K)', 'Noise Temp Std Dev', 'Noise Temp Min (K)',
            'Noise Temp Max (K)', 'Noise Temp Range (K)', ' ',
            'Gain Avg (dB)', 'Gain Std Dev', 'Gain Min (dB)',
            'Gain Max (dB)', 'Gain Range (dB)',
            'Noise Temp Avg (K)', 'Noise Temp Std Dev', 'Noise Temp Min (K)',
            'Noise Temp Max (K)', 'Noise Temp Range (K)', ' ',
            'Gain Avg (dB)', 'Gain Std Dev', 'Gain Min (dB)',
            'Gain Max (dB)', 'Gain Range (dB)',
            'Noise Temp Avg (K)', 'Noise Temp Std Dev', 'Noise Temp Min (K)',
            'Noise Temp Max (K)', 'Noise Temp Range (K)', ' ',
            'Gain Avg (dB)', 'Gain Std Dev', 'Gain Min (dB)',
            'Gain Max (dB)', 'Gain Range (dB)',
            'Noise Temp Avg (K)', 'Noise Temp Std Dev', 'Noise Temp Min (K)',
            'Noise Temp Max (K)', 'Noise Temp Range (K)', ' ',
            'Gain Avg (dB)', 'Gain Std Dev', 'Gain Min (dB)',
            'Gain Max (dB)', 'Gain Range (dB)',
            'Noise Temp Avg (K)', 'Noise Temp Std Dev', 'Noise Temp Min (K)',
            'Noise Temp Max (K)', 'Noise Temp Range (K)', ' ',
            'Gain Avg (dB)', 'Gain Std Dev', 'Gain Min (dB)',
            'Gain Max (dB)', 'Gain Range (dB)',
            'Noise Temp Avg (K)', 'Noise Temp Std Dev', 'Noise Temp Min (K)',
            'Noise Temp Max (K)', 'Noise Temp Range (K)', ' ',
            'Overall Time (s)', 'Start to BE Biasing (s)', 'BE Biasing (s)',
            'BE to Measurement LNA Biasing (s)', 
            'Measurement LNA Biasing (s)',
            'Thermal Setting (s)', 'First Measurement Loop (s)', 
            'Second Measurement Loop (s)']
        return res_ana_log_col_titles

    def results_ana_log_data(
            self, meas_settings: config_handling.MeasurementSettings,
            bias_id: int) -> list:
        """Returns results analysis log data row."""

        fwb = f'{self.freq_array[0]:.2f} -> {self.freq_array[-1]:.2f}'
        bw_1 = self.bw_1_min_max
        bw_2 = self.bw_2_min_max
        bw_3 = self.bw_3_min_max
        bw_4 = self.bw_4_min_max
        bw_5 = self.bw_5_min_max

        bws = [bw_1, bw_2, bw_3, bw_4, bw_5]
        bws_trm = []
        for bandwidth in bws:
            if bandwidth:
                bws_trm.append(f'{bandwidth[0]:.2f} -> {bandwidth[1]:.2f}')
            else:
                bws_trm.append('NA')

        col_data = [
            meas_settings.project_title, meas_settings.lna_id_str,
            str(meas_settings.session_id), str(bias_id),
            self.date_str, self.time_str, meas_settings.comment, None,
            fwb, bws_trm[0], bws_trm[1], bws_trm[2], bws_trm[3], bws_trm[4],
            None, *self.gain_post_proc.as_tuple(0),
            *self.noise_temp_post_proc.as_tuple(0), None,
            *self.gain_post_proc.as_tuple(1),
            *self.noise_temp_post_proc.as_tuple(1), None,
            *self.gain_post_proc.as_tuple(2),
            *self.noise_temp_post_proc.as_tuple(2), None,
            *self.gain_post_proc.as_tuple(3),
            *self.noise_temp_post_proc.as_tuple(3), None,
            *self.gain_post_proc.as_tuple(4),
            *self.noise_temp_post_proc.as_tuple(4), None,
            *self.gain_post_proc.as_tuple(5),
            *self.noise_temp_post_proc.as_tuple(5), None,
            *self.session_timings.as_tuple()]
        return col_data

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
            'Post Hot Measurement 2nd Extra Temp (K)',
            'Cold Loop Times (s)', 'Hot Loop Times (s)']
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
            self.hot.pre_post_temps.post_loop_extra_2_temps,
            self.cold.times, self.hot.times))

    @staticmethod
    def cal_settings_column_titles() -> list[str]:
        """Returns the calibration settings file column titles."""
        cal_settings_col_titles = [
            'Project Title', 'Cryostat Chain', 'Calibration ID',
            'Date', 'Time', 'Comment', ' ',
            'Center Frequency (GHz)', 'Marker Frequency (GHz)',
            'Resolution Bandwidth (MHz)', 'Video Bandwidth (Hz)',
            'Frequency Span (MHz)', 'Power Bandwidth (MHz)',
            'Attenuation (dB)', ' ', 'CRBE GV Set (V)', 'CRBE DV Set (V)',
            'CRBE DI Set (mA)', 'RTBE GV Set (V)', 'RTBE DV Set (V)',
            'RTBE DI Set (mA)', ' ', 'CRBE GV Meas (V)',
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
            'Post Hot Measurement 2nd Extra Temp (K)',
            'Cold Loop Times (s)', 'Hot Loop Times (s)']
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
            self.hot.pre_post_temps.post_loop_extra_2_temps,
            self.cold.times, self.hot.times))
# endregion


@dataclass()
class TimePair:
    @property
    def start_time(self) -> float:
        return self._start_time

    @start_time.setter
    def start_time(self, value: float) -> None:
        self._start_time = value

    @property
    def end_time(self) -> float
        return self._end_time

    @end_time.setter(self, value: float) -> None:
        self.time = value - self.start_time
        self._end_time = value

    @property
    def time(self) -> float
        return self._time

    @time.setter
    def time(self, value: float) -> None:
        self._time = value


class SessionTimings:
    """Records the timings of a session."""

    def __init__(self, measure_method: str) -> None:
        """Constructor for the SessionTimings class."""
        self.overall_time = TimePair()
        self.overall_time.start_time = time.perf_counter()

        self.start_to_be_bias = TimePair()
        self.start_to_be_bias.start_time = self.overall_time.start_time
        
        self.be_biasing = TimePair()

        self.be_to_meas_lna_biasing = TimePair()
        
        self.meas_lna_biasing = TimePair()
        
        self.thermal = TimePair()
        
        self.first_meas_loop = TimePair()

        self.second_meas_loop = TimePair()

    def add_to_bias_time(self, value: float) -> None:
        self.meas_lna_biasing.time = self.meas_lna_biasing.time + value

    def add_to_thermal_time(self, value: float) -> None:
        self.thermal.time = self.thermal.time + value

    @property
    def second_thermal(self) -> float:
        return self._second_thermal

    @second_thermal.setter
    def second_thermal(self, value:float) -> None:
        self._second_thermal = value

    def as_tuple(self):
        return self.overall_time.time, self.start_to_be_bias.time, \
            self.be_biasing.time, self.be_to_meas_lna_biasing.time, \
            self.meas_lna_biasing.time, self.thermal.time, \
            self.first_meas_loop.time, self.second_meas_loop.time