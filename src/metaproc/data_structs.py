from dataclasses import dataclass
import numpy as np
from typing import Optional

# region User settings objects.
@dataclass()
class SessionSettings:
    """User settings used to configure outputs."""
    session_id: int
    lna_1_id: str
    lna_2_id: Optional[str] = None
    num_of_lnas: Optional[int] = None
    num_of_stages: Optional[int] = None

    @property
    def num_of_lnas(self) -> int:
        return self._num_of_lnas

    @num_of_lnas.setter
    def num_of_lnas(self, value: int) -> None:
        self._num_of_lnas = value

    @property
    def num_of_stages(self) -> int:
        return self._num_of_stages

    @num_of_stages.setter
    def num_of_stages(self, value: int) -> None:
        self._num_of_stages = value

class MetaprocFileStructure:
    """Class to handle output file structure."""
    def __init__(
            self, input_set_log_path, input_res_log_path):
        """Constructor for the FileStructure class."""
        self.input_res_log_path = input_res_log_path
        self.input_set_log_path = input_set_log_path
# endregion


# region Bias objects.
@dataclass()
class Bias:
    gv: float
    dv: float
    di: float


class Biases:
    def __init__(self, gvs: np.array, dvs: np.array, dis: np.array) -> None:
        self.gvs = gvs
        self.dvs = dvs
        self.dis = dis
        self.bias_set = self.to_bias_set()

    def to_array(self, index: int) -> np.array:
        return np.array(self.gvs[index], self.dvs[index], self.dis[index])

    @classmethod
    def from_bias_set(cls, bias_set: np.array[Bias]):
        gvs = []
        dvs = []
        dis = []
        for bias in bias_set:
            gvs.append(bias.gv)
            dvs.append(bias.dv)
            dis.append(bias.di)
        return cls(np.array(gvs), np.array(dvs), np.array(dis))

    def to_bias_set(self):
        bias_set = []
        for i, _ in enumerate(self.dvs):
            bias_set.append(Bias(self.gvs[i], self.dvs[i], self.dis[i]))
        return np.array(bias_set)
# endregion


# region Bias collection objects.
@dataclass()
class BiasSet:
    set_biases: Biases
    cor_meas_biases: Biases
    meas_biases: np.array[Biases] = np.array([])
    bad_set_biases: np.array[Biases] = np.array([])
    bad_cor_meas_biases: np.array[Biases] = np.array([])


@dataclass
class BiasPlotBiases:
    stage_set_biases: Biases
    stage_meas_biases: Biases
    stage_bad_set_biases: Biases


@dataclass()
class LNABiases:
    set_biases: list[Biases]
    meas_biases: list[Biases]
    cor_meas_biases: list[Biases]
    bad_set_biases: list[Biases]
    bad_meas_biases: list[Biases]
# endregion


# region Results handling objects.
@dataclass()
class InputLogData:
    """The settings and results log data."""
    settings: np.array
    results: np.array


class StageResults:

    # region Internal class.
    @dataclass()
    class BiasColumns:
        """Columns in the settings log for each bias variable."""
        set_g_v: int
        set_d_v: int
        set_d_i: int
        meas_g_v: int
        meas_d_v: int
        meas_d_i: int   
    # endregion

    """Results set for a single stage."""
    def __init__(self, stage_number: int, log_data: InputLogData, 
                 drain_resistance: float) -> None:

        stage_settings = log_data.settings[
            log_data.settings[:,11] == stage_number]
        stage_results = log_data.results[
            log_data.results[:,11] == stage_number]

        self.avg_gains = stage_results[:,15]
        self.avg_noises = stage_results[:,20]

        # region Extract data bias data from settings log.
        if stage_number == 1:
            cols = self.BiasColumns(21, 22, 23, 41, 42, 43)
        elif stage_number == 2:
            cols = self.BiasColumns(24, 25, 26, 44, 45, 46)
        elif stage_number == 3:
            cols = self.BiasColumns(27, 28, 29, 47, 48, 49)
        else:
            raise Exception('Invalid stage number given.')

        corrected_meas_d_v = []
        for i, _ in enumerate(stage_settings):
            corrected_meas_d_v.append(
                stage_settings[i,cols.meas_d_v] 
                - (drain_resistance
                   * (stage_settings[i,cols.meas_d_i] / 1000)))

        set_biases = Biases(
            np.array(stage_settings[:,cols.set_g_v]),
            np.array(stage_settings[:,cols.set_d_v]),
            np.array(stage_settings[:,cols.set_d_i]))
        meas_biases = Biases(  
            np.array(stage_settings[:,cols.meas_g_v]),
            np.array(stage_settings[:,cols.meas_d_v]),
            np.array(stage_settings[:,cols.meas_d_i]))
        cor_meas_biases = Biases(
            np.array(stage_settings[:,cols.meas_g_v]),
            np.array(corrected_meas_d_v),
            np.array(stage_settings[:,cols.meas_d_i]))
        
        bad_set_biases = []
        bad_cor_meas_biases = []

        for i, _ in enumerate(set_biases.dvs):

            cor_meas_gv = cor_meas_biases.gvs[i]
            cor_meas_di = cor_meas_biases.dis[i]
            cor_meas_dv = cor_meas_biases.dvs[i]
            set_di = set_biases.dis[i]
            set_dv = set_biases.dvs[i]
            set_gv = set_biases.gvs[i]

            good_dv = bool(set_dv * 0.95 > cor_meas_dv > set_dv * 1.05)
            good_di = bool(set_di * 0.95 > cor_meas_di > set_di * 1.05)

            if not good_dv or not good_di:
                bad_set_biases.append(
                    Bias(set_gv, set_dv, set_di))
                bad_cor_meas_biases.append(
                    Bias(cor_meas_gv, cor_meas_dv, cor_meas_di))

        self.biases = BiasSet(set_biases, cor_meas_biases, meas_biases, 
                              bad_set_biases, bad_cor_meas_biases)

        # loop through each set bias, if not in bad biases, add to trimmed biases
        trimmed_biases = []
        for i, _ in enumerate(set_biases.dvs):
            in_bad_biases = False
            for j, _ in enumerate(bad_set_biases.dvs):
                if set_biases.to_array(i) == bad_set_biases.to_array(j):
                    in_bad_biases = True

            if not in_bad_biases:
                trimmed_biases.append(set_biases.to_array(i))

        self.trimmed_biases = trimmed_biases
        # endregion


@dataclass()
class LNAData:
    """Container for stage results."""
    stage_1_data: StageResults
    stage_2_data: Optional[StageResults] = None
    stage_3_data: Optional[StageResults] = None


class FullData:
    """Input data for each input file.
    
    Attributes:
        num_of_lnas (int): Number of LNAs in session.
        num_of_stages (int): Number of stages per LNA in session.
        lna_1_stages (StageResults): Results of each stage in LNA 1.
        lna_2_stages (StageResults): Results of each stage in LNA 2.
    """

    def __init__(
            self, req_session_id: int, full_logs: InputLogData) -> None:
        """Constructor for the FullData class.
        
        Args:
            req_session_id: The ID of the session to analyse.
            full_logs: The project settings/results log arrays.
        """

        # region Pull out the input session entries from the full logs.
        session_settings = full_logs.settings[
            full_logs.settings[:,2] == round(float(req_session_id),1)]
        session_results = full_logs.results[
            full_logs.results[:,2] == round(float(req_session_id), 1)]
        # endregion

        # region Validate lengths of settings and results are the same.
        if len(session_settings) != len(session_results):
            raise Exception('Settings and results lengths do not match.')
        # endregion

        # region Pull lna/stage layout info.
        self.num_of_lnas = int(round(session_settings[0,8], 0))
        first_lna = int(round(session_settings[0,9], 0))
        lna_1_drain_resistance = session_settings[0, 75]
        lna_2_drain_resistance = session_settings[0, 76]
        self.num_of_stages = int(round(session_settings[0,10], 0))
        # endregion

        # region Get settings/results rows for each LNA in session.
        # region LNA 1.
        lna_logs = []

        session_lna_1_settings = session_settings[session_settings[:,9] == 1]
        session_lna_2_settings = session_settings[session_settings[:,9] == 2]

        if first_lna == 1:
            session_lna_1_results = session_results[:len(session_lna_1_settings),:]
            session_lna_1_logs = InputLogData(session_lna_1_settings, 
                                              session_lna_1_results)
            
            if len(session_results) != len(session_lna_1_results):
                session_lna_2_results = session_results[len(session_lna_1_settings):,:]
                session_lna_2_logs = InputLogData(session_lna_2_settings, 
                                                  session_lna_2_results)
        # endregion
        # region LNA 2.
        elif first_lna == 2:
            session_lna_2_results = session_results[:len(session_lna_2_settings),:]
            session_lna_2_logs = InputLogData(session_lna_2_settings,
                                              session_lna_2_results)

            if len(session_results) != len(session_lna_2_results):
                session_lna_1_results = session_results[len(session_lna_1_settings):,:]
                session_lna_1_logs = InputLogData(session_lna_1_settings, 
                                                  session_lna_1_results)
        
        lna_logs.append(session_lna_1_logs)
        if len(session_lna_2_settings) != 0:
            lna_logs.append(session_lna_2_logs)
        # endregion
        # endregion

        # region Get stage results from LNA data.
        lna_1_set_biases = []
        lna_1_meas_biases = []
        lna_1_cor_meas_biases = []
        lna_1_bad_set_biases = []
        lna_1_bad_cor_meas_biases = []
        lna_2_set_biases = []
        lna_2_meas_biases = []
        lna_2_cor_meas_biases = []
        lna_2_bad_set_biases = []
        lna_2_bad_cor_meas_biases = []

        drain_resistances = [lna_1_drain_resistance, lna_2_drain_resistance]

        lna_results = []
        for i, lna_log in enumerate(lna_logs):

            stages_results = []

            for stage in range(int(self.num_of_stages)):

                stage_result = StageResults(stage+1, lna_log, 
                                            drain_resistances[i])
                stages_results.append(stage_result)

                if i == 0:
                    lna_1_set_biases.append(
                        stage_result.set_biases)
                    lna_1_meas_biases.append(
                        stage_result.meas_biases)
                    lna_1_cor_meas_biases.append(
                        stage_result.cor_meas_biases)
                    lna_1_bad_set_biases.append(
                        stage_result.bad_set_biases)
                    lna_1_bad_cor_meas_biases.append(
                        stage_result.bad_cor_meas_biases)

                if i == 1:
                    lna_2_set_biases.append(
                        stage_result.set_biases)
                    lna_2_meas_biases.append(
                        stage_result.meas_biases)
                    lna_2_cor_meas_biases.append(
                        stage_result.cor_meas_biases)
                    lna_2_bad_set_biases.append(
                        stage_result.bad_set_biases)
                    lna_2_bad_cor_meas_biases.append(
                        stage_result.bad_cor_meas_biases)

            lna_results.append(stages_results)

        self.lna_1_biases = LNABiases(lna_1_set_biases, 
                                      lna_1_meas_biases, 
                                      lna_1_cor_meas_biases,
                                      lna_1_bad_set_biases,
                                      lna_1_bad_cor_meas_biases)
        if self.num_of_lnas == 2:
            self.lna_2_biases = LNABiases(lna_2_set_biases, 
                                          lna_2_meas_biases, 
                                          lna_2_cor_meas_biases,
                                          lna_2_bad_set_biases,
                                          lna_2_bad_cor_meas_biases)
        # endregion

        # region Store stage results into structures.
        self.lna_1_stages = LNAData(*lna_results[0])
        if self.num_of_lnas == 2:
            self.lna_2_stages = LNAData(*lna_results[1])
        # endregion

    def _get_stage_data(self, lna_data: LNAData, stage: int) -> StageResults:
        """Return the stage data of a requested LNA."""
        if stage == 1:
            return lna_data.stage_1_data
        if stage == 2:
            return lna_data.stage_2_data
        if stage ==3:
            return lna_data.stage_3_data

    def _get_lna_data(self, lna: int) -> LNAData:
        """Return the LNA data of a requested LNA."""
        if lna == 1:
            return self.lna_1_stages
        elif lna == 2:
            return self.lna_2_stages

    def get_bias_acc_data(self, lna: int, stage: int) -> tuple[int, int]:
        """Return the bias accuracy plot data for an lna and stage."""
        stage_data = self._get_stage_data(self._get_lna_data(lna), stage)
        return stage_data.set_biases, stage_data.cor_meas_biases

    @property
    def lna_1_id(self) -> int:
        return self._lna_1_id

    @lna_1_id.setter
    def lna_1_id(self, value) -> None:
        self._lna_1_id = value

    @property
    def lna_2_id(self) -> int:
        return self._lna_2_id

    @lna_2_id.setter
    def lna_2_id(self, value) -> None:
        self._lna_2_id = value
# endregion