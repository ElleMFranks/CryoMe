# -*- coding: utf-8 -*-
"""lna_classes.py - Provides classes to store Stage/LNA information.

An LNA is made up of up to 3 stages, and a stage is made up of a drain
voltage and a gate voltage and/or drain current. Each chain in the
cryostat is made up of up to two LNAs in series. The chains are
connected to an 8 channel card on the LNA. There are four single stage
back end LNAs, one per chain inside the cryostat (CRBE) and one which
is external to the cryostat, connected to all the chains through a
switch, and is kept at room temperature (RTBE).

The power supply channel layout at present is as follows:
    LNA1 -> Stages 1, 2, 3 = Card (Cryostat Chain #) Channels 1, 2, 3
    LNA2 -> Stages 1, 2, 3 = Card (Cryostat Chain #) Channels 4, 5, 6
    CRBE -> Stage 1 = Card (Cryostat Chain #) Channel 7
    RTBE -> Stage 1 = Card 1 Channel 8

This module defines the data structures for stages and LNAs, both the
LNAs under test, and the backend LNAs are handled.
"""

# region Import modules.
from __future__ import annotations
from dataclasses import dataclass
from typing import (Optional, Union)
import copy as cp

import pyvisa as pv

import bias_ctrl as bc
import settings_classes as scl
import util as ut
# endregion


# region Base classes.
@dataclass()
class IndivBias:
    """Individual bias values for a stage.

    Constructor Arguments:
        target_d_v_at_lna (float):
        d_i (float):
        g_v (float):
    """
    d_i: float
    target_d_v_at_lna: float
    g_v: Union[str, float] = 'NA'


@dataclass()
class StageSettings:
    """Non-bias settings for a stage.

    Constructor Attributes:
        lna_position (str):
        d_i_limit (float):
        card_chnl (Optional[CardChnl]):
    """
    lna_position: str
    d_i_limit: float = 10
    card_chnl: Optional[bc.CardChnl] = None
# endregion


# region Mid level classes.
class StageBiasSet(IndivBias, StageSettings):
    """The gate and drain settings for a single lna stage.
    Attributes:
    """

    def __init__(self, stage_settings: StageSettings, bias: IndivBias) -> None:
        """Constructor for the StageBiasSet class.

        Args:
        """

        StageSettings.__init__(self, *ut.get_dataclass_args(stage_settings))

        # region Set drain resistance depending on passed position.
        if self.lna_position == 'LNA1':
            self.d_resistance = 12.6  # Ohms
        elif self.lna_position == 'LNA2':
            self.d_resistance = 2.6  # Ohms
        elif self.lna_position == 'CRBE':
            self.d_resistance = 12.5  # Ohms
        elif self.lna_position == 'RTBE':
            self.d_resistance = 12.5  # Ohms
        else:
            raise Exception('lna_position invalid.')
        # endregion

        # region Set args to attributes.
        IndivBias.__init__(self, *ut.get_dataclass_args(bias))
        # endregion

        # region Calculate additional attributes from args.
        self.d_v_at_psu = self.target_d_v_at_lna \
            + ((self.d_i * self.d_resistance) / 1000)
        # endregion

        self.d_i_lim = 8  # Default to 8mA just in case set incorrectly.

    @property
    def d_i(self) -> float:
        """Drain current / mA."""
        return self._d_i

    @d_i.setter
    def d_i(self, value: float) -> None:
        self._d_i = value

    @property
    def target_d_v_at_lna(self) -> float:
        """Drain voltage at LNA (after loss) / V."""
        return self._target_d_v_at_lna

    @target_d_v_at_lna.setter
    def target_d_v_at_lna(self, value) -> None:
        self._d_v_at_psu = (
            value + (self.d_resistance * self.d_i))
        self._target_d_v_at_lna = value

    @property
    def d_i_lim(self) -> float:
        """Drain current limit / mA."""
        return self._d_i_lim

    @d_i_lim.setter
    def d_i_lim(self, value: float) -> None:
        self._d_i_lim = value

    @property
    def g_v(self) -> float:
        """Gate voltage / V."""
        return self._g_v

    @g_v.setter
    def g_v(self, value: float) -> None:
        self._g_v = value

    @property
    def card_chnl(self) -> bc.CardChnl:
        """Card and channel on power supply."""
        return self._card_chnl

    @card_chnl.setter
    def card_chnl(self, value: Optional[bc.CardChnl]) -> None:
        if not (isinstance(value, bc.CardChnl) or value is None):
            raise Exception('Must be type CardChnl.')
        self._card_chnl = value

    def header(self) -> str:
        """Return the header information for the stage instance."""
        return str(f'GV = {self.g_v}V ' +
                   f'DV = {self.d_v_at_psu:+.3f}V ' +
                   f'DI = {self.d_i:.3f}mA ')

    def bias_strs(self) -> list[str]:
        """Returns the array of bias values for the instance."""
        return [
            f'{self.g_v:+.3f}', f'{self.d_v_at_psu:+3f}', f'{self.d_i:.3f}']


@dataclass
class LNAStages:
    """Settings for the collection of stages of an LNA.

    Constructor Attributes:
        stage_1 (StageBiasSet):
        stage_2 (Optional[StageBiasSet]):
        stage_3 (Optional[StageBiasSet]):
    """
    stage_1: StageBiasSet
    stage_2: Optional[StageBiasSet] = None
    stage_3: Optional[StageBiasSet] = None
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


# region Top level class.
class LNABiasSet(LNACryoLayout, LNAStages):
    """The bias settings for all stages in an LNA.

    Attributes:
        lna_position (str): Either "LNA1" if the first lna from the load
            in the cryostat, or "LNA2" if the second.
        lna_meas_column_data (list[str]): The LNA data which goes into
            the settings log for this measurement.

    """

    def __init__(
            self, lna_position: str, lna_cryo_layout: LNACryoLayout,
            single_stage_d_i_lim: float, lna_stages: LNAStages) -> None:
        """Constructor for the LNABiasSet class.

        Args:
            lna_position: Either "LNA1" if the first lna from the load
                in the cryostat, or "LNA2" if the second.

        Raises:
            Exception: If LNA2 is passed, but LNA is not a chain raises
                exception.
        """

        # region Set args to attributes.
        LNACryoLayout.__init__(
            self, *ut.get_dataclass_args(lna_cryo_layout))
        LNAStages.__init__(self, *ut.get_dataclass_args(lna_stages))

        self.lna_position = lna_position
        # endregion

        # region Set default attribute, initialise attributes for later.
        self.lna_meas_column_data = None
        # endregion

        # region Handle possible variable entry error.
        if self.lnas_per_chain != 2 and self.lna_position == 'LNA2':
            raise Exception(
                'LNA2 cannot exist unless there are 2 LNAs per chain.')
        # endregion

        # region Set additional attributes based on args.
        # region Handle channels/drain impedances based on LNA position.
        if self.lna_position == 'LNA1':
            self.stage_1.card_chnl = bc.CardChnl(self.cryo_chain, 1)
            self.stage_2.card_chnl = bc.CardChnl(self.cryo_chain, 2)
            self.stage_3.card_chnl = bc.CardChnl(self.cryo_chain, 3)
        elif self.lna_position == 'LNA2':
            self.stage_1.card_chnl = bc.CardChnl(self.cryo_chain, 4)
            self.stage_2.card_chnl = bc.CardChnl(self.cryo_chain, 5)
            self.stage_3.card_chnl = bc.CardChnl(self.cryo_chain, 6)
        elif self.lna_position == 'CRBE':
            self.stage_1.card_chnl = bc.CardChnl(self.cryo_chain, 7)
        elif self.lna_position == 'RTBE':
            self.stage_1.card_chnl = bc.CardChnl(1, 8)
        # endregion

        # region Handle same stage conditions.
        if self.stage_1_2_same and not self.stage_2_3_same:
            self.stage_2 = None
            self.stage_1.d_i_lim = 2 * single_stage_d_i_lim
        elif self.stage_2_3_same and not self.stage_1_2_same:
            self.stage_3 = None
            self.stage_2.d_i_lim = 2 * single_stage_d_i_lim
        elif self.stage_1_2_same and self.stage_2_3_same:
            self.stage_2 = None
            self.stage_3 = None
            self.stage_1.d_i_lim = 3 * single_stage_d_i_lim
        elif not self.stage_1_2_same and not self.stage_2_3_same:
            self.stage_1.d_i_lim = single_stage_d_i_lim
            if self.stage_2 is not None:
                self.stage_2.d_i_lim = single_stage_d_i_lim
            if self.stage_3 is not None:
                self.stage_3.d_i_lim = single_stage_d_i_lim
        else:
            raise Exception('')
        # endregion

        # region Set invalid stages to None.
        if self.stages_per_lna == 1:
            self.stage_2 = None
            self.stage_3 = None
        elif self.stages_per_lna == 2:
            self.stage_3 = None
        # endregion
        # endregion

        self.lna_meas_column_data = []

    @property
    def lna_meas_column_data(self):
        """The power supply measured bias variables."""
        return self._lna_meas_column_data

    @lna_meas_column_data.setter
    def lna_meas_column_data(self, value):
        self._lna_meas_column_data = value

    def sweep_setup(self, stage_ut, d_v_ut, d_i_ut, d_v_nom, d_i_nom):
        """Sets drain current and voltage to either nominal or UT value.
        """
        if stage_ut == 1:
            self.stage_1.d_i = d_i_ut
            self.stage_1.target_d_v_at_lna = d_v_ut
            if self.stage_2 is not None:
                self.stage_2.d_i = d_i_nom
                self.stage_2.target_d_v_at_lna = d_v_nom
            if self.stage_3 is not None:
                self.stage_3.d_i = d_i_nom
                self.stage_3.target_d_v_at_lna = d_v_nom
        if stage_ut == 2:
            self.stage_1.d_i = d_i_nom
            self.stage_1.target_d_v_at_lna = d_v_nom
            if self.stage_2 is not None:
                self.stage_2.d_i = d_i_ut
                self.stage_2.target_d_v_at_lna = d_v_ut
            if self.stage_3 is not None:
                self.stage_3.d_i = d_i_nom
                self.stage_3.target_d_v_at_lna = d_v_nom
        if stage_ut == 3:
            self.stage_1.d_i = d_i_nom
            self.stage_1.target_d_v_at_lna = d_v_nom
            if self.stage_2 is not None:
                self.stage_2.d_i = d_i_nom
                self.stage_2.target_d_v_at_lna = d_v_nom
            if self.stage_3 is not None:
                self.stage_3.d_i = d_i_ut
                self.stage_3.target_d_v_at_lna = d_v_ut

    def nominalise(self, d_v_nom, d_i_nom):
        """Sets drain current and voltage to nominals."""
        self.stage_1.d_i = d_i_nom
        self.stage_1.target_d_v_at_lna = d_v_nom
        if self.stage_2 is not None:
            self.stage_2.d_i = d_i_nom
            self.stage_2.target_d_v_at_lna = d_v_nom
        if self.stage_3 is not None:
            self.stage_3.d_i = d_i_nom
            self.stage_3.target_d_v_at_lna = d_v_nom

    def lna_bias_strs(self) -> list[str]:
        """Return the bias details of the LNA for settings/headers."""
        # region Construct and return bias details in strings.
        lna_bias = []
        lna_bias.extend(self.stage_1.bias_strs())

        # region Stage 2.
        if self.stage_2 is not None:
            lna_bias.extend(self.stage_2.bias_strs())
        else:
            lna_bias.extend(['NA', 'NA', 'NA'])
        # endregion

        # region Stage 3.
        if self.stage_3 is not None:
            lna_bias.extend(self.stage_3.bias_strs())
        else:
            lna_bias.extend(['NA', 'NA', 'NA'])
        # endregion

        return lna_bias
        # endregion

    def lna_set_column_data(self, is_calibration: bool = False) -> list[str]:
        """Returns the lna data for the settings log row."""
        # region Stack data for LNAs that exist, if state invalid 'NA'.
        if (self.lnas_per_chain == 2 and self.lna_position == 'LNA2') \
                or (self.lna_position in ['LNA1', 'CRBE', 'RTBE']) \
                or is_calibration:

            # region Stage 1.
            if self.stage_1.g_v is None:
                self.stage_1.g_v = 'NA'
            column_data = [f'{self.stage_1.g_v}',
                           f'{self.stage_1.target_d_v_at_lna:+.3f}',
                           f'{self.stage_1.d_i:+.3f}']
            if is_calibration or self.lna_position in ['CRBE', 'RTBE']:
                return column_data
            # endregion

            # region Stage 2.
            if self.stage_2 is None:
                column_data.append('NA')
                column_data.append('NA')
                column_data.append('NA')
            else:
                column_data.append(f'{self.stage_2.g_v}')
                column_data.append(f'{self.stage_2.target_d_v_at_lna:+.3f}')
                column_data.append(f'{self.stage_2.d_i:+.3f}')
            # endregion

            # region Stage 3.
            if self.stage_3 is None:
                column_data.append('NA')
                column_data.append('NA')
                column_data.append('NA')
            else:
                column_data.append(f'{self.stage_3.g_v}')
                column_data.append(f'{self.stage_3.target_d_v_at_lna:+.3f}')
                column_data.append(f'{self.stage_3.d_i:+.3f}')
            # endregion

            return column_data

        # region Handle variable entry error.
        raise Exception('')
        # endregion
        # endregion

    def lna_measured_column_data(
            self, psx_rm: Optional[pv.Resource] = None,
            is_calibration: bool = False) -> Optional[list[Union[float, str]]]:
        """Return the measured bias conditions of the LNA."""
        # region Measure and return bias conditions, or dummy values.
        meas_col_data = []
        # region Measure bias conditions.
        # region Stage 1.
        if psx_rm is not None and self.stage_1 is not None:
            meas_col_data.append(ut.safe_query(
                    f'Bias:MEASure:VG:'
                    f'CArd{self.stage_1.card_chnl.card}:'
                    f'CHannel{self.stage_1.card_chnl.chnl}?',
                    0.5, psx_rm, 'psx', True))
            meas_col_data.append(ut.safe_query(
                    f'Bias:MEASure:VD:'
                    f'CArd{self.stage_1.card_chnl.card}:'
                    f'CHannel{self.stage_1.card_chnl.chnl}?',
                    0.5, psx_rm, 'psx', True))
            meas_col_data.append((ut.safe_query(
                    f'Bias:MEASure:ID:'
                    f'CArd{self.stage_1.card_chnl.card}:'
                    f'CHannel{self.stage_1.card_chnl.chnl}?',
                    0.5, psx_rm, 'psx', True) * 1000))
            if is_calibration:
                self.lna_meas_column_data = meas_col_data
                return
            # endregion
        # endregion

        # region Stage 2.
        if psx_rm is not None and self.stage_2 is not None:
            meas_col_data.append(
                ut.safe_query(
                    f'Bias:MEASure:VG:'
                    f'CArd{self.stage_2.card_chnl.card}:'
                    f'CHannel{self.stage_2.card_chnl.chnl}?',
                    0.5, psx_rm, 'psx', True))
            meas_col_data.append(
                ut.safe_query(
                    f'Bias:MEASure:VD:'
                    f'CArd{self.stage_2.card_chnl.card}:'
                    f'CHannel{self.stage_2.card_chnl.chnl}?',
                    0.5, psx_rm, 'psx', True))
            meas_col_data.append(
                (ut.safe_query(
                    f'Bias:MEASure:ID:'
                    f'CArd{self.stage_2.card_chnl.card}:'
                    f'CHannel{self.stage_2.card_chnl.chnl}?',
                    0.5, psx_rm, 'psx', True) * 1000))
        elif psx_rm is not None and self.stage_2 is None:
            meas_col_data.extend(['NA', 'NA', 'NA'])
        # endregion

        # region Stage 3.
        if psx_rm is not None and self.stage_3 is not None:

            meas_col_data.append(ut.safe_query(
                    f'Bias:MEASure:VG:'
                    f'CArd{self.stage_3.card_chnl.card}:'
                    f'CHannel{self.stage_3.card_chnl.chnl}?',
                    0.5, psx_rm, 'psx', True))
            meas_col_data.append(ut.safe_query(
                    f'Bias:MEASure:VD:'
                    f'CArd{self.stage_3.card_chnl.card}:'
                    f'CHannel{self.stage_3.card_chnl.chnl}?',
                    0.5, psx_rm, 'psx', True))
            meas_col_data.append((ut.safe_query(
                    f'Bias:MEASure:ID:'
                    f'CArd{self.stage_3.card_chnl.card}:'
                    f'CHannel{self.stage_3.card_chnl.chnl}?',
                    0.5, psx_rm, 'psx', True) * 1000))
        elif psx_rm is not None and self.stage_3 is None:
            meas_col_data.extend(['NA', 'NA', 'NA'])
        # endregion
        # endregion

        # region Set measured data.
        for i, _ in enumerate(meas_col_data):
            if meas_col_data[i] != 'NA':
                meas_col_data[i] = f'{meas_col_data[i]:.3f}'

        self.lna_meas_column_data = meas_col_data
        # endregion

        # region Handle no psu resource manager.
        if psx_rm is None and (is_calibration or
                               self.lna_position in ['CRBE', 'RTBE']):
            # region Return recognisable dummy values.
            i = 0
            while i < 3:
                meas_col_data.append(i)
                i += 1
            self.lna_meas_column_data = meas_col_data
        elif psx_rm is None and not (is_calibration or
                                      self.lna_position in ['CRBE', 'RTBE']):
            i=0
            while i < 9:
                meas_col_data.append(i)
                i += 1
            self.lna_meas_column_data = meas_col_data
            # endregion
        # endregion
        # endregion

    def lna_header(self) -> list[str]:
        """Return the results csv file header."""
        # region Construct/return header strs dep on existing stages.
        results_header = [f'{self.stage_1.header()}']

        # region Stage 2.
        if self.stage_2 is not None:
            results_header.append(f'{self.stage_2.header()}')
        else:
            results_header.append('GV = NAV DV = NAV DI = NAmA ')
        # endregion

        # region Stage 3.
        if self.stage_3 is not None:
            results_header.append(f'{self.stage_3.header()}')
        else:
            results_header.append('GV = NAV DV = NAV DI = NAmA ')
        return results_header
        # endregion
        # endregion

    def lna_d_v_strs(self) -> list[str]:
        """Returns the drain voltage data for the lna instance."""
        # region Construct/return drain V strs dep on existing stages.
        if (self.lnas_per_chain == 2 and self.lna_position == 'LNA2') \
                or (self.lna_position == 'LNA1'):
            d_v_str_data = [f'{self.stage_1.target_d_v_at_lna:+.3f}']

            # region Stage 2.
            if self.stage_2 is not None:
                if self.stage_2.target_d_v_at_lna is not None:
                    d_v_str_data.append(
                        f'{self.stage_2.target_d_v_at_lna:+.3f}')
                else:
                    d_v_str_data.append('NA')
            else:
                d_v_str_data.append('NA')
            # endregion

            # region Stage 3.
            if self.stage_3 is not None:
                if self.stage_3.target_d_v_at_lna is not None:
                    d_v_str_data.append(
                        f'{self.stage_3.target_d_v_at_lna:+.3f}')
                else:
                    d_v_str_data.append('NA')
            else:
                d_v_str_data.append('NA')
            # endregion

            return d_v_str_data

        # region Catch variable entry error.
        raise Exception('')
        # endregion

        # endregion

    def lna_g_v_strs(self) -> list[str]:
        """Returns the gate voltage data for the lna instance."""
        # region Construct/return gate V strs dep on existing stages.
        if (self.lnas_per_chain == 2 and self.lna_position == 'LNA2') \
                or (self.lna_position == 'LNA1'):
            g_v_str_data = [f'{self.stage_1.g_v}']

            # region Stage 2.
            if self.stage_2 is not None:
                if self.stage_2.g_v is not None:
                    g_v_str_data.append(f'{self.stage_2.g_v}')
                else:
                    g_v_str_data.append('NA')
            else:
                g_v_str_data.append('NA')
            # endregion

            # region Stage 3.
            if self.stage_3 is not None:
                if self.stage_3.g_v is not None:
                    g_v_str_data.append(f'{self.stage_3.g_v}')
                else:
                    g_v_str_data.append('NA')
            else:
                g_v_str_data.append('NA')
            # endregion

            return g_v_str_data

        # region Handle variable error.
        raise Exception('')
        # endregion
        # endregion

    def lna_d_i_strs(self) -> list[str]:
        """Returns the drain current data for the lna instance."""
        # region Construct/return drain I data for the lna instance.
        if (self.lnas_per_chain == 2 and self.lna_position == 'LNA2') \
                or (self.lna_position == 'LNA1'):
            d_i_str_data = [f'{self.stage_1.d_i:+.3f}']

            # region Stage 2.
            if self.stage_2 is not None:
                if self.stage_2.d_i is not None:
                    d_i_str_data.append(f'{self.stage_2.d_i:+.3f}')
                else:
                    d_i_str_data.append('NA')
            else:
                d_i_str_data.append('NA')
            # endregion

            # region Stage 3.
            if self.stage_3 is not None:
                if self.stage_3.d_i is not None:
                    d_i_str_data.append(f'{self.stage_3.d_i:+.3f}')
                else:
                    d_i_str_data.append('NA')
            else:
                d_i_str_data.append('NA')
            # endregion

            return d_i_str_data

        # region Handle variable error.
        raise Exception('')
        # endregion
        # endregion
# endregion


# region Directly set LNA classes.
class ManualLNASettings:
    """The drain voltage and currents for manual LNA bias conditions.

    Note:
        The passed drain voltages get corrected for wire voltage drop.

    Attributes:
    """
    def __init__(
            self, manual_lna_biases: dict,
            lna_cryo_layout: LNACryoLayout, d_i_lim: float,
            ) -> None:
        """Constructor for the ManualLNASettings class.

        Args:
        """

        # region Set lna stages from args.
        self.lna_1_stages = LNAStages(
            StageBiasSet(StageSettings('LNA1', d_i_lim),
                         IndivBias(manual_lna_biases['man_l1_s1_d_i'],
                                   manual_lna_biases['man_l1_s1_d_v'])),
            StageBiasSet(StageSettings('LNA1', d_i_lim),
                         IndivBias(manual_lna_biases['man_l1_s2_d_i'],
                                   manual_lna_biases['man_l1_s2_d_v'])),
            StageBiasSet(StageSettings('LNA1', d_i_lim),
                         IndivBias(manual_lna_biases['man_l1_s3_d_i'],
                                   manual_lna_biases['man_l1_s3_d_v']))
        )

        self.lna_2_stages = LNAStages(
            StageBiasSet(StageSettings('LNA2', d_i_lim),
                         IndivBias(manual_lna_biases['man_l2_s1_d_i'],
                                   manual_lna_biases['man_l2_s1_d_v'])),
            StageBiasSet(StageSettings('LNA2', d_i_lim),
                         IndivBias(manual_lna_biases['man_l2_s2_d_i'],
                                   manual_lna_biases['man_l2_s2_d_v'])),
            StageBiasSet(StageSettings('LNA2', d_i_lim),
                         IndivBias(manual_lna_biases['man_l2_s3_d_i'],
                                   manual_lna_biases['man_l2_s3_d_v']))
        )
        # endregion

        # region Set LNAs from stages and args.
        self.lna_1_man = LNABiasSet(
            'LNA1', lna_cryo_layout, d_i_lim, self.lna_1_stages)

        self.lna_2_man = LNABiasSet(
            'LNA2', lna_cryo_layout, d_i_lim, self.lna_2_stages)
        # endregion


class NominalLNASettings:
    """Nominal LNA drain and gate voltages set to LNA objects."""
    def __init__(self, settings: scl.Settings) -> None:
        """Constructor for the NominalLNASettings class."""
        meas_settings = settings.meas_settings
        d_i_lim = settings.instr_settings.bias_psu_settings.d_i_lim

        self.lna_1_nom_stg = StageBiasSet(
            StageSettings('LNA1'),
            IndivBias(cp.copy(settings.sweep_settings.d_i_nominal),
                      cp.copy(settings.sweep_settings.d_v_nominal)))

        self.lna_2_nom_stg = StageBiasSet(
            StageSettings('LNA2'),
            IndivBias(cp.copy(settings.sweep_settings.d_i_nominal),
                      cp.copy(settings.sweep_settings.d_v_nominal)))

        self.lna_1_nom_bias = LNABiasSet(
            'LNA1', meas_settings.lna_cryo_layout, d_i_lim,
            LNAStages(cp.deepcopy(self.lna_1_nom_stg),
                      cp.deepcopy(self.lna_1_nom_stg),
                      cp.deepcopy(self.lna_1_nom_stg)))

        if meas_settings.lna_cryo_layout.lnas_per_chain == 2:
            self.lna_2_nom_bias = LNABiasSet(
                'LNA2', meas_settings.lna_cryo_layout, d_i_lim,
                LNAStages(cp.deepcopy(self.lna_2_nom_stg),
                          cp.deepcopy(self.lna_2_nom_stg),
                          cp.deepcopy(self.lna_2_nom_stg)))
        else:
            self.lna_2_nom_bias = None


class BackEndLNASettings:
    """The drain voltages/currents & gate voltages for the backend LNAs.

    Attributes:
        d_i_lim (float): D current limit for the LNAs in question.
    """

    def __init__(
            self, be_lna_biases: dict, be_d_i_lim: float = 20) -> None:
        """Constructor for the BackEndLNASettings class.

        Args:
            be_lna_biases: Dictionary containing user input BE LNA
                bias variables.
            be_d_i_lim: D current limit for the LNA in question.
        """

        self.rtbe_gv = be_lna_biases['rtbe_chna_g_v']
        self.crbe_gvs = [be_lna_biases['crbe_chn1_g_v'],
                         be_lna_biases['crbe_chn2_g_v'],
                         be_lna_biases['crbe_chn3_g_v']]

        # region Set args to attributes.
        self.d_i_lim = be_d_i_lim
        # endregion

        # region Construct additional parameters from args.
        # region Stages.
        rtbe_chain_a_stage_1 = StageBiasSet(
            StageSettings('RTBE', be_d_i_lim),
            IndivBias(be_lna_biases['rtbe_chna_d_i'],
                      be_lna_biases['rtbe_chna_d_v']))
        crbe_chain_1_stage_1 = StageBiasSet(
            StageSettings('CRBE', be_d_i_lim),
            IndivBias(be_lna_biases['crbe_chn1_d_i'],
                      be_lna_biases['crbe_chn1_d_v']))
        crbe_chain_2_stage_1 = StageBiasSet(
            StageSettings('CRBE', be_d_i_lim),
            IndivBias(be_lna_biases['crbe_chn2_d_i'],
                      be_lna_biases['crbe_chn2_d_v']))
        crbe_chain_3_stage_1 = StageBiasSet(
            StageSettings('CRBE', be_d_i_lim),
            IndivBias(be_lna_biases['crbe_chn3_d_i'],
                      be_lna_biases['crbe_chn3_d_v']))
        # endregion

        # region LNAs.
        self.rtbe_chain_a_lna = LNABiasSet(
            'RTBE', LNACryoLayout(1, 1, 1, False, False), be_d_i_lim,
            LNAStages(rtbe_chain_a_stage_1))
        self.crbe_chain_1_lna = LNABiasSet(
            'CRBE', LNACryoLayout(1, 1, 1, False, False), be_d_i_lim,
            LNAStages(crbe_chain_1_stage_1))
        self.crbe_chain_2_lna = LNABiasSet(
            'CRBE', LNACryoLayout(2, 1, 1, False, False), be_d_i_lim,
            LNAStages(crbe_chain_2_stage_1))
        self.crbe_chain_3_lna = LNABiasSet(
            'CRBE', LNACryoLayout(3, 1, 1, False, False), be_d_i_lim,
            LNAStages(crbe_chain_3_stage_1))
        # endregion
        # endregion
# endregion
