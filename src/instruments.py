# -*- coding: utf-8 -*-
"""instruments.py - Provides classes to store instrument information.

This module provides classes and error handling for the instrumentation
used in the measurement setup. The instrumentation falls into the
following categories:

    * Signal Analyser -
        Measures the output signal from the system. For the y factor
        measurements, the signal generator measures the power over a
        power bandwidth, centered at the down mixed frequency. At
        standard, center is 75MHz, 24MHz power BW, but will need to be
        re-evaluated in a different setup.

    * Signal Generator -
        Synthesizes the input signal, note that a frequency multiplier
        is in the chain, so the signal generated from the instrument is
        a fraction of the target frequency. Multiplier is 8x usually
        but can be specified.

    * Temperature Controller -
        Controls the temperature inside the cryostat. The temperature
        can be set and measured on four channels. One channel will be
        for the load, and then the other three for the LNAs. The warmup
        heater is always on the load.

    * Bias Power Supply
        The power supply which provides the gate voltage and drain
        voltage/current to a given amplifier. Limits are user defined.
        Different bias positions for an amplifier under test will
        yield differences in noise power and gain results, so by
        sweeping through bias positions the optimal operating
        conditions can be ascertained.

    * Switch
        The signal path switch, the position of this determines what
        cryostat signal chain is selected.

Of these instruments different types can be added (for example
'sig gen' or 'vna' as options in Signal Generator). This allows for
additional instruments to be implemented easily. Error handling is done
in two ways, minor error handling where the input variables are checked
over for obvious problems, and major error handling where the
instruments themselves have the error que polled, which should flag up
a lot of problems which haven't been considered.

The resource managers for each instrument are given their own class,
this is so you don't have to send the entire settings instance when
trying to access an instrument.
"""

# region Import modules.
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import logging

from pyvisa import Resource
import numpy as np

import bias_ctrl
import error_handling
import heater_ctrl
import util
# endregion


# region Base Level Classes.
# region Signal Analyser Settings.
@dataclass()
class SpecAnFreqSettings:
    """Frequency settings for the signal analyser.

    Constructor Attributes:
        center_freq (float): Center frequency/GHz.
        marker_freq (float): Marker frequency/GHz.
        freq_span (float): Frequency span/MHz.
    """
    center_freq: float
    marker_freq: float
    freq_span: float


@dataclass()
class SpecAnBWSettings:
    """Bandwidth settings for the signal analyser.

    Constructor Attributes:
        vid_bw (float): Video bandwidth/Hz.
        res_bw (float): Resolution bandwidth/MHz.
        power_bw (float): The bandwidth to measure the power over.
    """
    vid_bw: float
    res_bw: float
    power_bw: float


@dataclass()
class SpecAnAmplSettings:
    """Signal analyser amplitude settings.

    Constructor Attributes:
        atten (float): Attenuation/dB, either 0, 10, 20, 30dB.
        ref_lvl (float): Reference level/dBm.
    """
    atten: float
    ref_lvl: float
# endregion


# region Temperature Controller Settings.
@dataclass()
class TempCtrlChannels:
    """Temperature controller sensor channels.

    Constructor Attributes:
        load_lsch (str): A string containing the lakeshore channel of
            the temperature sensor on the load inside the cryostat.
        chn1_lna_lsch (str): A string containing the lakeshore channel
            of the temperature sensor on the first cryostat chain LNA
            under test.
        chn2_lna_lsch (str): A string containing the lakeshore channel
            of the temperature sensor on the second cryostat chain LNA
            under test.
        chn3_lna_lsch (str): A string containing the lakeshore channel
            of the temperature sensor on the third cryostat chain LNA
            under test.
    """
    load_lsch: str
    chn1_lna_lsch: str
    chn2_lna_lsch: str
    chn3_lna_lsch: str
    extra_sensors_en: bool


@dataclass()
class TempTargets:
    """Hot and cold measurement load temperature targets.

    Constructor Attributes:
        hot_target (float): The target hot temp in Kelvin of the load.
        cold_target (float): The target cold temp in Kelvin of the load.
        lna_target (float): The target lna temperature in Kelvin,
    """
    hot_target: float
    cold_target: float
    lna_target: float
# endregion


# region Signal Generator Settings.
@dataclass()
class FreqSweepSettings:
    """Frequency sweep settings for an individual measurement loop.

    Constructor Attributes:
        min_freq (float): GHz Minimum freq for the frequency sweep.
        max_freq (float): GHz maximum freq for the frequency sweep.
        freq_step_size (float): GHz size of frequency step to get from
            minimum frequency to maximum.
        inter_freq_factor (int): GHz multiplication factor in the
            signal chain, the signal generator outputs the target
            frequency divided by this factor.
    """
    min_freq: float
    max_freq: float
    freq_step_size: float
    inter_freq_factor: int


@dataclass()
class PSUMetaSettings:
    """Settings of the power supply not related to IV outputs directly.

    Constructor Attributes:
        psu_safe_init (bool): If true power supply steps all channels
            to 0V in safe steps before starting.
        bias_psu_en (bool): Setting for debugging, if bias_psu not
            connected then bias_psu queries give synthetic results and
            write commands are skipped.
        sig_gen_en (bool): Setting for debugging, if sig_an not
            connected then sig_gen queries give synthetic results and
            write commands are skipped.
    """
    psu_safe_init: bool
    bias_psu_en: bool
    buffer_time: float
# endregion


# region Bias PSU Settings.
@dataclass()
class GVSearchSettings:
    """Settings for the gate voltage to drain current adaptive search.

    Constructor Attributes:
        g_v_lower_lim (float): Lower gate voltage limit in volts.
        g_v_upper_lim (float): Upper gate voltage limit in volts.
        num_of_g_v_brd_steps (int): Number of steps in the primary gate
            voltage search sweep.
        num_of_g_v_mid_steps (int): Number of steps in the secondary
            gate voltage search sweep.
        num_of_g_v_nrw_steps (int): Number of steps in the tertiary
            gate voltage search sweep.
    """
    g_v_lower_lim: float
    g_v_upper_lim: float
    num_of_g_v_brd_steps: int
    num_of_g_v_mid_steps: int
    num_of_g_v_nrw_steps: int


@dataclass()
class PSULimits:
    """Power supply limits for voltage stepping and current limiting.

    Constructor Attributes:
        v_step_lim (float): The largest voltage step allowed on either
            the gate or drain.
        d_i_lim (float): The drain current limit in milli-amps.
    """
    v_step_lim: float
    d_i_lim: float
# endregion
# endregion


# region Overall Instrumentation Classes.
class SignalAnalyserSettings(SpecAnFreqSettings, SpecAnAmplSettings,
                             SpecAnBWSettings):
    """Class containing all signal analyser settings variables.

    Composed of SpecAnFreqSettings, SpecAnAmplSettings, and
    SpeAnBWSettings.

    Attributes:
        sig_an_en (bool): Whether the signal analyser is enabled for
            the measurement.
    """
    __doc__ += f'\n    SpecAnFreqSettings: {SpecAnFreqSettings.__doc__}\n'
    __doc__ += f'    SpecAnAmplSettings: {SpecAnAmplSettings.__doc__}\n'
    __doc__ += f'    SpecAnBWSettings: {SpecAnBWSettings.__doc__}'

    def __init__(
            self, sa_freq_settings: SpecAnFreqSettings,
            sa_bw_settings: SpecAnBWSettings,
            sa_ampl_settings: SpecAnAmplSettings,
            sig_an_en: bool) -> None:
        """Constructor for the SignalAnalyserSettings class.

        Args:
            sa_freq_settings: The frequency settings to be set on the
                signal analyser.
            sa_bw_settings: The bandwidth settings to be set on the
                signal analyser.
            sa_ampl_settings: The amplitude settings to be set on the
                signal analyser.
            sig_an_en: Whether the signal analyser is enabled in this
                measurement instance or not.
        """

        # region Check input settings.
        error_handling.validate_sa_freq_settings(sa_freq_settings)
        error_handling.check_sa_bw_settings(sa_bw_settings, sa_freq_settings)
        error_handling.check_sa_ampl_settings(sa_ampl_settings)

        if not isinstance(sig_an_en, bool):
            raise Exception('sig_an_en must be True or False.')
        # endregion

        # region Setup subclasses/initialise attributes.
        SpecAnFreqSettings.__init__(
            self, *util.get_dataclass_args(sa_freq_settings))

        SpecAnAmplSettings.__init__(
            self, *util.get_dataclass_args(sa_ampl_settings))

        SpecAnBWSettings.__init__(
            self, *util.get_dataclass_args(sa_bw_settings))

        self.sig_an_en = sig_an_en
        # endregion

    def spec_an_init(
            self, spec_an_rm: Resource, buffer_time: float) -> None:
        """Initialises the spectrum analyser to instance settings."""
        # region Send setup commands to Spectrum Analyser.
        log = logging.getLogger(__name__)
        sarm = spec_an_rm
        util.safe_write('*RST', buffer_time, sarm)
        util.safe_query('*OPC?', buffer_time, sarm, 'spec an')
        util.safe_write('*CLS', buffer_time, sarm)
        log.info(str(sarm.query('*IDN?'))[:-2])
        util.safe_write(
            f':FREQ:CENT {self.center_freq} GHz', buffer_time, sarm)
        util.safe_write(
            f':CALC:MARK1:X {self.marker_freq} Ghz', buffer_time, sarm)
        util.safe_write(f':BAND:RES {self.res_bw} MHz', buffer_time, sarm)
        util.safe_write(f':BAND:VID {self.vid_bw} Hz', buffer_time, sarm)
        util.safe_write(f':FREQ:SPAN {self.freq_span} Mhz', buffer_time, sarm)
        util.safe_write('INIT:CONT 1', buffer_time, sarm)
        util.safe_write(f':DISP:WIND1:TRAC:Y:RLEV {self.ref_lvl} dBm',
                        buffer_time, sarm)
        util.safe_write(':CALC:MARK1:FUNC BPOW', buffer_time, sarm)
        util.safe_write(f':CALC:MARK1:FUNC:BAND:SPAN {self.power_bw} MHz',
                        buffer_time, sarm)
        util.safe_write(f':POW:ATT {self.atten} dB', buffer_time, sarm)
        util.safe_write(':POW:GAIN:BAND LOW', buffer_time, sarm)
        util.safe_write(':POW:GAIN ON', buffer_time, sarm)
        error_status = util.safe_query(
            'SYSTem:ERRor?', buffer_time, sarm, 'spec an')
        if error_status != '+0,"No error"':
            raise Exception(f'Spec An Error Code {error_status}')
        util.safe_query('*OPC?', buffer_time, sarm, 'spec an')
        log.info('Spectrum analyser initialised successfully.')
        # endregion

    def header(self):
        """Return spectrum analyser settings as output header."""
        header = str(f'Center Frequency     = {self.center_freq}GHz' +
                     f'    Marker Frequency     = {self.marker_freq}GHz' +
                     f'    Resolution Bandwidth = {self.res_bw}MHz' +
                     f'    Video Bandwidth      = {self.vid_bw}Hz' +
                     f'    Frequency Span       = {self.freq_span}MHz' +
                     f'    Power Bandwidth      = {self.power_bw}MHz' +
                     f'    Attenuation          = {self.atten}dB')
        return header

    def spec_an_col_data(self) -> tuple:
        """Returns a list of the sig an settings for settings log row.
        """
        return self.center_freq, self.marker_freq, self.res_bw, self.vid_bw, \
            self.freq_span, self.power_bw, self.atten


class SignalGeneratorSettings(FreqSweepSettings):
    """Settings for the PNA-X N5245A network analyser.

    Inherits from FreqSweepSettings.

    Attributes:
        freq_array (ndarray): An equally spaced array of values from
            minimum to maximum frequency stepped by the step size.
        if_freq_array (ndarray): An equally spaced array of values from
            minimum to maximum intermediate frequency stepped by the
            intermediate frequency step size.
        sig_gen_en (bool): Setting for debugging, if sig_an not
            connected then sig_gen queries give synthetic results
            and write commands are skipped.
        sig_gen_pwr_lvls (list[float]): The trimmed list of power
            levels in dBm for the signal generator.
        vna_or_sig_gen (str): Either 'vna' or 'sig gen' depending on
            the instrument being used.
    """

    __doc__ += f'\n    FreqSweepSettings: {FreqSweepSettings.__doc__}'

    def __init__(
            self, freq_sweep_settings: FreqSweepSettings,
            vna_or_sig_gen: str, sig_gen_en: bool) -> None:
        """Constructor for the VNASettings class.

        Args:
            sig_gen_en: Setting for debugging, if sig_an not connected
                then sig_gen queries give synthetic results and write
                commands are skipped.
            vna_or_sig_gen: Either 'vna' or 'sig gen' depending on the
                instrument being used.
            freq_sweep_settings: The frequency sweep settings for the
                measurement instance.
        """

        # region Variable minor error handling.
        error_handling.check_freq_sweep_settings(freq_sweep_settings)

        if vna_or_sig_gen not in ['vna', 'sig gen']:
            raise Exception('vna_or_sig_gen must be "vna" or "sig gen"')

        if not isinstance(sig_gen_en, bool):
            raise Exception('sig_gen_en must be True or False.')
        # endregion

        # region Initialise subclass and set args as attributes.
        FreqSweepSettings.__init__(
            self, *util.get_dataclass_args(freq_sweep_settings))
        self.sig_gen_en = sig_gen_en
        self.vna_or_sig_gen = vna_or_sig_gen
        # endregion

        # region Calculate additional attributes from args.
        self.freq_array = np.arange(
            self.min_freq,
            self.max_freq + self.freq_step_size,
            self.freq_step_size)

        self.if_freq_array = self.freq_array / self.inter_freq_factor
        # endregion

        # region Initialise attribute to be set later.
        self.sig_gen_pwr_lvls = None
        # endregion

    @staticmethod
    def vna_init(sig_gen_rm: Resource, buffer_time: float):
        """Initialise the PNA-X N5245A VNA to 10GHz CW."""
        # region Print VNA ID and set it to CW mode at 10GHz.
        log = logging.getLogger(__name__)
        util.safe_write('*CLS', buffer_time, sig_gen_rm)
        log.info(util.safe_query(
            "*IDN?", buffer_time, sig_gen_rm, 'vna', False, True)[:-2])
        util.safe_write('SENS:SWEEP:TYPE CW', buffer_time, sig_gen_rm)
        util.safe_write('SENS:FREQ:CW 10 GHz', buffer_time, sig_gen_rm)
        error_status = util.safe_query(
            'SYST:ERR?', buffer_time, sig_gen_rm, 'sig gen')
        if error_status != '+0,"No error"':
            raise Exception(f'VNA Error Code {error_status}')
        log.info("VNA initialised successfully.")
        # endregion

    def sig_gen_init(self, sig_gen_rm: Resource, buffer_time: float) -> None:
        """Initialises the signal generator to 10GHz 0dBm."""
        util.safe_write(
            f'PL {self.sig_gen_pwr_lvls[0]} DM', buffer_time, sig_gen_rm)
        util.safe_write(
            f'CW {self.if_freq_array[0]} GZ', buffer_time, sig_gen_rm)

    def set_sig_gen_pwr_lvls(self, trimmed_pwr_lvls: list) -> None:
        """Set the power levels for the signal generator."""
        self.sig_gen_pwr_lvls = trimmed_pwr_lvls


class TempControllerSettings(TempCtrlChannels, TempTargets):
    """Settings for the Lakeshore temperature controller.

    Attributes:
        cryo_chain (int): The cryostat chain currently under test.
        temp_ctrl_en (bool): Setting for debugging, if temp controller
            not connected then temp_ctrl queries give synthetic results
            and write commands are skipped.
        lna_lsch (int): The lna UT lakeshore channel.
        extra_1_lsch (int): The first extra lakeshore channel.
        extra_2_lsch (int): The second extra lakeshore channel.
    """
    __doc__ += f'\n    TempCtrlChannels: {TempCtrlChannels.__doc__}\n'
    __doc__ += f'    TempTargets: {TempTargets.__doc__}'

    def __init__(self, temp_ctrl_channels: TempCtrlChannels,
                 temp_targets: TempTargets, cryo_chain: int,
                 temp_ctrl_en: bool) -> None:
        """Constructor for the TempControllerSettings class.

        Args:
            temp_ctrl_channels: The sensor channels on the lakeshore.
            temp_targets: The hot and cold temperature targets (K).
            cryo_chain: The cryostat chain currently under test.
            temp_ctrl_en: Setting for debugging, if temp controller not
                connected then temp_ctrl queries give synthetic results
                and write commands are skipped.
        """
        # region Minor error handling.
        error_handling.validate_temp_ctrl_channels(temp_ctrl_channels)
        error_handling.check_temp_targets(temp_targets)
        if cryo_chain not in [1, 2, 3]:
            raise Exception('Must select chain 1, 2, or 3.')
        if not isinstance(temp_ctrl_en, bool):
            raise Exception('temp_ctrl_en must be True or False.')
        # endregion

        # region Set args as attributes.
        TempCtrlChannels.__init__(
            self, *util.get_dataclass_args(temp_ctrl_channels))

        TempTargets.__init__(
            self, *util.get_dataclass_args(temp_targets))

        self.cryo_chain = cryo_chain
        self.temp_ctrl_en = temp_ctrl_en
        # endregion

        # region Set additional attributes from args.
        if self.cryo_chain == 1:
            self.extra_1_lsch = self.chn2_lna_lsch
            self.extra_2_lsch = self.chn3_lna_lsch
            self.lna_lsch = self.chn1_lna_lsch
        elif self.cryo_chain == 2:
            self.extra_1_lsch = self.chn1_lna_lsch
            self.extra_2_lsch = self.chn3_lna_lsch
            self.lna_lsch = self.chn2_lna_lsch
        elif self.cryo_chain == 3:
            self.extra_1_lsch = self.chn1_lna_lsch
            self.extra_2_lsch = self.chn2_lna_lsch
            self.lna_lsch = self.chn3_lna_lsch
        else:
            raise Exception('')
        # endregion

    def lakeshore_init(self, lakeshore_rm: Resource, buffer_time: float,
                       sample_or_warm_up: str = 'warm up') -> None:
        """Initialise the lakeshore to correct channels."""
        # region Print Lakeshore ID
        log = logging.getLogger(__name__)
        log.info(util.safe_query(
            '*IDN?', buffer_time, lakeshore_rm, 'lakeshore', False, True)[:-2])
        # endregion

        # region initialise sample/warmup heater.
        if sample_or_warm_up == 'sample':
            if self.cryo_chain == 1:
                heater_ctrl.heater_setup(
                    lakeshore_rm, self.chn1_lna_lsch, 'sample')
            elif self.cryo_chain == 2:
                heater_ctrl.heater_setup(
                    lakeshore_rm, self.chn2_lna_lsch, 'sample')
            elif self.cryo_chain == 3:
                heater_ctrl.heater_setup(
                    lakeshore_rm, self.chn3_lna_lsch, 'sample')
        elif sample_or_warm_up == 'warm up':
            heater_ctrl.heater_setup(lakeshore_rm, self.load_lsch, 'warmup')
        else:
            raise Exception('must be either "sample" or "warm up".')
        log.info("Lakeshore initialised.")
        # endregion


class BiasPSUSettings(GVSearchSettings, PSULimits, PSUMetaSettings):
    """The PSX bias power supply settings.

    Attributes:
        g_v_brd_step_size (float): The gate voltage step size for the
            broad level drain current search.
        g_v_mid_step_size (float): The gate voltage step size for the
            middle level drain current search.
        g_v_nrw_step_size (float): The gate voltage step size for the
            narrow level drain current search.
        g_v_brd_range (list[float]): The range of gate voltages  to
            sweep through in the broad level of the adaptive search.
    """

    __doc__ += f'\n    GVSearchSettings: {GVSearchSettings.__doc__}\n'
    __doc__ += f'    PSULimits: {PSULimits.__doc__}\n'
    __doc__ += f'    PSUMetaSettings: {PSUMetaSettings.__doc__}'

    def __init__(
            self, g_v_search_settings: GVSearchSettings, psu_limits: PSULimits,
            psu_meta_settings: PSUMetaSettings) -> None:
        """Constructor for the BiasPSUSettings class.

        Args:
            g_v_search_settings: The settings for sweeping g_v to find
                d_i.
            psu_limits: The power supply limits.
            psu_meta_settings: Extra information about the power supply
                configuration.
        """
        # region Variable minor error handling.
        error_handling.check_g_v_search_settings(g_v_search_settings)
        error_handling.check_psu_limits(psu_limits)
        error_handling.validate_psu_meta_settings(psu_meta_settings)
        # endregion

        # region Initialise subclasses.
        PSUMetaSettings.__init__(
            self, *util.get_dataclass_args(psu_meta_settings))
        GVSearchSettings.__init__(
            self, *util.get_dataclass_args(g_v_search_settings))
        PSULimits.__init__(
            self, *util.get_dataclass_args(psu_limits))
        # endregion

        # region Calculate additional attributes from args.
        self.g_v_brd_step_size = ((self.g_v_upper_lim - self.g_v_lower_lim)
                                  / self.num_of_g_v_brd_steps)
        self.g_v_mid_step_size = (self.g_v_brd_step_size
                                  / self.num_of_g_v_mid_steps)
        self.g_v_nrw_step_size = (self.g_v_mid_step_size
                                  / self.num_of_g_v_nrw_steps)
        self.g_v_brd_range = np.arange(self.g_v_lower_lim,
                                       self.g_v_upper_lim
                                       + self.g_v_brd_step_size,
                                       self.g_v_brd_step_size)
        # endregion

    @staticmethod
    def psx_init(psx_rm: Resource, buffer_time: float,
                 init_d_v: float, init_g_v: float):
        """Initialise the PSX bias power supply to initial values.

        Arguments:
            psx_rm: Resource manager for the psx bias power supply.
            buffer_time: Time in seconds between each command.
            init_d_v: The initial drain voltage in volts.
            init_g_v: The initial gate voltage in volts.
        """
        log = logging.getLogger(__name__)

        # region Send commands to set all channels to Vd=0 Vg=-0.25
        # Initialise into CV mode, all channels drain to 0v
        # and gate to pinch off (-0.25).
        util.safe_write("BIAS:SET:MODECV:CArd1 1", buffer_time, psx_rm)
        util.safe_write("BIAS:SET:MODECV:CArd2 1", buffer_time, psx_rm)
        util.safe_write("BIAS:SET:MODECV:CArd3 1", buffer_time, psx_rm)

        util.safe_write(f"BIAS:SET:VD:CArd1 {init_d_v}", buffer_time, psx_rm)
        util.safe_write(f"BIAS:SET:VD:CArd2 {init_d_v}", buffer_time, psx_rm)
        util.safe_write(f"BIAS:SET:VD:CArd3 {init_d_v}", buffer_time, psx_rm)
        util.safe_write(f"BIAS:SET:VG:CArd1 {init_g_v}", buffer_time, psx_rm)
        util.safe_write(f"BIAS:SET:VG:CArd2 {init_g_v}", buffer_time, psx_rm)
        util.safe_write(f"BIAS:SET:VG:CArd3 {init_g_v}", buffer_time, psx_rm)

        bias_ctrl.global_bias_en(psx_rm, buffer_time, 1)

        util.safe_write("BIAS:ENable:CArd1 0", buffer_time, psx_rm)
        util.safe_write("BIAS:ENable:CArd2 0", buffer_time, psx_rm)
        util.safe_write("BIAS:ENable:CArd3 0", buffer_time, psx_rm)
        util.safe_write("BIAS:ENable:SYStem 1", buffer_time, psx_rm)
        log.info(util.safe_query('*IDN?', buffer_time, psx_rm, 'psx'))
        log.info("PSX initialised. Global Enabled.")
        # endregion


@dataclass()
class SwitchSettings:
    """Class containing the settings for the cryostat chain switch.

    Constructor Arguments:
        cryo_chain (int): The cryostat chain currently under test.
        switch_en (bool): Setting for debugging, if switch not
            connected then switch queries give synthetic results and
            write commands are skipped.
    """
    cryo_chain: int
    switch_en: bool
# endregion


# region Top level classes.
@dataclass()
class InstrumentSettings:
    """An object containing all equipment settings classes.

    Constructor Arguments:
        buffer_time (float): Time in seconds between each command.
        sig_an_settings (SignalAnalyserSettings): The spectrum
            analyser settings.
        sig_gen_settings (VNASettings): The vna settings.
        temp_ctrl_settings (TempControllerSettings): The lakeshore
            settings.
        bias_psu_settings (BiasPSUSettings): The PSX bias supply
            settings.
        switch_settings (SwitchSettings): Settings or the cryostat
            signal chain switch.
    """
    sig_an_settings: SignalAnalyserSettings
    sig_gen_settings: SignalGeneratorSettings
    temp_ctrl_settings: TempControllerSettings
    bias_psu_settings: BiasPSUSettings
    switch_settings: SwitchSettings
    buffer_time: float


@dataclass
class ResourceManagers:
    """A class for the resource managers used.

    Constructor Arguments:
        sa_rm (Optional[Resource]): Spectrum analyser resource manager.
        sg_rm (Optional[Resource]): Sig gen resource manager.
        tc_rm (Optional[Resource]): Temp controller resource manager.
        psu_rm (Optional[Resource]): The psx resource manager.
    """
    # region Set args to attributes.
    sa_rm: Optional[Resource]
    sg_rm: Optional[Resource]
    tc_rm: Optional[Resource]
    psu_rm: Optional[Resource]
    # endregion

    def __del__(self):
        """Closes resource managers."""
        if self.sa_rm is not None:
            self.sa_rm.close()
        if self.sg_rm is not None:
            self.sg_rm.close()
        if self.tc_rm is not None:
            self.tc_rm.close()
        if self.psu_rm is not None:
            self.psu_rm.close()
# endregion
