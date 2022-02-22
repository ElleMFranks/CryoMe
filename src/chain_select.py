# -*- coding: utf-8 -*-
"""chain_select.py - Cryostat chain change handling.

First function, cryo_chain_switch(), safely changes the signal path
switch. The second, back_end_lna_setup(), biases both the room temp and
cryostat back end LNAs for the cryostat chain specified either using
a direct set method, or a safe set method.
"""

# region Import modules.
from __future__ import annotations
import logging
from time import sleep

import nidaqmx as ni
import pyvisa as pv

import bias_ctrl as bc
import instr_classes as ic
import settings_classes as sc
# endregion


def cryo_chain_switch(
        buffer_time: float, meas_settings: sc.MeasurementSettings) -> None:
    """Controls the switch between the signal and cryostat chain.

    Args:
        buffer_time: The time given for the equipment to buffer after
            a command.
        meas_settings: The measurement settings for the session.
    """
    # region Set up logging and unpack objects.
    log = logging.getLogger(__name__)
    chain = meas_settings.lna_cryo_layout.cryo_chain
    # endregion

    # region Set up ni daq
    switch_read = ni.Task()
    switch_write = ni.Task()
    switch_read.di_channels.add_di_chan('myDAQ1/port0/line4:7')
    switch_write.do_channels.add_do_chan('myDAQ1/port0/line0:3')
    # endregion

    # region Convert requested chain to required switch argument.
    # Args for switch_write.write() relating switch cmd to cryo ports.
    # Pos 1 = 14      P1-P2      P4-P3
    # Pos 2 = 13      P1-P3      NC-NC
    # Pos 3 = 11      P1-P4      P3-P2
    # Pos 4 = 7       NC-NC      P4-P3
    # Write 11, 13, 14 for cryostat chain 1, 2, 3 respectively
    if chain == 1:
        switch_position = 11
    elif chain == 2:
        switch_position = 13
    elif chain == 3:
        switch_position = 14
    else:
        raise Exception('Invalid switch position.')
    # endregion

    # region Set channel.
    switch_write.write([switch_position])
    sleep(buffer_time)
    # endregion

    # region Report cryostat status.
    switch_read_position = switch_read.read()
    if switch_read_position == 64:
        log.info('Cryostat chain 1 activated.')
    elif switch_read_position == 32:
        log.info('Cryostat chain 2 activated.')
    elif switch_read_position == 16:
        log.info('Cryostat chain 3 activated.')
    elif switch_read_position == 0:
        raise Exception('Switch not powered.')
    else:
        raise Exception('Invalid position read from switch.')
    # endregion

    # region Close switch ports.
    switch_write.close()
    switch_read.close()
    # endregion

    # region Set the LNA IDs in the Measurement Settings class.
    meas_settings.config_lna_ids(chain)
    # endregion


def back_end_lna_setup(
        settings: sc.Settings, psu_rm: pv.Resource,
        direct_set: bool = True) -> None:
    """Safely sets back end LNA biases for requested cryostat chain.

    Args:
        settings: Settings for the session.
        psu_rm: The biasing power supply resource manager.
        direct_set: Whether to direct set the biasing stages instead of
            safe set. Direct set is quicker, does not account for cable
            voltage drops.
    """
    # region Unpack passed objects.
    chain = settings.meas_settings.lna_cryo_layout.cryo_chain
    be_biases = settings.meas_settings.direct_lnas.be_lna_settings
    buffer_time = settings.instr_settings.buffer_time
    psu_settings = settings.instr_settings.bias_psu_settings
    # endregion

    # region Unpack LNA bias variables.
    rtbe_lna = be_biases.rtbe_chain_a_lna
    rtbe_chn_g_v = be_biases.rtbe_gv
    if direct_set:
        rtbe_chn_d_v = be_biases.rtbe_chain_a_lna.stage_1.target_d_v_at_lna
    else:
        rtbe_chn_d_v = be_biases.rtbe_chain_a_lna.stage_1.d_v_at_psu
    if chain == 1:
        crbe_lna = be_biases.crbe_chain_1_lna
        crbe_chn_g_v = be_biases.crbe_gvs[0]
    elif chain == 2:
        crbe_lna = be_biases.crbe_chain_2_lna
        crbe_chn_g_v = be_biases.crbe_gvs[1]
    elif chain == 3:
        crbe_lna = be_biases.crbe_chain_3_lna
        crbe_chn_g_v = be_biases.crbe_gvs[2]
    else:
        raise Exception('Invalid chain requested.')
    if direct_set:
        crbe_chn_d_v = crbe_lna.stage_1.target_d_v_at_lna
    else:
        crbe_chn_d_v = crbe_lna.stage_2.d_v_at_psu
    # endregion

    # region Send requests to bias control to set room-temp/cryo BELNAs.
    if psu_rm is not None and direct_set:
        # region If direct set uncor manual input g/dV to biasing.
        bc.direct_set_stage(
            psu_rm, bc.CardChnl(1, 8),
            ic.PSULimits(psu_settings.v_step_lim, 18), buffer_time,
            [bc.GOrDVTarget('g', rtbe_chn_g_v),
                bc.GOrDVTarget('d', rtbe_chn_d_v)])
        bc.direct_set_stage(
            psu_rm, bc.CardChnl(chain, 7),
            ic.PSULimits(psu_settings.v_step_lim, 18), buffer_time,
            [bc.GOrDVTarget('g', crbe_chn_g_v),
                bc.GOrDVTarget('d', crbe_chn_d_v)])
        # endregion

        # region Get measured back-end data.
        rtbe_lna.lna_measured_column_data(psu_rm, True)
        crbe_lna.lna_measured_column_data(psu_rm, True)
        # endregion

    elif psu_rm is not None and not direct_set:
        # region Otherwise send required drain I/V to bias_control
        # This will algorithmically find the requested current at
        # required drain voltage, need to remember in this case
        # drain voltage is corrected.
        bc.bias_set(psu_rm, rtbe_lna, psu_settings, buffer_time)
        bc.bias_set(psu_rm, crbe_lna, psu_settings, buffer_time)
        # endregion

        # region Get measured back-end data.
        rtbe_lna.lna_measured_column_data(psu_rm, True)
        crbe_lna.lna_measured_column_data(psu_rm, True)
        # endregion
    # endregion
