# -*- coding: utf-8 -*-
"""bias_ctrl.py - Handles the power supply and IV biasing algorithms.

This module provides functions to control the power supply. The module
contains mostly internal functions, and external to this module the only
functions that should be called are bias_set, global_bias_en, and
psu_safe_init.
"""

# region Import modules.
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
import logging
from typing import Union
from time import sleep

import numpy as np
import pyvisa as pv

import instr_classes as ic
import lna_classes as lc
import util as ut
# endregion


# region Define useful dataclasses.
@dataclass()
class GOrDVTarget:
    """Gate or drain voltage target.

    Constructor Attributes:
        g_or_d (str): Gate or drain voltage.
        v_target (float): Voltage target.
    """
    g_or_d: str
    v_target: float


@dataclass
class CardChnl:
    """Power supply card and channel.

    Constructor Attributes:
        card (int): The power supply card.
        chnl (Union[list[int], int]): The power supply channel/s.
    """
    card: int
    chnl: Union[list[int], int]
# endregion


def psu_safe_init(
        psu_rm: pv.Resource, buffer_time: float,
        psu_lims: ic.PSULimits, g_v_low_lim: float) -> None:
    """Safely disables all enabled channels, switches global en off.

    Args:
        psu_rm: Resource manager for the bias power supply.
        buffer_time: This is how long to wait in seconds after each
            command which is sent to any of the instruments.
        psu_lims: Power supply drain current and gate voltage step lim.
        g_v_low_lim: The lower limit for the gate voltage.
    """

    # region Define channels and cards on psu and set global enable on.
    log = logging.getLogger(__name__)
    channel = [1, 2, 3, 4, 5, 6, 7, 8]
    card = [1, 2, 3]
    global_bias_en(psu_rm, buffer_time, 1)
    # endregion

    # region Go through each channel on each card and safely disable.
    for state in product(card, channel):
        card_chnl = CardChnl(state[0], state[1])
        is_ch_on = ut.safe_query(
            f'Bias:ENable:CArd{card_chnl.card}:Channel{card_chnl.chnl}?',
            buffer_time, psu_rm, 'psx')

        # region If channel on, gate to min, drain off, local en off.
        if is_ch_on == '1\r':
            # Set gate voltage
            g_v_targ = GOrDVTarget('g', g_v_low_lim)
            d_v_targ = GOrDVTarget('d', 0)
            exc_psu_lims = ic.PSULimits(
                psu_lims.v_step_lim, 3 * psu_lims.d_i_lim)
            _safe_set_v(psu_rm, card_chnl, g_v_targ, exc_psu_lims, buffer_time)
            # Set drain voltage
            _safe_set_v(psu_rm, card_chnl, d_v_targ, exc_psu_lims, buffer_time)

            # Disable channel
            ut.safe_write(f'Bias:ENable:CA{state[0]}:CHannel{state[1]} 0',
                          buffer_time, psu_rm)
        # endregion

        log.info(f'Card {state[0]} Channel {state[1]} safe.')
    # endregion

    # region Disable supply
    global_bias_en(psu_rm, buffer_time, 0)
    log.info('Bias disabled\n')
    # endregion


def global_bias_en(
        psu_rm: pv.Resource, buffer_time: float,
        on_off: int) -> None:
    """Enables/disables psu."""
    ut.safe_write(f'BIAS:ENable:SYStem {on_off}', buffer_time, psu_rm)


def _local_bias_en(
        psu_rm: pv.Resource,
        card_chnl: CardChnl,
        on_or_off: int, buffer_time: float) -> None:
    """Enables/disables passed channels on card."""
    card = card_chnl.card
    channels = card_chnl.chnl
    # region If channel is passed as int, no need to loop, directly set.
    if isinstance(channels, int):  # Need this because no enumerate(int)
        ut.safe_write(
            f"Bias:ENable:CArd{card}:CHannel{channels} {on_or_off}",
            buffer_time, psu_rm)
    # endregion

    # region If channel passed as list of channels, loop through and en.
    else:
        for i, _ in enumerate(channels):
            ut.safe_write(
                f"Bias:ENable:CArd{card}:CHannel {channels[i]} {on_or_off}",
                buffer_time, psu_rm)
    # endregion


def _get_psu_v(
        psu_rm: pv.Resource, g_or_d: str, card_chnl: CardChnl,
        buffer_time: float, set_or_meas: str) -> float:
    """Returns gate(g) or drain(d) voltage from psu bias supply."""

    # region Get the measured value from the psu.
    if set_or_meas == 'meas':
        psu_v = ut.safe_query(
            f"Bias:Measure:V{g_or_d}:CArd{card_chnl.card}:"
            f"CHannel{card_chnl.chnl}?",
            buffer_time, psu_rm, 'psu', True)
    # endregion

    # region Get the set value from the psu.
    elif set_or_meas == 'set':
        psu_v = ut.safe_query(
            f"Bias:Set:V{g_or_d}:CArd{card_chnl.card}:"
            f"CHannel{card_chnl.chnl}?",
            buffer_time, psu_rm, 'psu', True)
    # endregion

    # region Catch invalid state.
    else:
        raise Exception(
            'Invalid argument for set_or_meas, either "set" or "meas".')
    # endregion

    # region Return the measured or set voltage.
    return psu_v
    # endregion


def _get_psu_d_i(
        psu_rm: pv.Resource, card_chnl: CardChnl,
        buffer_time: float) -> float:
    """Returns current measured by psu bias supply in mA"""
    # region Get and return the measured current from the psu.
    psu_d_i = ut.safe_query(
        f"Bias:Measure:Id:CArd{card_chnl.card}:CHannel{card_chnl.chnl}?",
        buffer_time, psu_rm, 'psu', True)
    return 1000 * psu_d_i
    # endregion


def _set_psu_v(
        psu_rm: pv.Resource, card_chnl: CardChnl,
        buffer_time: float, g_or_d_v_target: GOrDVTarget) -> None:
    """ Called by _safe_set_v. Sets gate/drain psu voltage directly."""
    # region Write the given voltage to the psu.
    ut.safe_write(f"BIAS:SET:V{g_or_d_v_target.g_or_d}:"
                  f"CA{card_chnl.card}:CH{card_chnl.chnl} "
                  f"{g_or_d_v_target.v_target}",
                  buffer_time, psu_rm)
    # endregion


def direct_set_stage(
        psu_rm: pv.Resource, card_chnl: CardChnl,
        psu_lims: ic.PSULimits, buffer_time: float,
        g_d_vs: list[GOrDVTarget]) -> None:
    """Set a stage by requesting gate and drain targets directly."""
    # region Safely set drain and then gate voltages using _safe_set_v.
    g_v_target = g_d_vs[0]
    d_v_target = g_d_vs[1]
    _safe_set_v(psu_rm, card_chnl, d_v_target, psu_lims, buffer_time)
    _safe_set_v(psu_rm, card_chnl, g_v_target, psu_lims, buffer_time)
    # endregion


def _get_step_to_target(v_set_diff: float, v_step_lim: float,
                        v_set_status: float) -> float:
    """Return direction and amount to step voltage towards target."""
    # region Return direction and amount to step voltage towards target.
    # region Define boolean conditions for readability.
    pos_targ_oof_rng = bool(
        abs(v_set_diff) > v_step_lim and v_set_diff > 0)
    pos_targ_in_rng = bool(
        abs(v_set_diff) <= v_step_lim and v_set_diff > 0)
    neg_targ_oof_rng = bool(
        abs(v_set_diff) > v_step_lim and v_set_diff < 0)
    neg_targ_in_rng = bool(
        abs(v_set_diff) <= v_step_lim and v_set_diff < 0)
    # endregion

    # region Depending on above, figure out step towards target.
    if pos_targ_oof_rng:
        v_set_status += v_step_lim
    elif pos_targ_in_rng:
        v_set_status += abs(v_set_diff)
    elif neg_targ_oof_rng:
        v_set_status -= v_step_lim
    elif neg_targ_in_rng:
        v_set_status -= abs(v_set_diff)
    # endregion

    return v_set_status
    # endregion


def _safe_set_v(
        psu_rm: pv.Resource, card_chnl: CardChnl,
        g_or_d_v_target: GOrDVTarget, psu_lims: ic.PSULimits,
        buffer_time: float) -> Union[float, str]:
    """ Safely sets psu to a target voltage in increments.

    When called will set either gate or drain voltage to specified
    target incrementally so that electric fields from transient voltage
    changes cannot damage the LNA internally.  Checks/handles the drain
    current against the limit.  The drain current corresponding to the
    target voltage is returned. If limit gets tripped the voltage
    reverts to where it was before, measures the current, and then
    either returns where it is or if the current is still higher than
    the limit, will throw an exception and turn everything off.

    Args:
        psu_rm: Resource manager for the psu.
        card_chnl: The card and channel to set to.
        g_or_d_v_target: 'g' or 'd' whether a gate or drain voltage is
            being set.
        psu_lims: The power supply drain current and voltage step lims.
        buffer_time: This is how long to wait in seconds after each
            command which is sent to any of the instruments.

    Returns:
        The measured drain current in mA or a notifier that the current
        is over limit.

    Raises:
        Current limit exception.  When this exception is raised, the
        current was found to be too high, then the voltage was backed
        off and the current was still too high.  Power supply is turned
        off globally to prevent further damage at this point.
    """
    # region Unpack objects and set up logging.
    log = logging.getLogger(__name__)
    v_target = g_or_d_v_target.v_target
    g_or_d = g_or_d_v_target.g_or_d
    d_i_lim = psu_lims.d_i_lim
    v_step_lim = psu_lims.v_step_lim
    # endregion

    # region Enable channel to be set.
    _local_bias_en(psu_rm, card_chnl, 1, buffer_time)
    # endregion

    # region Get gate or drain string from g_or_d for logging.
    if g_or_d == 'g':
        gate_or_drain = 'gate'
    elif g_or_d == 'd':
        gate_or_drain = 'drain'
    else:
        raise Exception('Invalid g_or_d argument.')
    # endregion

    log.cdebug(f'\nSTARTED SAFE VOLTAGE SET: '
               f'Chain {card_chnl.card} Channel {card_chnl.chnl} - '
               f'TARGET {gate_or_drain} V: {v_target:+.3f}V')

    # region Get initial measured current and measured/set voltages.
    v_meas_status = _get_psu_v(psu_rm, g_or_d, card_chnl, buffer_time, 'meas')
    v_set_status = _get_psu_v(psu_rm, g_or_d, card_chnl, buffer_time, 'set')
    d_i_status = _get_psu_d_i(psu_rm, card_chnl, buffer_time)
    v_set_initial = v_set_status
    # endregion

    # region Get diff between set & target, also set & meas voltages.
    v_set_diff = v_target - v_set_status
    set_meas_diff = v_set_status - v_meas_status
    # endregion

    # region Step to target voltage checking current for safety.
    while abs(v_set_diff) > 0.001:

        # region Check set and measured voltages are similar.
        # This can catch connection and LNA errors before going further.
        # Relevant on gate only as drain will have loss on cables so a
        # difference should be expected.
        if set_meas_diff > 0.015 and g_or_d == 'g':
            raise Exception(
                'Large difference between measured and set gate voltage.')
        # endregion

        # region Find the next voltage target.
        v_set_status = _get_step_to_target(
            v_set_diff, v_step_lim, v_set_status)
        # endregion

        # region Send command to set voltage to new voltage status.
        _set_psu_v(
            psu_rm, card_chnl, buffer_time, GOrDVTarget(g_or_d, v_set_status))
        sleep(1)
        # endregion

        # region Remeasure set/meas status and recalculate differences.
        v_set_status = _get_psu_v(
            psu_rm, g_or_d, card_chnl, buffer_time, 'set')
        v_meas_status = _get_psu_v(
            psu_rm, g_or_d, card_chnl, buffer_time, 'meas')
        set_meas_diff = v_set_status - v_meas_status
        v_set_diff = v_target - v_set_status
        # endregion

        # region Check the drain current with the new voltage status.
        d_i_status = _get_psu_d_i(psu_rm, card_chnl, buffer_time)
        v_psu_diff = v_set_status - v_meas_status
        # Notify if psu set and measured voltages are different and
        # provide more detail.  Otherwise, provide more concise detail.
        if abs(v_psu_diff) > 0.1:
            log.warning(
                'Significant difference between set and measured V detected:')
            log.info(f'Set {gate_or_drain} V: {v_set_status:+.3f}V    '
                     f'Measured: {v_meas_status:+.3f}V    '
                     f'Difference: {v_psu_diff:+.3f}V    '
                     f'Drain Current (Measured): {d_i_status:+.3f}mA')
        else:
            log.cdebug(f'Set {gate_or_drain} V to {v_set_status:+.3f}V    '
                       f'Target is {v_target:+.3f}V    '
                       f'Drain Current = {d_i_status:+.3f}mA')
        # endregion

        # region If drain current over limit back off to last voltage.
        d_i_over_limit = bool(d_i_status > d_i_lim)
        if d_i_over_limit:
            # region Set the voltage down a step and remeasure current.
            if g_or_d == 'g':
                _set_psu_v(psu_rm, card_chnl, buffer_time,
                           GOrDVTarget('g', v_set_initial))
            elif g_or_d == 'd' and v_set_status > v_step_lim:
                _set_psu_v(psu_rm, card_chnl, buffer_time,
                           GOrDVTarget('d', v_set_status - v_step_lim))
            elif g_or_d == 'd' and not v_set_status > v_step_lim:
                _set_psu_v(psu_rm, card_chnl, buffer_time,
                           GOrDVTarget(g_or_d, 0))

            d_i_status = _get_psu_d_i(psu_rm, card_chnl, buffer_time)
            # endregion

            # region If DI is still too high disable psu.
            d_i_status_still_over_limit = bool(d_i_status > d_i_lim)
            if d_i_status_still_over_limit:
                global_bias_en(psu_rm, buffer_time, 0)
                raise Exception(
                    'Drain current still too high even when backed off')
            return 'over limit'  # Notify problem by returning this.
            # endregion
        # endregion
    # endregion

    # region Report status in console and return drain current.
    log.cdebug(f'Set {gate_or_drain} V to {v_set_status:+.3f}V    '
               f'Drain Current = {d_i_status:+.3f}mA')
    log.cdebug('COMPLETED SAFE VOLTAGE SET\n')
    return d_i_status
    # endregion


def _adapt_search_stage(
        d_i_meas_arr: list, d_i_target: float, g_v_range: list,
        next_g_v_steps: int, final_stage: bool = False) -> tuple:
    """Handles a gate voltage / drain current adaptive search stage.

    For each gate voltage in the layer range set gate voltage, measure
    current, and calculate how far off the previous and present drain
    currents are to the target until you exceed the target current, then
    take the nearest to the target. If over current limit, ensure value
    used is upper lim and search is from low to high.

    Args:
        d_i_meas_arr: Array of previous current measurements from the
            level of the adaptive search the function is called in.
        d_i_target: The mA drain current limit for the measurement
            system.
        g_v_range: The range of gate voltages that are being tested,
            either the broad, middle, or narrow range.
        next_g_v_steps: Number of steps in the next layer range.
        final_stage: Is this the final stage? Will return the final gate
            voltage if so.

    Returns:
        The next layer gate voltage range or final value; the updated
        version of the measured current array; iterable of the next loop
        (either 1 if the target current has been exceeded, or 100 if
        not); whether the present drain current measured is closer to
        the target than the previous.
    """
    log = logging.getLogger(__name__)
    index = len(d_i_meas_arr) - 1
    is_g_v_range_increasing = bool(g_v_range[1] - g_v_range[0] > 0)

    # region Check if current now over limit and handle.
    # If current is over limit, make sure next search is low to high. Do
    # this by setting d_i to a really high number, forcing the condition
    # that the closest current to the target is the previous one because
    # the present is very high.
    if d_i_meas_arr[-1] == 'over limit':
        d_i_meas_arr[-1] = d_i_meas_arr[-2] + 100
    # endregion

    # region Figure out if the target current has been exceeded.
    prev_dist_from_targ = d_i_target - d_i_meas_arr[index - 1]
    pres_dist_from_targ = d_i_target - d_i_meas_arr[index]

    # If sign switches between previous and present measurement then
    # value of gate for drain is within this step.
    target_current_exceeded = bool(
        np.sign(prev_dist_from_targ) != np.sign(pres_dist_from_targ))
    # endregion

    # region Handle exceeded target current conditions.
    if target_current_exceeded and not final_stage:
        # region Increment level
        # If the previous distance to target is larger than the present
        # the direction of the next sweep is from present to previous <-
        # instead of previous to present ->.
        pres_closer_than_prev = bool(
            abs(prev_dist_from_targ) > abs(pres_dist_from_targ))

        log.cdebug('\nENTERING NEXT ADAPTIVE SEARCH LAYER')

        # region Setup next layer range
        if is_g_v_range_increasing:
            next_g_v_low_lim = g_v_range[index - 1]
            next_g_v_up_lim = g_v_range[index]
        else:
            next_g_v_low_lim = g_v_range[index]
            next_g_v_up_lim = g_v_range[index - 1]
        next_g_v_rng = np.linspace(
            next_g_v_low_lim, next_g_v_up_lim, next_g_v_steps)

        # If present closer than previous reverse list and use present
        # current measurement for first value in mid-current measurement
        # array, otherwise make previous broad current measurement as
        # first in mid-current measurement array.
        next_d_i_meas = []
        if pres_closer_than_prev and is_g_v_range_increasing:
            next_g_v_rng = next_g_v_rng[::-1]
            next_d_i_meas.append(d_i_meas_arr[index])

        elif not pres_closer_than_prev and is_g_v_range_increasing:
            next_d_i_meas.append(d_i_meas_arr[index - 1])

        elif pres_closer_than_prev and not is_g_v_range_increasing:
            next_d_i_meas.append(d_i_meas_arr[index])

        elif not pres_closer_than_prev and not is_g_v_range_increasing:
            next_g_v_rng = next_g_v_rng[::-1]
            next_d_i_meas.append(d_i_meas_arr[index - 1])
        # endregion

        return next_g_v_rng, next_d_i_meas, 1, \
            pres_closer_than_prev, d_i_meas_arr

    if target_current_exceeded and final_stage:
        # region Report and return closest to target gate voltage
        # region If present closest, return present gate voltage.
        if abs(prev_dist_from_targ) > abs(pres_dist_from_targ):
            log.info(f'COMPLETED STAGE SET\n'
                     f'TARGET: {d_i_target:.3f} mA - '
                     f'ACHIEVED: {d_i_meas_arr[index]:.3f} mA - '
                     f'GATE VOLTAGE: {g_v_range[index]:.3f} V.\n')
            return g_v_range[index], None, None, True, d_i_meas_arr
        # endregion

        # region If previous closest, return previous gate voltage.
        log.info(f'COMPLETED STAGE SET\n'
                 f'TARGET DI: {d_i_target:.3f} mA - '
                 f'ACHIEVED DI: {d_i_meas_arr[index - 1]:.3f} mA - '
                 f'GATE VOLTAGE: {g_v_range[index - 1]:.3f} V.\n')
        return g_v_range[index - 1], None, None, False, d_i_meas_arr
        # endregion
        # endregion
    # endregion

    # region If current target hasn't been exceeded then keep searching.
    # Return 100 so the next layer while loop doesn't start.
    return None, None, 100, None, d_i_meas_arr
    # endregion


def _safe_set_stage(
        psu_rm: pv.Resource, stage_bias: lc.StageBiasSet,
        psu_set: ic.BiasPSUSettings, card_chnl: CardChnl,
        psu_lims: ic.PSULimits) -> float:
    """Find required gate V for requested drain I at given drain V.

    Employs an adaptive search in order to find a gate voltage which
    results in a measured drain current which is as close to the
    requested drain current as possible. Three layers of search, broad,
    middle, and narrow are used.

    The broad range is stepped through from low to high, when the
    target current is exceeded, the gate voltage, and the one before it
    are used as new range limits for the middle level. The same thing
    happens for the middle level, resulting in a range for the narrow
    level. The narrow range is stepped through, and the closest drain
    current found for a gate voltage is determined, this is returned.

    Args:
        psu_rm: Resource manager for the biasing power supply.
        stage_bias: Bias variables (target voltages and currents).
        psu_set: Psu settings, contains the current limit, gate voltage
            limits, and wide/narrow voltage step sizes.
        card_chnl: The card and channel of the power supply to set.
        psu_lims: Drain current and voltage step psu limits.

    Returns:
        Gate voltage which with the specified drain voltage will produce
        the specified drain current.
    """

    # region Set up logging, instantiate arrays, and unpack objects.
    log = logging.getLogger(__name__)
    brd_d_i_meas = []
    buffer_time = psu_set.buffer_time
    d_i_target = stage_bias.d_i
    # endregion

    # region Set psu to target drain voltage accounting for wire v drop.
    _safe_set_v(
        psu_rm, card_chnl, GOrDVTarget('d', stage_bias.d_v_at_psu),
        psu_lims, buffer_time)
    # endregion

    # region Report drain current status.
    log.cdebug(f'\nSTARTED STAGE SET '
               f'- PSU CARD {card_chnl.card} CHANNEL {card_chnl.chnl} '
               f'- TARGET DRAIN CURRENT: {d_i_target:+.3f}mA')
    # endregion

    # region Get initial d_i measurement from first g_v setting.
    brd_g_v_range = psu_set.g_v_brd_range
    init_brd_d_i = _safe_set_v(
        psu_rm, card_chnl, GOrDVTarget('g', brd_g_v_range[0]),
        psu_lims, buffer_time)

    if init_brd_d_i == 'over limit':
        return brd_g_v_range[0]  # Return the lowest g_v possible.

    brd_d_i_meas.append(init_brd_d_i)
    # endregion

    # region Define inner current measurement function.
    def _mid_nrw_get_d_i(g_v_range: list[float], outer_d_i_meas: list[float],
                         index: int, outer_index: int) -> float:
        """Measures the current based on passed inner loop values."""
        # region Measure unmeasured currents.
        if index < len(g_v_range) - 1:
            # No _safe_set_v as know upper bound within current limit.
            _set_psu_v(psu_rm, card_chnl, buffer_time,
                       GOrDVTarget('g', g_v_range[index]))
            sleep(buffer_time)
            _d_i = _get_psu_d_i(psu_rm, card_chnl, buffer_time)
            log.cdebug(f'GV = {g_v_range[index]:+.3f}    DI = {_d_i:+.3f}')
        # endregion

        # region Final val of g_v_range already measured in outer loop.
        elif pres_closer_than_prev:
            _d_i = outer_d_i_meas[outer_index - 1]
        elif not pres_closer_than_prev:
            _d_i = outer_d_i_meas[outer_index]
        else:
            raise Exception('')
        # endregion
        return _d_i
    # endregion

    # region Employ adaptive search algorithm to get gate voltage.
    i = 1
    while i < psu_set.num_of_g_v_brd_steps + 1:
        # region Broad level.
        # region Get iteration current.
        d_i = _safe_set_v(
            psu_rm, card_chnl, GOrDVTarget('g', brd_g_v_range[i]), psu_lims,
            buffer_time)
        brd_d_i_meas.append(d_i)
        # endregion

        # region Process measured current to make adaptive search decisions.
        brd_lvl = _adapt_search_stage(brd_d_i_meas, d_i_target, brd_g_v_range,
                                      psu_set.num_of_g_v_mid_steps)
        # endregion

        # region Save variables from proc function returned tuple.
        mid_g_v_range = brd_lvl[0]  # Returned array if pres d_i > targ.
        mid_d_i_meas = brd_lvl[1]  # Returned array if pres d_i > targ.
        j = brd_lvl[2]  # If current exceeded this is 1, else 100.
        pres_closer_than_prev = brd_lvl[3]  # None if pres d_i < targ.
        brd_d_i_meas = brd_lvl[4]  # Updates if measured d_i over limit.
        # endregion
        # endregion

        while j < psu_set.num_of_g_v_mid_steps + 1:
            # region Middle level.
            # region Get and append iteration current to meas list.
            mid_d_i_meas.append(
                _mid_nrw_get_d_i(mid_g_v_range, brd_d_i_meas, j, i))
            # endregion

            # region Check whether to enter next layer.
            mid_lvl = _adapt_search_stage(mid_d_i_meas, d_i_target,
                                          mid_g_v_range,
                                          psu_set.num_of_g_v_nrw_steps)
            # endregion

            # region Save variables from proc function returned tuple.
            nrw_g_v_range = mid_lvl[0]
            nrw_d_i_meas = mid_lvl[1]
            k = mid_lvl[2]  # If current exceeded this is 1, else 100.
            pres_closer_than_prev = mid_lvl[3]
            mid_d_i_meas = mid_lvl[4]
            # endregion
            # endregion

            while k < psu_set.num_of_g_v_nrw_steps + 1:
                # region Narrow level.
                # region Get and append iteration current to meas list.
                nrw_d_i_meas.append(
                    _mid_nrw_get_d_i(nrw_g_v_range, mid_d_i_meas, k, j))
                # endregion

                # region Get and return final gate voltage.
                g_v_final = _adapt_search_stage(nrw_d_i_meas, d_i_target,
                                                nrw_g_v_range, 0, True)

                if g_v_final[0] is not None:
                    if not g_v_final[3]:
                        _set_psu_v(psu_rm, card_chnl, buffer_time,
                                   GOrDVTarget('g', g_v_final[0]))
                    return g_v_final[0]
                # endregion
                # endregion

                k += 1
            j += 1
        i += 1
    # endregion

    # region Handle condition if exit is never tripped.
    # If exit condition never tripped then the highest gate value in the
    # range must be the best we can do, so return that.
    return max(brd_g_v_range)
    # endregion


def bias_set(
        psu_rm: pv.Resource, target_lna_bias: lc.LNABiasSet,
        psu_set: ic.BiasPSUSettings, buffer_time: float) -> None:
    """Set the psu to the LNA bias values requested.

    When called will set each relevant channel of the psu to the
    required voltages/currents, and set the gate voltages required to
    get the requested current into the passed lna stage instances.

    Args:
        psu_rm: Resource manager for the psu.
        target_lna_bias: The target set of LNA biasing settings.
        psu_set: Psu settings, contains the current limit, gate voltage
            limits, and wide/narrow voltage step sizes.
        buffer_time: This is how long to wait in seconds after each
            command which is sent to any of the instruments.
    """

    # region Get power supply limits.
    psu_stg_1_lims = ic.PSULimits(
         psu_set.v_step_lim, target_lna_bias.stage_1.d_i_lim)
    if hasattr(target_lna_bias, 'stage_2'):
        if target_lna_bias.stage_2 is not None:
            psu_stg_2_lims = ic.PSULimits(
                psu_set.v_step_lim, target_lna_bias.stage_2.d_i_lim)
    if hasattr(target_lna_bias, 'stage_3'):
        if target_lna_bias.stage_3 is not None:
            psu_stg_3_lims = ic.PSULimits(
                psu_set.v_step_lim, target_lna_bias.stage_3.d_i_lim)
    # endregion

    # region Set each stage to target drain voltage and current
    # Return and store gate voltage
    _local_bias_en(psu_rm, target_lna_bias.stage_1.card_chnl, 1, buffer_time)
    target_lna_bias.stage_1.g_v = _safe_set_stage(
        psu_rm, target_lna_bias.stage_1, psu_set,
        target_lna_bias.stage_1.card_chnl, psu_stg_1_lims)

    if hasattr(target_lna_bias, 'stage_2'):
        if target_lna_bias.stage_2 is not None:
            _local_bias_en(psu_rm, target_lna_bias.stage_2.card_chnl, 1,
                           buffer_time)
            target_lna_bias.stage_2.g_v = _safe_set_stage(
                    psu_rm, target_lna_bias.stage_2, psu_set,
                    target_lna_bias.stage_2.card_chnl, psu_stg_2_lims)

    if hasattr(target_lna_bias, 'stage_3'):
        if target_lna_bias.stage_3 is not None:
            _local_bias_en(psu_rm, target_lna_bias.stage_3.card_chnl, 1,
                           buffer_time)
            target_lna_bias.stage_3.g_v = _safe_set_stage(
                    psu_rm, target_lna_bias.stage_3, psu_set,
                    target_lna_bias.stage_3.card_chnl, psu_stg_3_lims)
    # endregion
