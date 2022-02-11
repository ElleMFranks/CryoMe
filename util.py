# -*- coding: utf-8 -*-
"""util.py - Cryome miscellaneous utility functions.

Two methods which can be used to safely send a query or write command
via SCPI command to an instrument are provided.  The commands are sent
through a pyvisa Resource Manager, or a socket_communication Instrument
Socket.  One method is provided which prompts the user to accept a
variable if it is seen as potentially abnormal.

The query function in this module takes an SCPI command, tries to send
it, if it fails it will retry 10 times, and then try instrument specific
protocols five  more times until failing. This is done due to avoid some
observed intermittent faults during measurements which were non-serious
and solved by sending the same command again. After successfully sending
the query, the script pauses for the buffering time, allowing time for
the instruments to respond.

The write function simply sends the SCPI command, and pauses for the
buffering time.
"""

# region Import modules.
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional
import time

import pyvisa as pv

import instr_classes as ic
# endregion


def get_dataclass_args(datcls: dataclass):
    """Returns variables to set dataclass instance to subclass."""
    return [datcls.__getattribute__(att) for att in datcls.__dict__.keys()]


def yes_no(variable_name: str, variable_val: Any, variable_unit: str) -> None:
    """Prompts a y/n response to check if user wants to continue."""
    while True:
        _yn = input(f"{variable_name} is {variable_val} {variable_unit}, "
                    "possible incorrect value detected, "
                    "do you wish to proceed? (Enter y/n)").lower()
        if _yn == "n":
            raise Exception('Invalid variable value.')
        if _yn == 'y':
            return


def safe_query(
        command: str, buffer_time: float, res_manager: pv.Resource,
        instr: str, float_req: bool = False, str_req: bool = False,
        instr_settings: Optional[ic.InstrumentSettings] = None) -> Any:
    """Try query, if failed wait 5s and try again up to 10 times.

    Tries query multiple times, if spectrum analyser is being queried
    and multiple attempts to do this fail, spec an gets reset and tries
    again. Only need to send equipment settings with a spec an query.
    """
    # region Keep trying query until it works or too many tries.
    i = 0
    while True:
        try:
            res_manager.read_termination = '\n'
            output = res_manager.query(command)
            if float_req:
                output = float(output)
            elif str_req:
                output = str(output)
            time.sleep(buffer_time)
            return output
        except Exception as _e:
            print('Error')
            time.sleep(5)
            # region Handle retries for each instrument.
            i += 1
            if i > 10:
                if instr == 'spec an':
                    # Reset spec an and try again
                    if instr_settings is not None:
                        res_manager.write(':CONFigure:CHPower')
                        instr_settings.sig_an_settings.spec_an_init(
                            res_manager, buffer_time)
                    if i > 15:
                        raise Exception('Spec An Error') from _e

                if instr == 'lakeshore':
                    raise Exception('Lakeshore Error.') from _e

                if instr == 'vna':
                    raise Exception('VNA Error.') from _e

                if instr == 'sig gen':
                    raise Exception('Sig Gen Error.') from _e

                if instr == 'psx':
                    raise Exception('PSX Error.') from _e

                raise Exception('') from _e

            # endregion
            continue
    # endregion


def safe_write(command: str, buffer_time: float,
               res_manager: pv.Resource) -> None:
    """Writes a command to an instrument and waits the buffer time."""
    # region Write command and then sleep for buffer time.
    res_manager.write(command)
    time.sleep(buffer_time)
    # endregion
