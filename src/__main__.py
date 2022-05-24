# -*- coding: utf-8 -*-
"""
  _____ _______     ______  __  __ ______ 
 / ____|  __ \ \   / / __ \|  \/  |  ____|
| |    | |__) \ \_/ / |  | | \  / | |__   
| |    |  _  / \   /| |  | | |\/| |  __|  
| |____| | \ \  | | | |__| | |  | | |____ 
 \_____|_|  \_\ |_|  \____/|_|  |_|______|
                                          
CryoMe Automated Y Factor Noise Temperature Measurement Package.

Version 1.0

For more information on the package as a whole, see the doc folder in
the code directory. Settings are in config.yml, edit there to control
operation of this program.

Use of __future__ import to avoid problems with type hinting and
circular importing mean that this code will only work with Python 3.7
or higher.

Priority:Speed - High to low:Fast to slow (1 to 3)

ToDo:
    * ?:2 Algorithm to do all stage sweep.
    * 1:3 Finish metaproc.py.
    * 3:1 Biasing current limit to gate voltage speed optimisation.
    * 3:2 Re-lint code.
    * 3:1 Break up plotting in output_saving.py as in replot.py.
    * 3:1.5 Exception subclassing.
    * 1:1 Merge changes on lab computer (by 10/05/2022).
    * 2:2 Overall timing outputs in results log (save columns first).
    * 3:2 Analysis anomaly handling.
    * 1:1 Comment in config file.

Questions:
    * When running multiple config file, the text on the progress screen shows 
    two of every print out on the second set of measurements
    * Move switch to after safe bias check

    *Added command to turn the heaters on the Lakeshore off after the AT measurements, 
    Meas_algorithms line 323.  Also added 'included utils' at the top of the file. 

"""

# region Import modules
from __future__ import annotations
import logging
import os
import pathlib
import sys

import yaml

import config_handling
import session
# endregion


def main(config_index: int):
    """Main for CryoMe."""

    # region Print starting menu.
    print(__doc__)
    #input('Press enter to start a measurement...')
    # endregion

    # region Fix system path to make it consistent wherever linked from.
    os.chdir(os.path.dirname(os.path.dirname(sys.argv[0])))
    # endregion

    # region Set up logging.
    # region Set logging format.
    stream_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        '%m-%d %H:%M:%S')
    file_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s')
    # endregion

    # region Make lvl between debug and info for internal module debug.
    logging.addLevelName(15, "CDEBUG")

    def cdebug(self, message, *args, **kws):
        """Custom logging level for internal debugging."""
        # Access to internal class deemed necessary for creation of
        # additional custom logging level, as otherwise the logging
        # module always is referred to as root.
        self._log(15, message, args, **kws)  # pylint: disable=W0212
    logging.Logger.cdebug = cdebug
    # endregion

    # region Log handler for console output.
    logstream = logging.StreamHandler()
    logstream.setLevel(logging.INFO)
    logstream.setFormatter(stream_format)
    # endregion

    # region Create log to add handlers to.
    log = logging.getLogger()
    log.addHandler(logstream)
    log.setLevel(logging.DEBUG)
    # endregion
    # endregion

    # region Load in settings yaml file.
    with open(pathlib.Path(str(os.getcwd()) + f'\\config{config_index}.yml'),
              encoding='utf-8') as _f:
        yaml_config = yaml.safe_load(_f)
    # endregion

    # region Measure each chain as requested.
    for i, cryo_chain in enumerate(
            yaml_config['bias_sweep_settings']['chain_sequence']):

        #input(f'Please ensure chain {cryo_chain} is connected to the PSU, '
        #      f'then press enter.')

        # region Reset all class instances.
        if i > 0:
            del settings
        # endregion

        # region Set up measurement settings.
        settings = config_handling.settings_config(yaml_config, cryo_chain)
        # endregion

        # region Configure session ID and log file writer.
        settings.meas_settings.config_session_id(settings.file_struc)
        file_handler = logging.FileHandler(
            settings.file_struc.get_log_path(
                settings.meas_settings.session_id), encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_format)
        log.addHandler(file_handler)
        # endregion

        # region Check input files are closed.
        settings.file_struc.check_files_closed()
        # endregion

        # region Trigger measurement
        session.start_session(settings)
        #try:
        #    session.start_session(settings)
        ## Pylint broad-except disabled as sys.exc_info()[0] logged.
        #except Exception as _e:  # pylint: disable=broad-except
        #    log.exception(sys.exc_info()[0])
        #    input(f'Error {_e} logged, press Enter to exit...')
        # endregion

    input('Press Enter to exit...')

    # endregion


if __name__ == '__main__':

    i=1
    while i < 5:
        main(i)
        i+=1
