from __future__ import print_function
from dpx_control.dpx_test_pulse import DPXtestPulse
import dpx_control.dpx_settings
from dpx_control.system import System
from dpx_control.dpx_support import DPXsupport
from dpx_control.dpx_functions import DPXfunctions
from dpx_control.support import Support
from dpx_control.control import Control
from dpx_control.config import Config
import serial

GUI = False
class Dosepix(
        support.Support,
        dpx_functions.DPX_functions,
        dpx_support.DPXsupport,
        system.System,
        dpx_test_pulse.DPX_test_pulse):
    # === FLAGS ===
    USE_GUI = False

    def __init__(
            self,
            port_name,
            baud_rate=2e6,
            config_fn=None,
            thl_calib_files=None,
            params_file=None,
            bin_edges_file=None,
            Ikrum=None,
            eye_lens=False):

        self.port_name = port_name
        if self.port_name is None:
            # Call class without arguments to get a dummy instance.
            return

        self.bin_edges_file = bin_edges_file
        '''
        if params_file is None:
            self.bin_edges_file = bin_edges_file
            self.bin_edges = None
        else:
            self.bin_edges = bin_edges_file
            self.bin_edges_file = None
        '''

        self.params_file = params_file
        self.thl_calib_files = thl_calib_files
        self.port_name = port_name
        self.baud_rate = baud_rate
        self.config_fn = config_fn

        # Is eye lens dosimetry hardware used?
        self.eye_lens = eye_lens

        if GUI:
            self.getSettingsGUI()
            self.port_name, self.baud_rate, self.config_fn = self.setSettingsGUI()
        else:
            self.unset_GUI()

        # Instantiate classes
        self.config = Config(port_name, baud_rate, config_fn, eye_lens, Ikrum)
        self.control = Control()
        self.support = Support()
        self.dpx_functions = DPXfunctions()
        self.dpx_support = DPXsupport()
        self.system = System()
        self.dpx_test_pulse = DPXtestPulse()

        # Establish connection
        self.ser = serial.Serial(self.port_name, self.baud_rate)
        assert self.ser.is_open, 'Error: Could not establish serial connection!'

        # THL Calibration
        edges = self.config.get_config_DPX(self.thl_calib_files)
        self.THL_edges_low, self.THL_edges_high, self.THL_edges, self.THL_fit_params = edges

        # Initialize DPX and its connection
        self.params_dict, bin_edges = self.control.init_DPX(
            self.config,
            self.params_file,
            bin_edges_file,
            self.eye_lens)
        self.config.bin_edges = bin_edges

    def __del__(self):
        self.close()

    def set_GUI(self):
        self.USE_GUI = True

    def unset_GUI(self):
        self.USE_GUI = False

    def close(self):
        if self.port_name is None:
            return

        if not self.eye_lens:
            # = Shut down =
            self.control.HV_deactivate()
            print('Check if HV is deactivated...'),
            for i in range(5):
                if not self.control.HV_get_state():
                    print('done!')
                    break
                self.control.HV_deactivate()
            else:
                assert False, 'HV could not be deactivated'

        print('Measurement finished.')
        self.ser.close()
