from __future__ import print_function

GUI = False
# === Imports ===
import dpx_control.config
import dpx_control.control
import dpx_control.support
import dpx_control.dpx_functions
import dpx_control.dpx_support
import dpx_control.system
import dpx_control.dpx_settings
import dpx_control.dpx_test_pulse

class Dosepix(config.Config, control.Control, support.Support, dpx_functions.DPX_functions, dpx_support.DPX_support, system.System, dpx_test_pulse.DPX_test_pulse):
    # === FLAGS ===
    USE_GUI = False

    def __init__(self, portName, baudRate=2e6, configFn=None, thl_calib_files=None, params_file=None, bin_edges_file=None, Ikrum=None, eye_lens=False):
        self.portName = portName
        if self.portName is None:
            # Call class without arguments to get a dummy instance. 
            return

        """
        Creates instance of :class:`Dosepix`

        :param portName: Name of the port
        :type name: str
        """

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
        self.portName = portName
        self.baudRate = baudRate
        self.configFn = configFn

        # THL Calibration
        self.voltCalib, self.THLCalib = [], []
        self.THLEdgesLow, self.THLEdgesHigh = [], []
        self.THLFitParams = []
        self.THLEdges = [] * 3

        # Is eye lens dosimetry hardware used?
        self.eye_lens = eye_lens

        # Eye lens board only uses slot1
        if self.eye_lens:
            self.slot_range = [1]
        else:
            if configFn is None:
                self.slot_range = [1, 2, 3]
            else:
                self.slot_range = [idx for idx in range(1, len(self.configFn) + 1) if self.configFn[idx-1] is not None]

        print('Slots')
        print(self.slot_range)

        if GUI:
            self.getSettingsGUI()
            self.portName, self.baudRate, self.configFn = self.setSettingsGUI()
        else:
            self.unsetGUI()

        self.getConfigDPX(self.portName, self.baudRate, self.configFn, Ikrum)

    def __del__(self):
        self.close()

    def getSettingsGUI(self):
        serialPorts = self.getSerialPorts(self)

    def setSettingsGUI(self):
        return None, None, None

    def setGUI(self):
        self.USE_GUI = True

    def unsetGUI(self):
        self.USE_GUI = False

    def close(self):
        if self.portName is None:
            return

        if not self.eye_lens:
            # = Shut down =
            self.HVDeactivate()
            print('Check if HV is deactivated...'),
            for i in range(5):
                if not self.HVGetState():
                    print('done!')
                    break
                else:
                    self.HVDeactivate()
            else:
                assert 'HV could not be deactivated'

        print('Measurement finished.')

        self.ser.close()
