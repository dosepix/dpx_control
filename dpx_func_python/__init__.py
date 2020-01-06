from __future__ import print_function

GUI = False
# === Imports ===
import dpx_func_python.config
import dpx_func_python.control
import dpx_func_python.support
import dpx_func_python.dpx_functions
import dpx_func_python.dpx_support
import dpx_func_python.system
import dpx_func_python.dpx_settings
import dpx_func_python.dpx_test_pulse

class Dosepix(config.Config, control.Control, support.Support, dpx_functions.DPX_functions, dpx_support.DPX_support, system.System, dpx_test_pulse.DPX_test_pulse):
    # === FLAGS ===
    USE_GUI = False

    def __init__(self, portName, baudRate, configFn=None, thl_calib_files=None, params_file=None, bin_edges_file=None):
        """
        Creates instance of :class:`Dosepix`

        :param portName: Name of the port
        :type name: str
        """
        if params_file is None:
            self.bin_edges_file = bin_edges_file
        else:
            self.bin_edges = bin_edges_file

        self.params_file = params_file
        self.thl_calib_files = thl_calib_files
        self.portName = portName
        self.baudRate = baudRate
        self.configFn = configFn

        # THL Calibration
        self.voltCalib, self.THLCalib = [], []
        self.THLEdgesLow, self.THLEdgesHigh = [], []
        self.THLFitParams = []
        self.THLEdges = []
        
        if GUI:
            self.getSettingsGUI()
            self.portName, self.baudRate, self.configFn = setSettingsGUI()
        else:
            self.unsetGUI()

        self.getConfigDPX(self.portName, self.baudRate, self.configFn)

    def __del__(self):
        self.close()
        
    def getSettingsGUI(self):
        serialPorts = getSerialPorts(self)

    def setSettingsGUI(self):
        return None, None, None

    def setGUI(self):
        self.USE_GUI = True

    def unsetGUI(self):
        self.USE_GUI = False

    def close(self):
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
        
