import numpy as np
import time
import serial
import textwrap
from collections import namedtuple
import os
import os.path
import sys
import configparser 
import yaml
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.signal
import scipy.optimize
import scipy.constants
import scipy.special
import scipy.interpolate
import cPickle
import hickle

GUI = False
import config
import control
import support
import dpx_functions
import dpx_support
import system
import dpx_settings

class Dosepix(config.Config, control.Control, support.Support, dpx_functions.DPX_functions, dpx_support.DPX_support, system.System):
    # === Imports ===
    # from .config import *
    # from .support import *
    # from .dpx_support import *
    # from .control import *
    # from .system import *
    # from .dpx_functions import *

    # === FLAGS ===
    USE_GUI = False

    def __init__(self, portName, baudRate, configFn=None, bin_edges_file=None, params_file=None, thl_calib_files=None):
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

    def getSettingsGUI(self):
        serialPorts = getSerialPorts(self)

    def setSettingsGUI(self):
        return None, None, None

    def setGUI(self):
        self.USE_GUI = True

    def unsetGUI(self):
        self.USE_GUI = False

    def __del__(self):
        self.close()
        
    def close(self):
        # = Shut down =
        self.HVDeactivate()
        print 'Check if HV is deactivated...',
        for i in range(5):
            if not self.HVGetState():
                print 'done!'
                break
            else:
                self.HVDeactivate()
        else:
            assert 'HV could not be deactivated'

        print 'Measurement finished.'

        self.ser.close()
        
