#!/usr/bin/env python
import dpx_func_python
import hickle

# Important files
THL_CALIB_FILES = None # ['THLCalibration/THLCalib_%d.p' % slot for slot in [22, 6, 109]]
BIN_EDGES_FILE = None # 'Dennis1_binEdges.hck'

PORT = '/dev/ttyUSB0'
CONFIG_DIR = 'config/'
CONFIG_FN = 'DPXConfig.conf'
PARAMS_FILE = CONFIG_DIR + 'paramsDict_22_6_109_Ikrum.hck'
BIN_EDGES = CONFIG_DIR + 'binEdgesUniform_DPX22_6_109_Ikrum_10_20.hck'

CHIP_NUMS = [22, 6, 109]
IKRUM = [5, 10, 20] 

def main():
    if PARAMS_FILE.endswith('.p'):
        paramsDict = cPickle.load(open(PARAMS_FILE, 'rb'))
    else:
        paramsDict = hickle.load(PARAMS_FILE)

    bin_edges = hickle.load(BIN_EDGES)

    dpx = dpx_func_python.Dosepix(PORT, 2e6, CONFIG_DIR + CONFIG_FN, bin_edges_file=bin_edges, params_file=PARAMS_FILE, thl_calib_files=THL_CALIB_FILES)

    # Change Ikrum values
    for chip_idx in range(3):
        d = dpx.splitPerihperyDACs(dpx.peripherys + dpx.THLs[chip_idx], perc=False)
        d['I_krum'] = IKRUM[chip_idx]
        code = dpx.periheryDACsDictToCode(d, perc=False)
        dpx.peripherys = code[:-4]
        dpx.DPXWritePeripheryDACCommand(chip_idx + 1, code)
        print dpx.DPXReadPeripheryDACCommand(chip_idx + 1)
        print dpx.DPXReadOMRCommand(chip_idx + 1)

    for slot in range(1, 4):
        print dpx.DPXReadPeripheryDACCommand(slot)

    # dpx.measureDose(slot=[1, 2, 3], measurement_time=0, freq=False, frames=100000, logTemp=False, intPlot=False, conversion_factors=None)
    dpx.measureDoseEnergyShift(slot=[1, 2, 3], measurement_time=0, freq=False, frames=10000, logTemp=False, intPlot=False)
    dpx.close()

if __name__ == '__main__':
    main()

