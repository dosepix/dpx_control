#!/usr/bin/env python
import dpx_func_python
import numpy as np

PORT = '/dev/ttyUSB0'
CONFIG_DIR = 'config/'
CHIP_NUMS = [22, 6, 109]
IKRUM = 20
CONFIG_FN = [CONFIG_DIR + '%d/DPXConfig_%d_Ikrum%d.conf' % (CHIP, CHIP, IKRUM) if CHIP is not None else None for CHIP in CHIP_NUMS]
PARAMS_FILES = ['calibration_parameters/params_%d_Ikrum%d.json' % (CHIP, IKRUM) if CHIP is not None else None for CHIP in CHIP_NUMS]
THL_CALIB_FILES = [CONFIG_DIR + '%d/THLCalib_%d.json' % (CHIP, CHIP) if CHIP is not None else None for CHIP in CHIP_NUMS] 
BIN_EDGES_FILES = None

def main():
    # Establish connection
    dpx = dpx_func_python.Dosepix(PORT, 2e6, CONFIG_FN, thl_calib_files=THL_CALIB_FILES, 
            params_file=PARAMS_FILES, bin_edges_file=BIN_EDGES_FILES)

    peripherys = []
    for chip_idx in range(3):
        d = dpx.splitPerihperyDACs(dpx.peripherys[chip_idx] + dpx.THLs[chip_idx], perc=False)
        code = dpx.periheryDACsDictToCode(d, perc=False)
        peripherys.append( code[:-4] )

    THL0 = np.array(dpx.THLs, copy=True)
    for THLshift in range(-50, 11, 1):
        print(THLshift)
        for slot in range(3):
            THL = int(THL0[slot], 16)
            THL_idx = list(dpx.THLEdges[slot]).index(THL)
            THL = '%04x' % int(dpx.THLEdges[slot][THL_idx + THLshift])
            dpx.DPXWritePeripheryDACCommand(slot + 1, peripherys[slot] + THL)
        # Measure ToT
        dpx.measureToT(slot=[1, 2, 3], intPlot=False, cnt=10000, storeEmpty=False, logTemp=True, meas_time=60 * 5,
                       outDir='ToTMeasurement_THLShift_%d_det_22_6_109/' % THLshift)

    # Close connection
    dpx.close()

if __name__ == '__main__':
    main()

