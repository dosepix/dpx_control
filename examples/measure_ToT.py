#!/usr/bin/env python
import dpx_func_python

PORT = '/dev/ttyUSB0'
CONFIG_DIR = 'config/'
CHIP_NUMS = [22, 101, 109]
IKRUM = 20
CONFIG_FN = [CONFIG_DIR + '%d/DPXConfig_%d_Ikrum%d.conf' % (CHIP, CHIP, IKRUM) if CHIP is not None else None for CHIP in CHIP_NUMS]
PARAMS_FILES = ['calibration_parameters/params_%d_Ikrum%d.json' % (CHIP, IKRUM) if CHIP is not None else None for CHIP in CHIP_NUMS]
BIN_EDGES_FILES = None

def main():
    # Establish connection
    thl_calib_files = None 
    dpx = dpx_func_python.Dosepix(PORT, 2e6, CONFIG_FN, thl_calib_files=thl_calib_files, 
            params_file=PARAMS_FILES, bin_edges_file=BIN_EDGES_FILES)

    # Measure ToT
    dpx.measureToT(slot=[1, 2, 3], intPlot=True, cnt=10000, storeEmpty=False, logTemp=True, meas_time=60 * 60)

    # Close connection
    dpx.close()

if __name__ == '__main__':
    main()

