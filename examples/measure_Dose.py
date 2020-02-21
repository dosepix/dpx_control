#!/usr/bin/env python
import dpx_func_python

PORT = '/dev/ttyUSB0'
CONFIG_DIR = 'config/'
CHIP_NUMS = [22, 6, 109]
IKRUM = 50
CONFIG_FN = [CONFIG_DIR + '%d/DPXConfig_%d_Ikrum%d.conf' % (CHIP, CHIP, IKRUM) if CHIP is not None else None for CHIP in CHIP_NUMS]
PARAMS_FILES = ['calibration_parameters/params_%d_Ikrum%d.json' % (CHIP, IKRUM) if CHIP is not None else None for CHIP in CHIP_NUMS]
BIN_EDGES_FILES = ['bin_edges/bin_edges_uniform_energy_10_120.json'] * 3

def main():
    # Establish connection
    thl_calib_files = None 
    dpx = dpx_func_python.Dosepix(PORT, 2e6, CONFIG_FN, thl_calib_files=thl_calib_files,
            params_file=PARAMS_FILES, bin_edges_file=BIN_EDGES_FILES)

    # Measure Dose
    dpx.measureDose(slot=[1], intPlot=False, frames=200, logTemp=True, frame_time=10)

    # Close connection
    dpx.close()

if __name__ == '__main__':
    main()

