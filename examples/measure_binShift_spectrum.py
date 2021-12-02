#!/usr/bin/env python
import dpx_control
import sys
sys.path.insert(0, '../tools/')
import binEdgesShift_hist as besh

THL_CALIB_FILES = None 
PORT = '/dev/ttyUSB0'
CONFIG_DIR = 'config/'
CONFIG_FN = 'DPXConfig.conf'

IKRUM = 20
CHIP_NUMS = [22, 101, 109]
CONFIG_FN = [CONFIG_DIR + '%d/DPXConfig_%d_Ikrum%d.conf' % (CHIP, CHIP, IKRUM) if CHIP is not None else None for CHIP in CHIP_NUMS]
PARAMS_FILES = ['calibration_parameters/params_%d_Ikrum%d.json' % (CHIP, IKRUM) if CHIP is not None else None for CHIP in CHIP_NUMS]
BIN_EDGES_FILES = ['bin_edges/bin_edges_uniform_energy_10_120_4splits.json'] * 3

def main():
    # Establish connection
    thl_calib_files = None 
    dpx = dpx_control.Dosepix(PORT, 2e6, CONFIG_FN, thl_calib_files=thl_calib_files,
            params_file=PARAMS_FILES, bin_edges_file=BIN_EDGES_FILES)

    # Measure Dose
    outFn = 'doseMeasurementShift_AmMo'
    doseDict, timeDict = dpx.measureDoseEnergyShift(slot=[1, 2, 3], measurement_time=30., freq=False, frames=1, logTemp=True, intPlot=False,
                                                    fast=True, mlx=None, regions=1, outFn=outFn + '.json')
    bin_width = (120 - 10) / (15. * 4)
    besh.histogram_data(doseDict, timeDict, PARAMS_FILES, BIN_EDGES_FILES, rb=100, bw=bin_width, multi=True, split=4, out=outFn + '_hist.json')
    dpx.close()

if __name__ == '__main__':
    main()

