#!/usr/bin/env python
import dpx_func_python
import hickle

CORRECTION = True

# Important files
if CORRECTION:
    THL_CALIB_FILES = ['THLCalibration/THLCalib_%d.p' % slot for slot in [22, 6, 109]]
else:
    THL_CALIB_FILES = None 

PORT = '/dev/ttyUSB0'
CONFIG_DIR = 'config/'
CONFIG_FN = 'DPXConfig.conf'

IKRUM = 20
CHIP_NUMS = [22, 6, 109]
CONFIG_FN = [CONFIG_DIR + '%d/DPXConfig_%d_Ikrum%d.conf' % (CHIP, CHIP, IKRUM) if CHIP is not None else None for CHIP in CHIP_NUMS]
PARAMS_FILES = ['calibration_parameters/params_%d_Ikrum%d.json' % (CHIP, IKRUM) if CHIP is not None else None for CHIP in CHIP_NUMS]
BIN_EDGES_FILES = ['bin_edges/bin_edges_uniform_energy_10_120.json'] * 3

def main():
    # Establish connection
    thl_calib_files = None 
    dpx = dpx_func_python.Dosepix(PORT, 2e6, CONFIG_FN, thl_calib_files=thl_calib_files,
            params_file=PARAMS_FILES, bin_edges_file=BIN_EDGES_FILES)

    # THL correction
    def get_delta_E(I_leak):
        return 137.32 * I_leak - 13.87

    def get_THL(energy):
        return 4.859 * energy

    I_leak_params = (0.0664299, 0.09598637)

    import time
    mlx = dpx.megalix_connect('/dev/ttyUSB0')

    voltage = 120
    out_dir= 'doseMeasurementShift_edges_10_120'
    outFn = '%dkV_%dmA_slot%d.json'
    for current in range(1, 17):
        print 'Setting megalix to (%d, %d)' % (voltage, current)

        # I_l = I_leak[current - 1]
        if CORRECTION:
            I_l = I_leak_params[0] * current + I_leak_params[1]
            THLshift = int(get_THL(get_delta_E(I_l)))
            print 'THL shift =', THLshift

        # Turn on xrt
        dpx.megalix_set_kvpmA(mlx, voltage, current)
        time.sleep(1)
        
        # Longer pulse to measure current
        if not CORRECTION:
            dpx.megalix_xray_on(mlx)
            time.sleep(3)
            dpx.megalix_xray_off(mlx)
            time.sleep(.5)

        if CORRECTION:
            d = dpx.splitPerihperyDACs(dpx.peripherys + dpx.THLs[slot], perc=False)
            code = dpx.periheryDACsDictToCode(d, perc=False)

            # Find closest THL
            THLclose = min(list(dpx.THLEdges[slot]), key=lambda x:abs(x - (int(dpx.THLs[slot], 16) - THL_SHIFT)))
            newTHL = dpx.THLEdges[slot][list(dpx.THLEdges[slot]).index(THLclose) + THLshift]
            print( int(dpx.THLs[slot], 16), THLclose, newTHL )
            code = code[:-4] + '%04x' % newTHL

            dpx.peripherys = code[:-4]
            dpx.DPXWritePeripheryDACCommand(slot + 1, code)

        if CORRECTION:
            out_directory = out_dir + '_correction/'
        else:
            out_directory = out_dir + '_dummy/'

        # Measure Dose
        dpx.measureDoseEnergyShift(slot=[1, 2, 3], measurement_time=1., freq=False, frames=1, logTemp=True, intPlot=False,
                                    fast=True, outFn=out_directory + outFn % (voltage, current, Ikrum), mlx=mlx, regions=1)
        dpx.close()

if __name__ == '__main__':
    main()

