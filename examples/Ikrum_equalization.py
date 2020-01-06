#!/usr/bin/env python
import dpx_func_python
import os
try:
    input = raw_input
except NameError:
    pass

PORT = '/dev/ttyUSB0'
CHIP_NUMS = [22, 6, 109]
CONFIG_DIR = 'config/'
CONFIG_FN = [CONFIG_DIR + '/DPXConfig_%d.conf' % c for c in CHIP_NUMS]
IKRUM = [5, 10, 20] 

def main():
    # Create config dir if not already existing
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR)

    # THL measurements
    '''
    print('Ensure that detector %d is inserted at slot 1' % CHIP_NUMS[0]) 
    input('Press any key to proceed')
    for chip_idx in range(len(CHIP_NUMS)):
        print('Measuring THL of chip %d' % CHIP_NUMS[chip_idx])
        dpx = dpx_func_python.Dosepix(PORT, 2e6, CONFIG_DIR + '/' + CONFIG_FN)
        dpx.measureTHL(1, fn=CONFIG_DIR + 'THLCalib_%d.json' % CHIP_NUMS[chip_idx], plot=False)
        dpx.close()

        if chip_idx != len(CHIP_NUMS):
            print('Please disconnect board and insert detector %d into slot 1' % CHIP_NUMS[chip_idx])
            input('Reconnect and press any key to proceed')
        print()
    '''

    # Threshold equalization
    thl_calib_files = [CONFIG_DIR + '/THLCalib_%d.json' % CHIP for CHIP in CHIP_NUMS] 
    dpx = dpx_func_python.Dosepix(PORT, 2e6, thl_calib_files=thl_calib_files)

    # Change Ikrum values
    for chip_idx in range(3):
        d = dpx.splitPerihperyDACs(dpx.peripherys[chip_idx] + dpx.THLs[chip_idx], perc=False)
        d['I_krum'] = IKRUM[chip_idx]
        code = dpx.periheryDACsDictToCode(d, perc=False)
        dpx.peripherys[chip_idx] = code[:-4]
        dpx.DPXWritePeripheryDACCommand(chip_idx + 1, code)

    dpx.thresholdEqualizationConfig(CONFIG_FN, I_pixeldac=None, reps=1, intPlot=False, resPlot=True)
    dpx.close()

if __name__ == '__main__':
    main()

