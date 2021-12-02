#!/usr/bin/env python
import dpx_control
import os

try: 
    raw_input = input
except NameError: 
    pass

PORT = '/dev/ttyUSB0'
CONFIG_DIR = 'config/'
# CHIP_NUMS = [None, 154, None]
# CHIP_NUMS = [101, None, None]
CHIP_NUMS = [22, 101, 109]
CHIP_NUMS = [None, 101, None]
IKRUMS = [20] # [10, 20, 30, 40, 50]

# Set to large value if using for dose measurement (>= 20)
THL_OFFSET = 5

def main():
    # Create config dir if not already existing
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR)

    '''
    # THL measurements
    if CHIP_NUMS[0] is not None:
        print('Ensure that detector %d is inserted at slot 1' % CHIP_NUMS[0]) 
        raw_input('Press any key to proceed')
    for chip_idx in range(len(CHIP_NUMS) - 1):
        if CHIP_NUMS[chip_idx] is None:
            continue
        print('Measuring THL of chip %d' % CHIP_NUMS[chip_idx])
        dpx = dpx_control.Dosepix(PORT, 2e6)
        dpx.measureTHL(1, fn=CONFIG_DIR + 'THLCalib_%d.json' % CHIP_NUMS[chip_idx], plot=False)
        dpx.close()

        if chip_idx != len(CHIP_NUMS) and CHIP_NUMS[chip_idx+1] is not None:
                print('Please disconnect board and insert detector %d into slot 1' % CHIP_NUMS[chip_idx+1])
                raw_input('Reconnect and press any key to proceed')

    # Threshold equalization
    print('Please disconnect board and insert detectors in their specified slots')
    raw_input('Reconnect and press any key to proceed')
    '''

    for Ikrum in IKRUMS:
        CONFIG_FN = [CONFIG_DIR + '%d/DPXConfig_%d_Ikrum%d.conf' % (CHIP, CHIP, Ikrum) if CHIP is not None else None for CHIP in CHIP_NUMS]
        thl_calib_files = [CONFIG_DIR + '%d/THLCalib_%d.json' % (CHIP, CHIP) if CHIP is not None else None for CHIP in CHIP_NUMS] 
        dpx = dpx_control.Dosepix(PORT, 2e6, thl_calib_files=thl_calib_files, Ikrum=[Ikrum] * 3)
        dpx.thresholdEqualizationConfig(CONFIG_FN, I_pixeldac=None, reps=1, intPlot=False, resPlot=True, THL_offset=THL_OFFSET)
        dpx.close()

if __name__ == '__main__':
    main()

