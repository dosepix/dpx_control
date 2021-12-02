#!/usr/bin/env python
import dpx_control

PORT = '/dev/ttyUSB0'
CONFIG_FN = 'DPXConfig_22_6_109.conf'
CONFIG_DIR = 'config/'
CHIP_NUMS = [22, 6, 109]
IKRUM = [10, 30, 50] 
THL_SHIFT = 0 # 880

def main():
    # Establish connection
    thl_calib_files = [CONFIG_DIR + '/THLCalib_%d.hck' % CHIP for CHIP in CHIP_NUMS] 
    dpx = dpx_control.Dosepix(PORT, 2e6, CONFIG_DIR + '/' + CONFIG_FN, thl_calib_files=thl_calib_files)

    # Change Ikrum values
    new_peripherys = []
    for chip_idx in range(3):
        d = dpx.splitPerihperyDACs(dpx.peripherys + dpx.THLs[chip_idx], perc=False)
        d['I_krum'] = IKRUM[chip_idx]
        code = dpx.periheryDACsDictToCode(d, perc=False)
        code = code[:-4] + '%04x' % (int(dpx.THLs[chip_idx], 16) - THL_SHIFT)
        dpx.THLs[chip_idx] = '%04x' % (int(dpx.THLs[chip_idx], 16) - THL_SHIFT)

        dpx.peripherys = code[:-4]
        new_peripherys.append( code[:-4] )
        dpx.DPXWritePeripheryDACCommand(chip_idx + 1, code)
        print dpx.DPXReadPeripheryDACCommand(chip_idx + 1)
        print dpx.DPXReadOMRCommand(chip_idx + 1)

    for slot in range(1, 4):
        print dpx.DPXReadPeripheryDACCommand(slot)

    dpx.energySpectrumTHL(slot=1, THLhigh=int(dpx.THLs[0], 16),  THLlow=int(dpx.THLs[0], 16) - 1300, THLstep=1, timestep=.5, intPlot=True, outFn='energySpectrumTHL.hck', slopeFn=None)

    # Close connection
    dpx.close()

if __name__ == '__main__':
    main()

