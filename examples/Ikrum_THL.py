#!/usr/bin/env python
import dpx_control

PORT = '/dev/ttyUSB0'
CONFIG_FN = 'DPXConfig_22_6_109.conf'
CONFIG_DIR = 'config/'
CHIP_NUMS = [22, 6, 109]
IKRUM = [10, 30, 50] 
THL_SHIFT = 10

def main():
    # Establish connection
    thl_calib_files = None # [CONFIG_DIR + '/THLCalib_%d.hck' % CHIP for CHIP in CHIP_NUMS] 
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

    # THL scan
    THL = int(dpx.THLs[0], 16)
    THLlow = THL - 500
    THLhigh = THL + 500
    # dpx.energySpectrumTHL(slot=3, THLhigh=THLhigh, THLlow=THLlow, THLstep=1, timestep=0, intPlot=False)
    dpx.findNoise(slot=1, THLlow=THLlow, THLhigh=THLhigh, THLstep=1, timestep=0)
    return

    # XRT
    THLlow = 5700 # THL
    THLhigh = 6500 # 8000 # THLlow + 1000
    for voltage in [40]:
        for current in [20, 25, 30, 40, 50]:
            print 'Processing %d kV at %d mA' % (voltage, current)
            dpx.findNoise(slot=1, THLlow=THLlow, THLhigh=THLhigh, THLstep=10, timestep=0, outFn='noise_%d_%d.hck' % (voltage, current), megalix_port='/dev/ttyUSB1', megalix_settings=(voltage, current))
            raw_input('Done! Press any key to continue')
            print

    # Close connection
    dpx.close()

if __name__ == '__main__':
    main()

