#!/usr/bin/env python
import dpx_func_python

PORT = '/dev/tty.usbserial-A907PD5F'
CONFIG_FN = 'DPXConfig.conf'
def main():
    dpx = dpx_func_python.Dosepix(PORT, 2e6)
    dpx.thresholdEqualizationConfig(CONFIG_FN, I_pixeldac=None, reps=1, intPlot=False, resPlot=True)
    dpx.close()

if __name__ == '__main__':
    main()

