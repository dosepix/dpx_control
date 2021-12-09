#!/usr/bin/env python
import dpx_control

PORT = '/dev/ttyUSB0'
CONFIG_FN = 'DPXConfig.conf'
CONFIG_DIR = 'config/'
CHIP_NUMS = [10, 15, 20]

def main():
    thl_calib_files = [CONFIG_DIR + '/THLCalib_%d.hck' % CHIP for CHIP in CHIP_NUMS] 
    dpx = dpx_control.Dosepix(PORT, 2e6, CONFIG_DIR + '/' + CONFIG_FN, thl_calib_files=thl_calib_files)
    dpx.measureToT(slot=[1, 2, 3], intPlot=True, storeEmpty=False, logTemp=False)

    # dpx.TPtoToT(slot=1, column='all')
    # dpx.ADCWatch(1, ['I_krum', 'Temperature'], cnt=0)
    # dpx.measureADC(1, AnalogOut='V_fbk', perc=False, ADChigh=8191, ADClow=0, ADCstep=1, N=1, fn=None, plot=True)
    # dpx.energySpectrumTHL(1, THLhigh=8000, THLlow=int(dpx.THLs[0], 16) - 500, THLstep=25, timestep=1, intPlot=True)
    # dpx.measureADC(1, AnalogOut='V_cascode_bias', perc=True, ADChigh=0.06, ADClow=0., ADCstep=0.00001, N=1)
    # dpx.ToTtoTHL_pixelDAC(slot=1, THLstep=1, I_pixeldac=0.0001, valueLow=200, valueHigh=200, valueCount=1, energy=False, plot=False)
    # dpx.ToTtoTHL(slot=1, column='all', THLstep=1, valueLow=1.5e3, valueHigh=30e3, valueCount=20, energy=True, plot=False, outFn='ToTtoTHL.p')
    # dpx.energySpectrumTHL(1)
    # dpx.testPulseDosi(1, column='all')
    # dpx.testPulseToT(1, 10, column=0, DAC='I_krum', DACRange=range(3, 25) + range(25, 50, 5) + range(50, 120, 10), perc=False)
    # dpx.measureDose(slot=2, measurement_time=0, freq=False, frames=1000, logTemp=False, intPlot=True, conversion_factors=None) # ('conversion_factors/conversionFit_wSim_10.csv', 'conversion_factors/conversionFit_wSim_10.csv')) 
    # dpx.measureIntegration()
    # dpx.temperatureWatch(slot=1, column='all', frames=1000, energyRange=(50.e3, 50.e3), fn='TemperatureToT_DPX22_Ikrum_50keV.hck', intplot=True)
    # dpx.measureTHL(1, fn='THLCalib_22.p', plot=False)
    # dpx.measurePulseShape(1, column=1)

    dpx.close()

if __name__ == '__main__':
    main()

