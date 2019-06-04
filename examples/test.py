#!/usr/bin/env python
import dpx_func_python

# Important files
THL_CALIB_FILES = ['THLCalibration/THLCalib_%d.p' % slot for slot in [22, 6, 109]]
BIN_EDGES_FILE = None # 'Dennis1_binEdges.hck'
PARAMS_FILE = 'energyConversion/paramsDict_DPX22_6_109.hck' # 'testCalibFactors.p'

GEN_BIN_EDGES = False
GEN_BIN_EDGES_RANDOM = False
GEN_BIN_EDGES_UNIFORM = True
# BIN_EDGES_RANDOM_FN = 'binEdgesRandom_DPX22_6_109_v2.hck'
BIN_EDGES_RANDOM_FN = 'binEdgesUniform_DPX22_6_109_v2.hck'

if GEN_BIN_EDGES:
    import bin_edges_random as ber

    if PARAMS_FILE.endswith('.p'):
        paramsDict = cPickle.load(open(PARAMS_FILE, 'rb'))
    else:
        paramsDict = hickle.load(PARAMS_FILE)

    binEdgesDict = {}
    for slot in range(1, 3 + 1):
        if GEN_BIN_EDGES_RANDOM:
            binEdges = ber.getBinEdgesRandom(NPixels=256, edgeMin=12, edgeMax=100, edgeOvfw=430, uniform=False)
        elif GEN_BIN_EDGES_UNIFORM:
	    # Generate edges for multiple energy regions
            binEdges = []
	    energy_start, energy_range = 10, 90
	    for idx in range(4):
	    	edges = ber.getBinEdgesUniform(NPixels=256, edgeMin=energy_start + idx*energy_range, edgeMax=energy_start + (idx + 1)*energy_range, edgeOvfw=430)
		binEdges.append( edges )
        binEdgesDict['Slot%d' % slot] = binEdges
    hickle.dump(binEdgesDict, BIN_EDGES_RANDOM_FN)

# BIN_EDGES = hickle.load(BIN_EDGES_RANDOM_FN)
BIN_EDGES = {'Slot1': [12, 18, 21, 24.5, 33.5, 43, 53.5, 66.5, 81.5, 97, 113, 131.5, 151.5, 173, 200.5, 236, 430],
                'Slot2': [12, 17, 31, 40, 45.5, 50.5, 60.5, 68, 91, 102.5, 133, 148, 163, 196, 220, 257, 430],
                'Slot3': [32, 37, 47, 57.6, 68.5, 80, 91.5, 104, 117, 131, 145, 163.5, 183.5, 207.5, 234.5, 269.5, 430]}

def main():
    # Create object of class and establish connection
    dpx = dpx_func_python.Dosepix('/dev/tty.usbserial-A907PD5F', 2e6, 'DPXConfig_22_6_109.conf', thl_calib_files=['THLCalib_22.hck'] * 3)
    # dpx = Dosepix('/dev/ttyUSB0', 2e6, 'Configurations/DPXConfig_Dennis1_THL20.conf')
    # dpx = Dosepix('/dev/ttyUSB0', 2e6, 'DPXConfig_22_wCable.conf')
    # dpx.measureTestPulses(1, 0)

    # Change DAC values
    '''
    d = dpx.splitPerihperyDACs(dpx.peripherys + dpx.THLs[0], perc=False)
    d['I_krum'] = 80
    code = dpx.periheryDACsDictToCode(d, perc=False)
    dpx.peripherys = code[:-4]
    dpx.DPXWritePeripheryDACCommand(1, code)
    '''

    # dpx.TPtoToT(slot=1, column='all')
    # dpx.thresholdEqualizationConfig('DPXConfig_22_wLight.conf', I_pixeldac=None, reps=1, intPlot=False, resPlot=True)

    '''
    while True:
        for item in np.reshape( list( dpx.DPXReadDigitalThresholdsCommand(1) ), (256, 16*4) ):
            print ''.join(item)
        print
    '''

    # dpx.getPixelSlopes(1, I_pixeldac=0.)

    # dpx.meanSigmaMatrixPlot('pixelDAC_3.p', (0, 1., 15), (0, 63, 15))
    # dpx.meanSigmaLinePlot('pixelDAC_3.p', (0, 1., 15), (0, 63, 15))

    #while True:
        # print dpx.DPXReadPixelDACCommand(1)
        # dpx.DPXWriteOMRCommand(1, '0000')
        # print dpx.DPXReadPeripheryDACCommand(1)
        # print dpx.DPXReadOMRCommand(1)

    # dpx.ADCWatch(1, ['I_krum', 'Temperature'], cnt=0)
    # dpx.measureADC(1, AnalogOut='V_fbk', perc=False, ADChigh=8191, ADClow=0, ADCstep=1, N=1, fn=None, plot=True)
    # return
    # dpx.ADCWatch(1, ['V_ThA', 'V_TPref_fine', 'V_casc_preamp', 'V_fbk', 'V_TPref_coarse', 'V_gnd', 'I_preamp', 'I_disc1', 'I_disc2', 'V_TPbufout', 'V_TPbufin', 'I_krum', 'I_dac_pixel', 'V_bandgap', 'V_casc_krum', 'V_per_bias', 'V_cascode_bias', 'Temperature', 'I_preamp'], cnt=0)
    # dpx.energySpectrumTHL(1, THLhigh=8000, THLlow=int(dpx.THLs[0], 16) - 500, THLstep=25, timestep=1, intPlot=True)
    # dpx.ADCWatch(1, ['V_fbk', 'V_ThA', 'V_gnd', 'I_krum', 'V_casc_krum', 'I_preamp', 'V_casc_preamp', 'Temperature'], cnt=0)

    # dpx.measureADC(1, AnalogOut='V_cascode_bias', perc=True, ADChigh=0.06, ADClow=0., ADCstep=0.00001, N=1)
    # dpx.ToTtoTHL_pixelDAC(slot=1, THLstep=1, I_pixeldac=0.0001, valueLow=200, valueHigh=200, valueCount=1, energy=False, plot=False)
    # dpx.ToTtoTHL(slot=1, column='all', THLstep=1, valueLow=1.5e3, valueHigh=30e3, valueCount=20, energy=True, plot=False, outFn='ToTtoTHL.p')

    # dpx.energySpectrumTHL(1)
    paramsDict = None # hickle.load(PARAMS_FILE)
    # dpx.measureToT(slot=1, intPlot=True, storeEmpty=False, logTemp=False, paramsDict=paramsDict)

    # while True:
    #     dpx.TPtoToT(slot=1, column=0)

    # dpx.testPulseDosi(1, column='all')
    # dpx.testPulseSigma(1)
    # dpx.testPulseToT(1, 10, column=0, DAC='I_krum', DACRange=range(3, 25) + range(25, 50, 5) + range(50, 120, 10), perc=False)

    # dpx.measureDose(slot=2, measurement_time=0, freq=False, frames=1000, logTemp=False, intPlot=True, conversion_factors=None) # ('conversion_factors/conversionFit_wSim_10.csv', 'conversion_factors/conversionFit_wSim_10.csv')) 

    # dpx.measureIntegration()
    # dpx.temperatureWatch(slot=1, column='all', frames=1000, energyRange=(50.e3, 50.e3), fn='TemperatureToT_DPX22_Ikrum_50keV.hck', intplot=True)

    # dpx.measureTHL(1, fn='THLCalib_22.p', plot=False)
    dpx.thresholdEqualizationConfig('DPXConfig_test.conf', I_pixeldac=None, reps=1, intPlot=False, resPlot=True)

    # dpx.measurePulseShape(1, column=1)

    # Close connection
    dpx.close()
    return

if __name__ == '__main__':
    main()

