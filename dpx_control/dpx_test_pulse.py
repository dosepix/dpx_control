from __future__ import print_function
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

import dpx_control.dpx_settings as ds

class DPX_test_pulse(object):
    def testPulseInit(self, slot, column=0):
        # Set Polarity to hole and DosiMode in OMR
        OMRCode = self.OMR

        if not isinstance(self.OMR, basestring):
            OMRCode[0] = 'DosiMode'
        else:
            OMRCode = '%04x' % ((int(self.OMR, 16) & ~((0b11) << 22)) | (0b10 << 22))
        
        self.DPXWriteOMRCommand(slot, OMRCode)

        if column == 'all':
            columnRange = range(16)
        elif not isinstance(column, basestring):
            if isinstance(column, int):
                columnRange = [column]
            else:
                columnRange = column

        return columnRange

    def testPulseClose(self, slot):
        # Reset ConfBits
        self.DPXWriteConfigurationCommand(slot, self.confBits[slot-1])

        # Reset peripheryDACs
        self.DPXWritePeripheryDACCommand(slot, self.peripherys + self.THLs[slot-1])

    def maskBitsColumn(self, slot, column):
        # Set ConfBits to use test charge on preamp input if pixel is enabled
        # Only select one column at max
        confBits = np.zeros((16, 16))
        confBits.fill(getattr(ds._ConfBits, 'MaskBit'))
        confBits[column] = [getattr(ds._ConfBits, 'TestBit_Analog')] * 16

        # confBits = np.asarray( [int(num, 16) for num in textwrap.wrap(self.confBits[slot-1], 2)] )
        # confBits[confBits != getattr(ds._ConfBits, 'MaskBit')] = getattr(ds._ConfBits, 'TestBit_Analog')

        self.DPXWriteConfigurationCommand(slot, ''.join(['%02x' % conf for conf in confBits.flatten()]))

    def maskBitsPixel(self, slot, pixel=(0, 0)):
        confBits = np.zeros((16, 16))
        confBits.fill(getattr(ds._ConfBits, 'MaskBit'))
        confBits[pixel[0]][pixel[1]] = getattr(ds._ConfBits, 'TestBit_Analog')
        self.DPXWriteConfigurationCommand(slot, ''.join(['%02x' % conf for conf in confBits.flatten()]))

    def getTestPulseVoltageDAC(self, slot, DACVal, energy=False):
        # Set coarse and fine voltage of test pulse
        peripheryDACcode = int(self.peripherys + self.THLs[slot-1], 16)

        if energy:
            # Use nominal value of test capacitor
            # and DACVal as energy
            C = 5.14e-15

            deltaV = DACVal * scipy.constants.e / (C * 3.62)

            assert deltaV < 1.275, "TestPulse Voltage: The energy of the test pulse was set too high! Has to be less than or equal to 148 keV."

            # Set coarse voltage to 0
            voltageDiv = 2.5e-3
            DACVal = int((1.275 - deltaV) / voltageDiv)
        else:
            assert DACVal >= 0, 'Minimum THL value must be at least 0'
            assert DACVal <= 0x1ff, 'Maximum THL value mustn\'t be greater than %d' % 0x1ff

        # Delete current values
        peripheryDACcode &= ~(0xff << 32)   # coarse
        peripheryDACcode &= ~(0x1ff << 16)  # fine

        # Adjust fine voltage only
        peripheryDACcode |= (DACVal << 16)
        peripheryDACcode |= (0xff << 32)
        # print '%032x' % peripheryDACcode
        # print DACVal

        return '%032x' % peripheryDACcode

    def ToTtoTHL(self, slot=1, column=0, THLstep=1, valueLow=440, valueHigh=512, valueStep=1, energy=False, plot=False, outFn='ToTtoTHL.p'):
        # Description: Generate test pulses and measure their ToT
        #              values. Afterwards, do a THL-scan in order to
        #              find the corresponding THL value. Repeat 
        #              multiple times to find the correlation between
        #              ToT and THL
        # Parameters: 
        #   - energy: if set, iteration is performed over energies.
        #             Else, Test Pulse DAC values are modified directly.

        # Set low level to zero and high level to noise limit
        # for the coarse measurement
        THLlow = 0
        THLhigh = int(self.THLs[slot-1], 16)

        # Set AnalogOut to V_ThA
        OMRCode = self.OMR
        if not isinstance(OMRCode, basestring):
            OMRCode[4] = 'V_ThA'
        else:
            OMRCode = int(OMRCode, 16) & ~(0b11111 << 12)
            OMRCode |= getattr(ds._OMRAnalogOutSel, 'V_ThA')
            OMRCode = '%06x' % OMRCode
        self.DPXWriteOMRCommand(slot, OMRCode)

        valueRange = np.arange(valueLow, valueHigh, valueStep)

        # Number of test pulses for ToT measurement
        NToT = 3

        # Store results per column in dict
        resDict = {}

        # Activate DosiMode and select columns
        columnRange = self.testPulseInit(slot, column=column)

        for column in columnRange:
            THLstart = THLhigh - 1200
            THLstop = THLhigh

            # Select column
            self.maskBitsColumn(slot, column)

            # Store results in lists
            ToTListTotal, ToTErrListTotal = [], []
            THLListTotal, THLErrListTotal = [], []

            # Loop over test pulse energies
            for val in valueRange:
                # Set test pulse energy
                if energy:
                    print('Energy: %.2f keV' % (val/1000.))
                else:
                    val = int(val)
                    print('DAC: %04x' % val)

                # Activates DosiMode and selects columns
                columnRange = self.testPulseInit(slot, column=column)

                # Activate DosiMode
                if not isinstance(OMRCode, basestring):
                    OMRCode[0] = 'DosiMode'
                else:
                    OMRCode = '%06x' % (int(OMRCode, 16) & ~((0b11) << 22))
                self.DPXWriteOMRCommand(slot, OMRCode)
                
                # Set test pulse energy
                DACval = self.getTestPulseVoltageDAC(slot, val, energy)
                self.DPXWritePeripheryDACCommand(slot, DACval)

                # Generate test pulses
                dataList = []
                for i in range(NToT):
                    self.DPXDataResetCommand(slot)
                    self.DPXGeneralTestPulse(slot, 1000)
                    data = self.DPXReadToTDataDosiModeCommand(slot)[column]
                    dataList.append( data )

                # Determine corresponding ToT value
                testPulseToT = np.mean( dataList , axis=0)
                testPulseToTErr = np.std(dataList, axis=0) / np.sqrt( NToT )
                print('Test Pulse ToT:')
                print(testPulseToT)

                # Store in lists
                ToTListTotal.append( testPulseToT )
                ToTErrListTotal.append( testPulseToTErr )

                # Set PCMode
                # WARNING: 8bit counter only!
                if not isinstance(OMRCode, basestring):
                    OMRCode[0] = 'PCMode'
                else:
                    OMRCode = '%06x' % ((int(OMRCode, 16) & ~((0b11) << 22)) | (0b10 << 22))
                self.DPXWriteOMRCommand(slot, OMRCode)
                print(OMRCode)

                # Loop over THLs
                # Init result lists
                lastTHL = 0
                THLList = []
                dataList = []
                THLMeasList = []

                # Dummy readout of AnalogOut
                # self.MCGetADCvalue()
                # THLRange = np.arange(THLlow, THLhigh, THLstep)
                # THLRange = self.THLCalib[np.logical_and(self.THLCalib > THLlow, self.THLCalib < THLhigh)]
                if len(self.THLEdges) == 0 or self.THLEdges[slot - 1] is None:
                    THLRange = np.arange(THLlow, THLhigh)
                else:
                    THLRange = np.asarray(self.THLEdges[slot - 1])
                THLstop_ = THLhigh if THLstop > THLhigh else THLstop
                THLRange = THLRange[np.logical_and(THLRange >= THLstart, THLRange <= THLstop_)]

                # Do a fast measurement to estimate the position of the edge for each pixel
                THLFastList = []
                dataFastList = []
                THLFastStep = 5
                NPulses = 30        # Has to be less than 256

                for cnt, THL in enumerate(THLRange[::THLFastStep]): # :THLFastStep]):
                    # Set new THL
                    self.DPXWritePeripheryDACCommand(slot, DACval[:-4] + '%04x' % THL)

                    # Start measurement
                    self.DPXDataResetCommand(slot)
                    for i in range(NPulses):
                        self.DPXGeneralTestPulse(slot, 1000)
                    data = self.DPXReadToTDatakVpModeCommand(slot)[column]
                    data[data > NPulses] = NPulses

                    # Store data in lists
                    THLFastList.append( THL )
                    dataFastList.append( data )

                if not np.any(THLFastList):
                    continue

                print('THLFastList')
                print(THLFastList)
                print(np.asarray(dataFastList).T)

                # Check if array is empty
                if not np.count_nonzero(dataFastList):
                    # If so, energy is set too low
                    # valueCount -= 1
                    ToTListTotal.pop()
                    ToTErrListTotal.pop()
                    continue

                # Get derivative and find maximum for every pixel
                xFastDer = np.asarray(THLFastList[:-1]) + THLFastStep / 2.
                dataFastDer = [xFastDer[np.argmax(np.diff(data) / float(THLFastStep))] for data in np.asarray(dataFastList).T if np.count_nonzero(data)]

                print('dataFastDer')
                print(dataFastDer)

                # Get mean THL value
                meanTHL = np.mean( dataFastDer )

                # Set THL range to new estimated region
                regSel = np.logical_and(THLRange >= min(dataFastDer) - THLFastStep, THLRange <= max(dataFastDer) + THLFastStep)
                THLRange = THLRange[regSel]
                print('THLRange')
                print(THLRange)
                # THLRange = THLRange[abs(THLRange - meanTHL) < 3 * THLFastStep]

                # Do a slow but precise measurement with small THL steps
                for cnt, THL in enumerate(tqdm(THLRange[::THLstep])):
                    # for V in np.linspace(0.34, 0.36, 1000):
                    # THL = self.getTHLfromVolt(V)
                    # THLmeas = np.mean( [float(int(self.MCGetADCvalue(), 16)) for i in range(3) ] )
                    # THLMeasList.append( THLmeas )
                    # print THL, '%04x' % THL

                    # Set new THL
                    self.DPXWritePeripheryDACCommand(slot, DACval[:-4] + '%04x' % THL)
                    # print DACval[:-4] + '%04x' % THL
                    # print self.peripherys + '%04x' % THL

                    # Generate test pulse and measure ToT in integrationMode
                    # dataTemp = np.zeros(16)
                    self.DPXDataResetCommand(slot)
                    for i in range(NPulses):
                        self.DPXGeneralTestPulse(slot, 1000)
                    data = self.DPXReadToTDatakVpModeCommand(slot)[column]
                    # print data

                    # Store data in lists
                    THLList.append( THL )
                    dataList.append( data )

                    # lastTHL = THLmeas
                print()

                # Calculate derivative of THL spectrum
                xDer = np.asarray(THLList[:-1]) + 0.5*THLstep
                dataDer = [np.diff(data) / float(THLstep) for data in np.asarray(dataList).T]

                peakList = []
                peakErrList = []
                # Transpose to access data of each pixel
                for data in np.asarray(dataList).T:
                    # Convert to corrected THL
                    THLListCorr = [self.getVoltFromTHLFit(elm, slot) for elm in THLList]
                    THLListFit = np.linspace(min(THLListCorr), max(THLListCorr), 1000)

                    # Perform erf-Fit
                    # p0 = [THLListCorr[int(len(THLListCorr) / 2.)], 3.]
                    p0 = [np.mean(THLListCorr), 3.]
                    try:
                        popt, pcov = scipy.optimize.curve_fit(lambda x, b, c: self.erfFit(x, NPulses, b, c), THLListCorr, data, p0=p0)
                        perr = np.sqrt(np.diag(pcov))
                        print(popt)
                    except:
                        popt = p0
                        perr = len(p0) * [0]
                        print('Fit failed!')
                        pass

                    # Return fit parameters
                    # a: Amplitude
                    # b: x-offset
                    # c: scale
                    b, c = popt
                    peakList.append( b )
                    # Convert to sigma
                    peakErrList.append( perr[1] / np.sqrt(2) )

                    # Savitzky-Golay filter the data
                    # Ensure odd window length
                    windowLength = int(len(THLRange[::THLstep]) / 10.)
                    if not windowLength % 2:
                        windowLength += 1

                    # Plots
                    if plot:
                        try:
                            dataFilt = scipy.signal.savgol_filter(data, windowLength, 3)
                            plt.plot(THLListCorr, dataFilt)
                            plt.plot(*self.getDerivative(THLListCorr, dataFilt))
                        except:
                            pass

                        plt.plot(THLListFit, self.erfFit(THLListFit, NPulses, b, c), ls='-')
                        plt.plot(THLListFit, self.normalErf(THLListFit, NPulses, b, c), ls='-')
                        plt.plot(THLListCorr, data, marker='x', ls='')
                        plt.show()

                peakList, peakErrList = np.asarray(peakList), np.asarray(peakErrList)
                peakList[peakErrList > 50] = np.nan
                peakErrList[peakErrList > 50] = np.nan
                print(peakList, peakErrList)

                # Set start value for next scan
                print('Start, Stop')
                THLstart, THLstop = np.nanmin(THLList) - 400, np.nanmax(THLList) + 400
                print(THLstart, THLstop)

                # Convert back to THL
                # peakList = [THLRange[int(peak + 0.5)] + (peak % 1) for peak in peakList]

                THLListTotal.append( peakList )
                THLErrListTotal.append( peakErrList )
                print()

            # Transform to arrays
            THLListTotal = np.asarray(THLListTotal).T
            THLErrListTotal = np.asarray(THLErrListTotal).T
            ToTListTotal = np.asarray(ToTListTotal).T
            ToTErrListTotal = np.asarray(ToTErrListTotal).T

            print(THLListTotal)
            print(THLErrListTotal)
            print(ToTListTotal)
            print(ToTErrListTotal)

            resDict[column] = {'THL': THLListTotal, 'THLErr': THLErrListTotal, 'ToT': ToTListTotal, 'ToTErr': ToTErrListTotal, 'volt': valueRange}

        if plot:
            # Show results in plot for last accessed column
            fig, ax = plt.subplots()
            for i in range(len(THLListTotal)):
                ax.errorbar(THLListTotal[i], ToTListTotal[i], xerr=THLErrListTotal[i], yerr=ToTErrListTotal[i], color=self.getColor('Blues', len(ToTListTotal), i), marker='x')

            ax.grid()
            plt.xlabel('THL (DAC)')
            plt.ylabel('ToT')

            # TODO: Perform fit
            plt.show()

        # Save to pickle file
        if outFn:
            # Create dictionary
            # d = {'THL': THLListTotal, 'THLErr': THLErrListTotal, 'ToT': ToTListTotal, 'ToTErr': ToTErrListTotal}

            self.pickleDump(resDict, outFn)

        return resDict

    def ToTtoTHL_pixelDAC(self, slot=1, THLstep=1, I_pixeldac=0.1, valueLow=0, valueHigh=460, valueCount=460, energy=False, plot=False):
        pixelDACs = ['00', '3f']

        # Set I_pixeldac
        if I_pixeldac:
            dPeripherys = self.splitPerihperyDACs(self.peripherys + self.THLs[0], perc=True)
            dPeripherys['I_pixeldac'] = I_pixeldac
            code = self.periheryDACsDictToCode(dPeripherys, perc=True)
            self.peripherys = code[:-4]
            self.DPXWritePeripheryDACCommand(slot, code)

        dMeas = {}
        slopeDict = {'slope': []}

        for column in range(16):
            for pixelDAC in pixelDACs:
                # Set pixel DAC values
                self.DPXWritePixelDACCommand(slot, pixelDAC*256, file=False)

                d = self.ToTtoTHL(slot=slot, column=column, THLstep=THLstep, valueLow=valueLow, valueHigh=valueHigh, valueCount=valueCount, energy=energy, plot=plot)

                # Get THL positions
                dMeas[pixelDAC] = d['THL']

            # Get slope
            slope = (dMeas['00'] - dMeas['3f']) / 63.

            print('Pixel slope for column %d:' % column)
            print(slope)
            print('Mean slope: %.2f +/- %.2f' % (np.mean(slope), np.std(slope)))

            slopeDict['slope'].append( slope )

        self.pickleDump(slopeDict, 'pixelSlopes.p')

        return slope

    def TPtoToTPixel(self, slot=1, column=0, low=400, high=512, step=5, outFn='TPtoToT.p'):
        # Description: Generate test pulses and measure ToT afterwards
        #              to match test pulse voltage and ToT value.

        # Number of test pulses for ToT measurement
        NToT = 3

        # Test pulse voltage range
        # TPvoltageRange = list(reversed(np.arange(300, 490, 1))) # list(reversed(np.arange(490, 512, 1)))
        # TPvoltageRange = list(reversed(np.arange(470, 505, 2))) + list(reversed(np.arange(350, 470, 10)))
        TPvoltageRange = list(reversed(np.arange(low, high, step))) 

        # Store results per column in dict
        resDict = {}

        # Activate DosiMode and select columns
        columnRange = self.testPulseInit(slot, column=column)

        import time
        # Deselect all pixels
        confBits = np.zeros((16, 16))
        confBits.fill(getattr(ds._ConfBits, 'MaskBit'))

        ToTListTotal, ToTErrListTotal = [], []
        for val in TPvoltageRange:
            start_time = time.time()

            # Set test pulse voltage
            DACval = self.getTestPulseVoltageDAC(slot, val, energy=False)
            self.DPXWritePeripheryDACCommand(slot, DACval)

            dataList = []
            lastpixelx, lastpixely = 15, 15
            for i in range(NToT):
                self.DPXDataResetCommand(slot)
                for pixelx in range(16):
                    for pixely in range(16):
                        confBits[lastpixelx, lastpixely] = getattr(ds._ConfBits, 'MaskBit')
                        confBits[pixelx, pixely] = getattr(ds._ConfBits, 'TestBit_Analog')
                        self.DPXWriteConfigurationCommand(slot, ''.join(['%02x' % conf for conf in confBits.flatten()]))
                        self.DPXGeneralTestPulse(slot, 1000)
                        lastpixelx, lastpixely = pixelx, pixely
                data = np.asarray(self.DPXReadToTDataDosiModeCommand(slot), dtype=float)
                data[data > 1000] = np.nan
                dataList.append( data )

            ToT, ToTErr = np.nanmedian(dataList, axis=0), np.nanstd(dataList, axis=0) / np.sqrt(NToT)
            print(ToT[0])
            ToTListTotal.append( ToT ), ToTErrListTotal.append( ToTErr )
            print('Time:', (time.time() - start_time))
        resDict = [{'ToT': np.asarray(ToTListTotal)[:,column,:], 'ToTErr': np.asarray(ToTErrListTotal)[:,column,:], 'volt': TPvoltageRange} for column in range(16)]

        outFn = outFn.split('.')[0] + '.json' % T
        self.pickleDump(resDict, outFn)

    def TPTime(self, slot=1, column=0, voltage=100, timeRange=np.logspace(1, 3, 10), outFn='TPTime.json'):
        NToT = 10
        columnRange = self.testPulseInit(slot, column=column)
        DACval = self.getTestPulseVoltageDAC(slot, voltage, energy=False)
        self.DPXWritePeripheryDACCommand(slot, DACval)

        resDict = {column: {'ToT': [], 'ToTErr': [], 'time': timeRange} for column in columnRange}
        confBits = np.zeros((16, 16))
        for time in timeRange:
            print(time)
            for column in columnRange:
                confBits.fill(getattr(ds._ConfBits, 'MaskBit'))
                confBits[column] = [getattr(ds._ConfBits, 'TestBit_Analog')] * 16 + [getattr(ds._ConfBits, 'MaskBit')] * 0
                self.DPXWriteConfigurationCommand(slot, ''.join(['%02x' % conf for conf in confBits.flatten()]))

                dataList = []
                for i in range(NToT):
                    self.DPXDataResetCommand(slot)
                    self.DPXGeneralTestPulse(slot, time)
                    data = self.DPXReadToTDataDosiModeCommand(slot)
                    dataList.append( data[column] )

                ToT, ToTErr = np.median(dataList, axis=0), np.std(dataList, axis=0) / np.sqrt(NToT)

            resDict[column]['ToT'].append(ToT)
            resDict[column]['ToTErr'].append(ToTErr)

        self.pickleDump(resDict, outFn)

    def TPfindMax(self, slot=1, column=0):
        voltRange = np.arange(100, 400, 50)
        NToT = 3
        columnRange = self.testPulseInit(slot, column=column)
        confBits = np.zeros((16, 16))

        xMaxList = []
        for idx, volt in enumerate( voltRange ): 
            print('Scanning Test Pulse energy %d...' % volt)
            # Set test pulse voltage
            DACval = self.getTestPulseVoltageDAC(slot, volt, energy=False)
            self.DPXWritePeripheryDACCommand(slot, DACval)

            time = 1.
            x, y = [], []
            max_val = 0
            low_cnt = 0
            step_size = 5
            start_flag = True
            scans = 0
            it, maxIterations = 0, 30
            while True:
                columnList = []
                for column in columnRange:
                    confBits.fill(getattr(ds._ConfBits, 'MaskBit'))
                    confBits[column] = [getattr(ds._ConfBits, 'TestBit_Analog')] * 16 + [getattr(ds._ConfBits, 'MaskBit')] * 0
                    self.DPXWriteConfigurationCommand(slot, ''.join(['%02x' % conf for conf in confBits.flatten()]))

                    dataList = []
                    for i in range(NToT):
                        self.DPXDataResetCommand(slot)
                        self.DPXGeneralTestPulse(slot, time)
                        data = self.DPXReadToTDataDosiModeCommand(slot)
                        dataList.append( data[column] )

                    ToT, ToTErr = np.median(dataList, axis=0), np.std(dataList, axis=0) / np.sqrt(NToT)
                    columnList.append( ToT )
                
                x.append( time ), y.append( np.median(columnList) )
                it += 1
                if start_flag:
                    # Found peak
                    if len(y) > 2 and (y[-3] < y[-2]) and (y[-2] > y[-1]):
                        start_flag = False
                        time -= 1.5 * step_size
                        x = x[-3:]
                        y = y[-3:]
                        continue
                    else:
                        time += step_size
                else:
                    scans += 1
                    if scans == 2:
                        scans = 0

                        sort_idx = np.argsort(x)
                        x, y = list(np.asarray(x)[sort_idx][-5:]), list(np.asarray(y)[sort_idx][-5:])
                        if np.std(y) < 1.5 or (it > maxIterations):
                            break
                
                        zero = np.where(np.diff(np.sign(np.diff(y))))[0][0]

                        x = x[zero:zero+3]
                        y = y[zero:zero+3]

                        step_size *= 0.5
                        time = x[0] + 0.5 * step_size

                    else:
                        time += step_size

                continue

            # plt.plot(x, y, marker='x', ls='')
            spl = scipy.interpolate.UnivariateSpline(x, y, s=len(x) * 10, k=4)
            roots = spl.derivative().roots()
            x_max = roots[np.argmax(spl(roots))]

            # plt.plot(x, spl(x))
            # plt.plot(x_max, spl(x_max), marker='x', ls='', markersize=10, label=volt)

            xMaxList.append(x_max)
            # plt.show()

        print('Scanned Test Pulse energies:', voltRange)
        print('Maxima at:', xMaxList)

        idx = np.asarray(xMaxList[-2:]) > np.median(xMaxList[:2])
        idx = np.asarray([True] * (len(xMaxList) - 2) + list(idx))

        popt, pcov = scipy.optimize.curve_fit(self.linearFit, np.asarray(voltRange)[idx], np.asarray(xMaxList)[idx])
        perr = np.sqrt(np.diag(pcov))
        print('Test Pulse energy to Test Pulse duration relation:')
        print('t = (%.2f +/- %.2f) * E_TP + (%.2f +/- %.2f)' % (popt[0], perr[0], popt[1], perr[1]))
        return popt

    def TPtoToT(self, slot=1, column=0, low=400, high=512, step=5, outFn='TPtoToT.p'):
        # Description: Generate test pulses and measure ToT afterwards
        #              to match test pulse voltage and ToT value.

        # Number of test pulses for ToT measurement
        NToT = 10

        TPvoltageRange = list(reversed(np.arange(low, high, step))) 

        # Activate DosiMode and select columns
        columnRange = self.testPulseInit(slot, column=column)

        # Store results per column in dict
        resDict = {column: {'ToT': [], 'ToTErr': [], 'volt': TPvoltageRange} for column in columnRange}

        # Get times for maximum ToT
        print('Optimizing test pulse durations...')
        pTime = self.TPfindMax(slot=slot, column=0)

        print('Done!')
        print()

        import time
        it_idx = 0
        print('Test Pulse to ToT measurement:')
        for val in TPvoltageRange:
            start_time = time.time()

            # Set test pulse voltage
            DACval = self.getTestPulseVoltageDAC(slot, val, energy=False)
            self.DPXWritePeripheryDACCommand(slot, DACval)

            # Deselect all pixels
            confBits = np.zeros((16, 16))

            for column in tqdm(columnRange, desc='Column'):
                confBits.fill(getattr(ds._ConfBits, 'MaskBit'))
                confBits[column] = [getattr(ds._ConfBits, 'TestBit_Analog')] * 16 + [getattr(ds._ConfBits, 'MaskBit')] * 0
                self.DPXWriteConfigurationCommand(slot, ''.join(['%02x' % conf for conf in confBits.flatten()]))

                dataList = []
                for i in tqdm(range(NToT), leave=False):
                    self.DPXDataResetCommand(slot)
                    m, t = pTime
                    self.DPXGeneralTestPulse(slot, m * val + t)
                    data = self.DPXReadToTDataDosiModeCommand(slot)
                    dataList.append( data[column] )

                    it_idx += 1
                # print np.median(dataList, axis=0)

                ToT, ToTErr = np.median(dataList, axis=0), np.std(dataList, axis=0) / np.sqrt(NToT)

                resDict[column]['ToT'].append(ToT)
                resDict[column]['ToTErr'].append(ToTErr)

        self.pickleDump(resDict, outFn)

    def testPulseToT(self, slot, length, column='all', paramOutFn='testPulseParams.p', DAC=None, DACRange=None, perc=True):
        columnRange = self.testPulseInit(slot, column=column)

        # Measure multiple test pulses and return the average ToT
        energyRange = np.asarray(list(np.linspace(1.e3, 20.e3, 30)) + list(np.linspace(20e3, 100e3, 20)))
        energyRangeFit = np.linspace(1.e3, 100e3, 1000)
        # energyRange = np.arange(509, 0, -10)

        outDict = {'energy': energyRange, 'DAC': DACRange}

        # Set DAC value
        if DAC is not None:
            if DACRange is not None:
                outList = []
                for DACVal in DACRange:
                    d = self.splitPerihperyDACs(self.peripherys + self.THLs[0], perc=perc)
                    d[DAC] = DACVal
                    code = self.periheryDACsDictToCode(d, perc=perc)
                    self.peripherys = code[:-4]
                    self.DPXWritePeripheryDACCommand(slot, code)

                    dataArray = []
                    for column in  columnRange:
                        self.maskBitsColumn(slot, column)

                        ToTValues = []
                        for energy in energyRange:
                            # DACval = self.getTestPulseVoltageDAC(slot, energy)
                            DACval = self.getTestPulseVoltageDAC(slot, energy, True)
                            self.DPXWritePeripheryDACCommand(slot, DACval)

                            data = []
                            for i in range(10):
                                self.DPXDataResetCommand(slot)
                                self.DPXGeneralTestPulse(slot, 1000)
                                data.append( self.DPXReadToTDataDosiModeCommand(slot)[column] )
                            data = np.mean(data, axis=0)

                            # Filter zero entries
                            # data = data[data != 0]

                            ToTValues.append( data )

                        ToTValues = np.asarray( ToTValues )
                        print(ToTValues)
                        dataArray.append( ToTValues.T )
                        print(dataArray)

                        if False:
                            x = energyRange/float(1000)
                            for row in range(16):
                                y = []
                                for ToTValue in ToTValues:
                                    # print ToTValue
                                    y.append( ToTValue[row])

                                # Fit curve
                                try:
                                    popt, pcov = scipy.optimize.curve_fit(self.energyToToTFitAtan, x, y, p0=(3, 8, 50, 1))
                                    perr = np.sqrt( np.diag(pcov) )
                                except:
                                    popt = (3, 8, 50, 1)
                                    perr = np.zeros(len(popt))

                                # Store in dictionary
                                a, b, c, t = popt
                                paramOutDict['a'].append( a )
                                paramOutDict['b'].append( b )
                                paramOutDict['c'].append( c )
                                paramOutDict['t'].append( t )

                                print(popt, perr/popt*100)

                                plt.plot(x, y, marker='x')
                                plt.plot(energyRangeFit/float(1000), self.energyToToTFitAtan(energyRangeFit/float(1000), *popt))
                                plt.show()

                    outList.append(np.vstack(dataArray))

        outDict['data'] = outList

        # Dump to file
        if paramOutFn:
            json.dump(outDict, open(paramOutFn, 'wb'))
        return

        # Plot
        plt.xlabel('Energy (keV)')
        plt.ylabel('ToT')
        plt.grid()
        plt.show()

        self.testPulseClose(slot)
        return

    def measureTestPulses(self, slot, column='all'):
        columnRange = self.testPulseInit(slot, column=column)
        for column in columnRange:
            self.maskBitsColumn(slot, column)

            while True:
                self.DPXDataResetCommand(slot)
                # self.DPXGeneralTestPulse(slot, 1000)
                print(self.DPXReadToTDataDosiModeCommand(slot)[column])

    def measurePulseShape(self, slot, column='all'):
        assert len(self.THLEdges) > 0, 'Need THL calibration first!'

        columnRange = self.testPulseInit(slot, column=column)
        N = 10

        # Set energy
        energy = 10.e3
        DACVal = self.getTestPulseVoltageDAC(slot, energy, True)
        print('Energy DAC:', DACVal)
        # self.DPXWritePeripheryDACCommand(slot, DACval)

        # Define THLRange
        THLhigh = int(self.THLs[slot-1], 16)
        THLstep = 1
        THLstart = THLhigh - 1000
        THLstop = THLhigh

        THLRange = np.asarray(self.THLEdges[slot - 1])
        THLstop_ = THLhigh if THLstop > THLhigh else THLstop
        THLRange = THLRange[np.logical_and(THLRange >= THLstart, THLRange <= THLstop_)]

        for column in columnRange:
            self.maskBitsColumn(slot, column)

            ToTList, voltList = [], []
            for cnt, THL in enumerate(tqdm(THLRange[::THLstep])):

                # Set new THL
                self.DPXWritePeripheryDACCommand(slot, DACVal[:-4] + '%04x' % THL)

                dataList = []
                for n in range(N):
                    self.DPXDataResetCommand(slot)
                    self.DPXGeneralTestPulse(slot, 1000)
                    dataList.append( self.DPXReadToTDataDosiModeCommand(slot)[column] )

                mean = np.mean( dataList, axis=0 )
                if np.any(mean):
                    print(mean)
                    ToTList.append( mean )
                    voltList.append( self.getVoltFromTHLFit(THL, slot) )

            plt.plot(ToTList, -np.asarray(voltList))
            plt.show()

        self.testPulseClose(slot)

    def testPulseSigma(self, slot, column=[0, 1, 2], N=100):
        columnRange = self.testPulseInit(slot, column=column)

        fig, ax = plt.subplots()

        energyRange = np.linspace(5e3, 100e3, 50)
        meanListEnergy, sigmaListEnergy = [], []

        for energy in energyRange:
            DACval = self.getTestPulseVoltageDAC(slot, energy, True)
            print('Energy DAC:', DACVal)
            self.DPXWritePeripheryDACCommand(slot, DACval)

            meanList, sigmaList = [], []
            print('E = %.2f keV' % (energy / 1000.))
            for k, column in enumerate(tqdm(columnRange), desc='Column'):
                self.maskBitsColumn(slot, column)

                # Record ToT spectrum of pulse
                dataList = []
                for n in range(tqdm(N), leave=False):
                    self.DPXDataResetCommand(slot)
                    self.DPXGeneralTestPulse(slot, 100)
                    dataList.append( self.DPXReadToTDataDosiModeCommand(slot)[column] )

                dataList = np.asarray(dataList)
                for pixel in range(16):
                    # ax.hist(dataList[:,pixel], bins=30)
                    meanList.append( np.mean(dataList[:,pixel]) )
                    sigmaList.append( np.std(dataList[:,pixel]) )
                    # plt.show()

            print()
            print(meanList)
            meanListEnergy.append( meanList )
            sigmaListEnergy.append( sigmaList )

        meanListEnergy = np.asarray(meanListEnergy)
        sigmaListEnergy = np.asarray(sigmaListEnergy)
        # for pixel in range(len(meanListEnergy[0])):

        plt.errorbar(energyRange/1000., [np.mean(item) for item in sigmaListEnergy], xerr=energyRange/1000.*0.18, yerr=np.asarray([np.std(item) for item in sigmaListEnergy])/np.sqrt(len(columnRange)*16), marker='x', ls='')

        # Fit curve
        #popt, pcov = scipy.optimize.curve_fit(self.energyToToTFitAtan, energyRange/1000., [np.mean(item) for item in meanListEnergy], p0=(3, 8, 50, 1))
        #perr = np.sqrt( np.diag(pcov) )
        #print popt, perr/popt*100

        #plt.plot(energyRange/float(1000), self.energyToToTFitAtan(energyRange/float(1000), *popt))

        plt.xlabel('Energy (keV)')
        plt.ylabel('Standard deviation (ToT)')
        plt.show()

        self.testPulseClose(slot)

    def testPulseDosi(self, slot=1, column='all'):
        columnRange = self.testPulseInit(slot, column=column)

        # Select DosiMode
        '''
        OMRCode = self.OMR
        if not isinstance(OMRCode, basestring):
            OMRCode[0] = 'DosiMode'
        else:
            OMRCode = '%04x' % (int(OMRCode, 16) & ~((0b11) << 22))

        self.DPXWriteOMRCommand(slot, OMRCode)
        '''

        # Specify energy range
        energyRange = [25.e3] # list(np.linspace(20e3, 100e3, 10))

        ToTValues = []
        # Loop over energies
        for energy in energyRange:
            DACval = self.getTestPulseVoltageDAC(slot, energy, True)
            self.DPXWritePeripheryDACCommand(slot, DACval)

            # Clear data
            self.clearBins([slot])

            # Loop over columns
            outList = []
            for column in columnRange:
                self.maskBitsColumn(slot, 15 - column)
                # self.DPXWriteColSelCommand(slot, column)

                # Dummy readouts
                # self.DPXReadBinDataDosiModeCommand(slot)

                # Generate pulses
                for i in range(10):
                    # time.sleep(1)
                    self.DPXGeneralTestPulse(slot, 1000)

                # Read from column
                self.DPXWriteColSelCommand(slot, column)
                out = self.DPXReadBinDataDosiModeCommand(slot)
                print(np.mean(out), np.std(out))
                print(out)
                outList.append( out )

            dataMatrix = np.rec.fromarrays( outList )
            for i in range(len(dataMatrix)):
                print(np.asarray([list(entry) for entry in dataMatrix[i]]))
                plt.imshow(np.asarray([list(entry) for entry in dataMatrix[i]]))
                plt.show()

        self.testPulseClose(slot)
        return

