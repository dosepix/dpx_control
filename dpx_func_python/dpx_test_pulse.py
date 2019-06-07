import numpy as np

import dpx_settings as ds

class DPX_test_pulse(object):
    def testPulseInit(self, slot, column=0):
        # Set Polarity to hole and Photon Counting Mode in OMR
        OMRCode = self.OMR
        if not isinstance(self.OMR, basestring):
            OMRCode[3] = 'hole'
        else:
            OMRCode = '%04x' % (int(self.OMR, 16) & ~(1 << 17))

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
        # print confBits

        # confBits = np.asarray( [int(num, 16) for num in textwrap.wrap(self.confBits[slot-1], 2)] )
        # confBits[confBits != getattr(ds._ConfBits, 'MaskBit')] = getattr(ds._ConfBits, 'TestBit_Analog')

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
        print '%032x' % peripheryDACcode
        print DACVal

        return '%032x' % peripheryDACcode

    def TPtoToT(self, slot=1, column=0, outFn='TPtoToT.p'):
        # Description: Generate test pulses and measure ToT afterwards
        #              to match test pulse voltage and ToT value.

        # Measure temperature
        if type(self.OMR) is list:
            OMRCode_ = self.OMRListToHex(self.OMR)
        else:
            OMRCode_ = self.OMR
        OMRCode_ = int(OMRCode_, 16)

        OMRCode_ &= ~(0b11111 << 12)
        OMRCode_ |= getattr(ds._OMRAnalogOutSel, 'Temperature')
        self.DPXWriteOMRCommand(slot, hex(OMRCode_).split('0x')[-1])

        TList = []
        for i in range(100):
            TList.append( float(int(self.MCGetADCvalue(), 16)) )
        T = np.mean(TList)

        # Number of test pulses for ToT measurement
        NToT = 10

        # Test pulse voltage range
        # TPvoltageRange = list(reversed(np.arange(300, 490, 1))) # list(reversed(np.arange(490, 512, 1)))
        TPvoltageRange = list(reversed(np.arange(440, 512, 1))) + list(reversed(np.arange(250, 440, 5)))

        # Store results per column in dict
        resDict = {}

        # Activate DosiMode and select columns
        columnRange = self.testPulseInit(slot, column=column)

        # Loop over columns
        for column in columnRange:
            # Select column
            self.maskBitsColumn(slot, column)

            # Store results in lists
            ToTListTotal, ToTErrListTotal = [], []

            # Loop over test pulse voltages
            for val in TPvoltageRange:
                # Set test pulse voltage
                DACval = self.getTestPulseVoltageDAC(slot, val, energy=False)
                self.DPXWritePeripheryDACCommand(slot, DACval)

                # Generate test pulses
                dataList = []
                for i in range(NToT):
                    self.DPXDataResetCommand(slot)
                    self.DPXGeneralTestPulse(slot, 1000)
                    data = self.DPXReadToTDataDosiModeCommand(slot)[column]
                    dataList.append( data )

                # Determine corresponding ToT value
                testPulseToT = np.mean(dataList , axis=0)
                testPulseToTErr = np.std(dataList, axis=0) / np.sqrt( NToT )
                print 'Test Pulse ToT:'
                print testPulseToT, np.std(dataList, axis=0)

                # Store in lists
                ToTListTotal.append( testPulseToT )
                ToTErrListTotal.append( testPulseToTErr )

            resDict[column] = {'ToT': ToTListTotal, 'ToTErr': ToTErrListTotal, 'volt': TPvoltageRange}

            '''
            for i in range(16):
                plt.errorbar(TPvoltageRange, np.asarray(ToTListTotal).T[i], yerr=np.asarray(ToTErrListTotal).T[i], marker='x')
            plt.show()
            '''

        outFn = outFn.split('.')[0] + '_T%d.hck' % T
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
                        print ToTValues
                        dataArray.append( ToTValues.T )
                        print dataArray

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

                                print popt, perr/popt*100

                                plt.plot(x, y, marker='x')
                                plt.plot(energyRangeFit/float(1000), self.energyToToTFitAtan(energyRangeFit/float(1000), *popt))
                                plt.show()

                    outList.append(np.vstack(dataArray))

        outDict['data'] = outList

        # Dump to file
        if paramOutFn:
            cPickle.dump(outDict, open(paramOutFn, 'wb'))
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
                print self.DPXReadToTDataDosiModeCommand(slot)[column]

    def measurePulseShape(self, slot, column='all'):
        assert len(self.THLEdges) > 0, 'Need THL calibration first!'

        columnRange = self.testPulseInit(slot, column=column)
        N = 10

        # Set energy
        energy = 10.e3
        DACVal = self.getTestPulseVoltageDAC(slot, energy, True)
        print 'Energy DAC:', DACVal
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
            for cnt, THL in enumerate(THLRange[::THLstep]):

                # Set new THL
                self.DPXWritePeripheryDACCommand(slot, DACVal[:-4] + '%04x' % THL)
                self.statusBar(float(cnt) / len(THLRange[::THLstep]) * 100 + 1)

                dataList = []
                for n in range(N):
                    self.DPXDataResetCommand(slot)
                    self.DPXGeneralTestPulse(slot, 1000)
                    dataList.append( self.DPXReadToTDataDosiModeCommand(slot)[column] )

                mean = np.mean( dataList, axis=0 )
                if np.any(mean):
                    print mean
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
            print 'Energy DAC:', DACVal
            self.DPXWritePeripheryDACCommand(slot, DACval)

            meanList, sigmaList = [], []
            print 'E = %.2f keV' % (energy / 1000.)
            for k, column in enumerate(columnRange):
                self.maskBitsColumn(slot, column)

                # Record ToT spectrum of pulse
                dataList = []
                for n in range(N):
                    self.statusBar(float(k*N + n)/(len(columnRange) * N) * 100 + 1)
                    self.DPXDataResetCommand(slot)
                    self.DPXGeneralTestPulse(slot, 100)
                    dataList.append( self.DPXReadToTDataDosiModeCommand(slot)[column] )

                dataList = np.asarray(dataList)
                for pixel in range(16):
                    # ax.hist(dataList[:,pixel], bins=30)
                    meanList.append( np.mean(dataList[:,pixel]) )
                    sigmaList.append( np.std(dataList[:,pixel]) )
                    # plt.show()

            print
            print meanList
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
                print np.mean(out), np.std(out)
                print out
                outList.append( out )

            dataMatrix = np.rec.fromarrays( outList )
            for i in range(len(dataMatrix)):
                print np.asarray([list(entry) for entry in dataMatrix[i]])
                plt.imshow(np.asarray([list(entry) for entry in dataMatrix[i]]))
                plt.show()

        self.testPulseClose(slot)
        return

