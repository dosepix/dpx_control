import time
import numpy as np
import os
import os.path
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal
import scipy.optimize
import cPickle
import hickle

import dpx_settings as ds

class DPX_functions():
    # TODO: Correction factor of 16/15 necessary?
    def measureDose(self, slot=1, measurement_time=120, frames=10, freq=False, outFn='doseMeasurement.p', logTemp=False, intPlot=False, conversion_factors=None):
        """Perform measurement in DosiMode.

        Parameters
	----------
	slot : int or list
	    Use the specified slots on the read-out board. Can be either a 
	    single detector or multiple ones. If using the latter, specify 
	    the slots via a list.
	measurement_time : float
	    Measurement time between read-outs. The counts are integrated 
	    over the specified time and a frame is read out over 16 columns
	    via rolling shutter.
	frames : int
	    Number of frames to record.
	freq : bool
	    Store the count frequency by normalization via the measurement
	    time for each frame.
	outFn : str or None
	    Output file in which the measurement data is stored. If `None`,
	    no data is written to files. Otherwise the extension of the file
	    must be '.p' or '.hck' in order to use either 'cPickle' or 
	    'hickle' for storage. If file already exists, a number is 
	    appended to the file and incremented if necessary.
	logTemp : bool
	    Log the temperature and time for each frame. If `outFn` is set, 
	    the data is stored to a file which is named according to `outFn`
	    with the suffix '_temp' attached.
	intPlot : bool
	    Use interactive plotting. The total number of counts per pixel
	    is shown for each frame.
	conversion_factors : (str, str), optional
	    A tuple of two csv-files. The first file for the conversion factors
	    of the large pixels, the second one for the ones of the small pixels.
	    These conversion factors are applied to the correspodning counts matrices
	    and the dose is calculated for each frame. If `outFn` 
	    is set, the data is stored to a file which is named according
	    to 'outFn' with the suffix '_dose' attached.

	Returns
	-------
	out : dict
	    Measurement data. Keys `temp` or `dose` are only available if
	    `logTemp` or `conversion_factors` are set during function call.

	    data :
	        Number of counts per frame stored in nested list of shape
		`(number of frames, number of columns, number of pixels per column,
		number of bins per pixel)`, i.e. `(frames, 16, 16, 16)`.
	    temp : (Only if `logTemp` was set)
	    	Logged temperature in DAC values for each frame.
	    dose : (Only if `conversion_factors` was set)
	        Measured dose for each frame. Separated for small and large pixels.
        """

        # Set Dosi Mode in OMR
        # If OMR code is list
        OMRCode = self.OMR
        if not isinstance(OMRCode, basestring):
            OMRCode[0] = 'DosiMode'
        else:
            OMRCode = int(OMRCode, 16) & ~((0b11) << 22)

        # If slot is no list, transform it into one
        if not isinstance(slot, (list,)):
            slot = [slot]

        # Set ADC out to temperature if requested
        if logTemp:
            tempDict = {'temp': [], 'time': []}

            if type(self.OMR) is list:
                OMRCode_ = self.OMRListToHex(self.OMR)
            else:
                OMRCode_ = self.OMR
            OMRCode_ = int(OMRCode_, 16)

            OMRCode_ &= ~(0b11111 << 12)
            OMRCode_ |= getattr(ds._OMRAnalogOutSel, 'Temperature')
            self.DPXWriteOMRCommand(1, hex(OMRCode_).split('0x')[-1])

        # Initial reset 
        for sl in slot:
            self.DPXWriteOMRCommand(sl, OMRCode)
            self.DPXDataResetCommand(sl)
        self.clearBins(slot)

        # Load conversion factors if requested
        if conversion_factors is not None:
            if len(slot) < 3:
                print 'WARNING: Need three detectors to determine total dose!'
            cvLarge, cvSmall = np.asarray( pd.read_csv(conversion_factors[0], header=None) ), np.asarray( pd.read_csv(conversion_factors[1], header=None) )
            doseDict = {'Slot%d' % sl: [] for sl in slot}
            isLarge = np.asarray([self.isLarge(pixel) for pixel in range(256)])

        # Data storage
        outDict = {'Slot%d' % sl: [] for sl in slot}

        # Interactive plot
        if intPlot:
            print 'Warning: plotting takes time, therefore data should be stored as frequency instead of counts!'
            fig, ax = plt.subplots(1, len(slot), figsize=(5*len(slot), 5))
            plt.ion()
            imList = {}

            if len(slot) == 1:
                ax = [ax]

            for idx, a in enumerate(ax):
                imList[slot[idx]] = a.imshow(np.zeros((12, 16)))
                a.set_title('Slot%d' % slot[idx])
                a.set_xlabel('x (px)')
                a.set_ylabel('y (px)')
            fig.canvas.draw()
            plt.show(block=False)

        # = START MEASUREMENT =
        try:
            print 'Starting Dose Measurement!'
            print '========================='
            measStart = time.time()
            for c in range(frames):
                startTime = time.time()

                # Measure temperature?
                if logTemp:
                    temp = float(int(self.MCGetADCvalue(), 16))
                    tempDict['temp'].append( temp )
                    tempDict['time'].append( time.time() - measStart )

                # for sl in slot:
                #    self.DPXDataResetCommand(sl)
                #    self.clearBins([sl])
                time.sleep(measurement_time)

                # = Readout =
                doseList = []
                for sl in slot:
                    outList = []
                    showList = []

                    # Loop over columns
                    for col in range(16):
                        measTime = float(time.time() - startTime)
                        # self.DPXDataResetCommand(sl)
                        # self.clearBins([sl])

                        self.DPXWriteColSelCommand(sl, 16 - col)
                        out = np.asarray( self.DPXReadBinDataDosiModeCommand(sl), dtype=float )
                        # out[out >= 2**15] = np.nan

                        # Calculate frequency if requested
                        if freq:
                            out = out / measTime

                        # Bug: discard strange values in matrix
                        # print np.nanmean(out), np.nanstd(out), np.nansum(out)
                        # out[abs(out - np.nanmean(out)) > 3 * np.nanstd(out)] = 0
                        
                        # plt.imshow(out.T[2:-2].T)
                        # plt.show()
                        # print out

                        outList.append( out )
                        showList.append( np.nansum(out, axis=1)[2:-2] )

                    # Append to outDict
                    data = np.asarray(outList)
                    outDict['Slot%d' % sl].append( data )

                    # plt.imshow(showList)
                    # plt.show()

                    if conversion_factors is not None:
                        dLarge, dSmall = np.sum(np.asarray(data).reshape((256, 16))[isLarge], axis=0), np.sum(np.asarray(data).reshape((256, 16))[~isLarge], axis=0)
                        doseLarge = np.nan_to_num( np.dot(dLarge, cvLarge[:,(sl-1)]) / (time.time() - measStart) )
                        doseSmall = np.nan_to_num( np.dot(dSmall, cvSmall[:,(sl-1)]) / (time.time() - measStart) )
                        print 'Slot %d: %.2f uSv/s (large), %.2f uSv/s (small)' % (sl, doseLarge, doseSmall)
                        doseDict['Slot%d' % sl].append( (doseLarge, doseSmall) )

                    if intPlot:
                        print showList
                        imList[sl].set_data( showList )
                        # fig.canvas.flush_events()

                        # plt.imshow(showList)
                        # plt.show()

                if conversion_factors is not None:
                    print 'Total dose: %.2f uSv/s' % (np.sum([np.sum(doseDict['Slot%d' % sl]) for sl in slot]))
                    print

                if intPlot:
                    fig.canvas.draw()
                    # fig.canvas.flush_events()
                    # MAC: currently not working
                    # see: https://stackoverflow.com/questions/50490426/matplotlib-fails-and-hangs-when-plotting-in-interactive-mode
                    # plt.pause(0.0001)
                    # plt.show(block=True)

                print '%.2f Hz' % (c / (time.time() - measStart))

            # Loop finished
	    if outFn is not None:
                self.pickleDump(outDict, outFn)
                if logTemp:
                    self.pickleDump(tempDict, '%s_temp' % outFn.split('.p')[0] + '.p')
                if conversion_factors:
                    self.pickleDump(doseDict, '%s_dose' % outFn.split('.p')[0] + '.p')
	    returnDict = {'data': outDict}
	    if logTemp:
	    	returnDict['temp'] = tempDict
	    if conversion_factors:
	        returnDict['dose'] = doseDict
	    return returnDict

        except (KeyboardInterrupt, SystemExit):
            # Store data and plot in files
            print 'KeyboardInterrupt-Exception: Storing data!'
            print 'Measurement time: %.2f min' % ((time.time() - measStart) / 60.)
	    if outFn is not None:
                self.pickleDump(outDict, outFn)
                if logTemp:
                    self.pickleDump(tempDict, '%s_temp' % outFn.split('.p')[0] + '.p')
                if conversion_factors:
                    self.pickleDump(doseDict, '%s_dose' % outFn.split('.p')[0] + '.p')
                raise

            if intPlot:
                plt.close('all')

            # Reset OMR
            for sl in slot:
                self.DPXWriteOMRCommand(sl, self.OMR)

	    returnDict = {'data': outDict}
	    if logTemp:
	    	returnDict['temp'] = tempDict
	    if conversion_factors:
	        returnDict['dose'] = doseDict
	    return returnDict

    def measureIntegration(self, slot=1, measurement_time=10, frames=10, outFn='integrationMeasurement.p'):
        """Perform measurement in Integration Mode.

        Parameters
	----------
	slot : int or list
	    Use the specified slots on the read-out board. Can be either a 
	    single detector or multiple ones. If using the latter, specify 
	    the slots via a list.
	measurement_time : float
	    Duration for wich the measured ToT values are integrated per frame.
	frames : int
	    Number of frames to measure.
	outFn : str or None
	    Output file in which the measurement data is stored. If `None`,
	    no data is written to files. Otherwise the extension of the file
	    must be '.p' or '.hck' in order to use either 'cPickle' or 
	    'hickle' for storage. If file already exists, a number is 
	    appended to the file and incremented if necessary.

	Returns
	-------
	out : None
        """

        # Set Integration Mode in OMR
        # If OMR code is list
        OMRCode = self.OMR
        if not isinstance(OMRCode, basestring):
            OMRCode[0] = 'IntegrationMode'
        else:
            OMRCode = int(OMRCode, 16) & ((0b11) << 22)

        # If slot is no list, transform it into one
        if not isinstance(slot, (list,)):
            slot = [slot]

        for sl in slot:
            self.DPXWriteOMRCommand(sl, OMRCode)

        outDict = {'Slot%d' % sl: [] for sl in slot}

        # = START MEASUREMENT =
        print 'Starting ToT Integral Measurement!'
        print '=================================='
        for c in range(frames):
            # Start frame 
            OMRCode[1] = 'ClosedShutter'
            for sl in slot:
                # Reset data registers
                self.DPXDataResetCommand(sl)
                self.DPXWriteOMRCommand(slot, OMRCode)

            # Wait specified time
            time.sleep(measurement_time)

            # Read data and end frame
            OMRCode[1] = 'OpenShutter'
            for sl in slot:
                data = self.DPXReadToTDataIntegrationModeCommand(sl)
                # print data
                outDict['Slot%d' % sl].append( data.flatten() )
                self.DPXWriteOMRCommand(sl, OMRCode)

        if outFn:
            self.pickleDump(outDict, outFn)

    def measurePC(self, slot=1, measurement_time=10, frames=None, outFn='pcMeasurement.p'):
        """Perform measurement in Integration Mode.

        Parameters
	----------
	slot : int or list
	    Use the specified slots on the read-out board. Can be either a 
	    single detector or multiple ones. If using the latter, specify 
	    the slots via a list.
	measurement_time : float
	    Duration for wich the measured ToT values are integrated per frame.
	frames : int or None
	    Number of frames to measure. If set to None, an infinite loop
	    is used.
	outFn : str or None
	    Output file in which the measurement data is stored. If `None`,
	    no data is written to files. Otherwise the extension of the file
	    must be '.p' or '.hck' in order to use either 'cPickle' or 
	    'hickle' for storage. If file already exists, a number is 
	    appended to the file and incremented if necessary.

	Returns
	-------
	out : None
        """

        # Set PC Mode in OMR
        # If OMR code is list
        OMRCode = self.OMR
        if not isinstance(OMRCode, basestring):
            OMRCode[0] = 'PCMode'
        else:
            OMRCode = int(OMRCode, 16) & ((0b10) << 22)
            OMRCode = '%04x' % ((int(self.OMR, 16) & ~((0b11) << 22)) | (0b10 << 22))

        # If slot is no list, transform it into one
        if not isinstance(slot, (list,)):
            slot = [slot]

        for sl in slot:
            self.DPXWriteOMRCommand(sl, OMRCode)

        outDict = {'Slot%d' % sl: [] for sl in slot}
	
	# Specify frame range
	if frames is not None:
            frameRange = range(frames)
	else:
            frameRange = self.infinite_for()

        try:
            # = START MEASUREMENT =
            print 'Starting Photon Counting Measurement!'
            print '====================================='
            measStart = time.time()
            startTime = measStart
            for c in frameRange:
                # Start frame 
                for sl in slot:
                    # Reset data registers
                    self.DPXDataResetCommand(sl)

                # Wait specified time
                time.sleep(measurement_time)

                # Read data and end frame
                for sl in slot:
                    data = self.DPXReadToTDatakVpModeCommand(sl)
                    # print data
                    outDict['Slot%d' % sl].append( data.flatten() )
                    self.DPXWriteOMRCommand(sl, OMRCode)

                if c > 0 and not c % 100:
                    print '%.2f Hz' % ((time.time() - startTime) / 100)
                    startTime = time.time()

        except (KeyboardInterrupt, SystemExit):
            # Store data and plot in files
            print 'KeyboardInterrupt-Exception: Storing data!'
            print 'Measurement time: %.2f min' % ((time.time() - measStart) / 60.)

            self.pickleDump(outDict, outFn)
            raise

        if outFn:
            self.pickleDump(outDict, outFn)

    def measureToT(self, slot=1, cnt=10000, outDir='ToTMeasurement/', storeEmpty=False, logTemp=False, paramsDict=None, intPlot=False):
        """Perform measurement in ToTMode.

        Parameters
	----------
	slot : int or list
	    Use the specified slots on the read-out board. Can be either a 
	    single detector or multiple ones. If using the latter, specify 
	    the slots via a list.
	cnt : Number of frames 
	    ToT is measured in an endless loop. The data is written to file 
	    after 'cnt' frames were processed. Additionally, Keyboard 
	    Interrupts are caught in order to store data afterwards. 
	outDir : str or None
	    Output directory in which the measurement files are stored. 
	    A file is written after `cnt` frames. If file already exists, 
	    a number is appended to the file and incremented.
	storeEmpty : bool 
	    If set, empty frames are stored, otherwise discarded.
	logTemp : bool
	    Log the temperature and time for each frame. If `outFn` is set, 
	    the data is stored to a file which is named according to `outFn`
	    with the suffix '_temp' attached.
	intPlot : bool
	    Use interactive plotting. The total number of counts per pixel
	    is shown for each frame.
	paramsDict : dict, optional
	    If specified, measured ToT values are directly converted to their
	    corresponding energies. `paramsDict` contains the pixel indices as
	    keys where each value is a dictionary itself. These contain either
	    the keys `a, b, c, t` for standard calibration factors or `a, b, c, t, h, k`
	    if a test pulse calibration was performed.

	Returns
	-------
	out : None

	Notes
	-----
	The function is optimized for long term ToT measurements. It runs in 
	an endless loop and waits for a keyboard interrupt. After `cnt` frames
	where processed, data is written to a file in the directory specified
	via `outDir`. Using these single files allows for a memory sufficient
	method to store the data in histograms without loosing information and
	without the necessecity to load the whole dataset at once.
        """

        # Set Dosi Mode in OMR
        # If OMR code is list
        OMRCode = self.OMR
        if not isinstance(OMRCode, basestring):
            OMRCode[0] = 'DosiMode'
        else:
            OMRCode = '%04x' % ((int(OMRCode, 16) & ~((0b11) << 22)))

        # Check which slots to read out
        if isinstance(slot, int):
            slotList = [slot]
        elif not isinstance(slot, basestring):
            slotList = slot

        # Set mode in slots
        for slot in slotList:
            self.DPXWriteOMRCommand(slot, OMRCode)

        # Set ADC out to temperature if requested
        if logTemp:
            tempDict = {'temp': [], 'time': []}

            if type(self.OMR) is list:
                OMRCode_ = self.OMRListToHex(self.OMR)
            else:
                OMRCode_ = self.OMR
            OMRCode_ = int(OMRCode_, 16)

            OMRCode_ &= ~(0b11111 << 12)
            OMRCode_ |= getattr(ds._OMRAnalogOutSel, 'Temperature')
            self.DPXWriteOMRCommand(slot, hex(OMRCode_).split('0x')[-1])

        # GUI
        '''
        if self.USE_GUI:
            bins = np.arange(0, 400, 1)
            histData = np.zeros(len(bins)-1)
            yield bins, histData
        '''

        # Init plot
        if intPlot:
            plt.ion()

            fig, ax = plt.subplots()

            # Create empty axis
            line, = ax.plot(np.nan, np.nan, color='k') # , where='post')
            ax.set_xlabel('ToT')
            ax.set_ylabel('Counts')
            # ax.set_yscale("log", nonposy='clip')
            plt.grid()

            # Init bins and histogram data
            if paramsDict is not None:
                bins = np.linspace(20, 100, 300)
            else:
                bins = np.arange(0, 1000, 1)
            histData = np.zeros(len(bins)-1)

            ax.set_xlim(min(bins), max(bins))

        # Check if output directory exists
        outDir = self.makeDirectory(outDir)
        outFn = outDir.split('/')[0] + '.p'

        # Initial reset 
        for slot in slotList:
            self.DPXDataResetCommand(slot)

        # For KeyboardInterrupt exception
        print 'Starting ToT Measurement!'
        print '========================='
        measStart = time.time()
        try:
            while True:
                ToTDict = {'Slot%d' % slot: [] for slot in slotList}
                # if paramsDict is not None:
                #     energyDict = {'Slot%d' % slot: [] for slot in slotList}

                c = 0
                startTime = time.time()
                if intPlot or self.USE_GUI:
                    dataPlot = []

                while c <= cnt:
                    for slot in slotList:
                        # Read data
                        data = self.DPXReadToTDataDosiModeCommand(slot)

                        # Reset data registers
                        self.DPXDataResetCommand(slot)

                        # print np.median( data )
                        # data = self.DPXReadToTDataDosiModeMultiCommand(slot)
                        # print data[256:]
                        # print data[:256]
                        # print
                        # data = data[256:]

                        data = data.flatten()
                        # Discard empty frame?
                        if not np.any(data) and not storeEmpty:
                            continue

                        if paramsDict is not None:
                            energyData = []
                            p = paramsDict['Slot%d' % slot]
                            for pixel in range(256):
                                if pixel not in p.keys():
                                    energyData.append(np.nan)
                                    continue

                                pPixel = p[pixel]
                                if len( pPixel.keys() ) == 6:
                                    energyData.append( self.ToTtoEnergySimple(data[pixel], 
                                        pPixel['a'], pPixel['b'], pPixel['c'],
                                        pPixel['t'], pPixel['h'], pPixel['k']) )
                                else:
                                    energyData.append( self.ToTtoEnergySimple(data[pixel], 
                                        pPixel['a'], pPixel['b'], pPixel['c'],
                                        pPixel['t']) )
                            data = np.asarray(energyData)
                            print data

                        # Remove overflow
                        # data[data >= 4096] -= 4096

                        if intPlot or self.USE_GUI:
                            data_ = np.asarray(data[[self.isLarge(pixel) for pixel in range(256)]])
                            data_ = data_[~np.isnan(data_)]
                            dataPlot += data_.tolist()

                        ToTDict['Slot%d' % slot].append( np.nan_to_num(data).tolist() )

                    # Measure temperature?
                    if logTemp:
                        temp = float(int(self.MCGetADCvalue(), 16))
                        tempDict['temp'].append( temp )
                        tempDict['time'].append( time.time() - measStart )

                    if c > 0 and not c % 100:
                        print '%.2f Hz' % (100./(time.time() - startTime))
                        startTime = time.time()

                    # Increment loop counter
                    c += 1

                    # Update plot every 100 iterations
                    if intPlot or self.USE_GUI:
                        if not c % 10:
                            dataPlot = np.asarray(dataPlot)
                            # Remove empty entries
                            dataPlot = dataPlot[dataPlot > 0]

                            hist, bins_ = np.histogram(dataPlot, bins=bins)
                            histData += hist
                            dataPlot = []

                            '''
                            if self.USE_GUI:
                                yield bins_, histData
                            '''

                            if intPlot:
                                line.set_xdata(bins_[:-1])
                                line.set_ydata(histData)

                                # Update plot scale
                                ax.set_ylim(1, 1.1 * max(histData))
                                fig.canvas.draw()

                # Loop finished
                self.pickleDump(ToTDict, '%s/%s' % (outDir, outFn))
                if logTemp:
                    self.pickleDump(tempDict, '%s/%s_temp' % (outDir, outFn.split('.p')[0]) + '.p')
                print 'Measurement time: %.2f min' % ((time.time() - measStart) / 60.)
                for key in ToTDict.keys():
                    print 'Slot%d: %d events' % (slot, len(np.asarray(ToTDict[key]).flatten()) / 256.)

        except (KeyboardInterrupt, SystemExit):
            # Store data and plot in files
            print 'KeyboardInterrupt-Exception: Storing data!'
            print 'Measurement time: %.2f min' % ((time.time() - measStart) / 60.)
            for key in ToTDict.keys():
                print 'Slot%d: %d events' % (slot, len(np.asarray(ToTDict[key]).flatten()) / 256.)

            self.pickleDump(ToTDict, '%s/%s' % (outDir, outFn))
            if logTemp:
                self.pickleDump(tempDict, '%s/%s_temp' % (outDir, outFn.split('.p')[0]) + '.p')
            raise

        # Reset OMR
        for slot in slotList:
            self.DPXWriteOMRCommand(slot, self.OMR)

    def measureTHL(self, slot, fn=None, plot=False):
        if not fn:
            fn = THL_CALIB_FILE
        self.measureADC(slot, AnalogOut='V_ThA', perc=False, ADChigh=8191, ADClow=0, ADCstep=1, N=1, fn=fn, plot=plot)

    def measureADC(self, slot, AnalogOut='V_ThA', perc=False, ADChigh=8191, ADClow=0, ADCstep=1, N=1, fn=None, plot=False):
        # Display execution time at the end
        startTime = time.time()

        # Select AnalogOut
        if type(self.OMR) is list:
            OMRCode_ = self.OMRListToHex(self.OMR)
        else:
            OMRCode_ = self.OMR
        OMRCode_ = int(OMRCode_, 16)

        OMRCode_ &= ~(0b11111 << 12)
        OMRCode_ |= getattr(ds._OMRAnalogOutSel, AnalogOut)
        self.DPXWriteOMRCommand(slot, hex(OMRCode_).split('0x')[-1])

        # Get peripherys
        dPeripherys = self.splitPerihperyDACs(self.peripherys + self.THLs[0], perc=perc)

        if AnalogOut == 'V_cascode_bias':
            AnalogOut = 'V_casc_reset'
        elif AnalogOut == 'V_per_bias':
            AnalogOut = 'V_casc_preamp'

        ADCList = np.arange(ADClow, ADChigh, ADCstep)
        ADCVoltMean = []
        ADCVoltErr = []
        print 'Measuring ADC!'
        for cnt, ADC in enumerate(ADCList):
            self.statusBar(float(cnt)/len(ADCList) * 100 + 1)
            
            # Set threshold
            # self.DPXWritePeripheryDACCommand(slot, self.peripherys + '%04x' % ADC)

            # Set value in peripherys
            dPeripherys[AnalogOut] = ADC
            code = self.periheryDACsDictToCode(dPeripherys, perc=perc)
            self.peripherys = code[:-4]
            self.DPXWritePeripheryDACCommand(slot, code)

            # Measure multiple times
            ADCValList = []
            for i in range(N):
                ADCVal = float(int(self.MCGetADCvalue(), 16))
                ADCValList.append( ADCVal )

            ADCVoltMean.append( np.mean(ADCValList) )
            ADCVoltErr.append( np.std(ADCValList)/np.sqrt(N) )

        if plot:
            plt.errorbar(ADCList, ADCVoltMean, yerr=ADCVoltErr, marker='x')
            plt.show()

        # Sort lists
        ADCVoltMeanSort, ADCListSort = zip(*sorted(zip(ADCVoltMean, ADCList)))
        # print THLListSort
        if plot:
            plt.plot(ADCVoltMeanSort, ADCListSort)
            plt.show()

        d = {'Volt': ADCVoltMean, 'ADC': ADCList}
        if fn:
            self.pickleDump(d, fn)
        else:
            self.pickleDump(d, 'ADCCalib.p')

        print 'Execution time: %.2f min' % ((time.time() - startTime)/60.)

    def thresholdEqualizationConfig(self, configFn, reps=1, I_pixeldac=0.21, intPlot=False, resPlot=True):
        for i in range(1, 3 + 1):
            pixelDAC, THL, confMask = self.thresholdEqualization(slot=i, reps=reps, I_pixeldac=I_pixeldac, intPlot=intPlot, resPlot=resPlot)

            # Set values
            self.pixelDAC[i-1] = pixelDAC
            self.THLs[i-1] = THL
            self.confBits[i-1] = confMask

        self.writeConfig(configFn)

    def thresholdEqualization(self, slot, reps=1, THL_offset=20, I_pixeldac=0.5, intPlot=False, resPlot=True):
        """Perform threshold equalization of all pixels

        Parameters
	----------
	slot : int or list
	    Use the specified slots on the read-out board. Can be either a 
	    single detector or multiple ones. If using the latter, specify 
	    the slots via a list.
	reps : int
	    Number of repetitions to find the noise level for each pixel.
	THL_offset : int
	    Offset in THL which is subtracted from yielded THL value of 
	    equalization in order to gain robustness.
	I_pixeldac : float
	    Current of pixel DACs. If greater than 0.2, pixel DAC to THL
	    relation isn't linear anymore.
	intPlot : bool
	    Show pixel DAC to THL relation after equalization.
	resPlot : bool
	    Show distribution of noise level for all pixels before 
	    and after equalization.

	Returns
	-------
	pixelDAC : str
	    String specifying the pixel DAC values
	confMask : str
	    String specifying the conf-bits for each pixel
        """

        # THLlow, THLhigh = 5120, 5630
        THLlow, THLhigh = 4000, 6000
        # THLRange = np.arange(THLlow, THLhigh, 1) 

        if len(self.THLEdges) == 0 or self.THLEdges[slot - 1] is None:
            THLRange = np.arange(THLlow, THLhigh)
        else:
            THLRange = np.asarray(self.THLEdges[slot - 1])
            THLRange = np.around(THLRange[np.logical_and(THLRange >= THLlow, THLRange <= THLhigh)])

        THLstep = 1
        noiseLimit = 3
        spacing = 2
        existAdjust = True

        print '== Threshold equalization of detector %d ==' % slot

        # Set PC Mode in OMR in order to read kVp values
        # If OMR code is list
        if isinstance(self.OMR, (list,)):
            OMRCode = self.OMR
            OMRCode[0] = 'PCMode'
            self.DPXWriteOMRCommand(slot, OMRCode)
        else:
            self.DPXWriteOMRCommand(slot, '%04x' % ((int(self.OMR, 16) & ~((0b11) << 22)) | (0b10 << 22)))

        # Linear dependence:
        # Start and end points are sufficient
        pixelDACs = ['00', '3f']

        # Set I_pixeldac
        if I_pixeldac is not None:
            d = self.splitPerihperyDACs(self.peripherys + self.THLs[0], perc=True)
            d['I_pixeldac'] = I_pixeldac
            code = self.periheryDACsDictToCode(d, perc=True)
            self.peripherys = code[:-4]
            self.DPXWritePeripheryDACCommand(slot, code)

            # Perform equalization
            if I_pixeldac > 0.2:
                # Nonlinear dependence
                pixelDACs = ['%02x' % int(num) for num in np.linspace(0, 63, 9)]

        countsDict = self.getTHLLevel(slot, THLRange, pixelDACs, reps, intPlot)
        print countsDict['00'].keys(), THLRange
        gaussDict, noiseTHL = self.getNoiseLevel(countsDict, THLRange, pixelDACs, noiseLimit)
        # print 'noiseTHL1:', noiseTHL

        # Transform values to indices and get meanDict
        meanDict, noiseTHL = self.valToIdx(slot, pixelDACs, THLRange, gaussDict, noiseTHL)

        if len(pixelDACs) > 2:
            def slopeFit(x, m, t):
                return m*x + t
            slope = np.zeros((16, 16))
            offset = np.zeros((16, 16))

        else: 
            slope = (noiseTHL['00'] - noiseTHL['3f']) / 63.
            offset = noiseTHL['00']

        x = [int(key, 16) for key in pixelDACs]
        if len(pixelDACs) > 2:
            # Store fit functions in list
            polyCoeffList = []

        for pixelX in range(16):
            for pixelY in range(16):
                y = []
                for pixelDAC in pixelDACs:
                    y.append(noiseTHL[pixelDAC][pixelX, pixelY])

                if len(pixelDACs) > 2:
                    x, y = np.asarray(x), np.asarray(y)
                    # Remove nan values
                    notNanIdx = np.isfinite(x) & np.isfinite(y)

                    # Perform polyfit
                    polyCoeff = np.polynomial.polynomial.polyfit(x[notNanIdx], y[notNanIdx], 3)
                    polyCoeffList.append( polyCoeff )

                    polyFit = np.polynomial.polynomial.Polynomial(polyCoeff)

                # plt.plot(adjust[pixelX, pixelY], mean, marker='x')
                if resPlot:
                    # Add offset to getColor to get rid of bright colors
                    plt.plot(x, y, alpha=.5, color=self.getColor('Blues', 5 + 256, pixelX * 16 + pixelY + 5))

                    if len(pixelDACs) > 2:
                        plt.plot(x, y, alpha=.5, color=self.getColor('Blues', 5 + 256, pixelX * 16 + pixelY + 5), ls='', marker='x')

                        xFit = np.linspace(min(x), max(x), 1000)
                        plt.plot(xFit, polyFit(xFit), ls='-', color=self.getColor('Blues', 5 + 256, pixelX * 16 + pixelY + 5))

        mean = 0.5 * (meanDict['00'] + meanDict['3f'])
        # print meanDict['00'], meanDict['3f'], mean

        if resPlot:
            plt.xlabel('DAC')
            plt.ylabel('THL')
            plt.axhline(y=mean, ls='--')
            plt.grid()
            plt.show()

        if len(pixelDACs) == 2:
            # Get adjustment value for each pixel
            adjust = np.asarray( (offset - mean) / slope + 0.5 )
        else:
            # Get intersection of fit function and mean value
            # Describe mean in terms of a polynomial of 3rd grade
            polyMean = [mean, 0., 0., 0.]
            adjust = np.zeros(256)

            for pixel in range(256):
                polyCoeff = polyCoeffList[pixel]

                if np.any(np.isnan(polyCoeff)):
                    adjust[pixel] = 0
                else:
                    roots = np.asarray( np.polynomial.polynomial.polyroots(polyCoeff - polyMean) )
                    roots = roots[np.logical_and(roots <= 63, roots >= 0)]
                    if list(roots):
                        adjust[pixel] = roots[0]
                    else:
                        adjust[pixel] = 0

                    print adjust[pixel]
                
            adjust = np.reshape(adjust, (16, 16))

        # Consider extreme values
        adjust[np.isnan(adjust)] = 0
        adjust[adjust > 63] = 63
        adjust[adjust < 0] = 0

        # Convert to integer
        adjust = adjust.astype(dtype=int)

        # Set new pixelDAC values
        pixelDACNew = ''.join(['%02x' % entry for entry in adjust.flatten()])

        # Repeat procedure to get noise levels
        countsDictNew = self.getTHLLevel(slot, THLRange, pixelDACNew, reps, intPlot)

        gaussDictNew, noiseTHLNew = self.getNoiseLevel(countsDictNew, THLRange, pixelDACNew, noiseLimit)

        # Transform values to indices
        meanDictNew, noiseTHLNew = self.valToIdx(slot, [pixelDACNew], THLRange, gaussDictNew, noiseTHLNew)

        # Plot the results of the equalization
        if resPlot:
            bins = np.linspace(min(gaussDict['3f']), max(gaussDict['00']), 100)

            for pixelDAC in ['00', '3f']:
                plt.hist(gaussDict[pixelDAC], bins=bins, label='%s' % pixelDAC, alpha=0.5)

            plt.hist(gaussDictNew[pixelDACNew], bins=bins, label='After equalization', alpha=0.5)

            plt.legend()

            plt.xlabel('THL')
            plt.ylabel('Counts')

            plt.show()

        # Create confBits
        confMask = np.zeros((16, 16)).astype(str)
        confMask.fill('00')
        confMask[abs(noiseTHLNew[pixelDACNew] - mean) > 10] = '%02x' % (0b1 << 2)
        confMask = ''.join(confMask.flatten())

        '''
        # Find center point of next sawtooth in THL curve
        # in negative direction of THL
        THLNew = THLRange[int(mean)]

        # Calculate centers of edge intervals
        edgesCenter = np.asarray([ 0.5 * (self.THLEdgesHigh[k] + self.THLEdgesLow[k]) for k in range(len(self.THLEdgesLow)) ])

        # Eliminate values larger than THLNew
        edgesCenter = edgesCenter[edgesCenter <= THLNew]

        # Get closest center value to THLNew
        THLNew = min(edgesCenter, key=lambda x:abs(x-THLNew))
        '''
        # THLNew = self.THLEdges[list(THLRange).index(int(self.getTHLfromVolt(mean))) - 20]
        THLNew = int(np.mean(gaussDictNew[pixelDACNew]) - THL_offset)

        print
        print 'Summary:'
        print 'pixelDACs:', pixelDACNew
        print 'confMask:', confMask
        print 'Bad pixels:', np.argwhere((abs(noiseTHLNew[pixelDACNew] - mean) > 10) == True)
        print 'THL:', '%04x' % THLNew

        # Restore OMR values
        self.DPXWriteOMRCommand(slot, self.OMR)

        # Subtract value from THLMin to guarantee robustness
        return pixelDACNew, '%04x' % THLNew, confMask

