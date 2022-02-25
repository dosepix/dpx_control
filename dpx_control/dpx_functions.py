from __future__ import print_function

import time
import numpy as np
import os
import os.path
import matplotlib.pyplot as plt
import scipy.signal
import scipy.optimize
import json
from tqdm import tqdm
    
import dpx_control.dpx_settings as ds

try:
    raw_input = input
except:
    pass

try:
    basestring
except NameError:
    basestring = str

class DPX_functions():
    def measureDose(self, slot=1, frame_time=10, frames=10, freq=False,
        outFn='doseMeasurement.json', logTemp=False, intPlot=False,
        conversion_factors=None, use_gui=False):
        """Wrapper for measureDose_func
        """

        gen = self.measureDose_func(slot=slot, frame_time=frame_time,
        frames=frames, freq=freq, outFn=outFn, logTemp=logTemp,
        intPlot=intPlot, conversion_factors=conversion_factors,
        use_gui=use_gui)

        if use_gui:
            *_, returnDict = gen
            return returnDict
        return gen

    # TODO: Correction factor of 16/15 necessary?
    def measureDose_func(self,
        slot=1, frame_time=10, frames=10, freq=False,
        outFn='doseMeasurement.json', logTemp=False, intPlot=False,
        conversion_factors=None, use_gui=False):
        """Perform measurement in DosiMode.

        Parameters
        ----------
        slot : int or list
            Use the specified slots on the read-out board. Can be either a 
            single detector or multiple ones. If using the latter, specify 
            the slots via a list.
        frame_time : float
            Measurement time between read-outs. The counts are integrated 
            over the specified time and a frame is read out over 16 columns
            via rolling shutter.
        frames : int
            Number of frames to record. If set to None, an infinite measurement
            is started
        freq : bool
            Store the count frequency by normalization via the measurement
            time for each frame.
        outFn : str or None
            Output file in which the measurement data is stored in json-format. If `None`,
            no data is written to files
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
        for sl in slot:
            OMRCode = self.OMR[sl - 1]
            if not isinstance(OMRCode, basestring):
                OMRCode[0] = 'DosiMode'
            else:
                OMRCode = int(OMRCode, 16) & ~((0b11) << 22)

            # If slot is no list, transform it into one
            if not isinstance(slot, (list,)):
                slot = [slot]

            # Set OMR
            self.OMR[sl - 1] = OMRCode
            self.DPXWriteOMRCommand(sl, OMRCode)
            self.DPXDataResetCommand(sl)

        # Set ADC out to temperature if requested
        if logTemp:
            tempDict = {'temp': [], 'time': []}

            if type(self.OMR[0]) is list:
                OMRCode_ = self.OMRListToHex(self.OMR[0])
            else:
                OMRCode_ = self.OMR[0]
            # OMRCode_ = int(OMRCode_, 16)

            OMRCode_ &= ~(0b11111 << 12)
            OMRCode_ |= getattr(ds._OMRAnalogOutSel, 'Temperature')
            self.DPXWriteOMRCommand(1, hex(OMRCode_).split('0x')[-1])

        # Initial reset 
        self.clearBins(slot)

        # Load conversion factors if requested
        if conversion_factors is not None:
            if len(slot) < 3:
                print('WARNING: Need three detectors to determine total dose!')
            cvLarge, cvSmall = np.genfromtxt(conversion_factors[0]), np.genfromtxt(conversion_factors[1])
            doseDict = {'Slot%d' % sl: [] for sl in slot}
            isLarge = np.asarray([self.isLarge(pixel) for pixel in range(256)])

        # Data storage
        outDict = {'Slot%d' % sl: [] for sl in slot}

        # Interactive plot
        if intPlot:
            print('Warning: plotting takes time, therefore data should be stored as frequency instead of counts!')
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

        # Specify frame range
        if frames is not None:
                frameRange = tqdm(range(frames))
        else:
                frameRange = self.infinite_for()

        # = START MEASUREMENT =
        try:
            print('Starting Dose Measurement!')
            print('=========================')
            measStart = time.time()
            for c in frameRange:
                startTime = time.time()

                # Measure temperature?
                if logTemp:
                    temp = float(int(self.MCGetADCvalue(), 16))
                    tempDict['temp'].append( temp )
                    tempDict['time'].append( time.time() - measStart )

                # for sl in slot:
                #    self.DPXDataResetCommand(sl)
                #    self.clearBins([sl])
                time.sleep(frame_time)

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

                        self.DPXWriteColSelCommand(sl, 15 - col)
                        out = np.asarray( self.DPXReadBinDataDosiModeCommand(sl), dtype=float )
                        # out[out >= 2**15] = np.nan

                        # Calculate frequency if requested
                        if freq:
                            out = out / measTime

                        # Bug: discard strange values in matrix
                        # print(np.nanmean(out), np.nanstd(out), np.nansum(out))
                        # out[abs(out - np.nanmean(out)) > 3 * np.nanstd(out)] = 0
                        
                        # plt.imshow(out.T[2:-2].T)
                        # plt.show()

                        outList.append( out )
                        showList.append( np.nansum(out, axis=1)[2:-2] )

                    # Append to outDict
                    data = np.asarray( outList )
                    outDict['Slot%d' % sl].append( data )
                    if use_gui:
                        yield( data )

                    # plt.imshow(showList)
                    # plt.show()

                    if conversion_factors is not None:
                        dLarge, dSmall = np.sum(np.asarray(data).reshape((256, 16))[isLarge], axis=0), np.sum(np.asarray(data).reshape((256, 16))[~isLarge], axis=0)
                        doseLarge = np.nan_to_num( np.dot(dLarge, cvLarge[:,(sl-1)]) / (time.time() - measStart) )
                        doseSmall = np.nan_to_num( np.dot(dSmall, cvSmall[:,(sl-1)]) / (time.time() - measStart) )
                        print('Slot %d: %.2f uSv/s (large), %.2f uSv/s (small)' % (sl, doseLarge, doseSmall))
                        doseDict['Slot%d' % sl].append( (doseLarge, doseSmall) )

                    if intPlot:
                        print(showList)
                        imList[sl].set_data( showList )
                        # fig.canvas.flush_events()

                        # plt.imshow(showList)
                        # plt.show()

                if conversion_factors is not None:
                    print('Total dose: %.2f uSv/s' % (np.sum([np.sum(doseDict['Slot%d' % sl]) for sl in slot])))
                    print()

                if intPlot:
                    fig.canvas.draw()
                    # fig.canvas.flush_events()
                    # MAC: currently not working
                    # see: https://stackoverflow.com/questions/50490426/matplotlib-fails-and-hangs-when-plotting-in-interactive-mode
                    # plt.pause(0.0001)
                    # plt.show(block=True)

                # print('%.2f Hz' % (c / (time.time() - measStart)))

            # Loop finished
            if outFn is not None:
                self.pickleDump(outDict, outFn)
                outFn_split = outFn.split('.')
                if logTemp:
                    self.pickleDump(tempDict, '%s_temp' % outFn_split[0] + '.%s' % outFn_split[1])
                if conversion_factors:
                    self.pickleDump(doseDict, '%s_dose' % outFn_split[0] + '.%s' % outFn_split[1])
            returnDict = {'data': outDict}

            if logTemp:
                returnDict['temp'] = tempDict
            if conversion_factors:
                returnDict['dose'] = doseDict
            yield returnDict

        except (KeyboardInterrupt, SystemExit):
            # Store data and plot in files
            print('KeyboardInterrupt-Exception: Storing data!')
            print('Measurement time: %.2f min' % ((time.time() - measStart) / 60.))
            if outFn is not None:
                self.pickleDump(outDict, outFn)
                outFn_split = outFn.split('.')
                if logTemp:
                    self.pickleDump(tempDict, '%s_temp' % outFn_split[0] + '.%s' % outFn_split[1])
                if conversion_factors:
                    self.pickleDump(doseDict, '%s_dose' % outFn_split[0] + '.%s' % outFn_split[1])
                raise

            if intPlot:
                plt.close('all')

            # Reset OMR
            for sl in slot:
                self.DPXWriteOMRCommand(sl, self.OMR[sl-1])

            returnDict = {'data': outDict}
            if logTemp:
                returnDict['temp'] = tempDict
            if conversion_factors:
                returnDict['dose'] = doseDict
            yield returnDict

    # If binEdgesDict is provided, perform shifting of energy regions between
    # frames. I.e. the measured region is shifted in order to increase the
    # total energy region.
    def measureDoseEnergyShift(self, slot=1, measurement_time=120, frames=10, regions=3, freq=False, outFn='doseMeasurementShift.json', logTemp=False, intPlot=False, fast=False, mlx=None):
        # Set Dosi Mode in OMR
        # If OMR code is list
        print(self.OMR)
        for sl in slot:
            OMRCode = self.OMR[sl - 1]
            if not isinstance(OMRCode, basestring):
                OMRCode[0] = 'DosiMode'
            else:
                OMRCode = int(OMRCode, 16) & ~((0b11) << 22)

            # If slot is no list, transform it into one
            if not isinstance(slot, (list,)):
                slot = [slot]

            # Set OMR
            self.OMR[sl - 1] = '%06x' % OMRCode
            self.DPXWriteOMRCommand(sl, OMRCode)
            self.DPXDataResetCommand(sl)

        # Set ADC out to temperature if requested
        if logTemp:
            tempDict = {'temp': [], 'time': []}

            if type(self.OMR[0]) is list:
                OMRCode_ = self.OMRListToHex(self.OMR[0])
            else:
                OMRCode_ = int(self.OMR[0], 16)
            # OMRCode_ = int(OMRCode_, 16)

            OMRCode_ &= ~(0b11111 << 12)
            OMRCode_ |= getattr(ds._OMRAnalogOutSel, 'Temperature')
            self.DPXWriteOMRCommand(1, hex(OMRCode_).split('0x')[-1])

        # Initial reset
        self.clearBins(slot)

        # Data storage
        outDict = {'Slot%d' % sl: {'Region%d' % reg: [] for reg in range(regions)} for sl in slot}
        timeDict = {'Slot%d' % sl: {'Region%d' % reg: [] for reg in range(regions)} for sl in slot}

        # = START MEASUREMENT =
        try:
            print('Starting Dose Measurement!')
            print('=========================')
            if not fast:
                region_stack = {sl: [0] for sl in slot}
            else:
                region_stack = {sl: list(range(1, regions)) for sl in slot}

            region_idx = {sl: 0 for sl in slot}
            last_region_idx = {sl: None for sl in slot}
            measStart = time.time()
            STACK_APPEND = {sl: True for sl in slot}
            startTime = {sl: time.time() for sl in slot}
            stack_size = 10

            for c in range(frames * regions):
                # Measure temperature?
                if logTemp:
                    temp = float(int(self.MCGetADCvalue(), 16))
                    tempDict['temp'].append( temp )
                    tempDict['time'].append( time.time() - measStart )

                # Wait
                if mlx is not None:
                    mlx(measurement_time * 1000)
                    time.sleep(measurement_time)
                    # self.megalix_xray_on(mlx)
                    # time.sleep(measurement_time)
                    # self.megalix_xray_off(mlx)
                else:
                    time.sleep(measurement_time)

                for sl in slot:
                    outList = []

                    '''
                    if measurement_time > 0:
                        self.clearBins([sl])
                        time.sleep(measurement_time)
                        t = time.time()
                    '''

                    # Loop over columns
                    for col in range(16):
                        self.DPXWriteColSelCommand(sl, 15 - col)
                        out = np.asarray( self.DPXReadBinDataDosiModeCommand(sl), dtype=float )
                        print(np.nanmean(out), np.nanstd(out), np.nansum(out))
                        outList.append( out )
                    print()

                    # Stop time
                    # if measurement_time > 0:
                    #    timeDict['Slot%d' % sl]['Region%d' % region_idx[sl]].append( time.time() - t )
                    # else:
                    timeDict['Slot%d' % sl]['Region%d' % region_idx[sl]].append( time.time() - startTime[sl])

                    # Append to outDict
                    data = np.asarray(outList)
                    print('Append to region:', region_idx[sl])
                    outDict['Slot%d' % sl]['Region%d' % region_idx[sl]].append( data )

                    # = Measurement time handler =
                    # If stack is empty, start from beginning
                    if not len(region_stack[sl]):
                        region_idx[sl] = 0
                    else:
                        # Get current energy region
                        last_region_idx[sl] = region_idx[sl]
                        region_idx[sl] = region_stack[sl].pop(0)

                    if not fast and (len(region_stack[sl]) == 1 or (region_idx[sl] != last_region_idx[sl] and region_idx[sl] != region_stack[sl][-1])):
                        STACK_APPEND[sl] = True

                    # Get number of counts in frame and overflow bins
                    if regions > 1:
                        sum_frame = np.sum(np.reshape(data, (256, -1)), axis=0)
                        N_frame = np.sum(sum_frame[1:])
                        N_ovfw = sum_frame[0] 

                        if not fast:
                            if N_ovfw == 0 or N_frame == 0:
                                # region_stack[sl].append( region_idx )
                                continue

                        # Calculate fraction
                        p_next = N_ovfw / N_frame
                        print('p_next =', p_next)

                    if fast:
                        region_stack[sl] += range(regions)
                    else:
                        if p_next <= 0.1 and STACK_APPEND[sl]:
                            # print('Continue')
                            # Start from the beginning
                            region_stack[sl] += [0]
                            stack_size = 10
                            STACK_APPEND[sl] = False
                            # continue

                        # Select next region
                        if STACK_APPEND[sl]:
                            if region_idx[sl] == 0 or region_idx[sl] + 1 > regions:
                                stack_size = 10. / p_next

                            if region_idx[sl] + 1 > regions:
                                region_stack[sl] += 0
                            else:
                                next_region = region_idx[sl] + 1
                                region_stack[sl] += [region_idx[sl]] * (int(stack_size / (1. / p_next)) - 1)
                                region_stack[sl] += [next_region] * int(p_next * stack_size)

                            STACK_APPEND[sl - 1] = False

                    if sl == 1:
                        print(region_stack[sl])
                        print(region_stack[sl][0], region_idx[sl])

                    if regions > 1:
                        if region_stack[sl][0] != region_idx[sl]:
                            self.setBinEdgesToT(sl, self.binEdges['Slot%d' % sl][region_idx[sl]])
                        self.clearBins([sl])
                    startTime[sl] = time.time()

            # Loop finished
            outFn_split = outFn.split('.')
            if logTemp:
                self.pickleDump(tempDict, '%s_temp' % outFn_split[0] + '.%s' % outFn_split[1], overwrite=True)

            self.pickleDump(timeDict, '%s_time' % outFn_split[0] + '.%s' % outFn_split[1], overwrite=True)
            self.pickleDump(outDict, outFn, overwrite=True)

        except (KeyboardInterrupt, SystemExit):
            # Store data and plot in files
            print('KeyboardInterrupt-Exception: Storing data!')
            print('Measurement time: %.2f min' % ((time.time() - measStart) / 60.))
            outFn_split = outFn.split('.')
            if logTemp:
                self.pickleDump(tempDict, '%s_temp' % outFn_split[0] + '.%s' % outFn_split[1], overwrite=True)

            self.pickleDump(timeDict, '%s_time' % outFn_split[0] + '.%s' % outFn_split[1], overwrite=True)
            self.pickleDump(outDict, outFn, overwrite=True)
            # self.pickleDump(self.binEdges, '%s_binEdges' % outFn.split('.p')[0] + '.p')
            raise

        # Reset OMR
        for sl in slot:
            self.DPXWriteOMRCommand(sl, self.OMR[sl-1])
        return outDict, timeDict

    def measureIntegration(self, slot=1, measurement_time=10, frames=10, outFn='integrationMeasurement.json'):
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
            Output file in which the measurement data is stored in json-format. If `None`,
            no data is written to files

        Returns
        -------
        out : None
        """

        # Set Integration Mode in OMR
        for sl in slot:
            OMRCode = self.OMR[sl-1]
            OMRCode = 'f9ffc0'
            print(int(OMRCode, 16))
            self.DPXWriteOMRCommand(sl, int(OMRCode, 16))
            print( self.DPXReadOMRCommand(sl) )


        outDict = {'Slot%d' % sl: [] for sl in slot}

        # = START MEASUREMENT =
        print('Starting ToT Integral Measurement!')
        print('==================================')
        for c in range(frames):
            # Start frame
            for sl in slot:
                # Reset data registers
                self.DPXDataResetCommand(sl)

            # Wait specified time
            time.sleep(measurement_time)

            # Read data and end frame
            for sl in slot:
                data = self.DPXReadToTDataIntegrationModeCommand(sl)
                print(data)
                outDict['Slot%d' % sl].append( data.flatten() )

        if outFn:
            self.pickleDump(outDict, outFn)

    def measurePC(self, slot=1, measurement_time=10, frames=None, outFn='pcMeasurement.json', intPlot=False):
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
            Output file in which the measurement data is stored in json-format. If `None`,
            no data is written to files. If file already exists, a number is 
            appended to the file and incremented if necessary.

        Returns
        -------
        out : measurement data
        """

        # Set PC Mode in OMR
        # If OMR code is list
        for sl in slot:
            OMRCode = self.OMR[sl-1]
            if not isinstance(OMRCode, basestring):
                OMRCode[0] = 'PCMode'
            else:
                OMRCode = int(OMRCode, 16) & ((0b10) << 22)
                OMRCode = '%04x' % ((int(self.OMR[sl-1], 16) & ~((0b11) << 22)) | (0b10 << 22))

            # If slot is no list, transform it into one
            if not isinstance(slot, (list,)):
                slot = [slot]

            self.DPXWriteOMRCommand(sl, OMRCode)

        outDict = {'Slot%d' % sl: [] for sl in slot}

        # Specify frame range
        if frames is not None:
                frameRange = tqdm(range(frames))
        else:
                frameRange = self.infinite_for()

        # Interactive plot
        if intPlot:
            print('Warning: plotting takes time, therefore data should be stored as frequency instead of counts!')
            fig, ax = plt.subplots(1, len(slot), figsize=(5*len(slot), 5))
            plt.ion()
            imList = {}

            if len(slot) == 1:
                ax = [ax]

            for idx, a in enumerate(ax):
                imList[slot[idx]] = a.imshow(np.zeros((16, 16)), vmin=0, vmax=256) # np.zeros((16, 16)))
                a.set_title('Slot%d' % slot[idx])
                a.set_xlabel('x (px)')
                a.set_ylabel('y (px)')
            fig.canvas.draw()
            # plt.show(block=False)

        try:
            # = START MEASUREMENT =
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

                    if intPlot:
                        imList[sl].set_data( np.asarray(data).reshape(16, 16) )
                        print(np.asarray(data).reshape(16, 16))

                if intPlot:
                    fig.canvas.draw()
                    plt.show()

                if c > 0 and not c % 10:
                    print('%.2f Hz' % (10. / (time.time() - startTime)))
                    startTime = time.time()

        except (KeyboardInterrupt, SystemExit):
            # Store data and plot in files
            print('KeyboardInterrupt-Exception: Storing data!')
            print('Measurement time: %.2f min' % ((time.time() - measStart) / 60.))
            self.pickleDump(outDict, outFn)
            raise

        if outFn is not None and outFn:
            self.pickleDump(outDict, outFn)
        else:
            return outDict

    def measureToT(self, slot=1, cnt=10000, outDir='ToTMeasurement/', 
        storeEmpty=False, logTemp=False, intPlot=False, meas_time=None,
        external_THL=False, make_hist=False, use_gui=False):
        """Wrapper for measureToT_func
        """
        gen = self.measureToT_func(slot=slot, cnt=cnt, outDir=outDir,
            storeEmpty=storeEmpty, logTemp=logTemp, intPlot=intPlot,
            meas_time=meas_time, external_THL=external_THL,
            make_hist=make_hist, use_gui=use_gui)
        if not use_gui:
            *_, ToTDict = gen
            return ToTDict
        return gen

    def measureToT_func(self, slot=1, cnt=10000, outDir='ToTMeasurement/',
        storeEmpty=False, logTemp=False, intPlot=False, meas_time=None,
        external_THL=False, make_hist=False, use_gui=False):
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
        meas_time : float, optional
            If set, the measurement finishes after the given time in seconds.
        external_THL : bool
            Use external input voltage to set THL.

        Returns
        -------
        out : None

        Notes
        -----
        The function is optimized for long term ToT measurements. It runs in 
        an endless loop and waits for a keyboard interrupt. After `cnt` frames
        were processed, data is written to a file in the directory specified
        via `outDir`. Using these single files allows for a memory sufficient
        method to store the data in histograms without loosing information and
        without the necessecity to load the whole dataset at once.
        """

        # Check which slots to read out
        if isinstance(slot, int):
            slotList = [slot]
        elif not isinstance(slot, basestring):
            slotList = slot

        # Set Dosi Mode in OMR
        # If OMR code is list
        for slot in slotList:
            OMRCode = self.OMR[slot-1]
            if not isinstance(OMRCode, basestring):
                OMRCode[0] = 'DosiMode'
            else:
                OMRCode = int(OMRCode, 16) & ~((0b11) << 22)
                # OMRCode = '%04x' % ((int(OMRCode, 16) & ~((0b11) << 22)))

            # Set mode in slots
            self.OMR[slot - 1] = '%04x' % OMRCode
            self.DPXWriteOMRCommand(slot, OMRCode)

        # Get OMR
        if logTemp or external_THL:
            sl = 1
            if type(self.OMR[sl - 1]) is list:
                OMRCode_ = self.OMRListToHex(self.OMR[sl - 1])
            else:
                OMRCode_ = self.OMR[sl - 1]
            OMRCode_ = int(OMRCode_, 16)

        # Set ADC out to temperature if requested
        if logTemp:
            tempDict = {'temp': [], 'time': []}

            OMRCode_ &= ~(0b11111 << 12)
            OMRCode_ |= getattr(ds._OMRAnalogOutSel, 'Temperature')
            self.DPXWriteOMRCommand(1, hex(OMRCode_).split('0x')[-1])

        # Set V_ThA to be controlled from outside
        if external_THL:
            OMRCode_ &= ~(0b11111 << 7)
            OMRCode_ |= getattr(ds._OMRAnalogInSel, 'V_ThA')
            for slot in slotList:
                self.DPXWriteOMRCommand(slot, hex(OMRCode_).split('0x')[-1])

        # GUI
        if intPlot:
            # Init bins and histogram data
            if self.paramsDict is not None:
                bins = np.linspace(6, 100, 100)
            else:
                bins = np.arange(0, 400, 1)
            histData = {slot: np.zeros(len(bins)-1) for slot in slotList}
            if use_gui:
                yield bins, histData

        # Init plot
        if intPlot:
            plt.ion()

            fig, ax = plt.subplots(1, len(slotList), figsize=(5*len(slotList), 5))
            if not type(ax) == np.ndarray:
                ax = np.asarray([ax])
            ax = {slotList[idx]: ax[idx] for idx in range(len(slotList))}
            
            # Create empty axis
            ax[slotList[0]].set_ylabel('Counts')
            lines = {}
            for slot in slotList:
                lines[slot] = ax[slot].plot(np.nan, np.nan, color='k')[-1] # , where='post')
                # ax.set_yscale("log", nonposy='clip')
                ax[slot].set_xlim(min(bins), max(bins))
                ax[slot].set_xlabel('ToT')
                ax[slot].grid()

        # Check if output directory exists
        if not use_gui:
            outDir = self.makeDirectory(outDir)
            if outDir.endswith('/'):
                outDir_ = outDir[:-1]
            if '/' in outDir_:
                outFn = outDir_.split('/')[-1] + '.json'
            else:
                outFn = outDir_ + '.json'

        # Initial reset 
        for slot in slotList:
            self.DPXDataResetCommand(slot)

        for slot in slotList:
            print('Slot%d' % slot)
            print('Peripheries:', self.DPXReadPeripheryDACCommand(slot))
            print('OMR:', self.DPXReadOMRCommand(slot))

        # For KeyboardInterrupt exception
        print('Starting ToT Measurement!')
        print('=========================')
        measStart = time.time()
        
        try:
            while True:
                if meas_time is not None:
                    if time.time() - measStart > meas_time:
                        break

                if make_hist:
                    ToTDict = {'Slot%d' % slot: np.zeros((256, 8192)) for slot in slotList}
                else:
                    ToTDict = {'Slot%d' % slot: [] for slot in slotList}

                if self.paramsDict is not None:
                    energyDict = {'Slot%d' % slot: [] for slot in slotList}

                # if paramsDict is not None:
                #     energyDict = {'Slot%d' % slot: [] for slot in slotList}

                c = 0
                startTime = time.time()
                if intPlot:
                    dataPlot = {slot: [] for slot in slotList}

                while c <= cnt:
                    # Break if meas_time is surpassed
                    if meas_time is not None:
                        if time.time() - measStart > meas_time:
                            break

                    for slot in slotList:
                        # d = self.DPXReadPeripheryDACCommand(slot=2)
                        # print( d )

                        # Read data
                        data = self.DPXReadToTDataDosiModeCommand(slot)

                        # Reset data registers
                        self.DPXDataResetCommand(slot)

                        # Flatten data
                        data = data.flatten()

                        # Discard empty frame?
                        if not np.any(data) and (not storeEmpty or use_gui):
                            continue

                        if self.paramsDict is not None:
                            energyData = []
                            p = self.paramsDict['Slot%d' % slot]
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
                            # data = np.asarray(energyData)
                            energyDict['Slot%d' % slot].append( np.nan_to_num(energyData).tolist() )
                            # print data

                        if use_gui:
                            yield data

                        if intPlot:
                            if self.paramsDict is not None:
                                data_ = np.asarray(energyData).flatten()
                            else:
                                data_ = data
                            data_ = np.asarray(data_[[self.isLarge(pixel) for pixel in range(256)]])
                            data_ = data_[~np.isnan(data_)]
                            dataPlot[slot] += data_.tolist()

                        if make_hist:
                            # Remove overflow
                            data[data >= 8192] -= 8192
                            data[data >= 8192] = 0
                            for pixel in range(256):
                                ToTDict['Slot%d' % slot][pixel][data[pixel]] += 1
                        else:
                            ToTDict['Slot%d' % slot].append( np.nan_to_num(data).tolist() )

                    # Measure temperature?
                    if logTemp:
                        temp = float(int(self.MCGetADCvalue(), 16))
                        tempDict['temp'].append( temp )
                        tempDict['time'].append( time.time() - measStart )

                    if c > 0 and not c % 100:
                        print('%.2f Hz' % (100./(time.time() - startTime)))
                        startTime = time.time()

                    # Increment loop counter
                    c += 1

                    # Update plot every 10 iterations
                    if intPlot:
                        if not c % 10:
                            for slot in slotList:
                                dp = np.asarray(dataPlot[slot])
                                # Remove empty entries
                                dp = dp[dp > 0]

                                hist, bins_ = np.histogram(dp, bins=bins)
                                histData[slot] += hist
                                dataPlot[slot] = []

                                if intPlot:
                                    lines[slot].set_xdata(bins_[:-1])
                                    lines[slot].set_ydata(histData[slot])

                                    # Update plot scale
                                    ax[slot].set_ylim(1, 1.1 * max(histData[slot]))
                                    fig.canvas.draw()

                # Loop finished
                if not use_gui:
                    self.pickleDump(ToTDict, '%s/%s' % (outDir, outFn))
                    outFn_split = outFn.split('.')
                    if logTemp:
                        self.pickleDump(tempDict, '%s/%s_temp' % (outDir, outFn_split[0]) + '.%s' % outFn_split[-1], overwrite=True)
                    print('Measurement time: %.2f min' % ((time.time() - measStart) / 60.))
                    for key in ToTDict.keys():
                        print('%s: %d events' % (key, np.count_nonzero(ToTDict[key])))
                yield ToTDict

        except (KeyboardInterrupt, SystemExit):
            if not use_gui:
                # Store data and plot in files
                print('KeyboardInterrupt-Exception: Storing data!')
                print('Measurement time: %.2f min' % ((time.time() - measStart) / 60.))
                for key in ToTDict.keys():
                    print('%s: %d events' % (key, np.count_nonzero(ToTDict[key])))

                outFn_split = outFn.split('.')
                self.pickleDump(ToTDict, '%s/%s' % (outDir, outFn))
                self.pickleDump(energyDict, '%s/%s_energy' % (outDir, outFn_split[0]) + '.%s' % outFn_split[-1])
                if logTemp:
                    self.pickleDump(tempDict, '%s/%s_temp' % (outDir, outFn_split[0]) + '.%s' % outFn_split[-1], overwrite=True)
            yield ToTDict

        # Reset OMR
        # for slot in slotList:
        #    self.DPXWriteOMRCommand(slot, self.OMR)
            
    def energySpectrumTHL(self, slot=1, THLhigh=4975, THLlow=2000, THLstep=1, timestep=0.1, intPlot=True, outFn='energySpectrumTHL.json', slopeFn='pixelSlopes.json'):
        # Description: Measure cumulative energy spectrum 
        #              by performing a threshold scan. The
        #              derivative of this spectrum resembles
        #              the energy spectrum.

        # THLhigh = int(self.THLs[slot-1], 16)

        assert THLhigh <= 8191, "energySpectrumTHL: THLHigh value set too high!"

        # Take data in photon counting mode: 
        #     total number of events over threshold
        OMRCode = self.OMR[slot-1]
        if not isinstance(OMRCode, basestring):
            OMRCode[0] = 'PCMode'
            # Select AnalogOut
            OMRCode[4] = 'V_ThA'

        else:
            OMRCode = int(OMRCode, 16) & ((0b10) << 22)
            OMRCode = '%04x' % ((int(self.OMR[slot-1], 16) & ~((0b11) << 22)) | (0b10 << 22))

            # Select AnalogOut
            OMRCode = int(OMRCode, 16) & ~(0b11111 << 12)
            OMRCode |= getattr(ds._OMRAnalogOutSel, 'V_ThA')
            OMRCode = '%04x' % OMRCode

        self.DPXWriteOMRCommand(slot, OMRCode)
        self.DPXWriteColSelCommand(slot, 0)

        lastTHL = 0
        if intPlot:
            plt.ion()
            fig, ax = plt.subplots()
            axDer = ax.twinx()

            ax.set_yscale("log", nonposy='clip')
            # axDer.set_yscale("log", nonposy='clip')

            # Empty plot
            lineCum, = ax.plot(np.nan, np.nan, label='Cumulative', color='cornflowerblue')
            lineDer, = axDer.plot(np.nan, np.nan, label='Derivative', color='crimson')
            # lineTHL, = axTHL.plot(np.nan, np.nan, label='THL', color='orange')

            # Settings
            plt.xlabel('THL (DAC)')
            ax.set_ylabel('Counts / s')
            axDer.set_ylabel('Derivative (a.u.)')
            # axTHL.set_ylabel('THL (V)')

            plt.grid()

        # Also use pixelDACs?
        if slopeFn and THLstep == 1:
            if os.path.isfile(slopeFn):
                if slopeFn.endswith('.p'):
                    slopeDict = json.load( open(slopeFn, 'rb') )

                slopes = np.reshape(slopeDict['slope'], (16, 16))
                # Remove values with large slope...
                slopes[slopes > 3] = np.nan
                # ...and replace with mean
                slopes[np.isnan(slopes)] = np.nanmean( slopes )

                # Mean slope
                slopeMean = np.mean( slopes.flatten() )
                print('SlopeMean:')
                print(slopeMean)

                # pixelDAC commands
                # Start each pixel at 0x1f
                pixelDACs = ['1f' * 256]
                THLOffsets = [[0]*256]
                for m in reversed(range(1, int(1./slopeMean) + 1)):
                    pixelDAC = np.asarray( [0x1f + 1./slope * 1./m for slope in slopes.flatten()], dtype=int )

                    offset = np.reshape([(pixelDAC[i] - 0x1f) * slopes.flatten()[i] for i in range(len(pixelDAC))], (16, 16))
                    THLOffsets.append( offset )

                    # Check if pixel is within range
                    pixelDAC[pixelDAC < 0] = 0
                    pixelDAC[pixelDAC > 63] = 63
                    # Convert to string
                    pixelDACs.append( ''.join(['{:02x}'.format(item) for item in pixelDAC]) )

                print(pixelDACs)

                # THLOffsets = [ np.reshape([(int(1./slope * m) - 0x1f) * slope for slope in slopes.flatten()], (16, 16)) for m in range(int(1./slopeMean)) ]

                print('THLOffsets:')
                print(THLOffsets)
                print()

            else:
                pixelDACs = [None]
        else:
            pixelDACs = [None]

        # Loop over DAC values
        THLList = []
        THLVList = []
        dataList = []

        if len(self.THLEdges) == 0 or self.THLEdges[slot - 1] is None:
            print(THLlow, THLhigh, THLstep)
            THLRange = np.arange(THLlow, THLhigh)
        else:
            THLRange = np.asarray(self.THLEdges[slot - 1])
            THLRange = THLRange[np.logical_and(THLRange > THLlow, THLRange < THLhigh)]
        print(THLRange)

        # Savitzky-Golay filter the data
        # Ensure odd window length
        windowLength = int(len(THLRange[::THLstep]) / 10.)
        if not windowLength % 2:
            windowLength += 1

        # Empty bins
        '''
        self.DPXDataResetCommand(slot)
        for col in range(16):
            self.DPXWriteColSelCommand(slot, col)
            self.DPXReadBinDataDosiModeCommand(slot)
        '''

        # Create dictionary to store data
        pixelDataList = []
        deltaTimeList = []

        # Catch keyboardInterrupt exception if necessary
        measStart = time.time()

        # Select PC mode
        OMRCode = self.OMR[slot-1]
        if not isinstance(OMRCode, basestring):
            OMRCode[0] = 'PCMode'
        else:
            OMRCode = int(OMRCode, 16) & ((0b10) << 22)
            OMRCode = '%04x' % ((int(self.OMR[slot-1], 16) & ~((0b11) << 22)) | (0b10 << 22))

        try:
            # Measure time
            for THL in THLRange[::-THLstep]:
                # Convert THL to corrected THL value
                # THL = int(self.getVoltFromTHLFit(THL, slot))

                # Set threshold
                self.DPXWritePeripheryDACCommand(slot, self.peripherys[slot-1] + '%04x' % THL)
                print(self.peripherys[slot-1] + '%04x' % THL)
                self.THLs[slot - 1] = '%04x' % THL

                startTime = time.time()
                for i, pixelDAC in enumerate(pixelDACs):
                    # THL matrix for all pixels
                    THLArray = np.zeros((16, 16))
                    print(THL)
                    THLArray.fill( THL )

                    if pixelDAC:
                        self.DPXWritePixelDACCommand(slot, pixelDAC)
                        THLArray = np.asarray( THLArray ) + np.reshape(THLOffsets[i] , (16, 16))

                    THLList.append( THLArray.flatten() )

                    # Start frame 
                    # OMRCode[1] = 'ClosedShutter'
                    # self.DPXWriteOMRCommand(slot, OMRCode)
                    

                    # Wait to accumulate data
                    pixelData = np.zeros((16, 16))
                    self.DPXDataResetCommand(slot)
                    frame_start = time.time()
                    # while time.time() - frame_start < timestep:
                    # print self.DPXReadPeripheryDACCommand(slot)
                    for loop in range(100):
                        # Read out in PCMode
                        self.DPXDataResetCommand(slot)
                        time.sleep(timestep)
                        pixelData += self.DPXReadToTDatakVpModeCommand(slot)
                        # pixelData = self.DPXReadToTDataIntegrationModeCommand(slot)
                    print(pixelData)

                    # Loop over pixel columns
                    '''
                    for col in range(16):
                        self.DPXWriteColSelCommand(slot, col) 
                        data = np.sum(self.DPXReadBinDataDosiModeCommand(slot), axis=1)

                        # Gather in pixel matrix
                        pixelData.append( data )
                    '''

                    # Scale data with readout time
                    delta_time = float(time.time() - startTime)
                    pixelData = np.asarray(pixelData) / delta_time
                    deltaTimeList.append( delta_time )

                    if intPlot:
                        # Large pixels only
                        plotData = pixelData[:,range(2, 14)]

                    # After readout, new measurement begins
                    # startTime = time.time()

                    # Flatten and sum data to get total number of events
                    data = pixelData.flatten()
                    print(data)
                    pixelDataList.append( data )

                    # OMRCode[1] = 'OpenShutter'
                    # self.DPXWriteOMRCommand(slot, OMRCode)
                    # End frame

                    try:
                        # Update plot
                        pixel = -55
                        if intPlot:
                            # Plot only one pixel
                            # dataList.append( plotData.flatten()[pixel] )
                            dataList.append( np.median( plotData.flatten() ) )

                            if len(THLList[1:]) < windowLength:
                                continue

                            dataFilt = scipy.signal.savgol_filter(dataList[1:], windowLength, 3)

                            THLList_ = np.arange(len(np.asarray(THLList[1:])[:,pixel]))

                            # Cumulative
                            lineCum.set_xdata( THLList_ )
                            lineCum.set_ydata( dataFilt )
                            
                            # Derivative
                            dataDer = np.diff( dataFilt )
                            lineDer.set_xdata( np.asarray(THLList_[:-1]) + 0.5)
                            lineDer.set_ydata( dataDer )

                            # Measure threshold voltage
                            # lineTHL.set_xdata( THLList )
                            # lineTHL.set_ydata( THLVList )

                            ax.set_xlim(min(THLList_), max(THLList_))
                            ax.set_ylim(0.9*min(dataFilt), 1.1*max(dataFilt))

                            # axTHL.set_xlim(min(THLList), max(THLList))
                            # axTHL.set_ylim(0.9*min(THLVList), 1.1*max(THLVList))

                            if dataDer.size:
                                dataDer *= max(dataFilt)/max(dataDer)
                                axDer.set_xlim(min(THLList_), max(THLList_))
                                axDer.set_ylim(0.9*min(dataDer[1:]), 1.1*max(dataDer[1:]))

                            fig.canvas.draw()
                    except:
                        pass

            # PixelDataList to dictionary
            pixelDict = {i: {'THL': np.asarray(THLList)[:,i][1:], 'data': np.asarray(pixelDataList)[:,i]} for i in range(256)}
            pixelDict['time'] = deltaTimeList

            # Save to file
            self.pickleDump(pixelDict, outFn)
            raw_input('')

        except (KeyboardInterrupt, SystemExit):
            # Store data and plot in files
            print('KeyboardInterrupt-Exception: Storing data!')
            print('Measurement time: %.2f min' % ((time.time() - measStart) / 60.))

            # PixelDataList to dictionary
            pixelDict = {i: {'THL': np.asarray(THLList)[:,i], 'data': np.asarray(pixelDataList)[:,i]} for i in range(256)}

            self.pickleDump(pixelDict, outFn)
            raise

        # Reset OMR and peripherys
        self.DPXWriteOMRCommand(slot, self.OMR[slot-1])
        self.DPXWritePeripheryDACCommand(slot, self.peripherys[slot-1] + self.THLs[slot-1])

    def findNoise(self, slot, THLlow, THLhigh, THLstep, timestep, outFn='noise.json', megalix_port=None, megalix_settings=None):
        if megalix_port is not None:
            mlx = self.megalix_connect(megalix_port)
            self.megalix_set_kvpmA(mlx, megalix_settings[0], megalix_settings[1])

        # Select PCMode
        OMRCode = self.OMR[slot-1]
        if not isinstance(OMRCode, basestring):
            OMRCode[0] = 'PCMode'

        else:
            OMRCode = int(OMRCode, 16) & ((0b10) << 22)
            OMRCode = '%04x' % ((int(self.OMR[slot-1], 16) & ~((0b11) << 22)) | (0b10 << 22))

        self.DPXWriteOMRCommand(slot, OMRCode)
        self.DPXWriteColSelCommand(slot, 0)

        # Get THLRange 
        if len(self.THLEdges) == 0 or self.THLEdges[slot - 1] is None:
            print(THLlow, THLhigh, THLstep)
            THLRange = np.arange(THLlow, THLhigh)
        else:
            THLRange = np.asarray(self.THLEdges[slot - 1])
            THLRange = THLRange[np.logical_and(THLRange > THLlow, THLRange < THLhigh)]

        print('Finding noise level')
        start_time = time.time()
        if megalix_port is not None:
            self.megalix_xray_on(mlx)

        THLList, noiseList, noiseErrList = [], [], []
        try:
            THLFastStep = 10
            pixelHitList = []
            for THL in reversed(THLRange[::THLFastStep]):
                # Set threshold
                self.DPXWritePeripheryDACCommand(slot, self.peripherys[slot-1] + '%04x' % THL)

                # Measurement
                self.DPXDataResetCommand(slot)
                data = self.DPXReadToTDatakVpModeCommand(slot)

                # Count noisy pixels
                pixelHitList.append( np.count_nonzero(data) )

            print(pixelHitList)
            args = np.argwhere(np.asarray(pixelHitList) > 0).flatten()
            print(args)
            # THLRange = np.arange(THLlow, THLRange[::THLFastStep][args[-1]])
            # Skip fast scan:
            # THLRange = np.arange(THLRange[::THLFastStep][args[0] - 1], THLRange[::THLFastStep][args[-1]])
            print(THLlow, THLRange[::THLFastStep])
            self.DPXDataResetCommand(slot)

            for THL in reversed(THLRange[::THLstep]):
                # Set threshold
                self.DPXWritePeripheryDACCommand(slot, self.peripherys[slot-1] + '%04x' % THL)

                # Convert THL to corrected THL value
                THL = self.getVoltFromTHLFit(THL, slot)
                THLList.append( THL )

                # Measurement
                dataStack = []
                for idx in range(1):
                    dataStack.append( self.DPXReadToTDatakVpModeCommand(slot) )
                noiseList.append( np.mean(dataStack, axis=0) )
                noiseErrList.append( np.std(dataStack, axis=0) / np.sqrt(30) )

                # Wait before doing next iteration
                self.DPXDataResetCommand(slot)
                time.sleep(timestep)

        except (KeyboardInterrupt, SystemExit):
            if megalix_port is not None:
                self.megalix_xray_off(mlx)
            self.pickleDump({'THL': THLList, 'data': noiseList}, outFn)

        print('Elapsed time: %.2f s' % (time.time() - start_time))
        if megalix_port is not None:
            self.megalix_xray_off(mlx)
        self.pickleDump({'THL': THLList, 'data': noiseList, 'err': noiseErrList}, outFn)

    def temperatureWatch(self, slot, column='all', frames=10, energyRange=(15.e3, 125.e3, 10), fn='TemperatureToT.json', intplot=True):
        if intplot:
            # Interactive plot
            plt.ion()
            fig, ax = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw = {'width_ratios': [1, 1]})

            # ax0 label
            ax[0].set_xlabel('Time (s)')
            ax[0].set_ylabel('Temperature (DAC)')

            # ax1 label
            ax[1].set_xlabel(r'$\mu_\mathrm{ToT}$')
            ax[1].set_ylabel('Temperature (DAC)')

            # Temperature over time
            line, = ax[0].plot(np.nan, np.nan)
            # Correlation plot
            lineCorrList = []
            for i in range(256):
                lineCorr, = ax[1].plot(np.nan, np.nan, color=self.getColor('viridis', 256, i))
                lineCorrList.append( lineCorr )

        if type(self.OMR[slot-1]) is list:
            OMRCode_ = self.OMRListToHex(self.OMR[slot-1])
        else:
            OMRCode_ = self.OMR[slot-1]
        OMRCode_ = int(OMRCode_, 16)

        OMRCode_ &= ~(0b11111 << 12)
        OMRCode_ |= getattr(ds._OMRAnalogOutSel, 'Temperature')
        self.DPXWriteOMRCommand(slot, hex(OMRCode_).split('0x')[-1])

        # Init test pulses
        columnList = self.testPulseInit(slot, column=column)
        # self.maskBitsColumn(slot, column)

        # Get energy within specified range
        energyRange = np.linspace(*energyRange)

        # Number of test pulses per measurement
        Npulses = 5

        try:
            # Loop over measurements
            startTime = time.time()
            energyDict = {}

            logDict = {'time': [], 'temp': [], 'tempErr': [], 'ToT': [], 'ToTErr': [], 'energy': []}

            for energy in energyRange:
                DACval = self.getTestPulseVoltageDAC(slot, energy, True)
                self.DPXWritePeripheryDACCommand(slot, DACval)

                if intplot:
                    fig.suptitle('Test Pulse Energy: %.2f keV' % (energy*1.e-3))

                for c in range( frames ):
                    # Time
                    logDict['time'].append( time.time() - startTime )

                    # Energy
                    logDict['energy'].append( energy )

                    # Measure temperature at start
                    TList = []
                    for i in range(100):
                        TList.append( float(int(self.MCGetADCvalue(), 16)) )

                    # Temperature
                    logDict['temp'].append( np.mean(TList) )
                    logDict['tempErr'].append( np.std(TList) / np.sqrt(100) )

                    # Plot
                    if intplot:
                        line.set_xdata(logDict['time'])
                        line.set_ydata(logDict['temp'])
                        ax[0].set_xlim(min(logDict['time']), max(logDict['time']))
                        ax[0].set_ylim(0.95*min(logDict['temp']), 1.05*max(logDict['temp']))

                    # Test pulses
                    dataList = []
                    for i in range(Npulses):
                        self.DPXDataResetCommand(slot)
                        for col in columnList:
                            self.maskBitsColumn(slot, col)
                            self.DPXGeneralTestPulse(slot, 1000)

                        data = np.asarray(self.DPXReadToTDataDosiModeCommand(slot), dtype=float)[columnList].flatten()
                        # Get rid of outliers
                        data[abs(data - np.mean(data)) > 3 * np.std(data)] = np.nan
                        dataList.append( np.asarray(self.DPXReadToTDataDosiModeCommand(slot))[columnList].flatten() )

                    dataMean = np.mean(dataList, axis=0)
                    dataErr = np.std(dataList, axis=0) / np.sqrt( Npulses )

                    logDict['ToT'].append( dataMean )
                    logDict['ToTErr'].append( dataErr )

                    if intplot:
                        # Correlation plot
                        for i in range(len(columnList) * 16):
                            lineCorrList[i].set_xdata(np.asarray(logDict['ToT'])[:,i])
                            lineCorrList[i].set_ydata(logDict['temp'])

                        print(logDict['ToT'])
                        ax[1].set_xlim(0.99*np.min(logDict['ToT']), 1.01*np.max(logDict['ToT']))
                        ax[1].set_ylim(0.95*min(logDict['temp']), 1.05*max(logDict['temp']))

                        fig.canvas.draw()

            if fn:
                self.pickleDump(energyDict, fn)

        except (KeyboardInterrupt, SystemExit):
            # Store data and plot in files
            print('KeyboardInterrupt')
            if fn:
                self.pickleDump(logDict, fn)
            raise

    # If cnt = 0, loop infinitely
    def ADCWatch(self, slot, OMRAnalogOutList, cnt=0, fn='ADCWatch.json'):
        # Interactive plot
        plt.ion()
        fig, ax = plt.subplots()

        # Plot settings
        plt.xlabel('Time (s)')

        # Init plot
        OMRAnalogOutDict = {}

        lineDict = {}   # Lines of plots
        startDict = {}  # Used to normalize the ADC values

        startTime = time.time()

        for OMRAnalogOut in OMRAnalogOutList:
            dataDict = {}

            # Select AnalogOut
            print('OMR Manipulation:')
            if type(self.OMR[slot-1]) is list:
                OMRCode_ = self.OMRListToHex(self.OMR[slot-1])
            else:
                OMRCode_ = self.OMR[slot-1]
            OMRCode_ = int(OMRCode_, 16)

            OMRCode_ &= ~(0b11111 << 12)
            OMRCode_ |= getattr(ds._OMRAnalogOutSel, OMRAnalogOut)
            # OMRCode_ &= ~(0b11111 << 7)
            # OMRCode_ |= getattr(ds._OMRAnalogInSel, OMRAnalogOut)
            self.DPXWriteOMRCommand(slot, hex(OMRCode_).split('0x')[-1])

            # Data
            # Take mean of 10 samples for first point
            meanList = []
            for i in range(10):
                val = float(int(self.MCGetADCvalue(), 16))
                meanList.append( val )
            print(meanList)

            startVal = float(np.mean(meanList))
            startDict[OMRAnalogOut] = startVal
            dataDict['data'] = [startVal]

            # Time
            currTime = time.time() - startTime
            dataDict['time'] = [currTime]

            OMRAnalogOutDict[OMRAnalogOut] = dataDict

            line, = ax.plot([currTime], [1.], label=OMRAnalogOut)
            lineDict[OMRAnalogOut] = line

        plt.legend()
        plt.grid()
        fig.canvas.draw()

        print(OMRAnalogOutDict)

        # Run measurement forever if cnt == 0
        loopCnt = 0
        while (True if cnt == 0 else loopCnt < cnt):
            try:
                for i, OMRAnalogOut in enumerate(OMRAnalogOutList):

                    # Select AnalogOut
                    if type(self.OMR[slot-1]) is list:
                        OMRCode_ = self.OMRListToHex(self.OMR[slot-1])
                    else:
                        OMRCode_ = self.OMR[slot-1]
                    OMRCode_ = int(OMRCode_, 16)
                    OMRCode_ = OMRCode_ & ~(0b11111 << 12)
                    OMRCode_ |= getattr(ds._OMRAnalogOutSel, OMRAnalogOut)
                    print(hex(OMRCode_).split('0x')[-1])
                    # OMRCode_ &= ~(0b11111 << 7)
                    # OMRCode_ |= getattr(ds._OMRAnalogInSel, OMRAnalogOut)
                    self.DPXWriteOMRCommand(slot, hex(OMRCode_).split('0x')[-1])

                    # Time 
                    currTime = time.time() - startTime

                    # Data
                    currVal = float(int(self.MCGetADCvalue(), 16))
                    print(currVal, startDict[OMRAnalogOut])

                    # Set values in dict
                    OMRAnalogOutDict[OMRAnalogOut]['time'].append(currTime)

                    OMRAnalogOutDict[OMRAnalogOut]['data'].append(currVal)

                    # Refresh line
                    lineDict[OMRAnalogOut].set_xdata(OMRAnalogOutDict[OMRAnalogOut]['time'])

                    if len(OMRAnalogOutDict[OMRAnalogOut]['data']) > 21:
                        lineDict[OMRAnalogOut].set_ydata(scipy.signal.savgol_filter(np.asarray(OMRAnalogOutDict[OMRAnalogOut]['data'])/startDict[OMRAnalogOut], 21, 5))
                    else: 
                        lineDict[OMRAnalogOut].set_ydata(np.asarray(OMRAnalogOutDict[OMRAnalogOut]['data'])/startDict[OMRAnalogOut])

                # Scale plot axes
                plt.xlim(min([min(OMRAnalogOutDict[key]['time']) for key in OMRAnalogOutDict.keys()]), max([max(OMRAnalogOutDict[key]['time']) for key in OMRAnalogOutDict.keys()]))

                plt.ylim(min([min(np.asarray(OMRAnalogOutDict[key]['data'])/startDict[OMRAnalogOut]) for key in OMRAnalogOutDict.keys()]), max([max(np.asarray(OMRAnalogOutDict[key]['data'])/startDict[OMRAnalogOut]) for key in OMRAnalogOutDict.keys()]))

                fig.canvas.draw()
                loopCnt += 1

            except (KeyboardInterrupt, SystemExit):
                # Store data and plot in files
                print('KeyboardInterrupt')
                if fn:
                    self.pickleDump(OMRAnalogOutDict, fn)
                raise

        if fn:
            self.pickleDump(OMRAnalogOutDict, fn)

    def measureTHL(self, slot, fn=None, plot=False, use_gui=False):
        if not fn:
            fn = self.THL_CALIB_FILE
        gen = self.measureADC_func(slot, AnalogOut='V_ThA', perc=False, ADChigh=8191, ADClow=0, ADCstep=1, N=1, fn=fn, plot=plot, use_gui=use_gui)
        if use_gui:
            return gen
        *_, out_dict = gen
        return out_dict

    def measureADC(self, slot, AnalogOut='V_ThA', perc=False, ADChigh=8191,
        ADClow=0, ADCstep=1, N=1, fn=None, plot=False, use_gui=False):
        gen = self.measureADC_func(slot=slot, AnalogOut=AnalogOut,
            perc=perc, ADChigh=ADChigh, ADClow=ADClow, ADCstep=ADCstep,
            N=N, fn=fn, plot=plot, use_gui=use_gui)
        if use_gui:
            return gen
        *_, out_dict = gen
        return out_dict

    def measureADC_func(self, slot, AnalogOut='V_ThA', perc=False, ADChigh=8191, ADClow=0, ADCstep=1, N=1, fn=None, plot=False, use_gui=False):
        # Display execution time at the end
        startTime = time.time()

        # Select AnalogOut
        if type(self.OMR[slot-1]) is list:
            OMRCode_ = self.OMRListToHex(self.OMR[slot-1])
        else:
            OMRCode_ = self.OMR[slot-1]
        OMRCode_ = int(OMRCode_, 16)

        OMRCode_ &= ~(0b11111 << 12)
        OMRCode_ |= getattr(ds._OMRAnalogOutSel, AnalogOut)
        self.DPXWriteOMRCommand(slot, hex(OMRCode_).split('0x')[-1])

        # Get peripherys
        dPeripherys = self.splitPerihperyDACs(self.peripherys[slot-1] + self.THLs[0], perc=perc)

        if AnalogOut == 'V_cascode_bias':
            AnalogOut = 'V_casc_reset'
        elif AnalogOut == 'V_TPref_fine':
            AnalogOut = 'V_tpref_fine'
        elif AnalogOut == 'V_per_bias':
            AnalogOut = 'V_casc_preamp'

        ADCList = np.arange(ADClow, ADChigh, ADCstep)
        ADCVoltMean = []
        ADCVoltErr = []
        print('Measuring ADC!')

        if not use_gui:
            loop_range = tqdm(ADCList)
        else:
            loop_range = ADCList
        for cnt, ADC in enumerate( loop_range ):
            # Set threshold
            # self.DPXWritePeripheryDACCommand(slot, self.peripherys + '%04x' % ADC)

            # Set value in peripherys
            dPeripherys[AnalogOut] = ADC
            code = self.periheryDACsDictToCode(dPeripherys, perc=perc)
            self.peripherys[slot-1] = code[:-4]
            self.DPXWritePeripheryDACCommand(slot, code)

            # Measure multiple times
            ADCValList = []
            for i in range(N):
                ADCVal = float(int(self.MCGetADCvalue(), 16))
                ADCValList.append( ADCVal )

            ADCVoltMean.append( np.mean(ADCValList) )
            ADCVoltErr.append( np.std(ADCValList)/np.sqrt(N) )

            if use_gui:
                yield {'Volt': ADCVoltMean[-1], 'ADC': int(ADC)}

        if plot and not use_gui:
            plt.errorbar(ADCList, ADCVoltMean, yerr=ADCVoltErr, marker='x')
            plt.show()

        # Sort lists
        ADCVoltMeanSort, ADCListSort = zip(*sorted(zip(ADCVoltMean, ADCList)))
        out_dict = {'Volt': ADCVoltMeanSort, 'ADC': ADCListSort}
        print(out_dict)
        if not use_gui:
            # print THLListSort
            if plot:
                plt.plot(ADCVoltMeanSort, ADCListSort)
                plt.show()

            if fn:
                self.pickleDump(out_dict, fn, overwrite=True)
            else:
                self.pickleDump(out_dict, 'ADCCalib.json', overwrite=True)

            print('Execution time: %.2f min' % ((time.time() - startTime)/60.))
        yield out_dict

    def thresholdEqualizationConfig(self, configFn, reps=1, THL_offset=20, I_pixeldac=0.21, intPlot=False, resPlot=True):
        for slot in self.slot_range:
            if configFn[slot - 1] is None:
                 continue
            pixelDAC, THL, confMask = self.thresholdEqualization(slot=slot, reps=reps, THL_offset=THL_offset, 
                    I_pixeldac=I_pixeldac, intPlot=intPlot, resPlot=resPlot)

            # Set values
            self.pixelDAC[slot - 1] = pixelDAC
            self.THLs[slot - 1] = THL
            self.confBits[slot - 1] = confMask

            print(self.peripherys)
            self.writeConfig(configFn[slot - 1], slot=slot)

    def thresholdEqualization(self, slot, reps=1, THL_offset=20,
        I_pixeldac=0.21, intPlot=False, resPlot=True, use_gui=False):
        gen = self.thresholdEqualization_func(slot=slot, reps=reps,
            THL_offset=THL_offset, I_pixeldac=I_pixeldac, intPlot=intPlot,
            resPlot=resPlot, use_gui=use_gui)
        if use_gui:
            return gen
        *_, out_dict = gen
        return out_dict

    def thresholdEqualization_func(self, slot, reps=1, THL_offset=20, I_pixeldac=0.21, intPlot=False, resPlot=True, use_gui=False):
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

        print('== Threshold equalization of detector %d ==' % slot)
        if use_gui:
            yield {'stage': 'Init'}

        # Set PC Mode in OMR in order to read kVp values
        # If OMR code is list
        if isinstance(self.OMR[slot-1], (list,)):
            OMRCode = self.OMR[slot-1]
            OMRCode[0] = 'PCMode'
            self.DPXWriteOMRCommand(slot, OMRCode)
        else:
            OMRCode = '%04x' % ((int(self.OMR[slot-1], 16) & ~((0b11) << 22)) | (0b10 << 22))
            self.DPXWriteOMRCommand(slot, OMRCode)
        self.OMR[slot-1] = OMRCode

        # Linear dependence:
        # Start and end points are sufficient
        pixelDACs = ['00', '3f']

        # Set I_pixeldac
        if I_pixeldac is not None:
            d = self.splitPerihperyDACs(self.peripherys[slot-1] + self.THLs[0], perc=True)
            d['I_pixeldac'] = I_pixeldac
            code = self.periheryDACsDictToCode(d, perc=True)
            self.peripherys[slot-1] = code[:-4]
            self.DPXWritePeripheryDACCommand(slot, code)

            # Perform equalization
            if I_pixeldac > 0.2:
                # Nonlinear dependence
                pixelDACs = ['%02x' % int(num) for num in np.linspace(0, 63, 9)]

        # Return status to GUI
        if use_gui:
            yield {'stage': 'THL_pre_start'}
            countsDict_gen = self.getTHLLevel(slot, THLRange, pixelDACs, reps, intPlot=False, use_gui=True)
            for res in countsDict_gen:
                if 'status' in res.keys():
                    yield {'stage': 'THL_pre', 'status': np.round(res['status'], 4)}
                elif 'DAC' in res.keys():
                    yield {'stage': 'THL_pre_loop_start', 'status': res['DAC']}
                else:
                    countsDict = res['countsDict']
            # countsDict = self.getTHLLevel_gui(slot, THLRange, pixelDACs, reps)
        else:
            gen = self.getTHLLevel(slot, THLRange, pixelDACs, reps, intPlot, use_gui=False)
            *_, countsDict = gen
        gaussDict, noiseTHL = self.getNoiseLevel(countsDict, THLRange, pixelDACs, noiseLimit)

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
                if resPlot and not use_gui:
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

                    print(adjust[pixel])
                
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
        if use_gui:
            yield {'stage': 'THL_start'}
            countsDict_gen = self.getTHLLevel(slot, THLRange, pixelDACNew, reps, intPlot=False, use_gui=True)
            for res in countsDict_gen:
                if 'status' in res.keys():
                    yield {'stage': 'THL', 'status': np.round(res['status'], 4)}
                elif 'DAC' in res.keys():
                    yield {'stage': 'THL_loop_start', 'status': res['DAC']}
                else:
                    countsDictNew = res['countsDict']
        else:
            countsDict_gen = self.getTHLLevel(slot, THLRange, pixelDACNew, reps, intPlot)
            *_, countsDictNew = countsDict_gen
        gaussDictNew, noiseTHLNew = self.getNoiseLevel(countsDictNew, THLRange, pixelDACNew, noiseLimit)

        # Transform values to indices
        meanDictNew, noiseTHLNew = self.valToIdx(slot, [pixelDACNew], THLRange, gaussDictNew, noiseTHLNew)

        # Plot the results of the equalization
        if resPlot and not use_gui:
            bins = np.linspace(min(gaussDict['3f']), max(gaussDict['00']), 100)

            for pixelDAC in ['00', '3f']:
                plt.hist(gaussDict[pixelDAC], bins=bins, label='%s' % pixelDAC, alpha=0.5)

            plt.hist(gaussDictNew[pixelDACNew], bins=bins, label='After equalization', alpha=0.5)

            plt.legend()

            plt.xlabel('THL')
            plt.ylabel('Counts')

            plt.show()

        '''
        # Find center point of next sawtooth in THL curve
        # in negative direction of THL
        THLNew = THLRange[int(mean)]

        # Calculate centers of edge intervals
        edgesCenter = np.asarray([ 0.5 * (self.THLEdgesHigh[k] + self.THLEdgesLow[k]) for k in range(len(self.THLEdgesLow)) ])

        # Eliminate values larger than THLNew
        edgesCenter = edgesCenter[edgesCenter <= THLNew]

        # Get closest center value to THLNew
        THLNew = min(edgesCenter, key=lambda x:abs(x - THLNew))
        '''

        if use_gui:
            yield {'stage': 'confBits'}

        # Create confBits
        confMask = np.zeros((16, 16)).astype(str)
        confMask.fill('00')

        # Check for noisy pixels after equalization. If there still are any left, reduce THL value even further
        # to reduce their amount. If a pixel is really noisy, it shouldn't change its state even when THL is lowered.
        # Therefore, if pixels don't change their behavior after 5 decrements of THL, switch those pixels off.
        if use_gui:
            yield {'stage': 'noise'}

        THLNew = int(np.mean(gaussDictNew[pixelDACNew]))
        
        if self.THLEdges[slot-1] is not None:
            print('Getting rid of noisy pixels...')
            self.DPXWritePeripheryDACCommand(slot, self.peripherys[slot-1] + ('%04x' % int(THLNew)))

            pc_noisy_last = []
            noisy_count = 0
            while True:
                pc_data = np.zeros((16, 16))
                self.DPXDataResetCommand(slot)
                for cnt in range(30):
                    pc_data += np.asarray( self.DPXReadToTDatakVpModeCommand(slot) )
                    self.DPXDataResetCommand(slot)
                pc_sum = pc_data.flatten()

                # Noisy pixels
                pc_noisy = np.argwhere(pc_sum > 0).flatten()
                print('THL: %d' % THLNew)
                print('Noisy pixels index:')
                print(pc_noisy)
                # Compare with previous read-out
                noisy_common = sorted(list(set(pc_noisy_last) & set(pc_noisy)))
                print( noisy_common )
                print(len(pc_noisy), len(noisy_common))
                print(noisy_count)
                print()

                # If noisy pixels don't change, increase counter
                # if len(list(set(noisy_common) & set(pc_noisy))) > 0:
                if len(pc_noisy) == len(noisy_common) and len(pc_noisy) == len(pc_noisy_last):
                    noisy_count += 1
                pc_noisy_last = np.array(pc_noisy, copy=True)
                
                # If noisy pixels don't change for 5 succeeding steps, interrupt
                if noisy_count == 3 or not len(pc_noisy):
                    break
                else:
                    # Reduce THL by 1
                    THLNew = self.THLEdges[slot-1][list(self.THLEdges[slot-1]).index(THLNew) - 1]
                    self.DPXWritePeripheryDACCommand(slot, self.peripherys[slot-1] + ('%04x' % int(THLNew)))

            # Subtract additional offset to THL
            THLNew = self.THLEdges[slot-1][list(self.THLEdges[slot-1]).index(THLNew) - THL_offset]

            # Switch off noisy pixels
            confMask[(pc_sum > 10).reshape((16, 16))] = '%02x' % (0b1 << 2)
        else:
            THLNew = int(np.mean(gaussDictNew[pixelDACNew]) - THL_offset)
            confMask[abs(noiseTHLNew[pixelDACNew] - mean) > 10] = '%02x' % (0b1 << 2)

        # Transform into string
        confMask = ''.join(confMask.flatten())

        print()
        print('Summary:')
        print('pixelDACs:', pixelDACNew)
        print('confMask:', confMask)
        print('Bad pixels:', np.argwhere((abs(noiseTHLNew[pixelDACNew] - mean) > 10) == True))
        print('THL:', '%04x' % int(THLNew))

        # Restore OMR values
        self.DPXWriteOMRCommand(slot, self.OMR[slot-1])

        if use_gui:
            yield {'stage': 'finished', 
                'pixelDAC': pixelDACNew, 
                'THL': '%04x' % int(THLNew), 
                'confMask': confMask}
        else:
            yield pixelDACNew, '%04x' % int(THLNew), confMask

    def THSEqualization(self, slot):
        # Set PC Mode in OMR in order to read kVp values
        # If OMR code is list
        if isinstance(self.OMR[slot-1], (list,)):
            OMRCode = self.OMR[slot-1]
            OMRCode[0] = 'PCMode'
            self.DPXWriteOMRCommand(slot, OMRCode)
        else:
            self.DPXWriteOMRCommand(slot, (int(self.OMR[slot-1], 16) & ~((0b11) << 22)) | (0b10 << 22))

        # Get THLRange
        THLlow, THLhigh = 4000, 8000
        # THLRange = np.arange(THLlow, THLhigh, 1) 
        THLRange = np.asarray(self.THLEdges[slot - 1])
        THLRange = THLRange[np.logical_and(THLRange >= THLlow, THLRange <= THLhigh)]

        # Transform peripheryDAC values to dictionary
        print(self.peripherys[slot-1] + self.THLs[0])
        d = self.splitPerihperyDACs(self.peripherys[slot-1] + self.THLs[0], perc=True)

        minIpixeldac, maxIpixeldac, NIpixeldac = 0, 1., 15
        minDAC, maxDAC, NDAC = 0, 63, 15

        I_pixeldacList = np.linspace(minIpixeldac, maxIpixeldac, NIpixeldac)
        pixelDACs = ['%02x' % elm for elm in np.asarray(np.linspace(minDAC, maxDAC, NDAC), dtype=int)]
        stdList = []

        # Store results in dicts
        gaussDicts = {}

        for idx, I_pixeldac in enumerate( I_pixeldacList ):
            # Change pixelDAC step value
            d['I_pixeldac'] = I_pixeldac
            code = self.periheryDACsDictToCode(d, perc=True)
            print(code)

            # Set peripheryDACs
            self.peripherys[slot-1] = code[:-4]
            self.DPXWritePeripheryDACCommand(slot, code)

            # Set pixelDACs to maximum value
            # pixelDACs = ['00']

            countsDict = self.getTHLLevel(slot, THLRange, pixelDACs, reps, intPlot)
            gaussDict, noiseTHL = self.getNoiseLevel(countsDict, THLRange, pixelDACs, noiseLimit)
            meanDict, noiseTHL = self.valToIdx(slot, pixelDACs, THLRange, gaussDict, noiseTHL)
            gaussCorrDict = {pixelDAC: [self.getVoltFromTHLFit(elm, slot) if elm else np.nan for elm in gaussDict[pixelDAC]] for pixelDAC in pixelDACs}

            # Add to I_pixeldac-dict
            gaussDicts[I_pixeldac] = gaussCorrDict

            # Add standard deviation to list
            # stdList.append( np.std(dic) )

        # Save to file
        self.pickleDump({'mean': meanMatrix, 'sigma': sigmaMatrix}, 'pixelDAC.p')

        return

    # Get information about the slope THL/pixelDAC for each pixel.
    # Notice that the slope is only linear for I_pixeldac values 
    # about less than 20%. 
    def getPixelSlopes(self, slot, I_pixeldac=0.2):
        if not isinstance(I_pixeldac, list, ):
            I_pixeldacList = [I_pixeldac]
        else:
            I_pixeldacList = I_pixeldac

        assert max(I_pixeldacList) <= 0.2, "I_pixeldac has to be set less than or equal 20% to ensure linearity of the pixels."

        # Set PC Mode in OMR in order to read kVp values
        # If OMR code is list
        if not isinstance(self.OMR[slot-1], (list,)):
            OMRCode = self.OMR[slot-1]
            OMRCode[0] = 'PCMode'
            self.DPXWriteOMRCommand(slot, OMRCode)
        else:
            self.DPXWriteOMRCommand(slot, (int(self.OMR[slot-1], 16) & ~((0b11) << 22)) | (0b10 << 22))

        # Get THLRange
        THLlow, THLhigh = 4000, 8000
        # THLRange = np.arange(THLlow, THLhigh, 1) 
        THLRange = np.asarray(self.THLEdges[slot - 1])
        THLRange = THLRange[np.logical_and(THLRange >= THLlow, THLRange <= THLhigh)]

        # Only use first and last pixelDAC values
        pixelDACs = ['00', '3f']

        # Transform peripheryDAC values to dictionary
        print(self.peripherys[slot-1] + self.THLs[0])
        dPeripherys = self.splitPerihperyDACs(self.peripherys[slot-1] + self.THLs[0], perc=True)

        # Init plot
        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_axes([0.1, 0.14, 0.7, 0.8])
        # Add colorbar later
        axCBar = fig.add_axes([0.85, 0.1, 0.05, 0.8])

        slopeList = []
        # Loop over I_pixeldacs
        for idx, I_pixeldac in enumerate( I_pixeldacList ):
            # Set I_pixeldac in peripherys
            dPeripherys['I_pixeldac'] = I_pixeldac
            code = self.periheryDACsDictToCode(dPeripherys, perc=True)
            self.peripherys[slot-1] = code[:-4]
            self.DPXWritePeripheryDACCommand(slot, code)

            # Get noise levels for each pixel
            countsDict = self.getTHLLevel(slot, THLRange, pixelDACs, 1, False)
            gaussDict, noiseTHL = self.getNoiseLevel(countsDict, THLRange, pixelDACs, 1)
            meanDict, noiseTHL = self.valToIdx(slot, pixelDACs, THLRange, gaussDict, noiseTHL)

            # Get slope for each pixel
            slope = (noiseTHL['00'] - noiseTHL['3f']) / 63.
            slopeList.append( slope )
            offset = noiseTHL['00']

            slope = np.nan_to_num( slope )
            print(slope)

            # Show histogram of slopes
            color = self.getColor('viridis', len(I_pixeldacList), idx)
            hist, bins = np.histogram(slope) # , bins=np.linspace(0., .15, 30))
            ax.plot(bins[:-1], hist, drawstyle='steps', color=color, alpha=0.5)
            ax.axvline(x=np.nanmean(slope))
            ax.axvline(x=np.nanmean(slope)+np.nanstd(slope), ls='--', color=color)
            ax.axvline(x=np.nanmean(slope)-np.nanstd(slope), ls='--', color=color)

        # Add colorbar
        self.getColorBar(axCBar, I_pixeldacList[0]*100, I_pixeldacList[-1]*100, N=len(I_pixeldacList), label=r'$I_\mathrm{pixelDAC}$ (%)')

        plt.show()
        if len(slopeList) == 1:
            return slopeList[0]
        else:
            return slopeList
