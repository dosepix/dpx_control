from __future__ import print_function

import time
import numpy as np
import os
import os.path
import matplotlib.pyplot as plt
import scipy.signal
import scipy.optimize
import tqdm
import multiprocessing as mp
import itertools
    
import dpx_control.dpx_settings as ds

try:
  basestring
except NameError:
  basestring = str

class DosepixMulti(object):
    def __init__(self, dpxObjects, slotList):
        self.dpxObjects = np.asarray( dpxObjects )
        print( self.dpxObjects )
        print(slotList)
        self.slotList = slotList
        self.N_board = len(self.dpxObjects)

    # === MEASUREMENT FUNCTIONS ===
    def measureDose(self, frames=1000, sync=True):
        """Perform measurement in DosiMode for multiple read-out boards at once

        Parameters
        ----------
        dpxObjects: list
            Contains the dpx objects of the previously connected boards
        slot : list
            Use the specified slots for the used read-out boards. List can 
            either contain a single detector or multiple ones per board. If 
            using the latter, specify the slots via a list.

        Returns
        -------
        out : dict
            Measurement data. Keys `temp` or `dose` are only available if
            `logTemp` or `conversion_factors` are set during function call.
        """

        # Apply settings necessary for DosiMode to all detectors
        self.initMeasureDose_multi()

        func = self.readFrameDosi_multi
        start = time.time()
        print('=== Starting Dose measurement ===')
        frame_shape = (16, 16, 16)
        print(frame_shape)
        if sync:
            out, timeLog = self.measureSync(func, frames, frame_shape)
        else:
            out = self.measureAsync(func, frames, frame_shape)
        self.dpxObjects[0].pickleDump(out, 'doseMeasurementMulti.json')

        print('Done!')
        print('Elapsed time: %.1f s' % (time.time() - start))

    def measureToT(self, frames, outDir='measureToTMulti/', sync=True):
        # Apply settings necessary for DosiMode to all detectors
        self.initMeasureToT_multi(logTemp=False)

        func = self.readFrameToT_multi
        start = time.time()
        print('=== Starting ToT measurement ===')
        frame_max = 10000
        if frames is None:
            frame_chunk = itertools.repeat(frame_max)

        else:
            # Split frames into chunks
            frame_chunk = [frame_max] * (frames // frame_max)
            if frames % frame_max:
                frame_chunk.append( frames % frame_max )
    
        # Loop over frame chunks and save output for each
        outDir = self.dpxObjects[0].makeDirectory(outDir)
        try:
            for fr in frame_chunk:
                if sync:
                    out, timeLog = self.measureSync(func, fr, [256])
                    self.dpxObjects[0].pickleDump(timeLog, outDir + 'time.json')
                else:
                    out = self.measureAsync(func, fr, [256])
                self.dpxObjects[0].pickleDump(out, outDir + 'data.json')
        except (KeyboardInterrupt, SystemExit):
            print('KeyboardInterrupt-Exception')
            pass

        print('Done!')
        print('Elapsed time: %.1f s' % (time.time() - start))
        return

    # === INIT FUNCTIONS ===
    def initMeasureDose_multi(self, logTemp=False):
        # Set Dosi Mode in OMR
        for idx, dpx in enumerate(self.dpxObjects):
            slot = self.slotList[idx]
            # If OMR code is list
            for sl in slot:
                OMRCode = dpx.OMR[sl - 1]
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

                    if type(dpx.OMR[sl - 1]) is list:
                        OMRCode_ = dpx.OMRListToHex(dpx.OMR[sl - 1])
                    else:
                        OMRCode_ = dpx.OMR[sl - 1]
                    OMRCode_ = int(OMRCode_, 16)

                    OMRCode_ &= ~(0b11111 << 12)
                    OMRCode_ |= getattr(ds._OMRAnalogOutSel, 'Temperature')
                    dpx.DPXWriteOMRCommand(1, hex(OMRCode_).split('0x')[-1])

                # Set OMR
                dpx.DPXWriteOMRCommand(sl, OMRCode)
                dpx.DPXDataResetCommand(sl)

            # Initial reset 
            dpx.clearBins(slot)

    def initMeasureToT_multi(self, logTemp=False):
        for idx, dpx in enumerate(self.dpxObjects):
            slot = self.slotList[idx]
            # Check which slots to read out
            if isinstance(slot, int):
                slotList = [slot]
            elif not isinstance(slot, basestring):
                slotList = slot

            # Set Dosi Mode in OMR
            # If OMR code is list
            for slot in slotList:
                OMRCode = dpx.OMR[slot-1]
                if not isinstance(OMRCode, basestring):
                    OMRCode[0] = 'DosiMode'
                else:
                    OMRCode = '%04x' % ((int(OMRCode, 16) & ~((0b11) << 22)))

                # Set mode in slots
                dpx.DPXWriteOMRCommand(slot, OMRCode)

            # Get OMR
            sl = 1
            if type(dpx.OMR[sl-1]) is list:
                OMRCode_ = dpx.OMRListToHex(self.OMR[sl-1])
            else:
                OMRCode_ = dpx.OMR[sl-1]
            OMRCode_ = int(OMRCode_, 16)

            # Set ADC out to temperature if requested
            if logTemp:
                tempDict = {'temp': [], 'time': []}
                OMRCode_ &= ~(0b11111 << 12)
                OMRCode_ |= getattr(ds._OMRAnalogOutSel, 'V_ThA') # 'Temperature')
                for slot in slotList:
                    dpx.DPXWriteOMRCommand(slot, hex(OMRCode_).split('0x')[-1])
    
    def readFrameDosi_multi(self, dpx_idx, output):
        dpx = self.dpxObjects[dpx_idx]
        slot = self.slotList[dpx_idx]

        outList = [[]] * len(slot)
        for sl_idx, sl in enumerate( slot ):
            # Loop over columns
            for col in range(16):
                dpx.DPXWriteColSelCommand(sl, 15 - col)
                out = np.asarray( dpx.DPXReadBinDataDosiModeCommand(sl), dtype=float )
                outList[sl_idx].append( out )
                # output.put( (dpx_idx, out) )
        output_np = np.frombuffer(output)
        np.copyto(output_np, np.asarray(outList).flatten())
        return

    def readFrameToT_multi(self, dpx_idx, output):
        dpx = self.dpxObjects[dpx_idx]
        slot = self.slotList[dpx_idx]
        outList = []
        for sl in slot:
            # Read data
            data = dpx.DPXReadToTDataDosiModeCommand(sl)

            # Reset data registers
            dpx.DPXDataResetCommand(sl)

            # Append to out
            data = data.flatten()
            outList.append( data )

        output_np = np.frombuffer(output)
        np.copyto(output_np, np.asarray(outList).flatten())
        return

    # === MAIN FUNCTIONS ===
    def measureAsync(self, func, frames, shape):
        # Create output arrays
        self.outputs = []
        for idx in range(self.N_board):
            self.outputs.append( mp.RawArray('d', len(self.slotList[idx] * np.prod(shape))) )
        frameOut = {dpx_idx: [] for dpx_idx in range(self.N_board)}

        # Asynchronous
        # Start
        procs = []
        for idx in range(len(self.slotList)):
            proc = mp.Process(target=func, args=[idx, self.outputs[idx]])
            proc.start()
            procs.append( proc )

        # Loop over frames
        if frames is None:
            frame_cnt = [True] * self.N_board
        else:
            frame_cnt = [frames] * self.N_board
        try:
            while sum(frame_cnt) > 0:
                for proc_idx, proc in enumerate(procs):
                    proc.join(timeout=0)
                    if not proc.is_alive() and frame_cnt[proc_idx] > 0:
                        out = np.frombuffer(self.outputs[proc_idx]).reshape((len(self.slotList[proc_idx]), *shape))
                        frameOut[proc_idx].append( out )

                        procs[proc_idx] = mp.Process(target=func, args=[proc_idx, self.outputs[proc_idx]])
                        procs[proc_idx].start()
                        if frames is not None:
                            frame_cnt[proc_idx] -= 1
                    else:
                        continue
        except (KeyboardInterrupt, SystemExit):
            print('KeyboardInterrupt-Exception')
            pass

        return frameOut

    def measureSync(self, func, frames, shape):
        frameOut = {dpx_idx: [] for dpx_idx in range(self.N_board)}

        # Loop length
        if frames is None:
            # Infinite loop
            frameRange = itertools.count()
        else:
            frameRange = tqdm.trange(frames)

        try:
            timeLog = []
            # Start measurement
            start_time = time.time()
            for frame in frameRange:
                # Create output arrays
                outputs = []
                for idx in range(self.N_board):
                    outputs.append( mp.RawArray('d', len(self.slotList[idx] * np.prod(shape))) )

                procs = []
                for idx in range(len(self.slotList)):
                    proc = mp.Process(target=func, args=[idx, outputs[idx]])
                    proc.start()
                    procs.append( proc )

                # Synchronous
                for proc in procs:
                    proc.join()

                # Collect results
                for idx in range(self.N_board):
                    out = np.frombuffer(outputs[idx]).reshape((len(self.slotList[idx]), *shape))
                    frameOut[idx].append( out )

                timeLog.append( time.time() - start_time )
                # if not (frame % 10):
                #    print( '%.2f Hz' % ((frame + 1) / (time.time() - start_time)) )
        except (KeyboardInterrupt, SystemExit):
            print('KeyboardInterrupt-Exception')
            pass

        return frameOut, timeLog

