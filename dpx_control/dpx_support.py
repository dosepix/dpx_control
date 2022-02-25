from __future__ import print_function
import scipy.constants
import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.optimize
from tqdm import tqdm

try:
  basestring
except NameError:
  basestring = str

class DPX_support(object):
    def fillParamDict(self, paramDict):
        # Get mean factors
        meanDict = {}

        if 'h' in paramDict[paramDict.keys()[0]].keys():
            keyList = ['a', 'c', 'b', 'h', 'k', 't']
        else:
            keyList = ['a', 'c', 'b', 't']

        for val in keyList:
            valList = []
            for idx in paramDict.keys():
                valList.append( paramDict[idx][val] )
            valList = np.asarray(valList)
            valList[abs(valList) == np.inf] = np.nan
            meanDict[val] = np.nanmean(valList)

        # Find non existent entries and loop
        for pixel in set(np.arange(256)) - set(paramDict.keys()):
            paramDict[pixel] = {val: meanDict[val] for val in keyList}

        return paramDict

    def getTHLfromVolt(self, V, slot):
        return self.THLCalib[slot - 1][abs(V - self.voltCalib).argmin()]

    def getVoltFromTHL(self, THL, slot):
        return self.voltCalib[slot - 1][np.argwhere(self.THLCalib == THL)]

    def getVoltFromTHLFit(self, THL, slot):
        if len(self.THLEdgesLow) == 0 or self.THLEdgesLow[slot - 1] is None:
            return THL

        edges = zip(self.THLEdgesLow[slot - 1], self.THLEdgesHigh[slot - 1])
        for i, edge in enumerate(edges):
            if THL >= edge[0] and THL <= edge[1]:
                break
        # else:
        #     return None

        params = self.THLFitParams[slot - 1][i]
        if i == 0:
            return self.erfStdFit(THL, *params)
        else:
            return self.linearFit(THL, *params)

    def EnergyToToTSimple(self, x, a, b, c, t, h=1, k=0):
        return np.where(x >= self.getTHL(a, b, c, t), a*x + b + float(c)/(x - t), 0)

    def ToTtoEnergySimple(self, x, a, b, c, t, h=1, k=0):
        return 1./(2*a) * ( t*a + x - b + np.sqrt((b + t*a - x)**2 - 4*a*c) )

    def getTHL(self, a, b, c, t):
        return 1./(2*a) * ( t*a - b + np.sqrt((b + t*a)**2 - 4*a*c) )

    def energyToToTFitAtan(self, x, a, b, c, d):
        return np.where(x > b, a*(x - b) + c*np.arctan((x - b)/d), 0)

    def energyToToTFitHyp(self, x, a, b, c, d):
        return np.where(x > d, a*x + b + float(c)/(x - d), 0)

    def THLCalibToEdges(self, THLDict):
        volt, thl = THLDict['Volt'], THLDict['ADC']

        # Sort by THL
        thl, volt = zip(*sorted(zip(thl, volt)))

        # Find edges by taking derivative
        diff = abs(np.diff(volt))
        if self.eye_lens:
            thres = 100
        else:
            thres = 200
        edges = np.argwhere(diff > thres).flatten() + 1

        # Store fit results in dict
        d = {}

        edges = list(edges)
        edges.insert(0, 0)
        edges.append( 8190 )

        THLEdgesLow, THLEdgesHigh = [0], []

        x1 = np.asarray( thl[edges[0]:edges[1]] )
        y1 = np.asarray( volt[edges[0]:edges[1]] )
        popt1, pcov1 = scipy.optimize.curve_fit(self.erfStdFit, x1, y1)
        d[0] = popt1

        for i in range(1, len(edges) - 2):
            # Succeeding section
            x2 = np.asarray( thl[edges[i]:edges[i+1]] )
            y2 = np.asarray( volt[edges[i]:edges[i+1]] )

            popt2, pcov2 = scipy.optimize.curve_fit(self.linearFit, x2, y2)
            m1, m2, t1, t2 = popt1[0], popt2[0], popt1[1], popt2[1]
            d[i] = popt2

            # Get central position
            # Calculate intersection to get edges
            if i == 1:
                    Vcenter = 0.5*(self.erfStdFit(edges[i], *popt1) + self.linearFit(edges[i], m2, t2))
                    THLEdgesHigh.append( scipy.optimize.fsolve(lambda x: self.erfStdFit(x, *popt1) - Vcenter, 100)[0] )
            else:
                    Vcenter = 1./(m1 + m2) * (2*edges[i]*m1*m2 + t1*m1 + t2*m2)
                    THLEdgesHigh.append( (Vcenter - t1)/m1 )

            THLEdgesLow.append( (Vcenter - t2)/m2 )
            popt1, pcov1 = popt2, pcov2

        THLEdgesHigh.append( 8190 )

        return THLEdgesLow, THLEdgesHigh, d

    def valToIdx(self, slot, pixelDACs, THLRange, gaussDict, noiseTHL):
        # Transform values to indices
        meanDict = {}
        for pixelDAC in pixelDACs:
            d = np.asarray([self.getVoltFromTHLFit(elm, slot) if elm else np.nan for elm in gaussDict[pixelDAC] ], dtype=np.float)
            meanDict[pixelDAC] = np.nanmean(d)

            for pixelX in range(16):
                for pixelY in range(16):
                    elm = noiseTHL[pixelDAC][pixelX, pixelY]
                    if elm:
                        noiseTHL[pixelDAC][pixelX, pixelY] = self.getVoltFromTHLFit(elm, slot)
                    else:
                        noiseTHL[pixelDAC][pixelX, pixelY] = np.nan

        return meanDict, noiseTHL

    def getTHLLevel(self, slot, THLRange, pixelDACs=['00', '3f'], reps=1, intPlot=False, use_gui=False):
        # Force no plot if GUI is used
        if use_gui:
            intPlot = False

        countsDict = {}

        # Interactive plot showing pixel noise
        if intPlot:
            plt.ion()
            fig, ax = plt.subplots()

            plt.xlabel('x (pixel)')
            plt.ylabel('y (pixel)')
            im = ax.imshow(np.zeros((16, 16)), vmin=0, vmax=255)

        if isinstance(pixelDACs, basestring):
            pixelDACs = [pixelDACs]

        # Loop over pixelDAC values
        for pixelDAC in pixelDACs:
            countsDict[pixelDAC] = {}
            print('Set pixel DACs to %s' % pixelDAC)
            
            # Set pixel DAC values
            if len(pixelDAC) > 2:
                pixelCode = pixelDAC
            else:
                pixelCode = pixelDAC*256
            self.DPXWritePixelDACCommand(slot, pixelCode, file=False)

            '''
            resp = ''
            while resp != pixelCode:
                resp = self.DPXReadPixelDACCommand(slot)
            ''' 

            # Dummy readout
            for j in range(3):
                self.DPXReadToTDatakVpModeCommand(slot)
                time.sleep(0.2)

            # Noise measurement
            # Loop over THL values
            print('Loop over THLs')

            # Fast loop
            countsList = []
            THLRangeFast = THLRange[::10]
            for cnt, THL in enumerate(THLRangeFast):
                self.DPXWritePeripheryDACCommand(slot, self.peripherys[slot-1] + ('%04x' % int(THL)))
                self.DPXDataResetCommand(slot)
                time.sleep(0.001)

                # Read ToT values into matrix
                countsList.append( self.DPXReadToTDatakVpModeCommand(slot).flatten() )

            countsList = np.asarray( countsList ).T
            THLRangeFast = [ THLRangeFast[item[0][0]] if np.any(item) else np.nan for item in [np.argwhere(counts > 3) for counts in countsList] ]

            # Precise loop
            if use_gui:
                yield {'DAC': pixelDAC}

            THLRangeSlow = np.around(THLRange[np.logical_and(THLRange >= (np.nanmin(THLRangeFast) - 10), THLRange <= np.nanmax(THLRangeFast))])

            NTHL = len(THLRangeSlow)
            # Do not use tqdm with GUI
            if use_gui:
                loop_range = THLRangeSlow
            else:
                loop_range = tqdm(THLRangeSlow)
            for cnt, THL in enumerate( loop_range ):
                # Repeat multiple times since data is noisy
                counts = np.zeros((16, 16))
                for lp in range(reps):
                    self.DPXWritePeripheryDACCommand(slot, self.peripherys[slot-1] + ('%04x' % int(THL)))
                    self.DPXDataResetCommand(slot)
                    time.sleep(0.001)

                    # Read ToT values into matrix
                    counts += self.DPXReadToTDatakVpModeCommand(slot)

                    if intPlot:
                        im.set_data(counts)
                        ax.set_title('THL: ' + hex(THL))
                        fig.canvas.draw()

                counts /= float(reps)
                countsDict[pixelDAC][int(THL)] = counts

                # Return status as generator when using GUI
                if use_gui:
                    yield {'status': float(cnt) / len(loop_range)}
            print()
        if use_gui:
            yield {'countsDict': countsDict}
        yield countsDict

    def getNoiseLevel(self, countsDict, THLRange, pixelDACs=['00', '3f'], noiseLimit=3):
        if isinstance(pixelDACs, basestring):
            pixelDACs = [pixelDACs]

        # Get noise THL for each pixel
        noiseTHL = {key: np.zeros((16, 16)) for key in pixelDACs}

        gaussDict, gaussSmallDict, gaussLargeDict = {key: [] for key in pixelDACs}, {key: [] for key in pixelDACs}, {key: [] for key in pixelDACs}

        # Loop over each pixel in countsDict
        for pixelDAC in pixelDACs:
            for pixelX in range(16):
                for pixelY in range(16):
                    for idx, THL in enumerate(THLRange):
                        if not THL in countsDict[pixelDAC].keys():
                            continue

                        if countsDict[pixelDAC][THL][pixelX, pixelY] >= noiseLimit and noiseTHL[pixelDAC][pixelX, pixelY] == 0:
                                noiseTHL[pixelDAC][pixelX, pixelY] = THL

                                gaussDict[pixelDAC].append(THL)
                                if pixelY in [0, 1, 14, 15]:
                                    gaussSmallDict[pixelDAC].append(THL)
                                else:
                                    gaussLargeDict[pixelDAC].append(THL)

        return gaussDict, noiseTHL

