from __future__ import print_function
import scipy.constants
import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.optimize
from tqdm import tqdm
from support import Support as supp

try:
    basestring
except NameError:
    basestring = str

class DPXsupport(object):
    def __init__(self):
        return

    @classmethod
    def fill_param_dict(cls, param_dict):
        # Get mean factors
        mean_dict = {}

        if 'h' in param_dict[param_dict.keys()[0]].keys():
            key_list = ['a', 'c', 'b', 'h', 'k', 't']
        else:
            key_list = ['a', 'c', 'b', 't']

        for val in key_list:
            val_list = []
            for idx in param_dict.keys():
                val_list.append(param_dict[idx][val])
            val_list = np.asarray(val_list)
            val_list[abs(val_list) == np.inf] = np.nan
            mean_dict[val] = np.nanmean(val_list)

        # Find non existent entries and loop
        for pixel in set(np.arange(256)) - set(param_dict.keys()):
            param_dict[pixel] = {val: mean_dict[val] for val in key_list}

        return param_dict

    def get_THL_from_volt(self, V, slot):
        return self.THL_calib[slot - 1][abs(V - self.volt_calib).argmin()]

    def getVoltFromTHL(self, THL, slot):
        return self.volt_calib[slot - 1][np.argwhere(self.THL_calib == THL)]

    def getVoltFromTHLFit(self, THL, slot):
        if len(self.THL_edges_low) == 0 or self.THL_edges_low[slot - 1] is None:
            return THL

        edges = zip(self.THL_edges_low[slot - 1], self.THL_edges_high[slot - 1])
        for i, edge in enumerate(edges):
            if THL >= edge[0] and THL <= edge[1]:
                break
        # else:
        #     return None

        params = self.THL_fit_params[slot - 1][i]
        if i == 0:
            return supp.erfStdFit(THL, *params)
        else:
            return supp.linear_fit(THL, *params)

    def energy_to_ToT_simple(self, x, a, b, c, t):
        return np.where(x >= self.getTHL(a, b, c, t),
                        a * x + b + float(c) / (x - t), 0)

    def ToT_to_energy_simple(self, x, a, b, c, t):
        return 1. / (2 * a) * (t * a + x - b +
                               np.sqrt((b + t * a - x)**2 - 4 * a * c))

    def getTHL(self, a, b, c, t):
        return 1. / (2 * a) * (t * a - b + np.sqrt((b + t * a)**2 - 4 * a * c))

    def energyToToTFitAtan(self, x, a, b, c, d):
        return np.where(x > b, a * (x - b) + c * np.arctan((x - b) / d), 0)

    def energyToToTFitHyp(self, x, a, b, c, d):
        return np.where(x > d, a * x + b + float(c) / (x - d), 0)

    @classmethod
    def THL_calib_to_edges(cls, THL_dict, eye_lens=False):
        volt, thl = THL_dict['Volt'], THL_dict['ADC']

        # Sort by THL
        thl, volt = zip(*sorted(zip(thl, volt)))

        # Find edges by taking derivative
        diff = abs(np.diff(volt))
        if eye_lens:
            thres = 100
        else:
            thres = 200
        edges = np.argwhere(diff > thres).flatten() + 1

        # Store fit results in dict
        d = {}

        edges = list(edges)
        edges.insert(0, 0)
        edges.append(8190)

        THL_edges_low, THL_edges_high = [0], []

        x1 = np.asarray(thl[edges[0]:edges[1]])
        y1 = np.asarray(volt[edges[0]:edges[1]])
        popt1, pcov1 = scipy.optimize.curve_fit(supp.erfStdFit, x1, y1)
        d[0] = popt1

        for i in range(1, len(edges) - 2):
            # Succeeding section
            x2 = np.asarray(thl[edges[i]:edges[i + 1]])
            y2 = np.asarray(volt[edges[i]:edges[i + 1]])

            popt2, pcov2 = scipy.optimize.curve_fit(supp.linear_fit, x2, y2)
            m1, m2, t1, t2 = popt1[0], popt2[0], popt1[1], popt2[1]
            d[i] = popt2

            # Get central position
            # Calculate intersection to get edges
            if i == 1:
                Vcenter = 0.5 * \
                    (supp.erfStdFit(edges[i], *popt1) + supp.linear_fit(edges[i], m2, t2))
                THL_edges_high.append(
                    scipy.optimize.fsolve(
                        lambda x: supp.erfStdFit(
                            x, *popt1) - Vcenter, 100)[0])
            else:
                Vcenter = 1. / (m1 + m2) * \
                    (2 * edges[i] * m1 * m2 + t1 * m1 + t2 * m2)
                THL_edges_high.append((Vcenter - t1) / m1)

            THL_edges_low.append((Vcenter - t2) / m2)
            popt1, pcov1 = popt2, pcov2

        THL_edges_high.append(8190)

        return THL_edges_low, THL_edges_high, d

    def valToIdx(self, slot, pixel_DACs, THLRange, gaussDict, noiseTHL):
        # Transform values to indices
        mean_dict = {}
        for pixel_DAC in pixel_DACs:
            d = np.asarray([self.getVoltFromTHLFit(
                elm, slot) if elm else np.nan for elm in gaussDict[pixel_DAC]], dtype=np.float)
            mean_dict[pixel_DAC] = np.nanmean(d)

            for pixelX in range(16):
                for pixelY in range(16):
                    elm = noiseTHL[pixel_DAC][pixelX, pixelY]
                    if elm:
                        noiseTHL[pixel_DAC][pixelX,
                                           pixelY] = self.getVoltFromTHLFit(elm, slot)
                    else:
                        noiseTHL[pixel_DAC][pixelX, pixelY] = np.nan

        return mean_dict, noiseTHL

    def getTHLLevel(
            self,
            slot,
            THLRange,
            pixel_DACs=[
                '00',
                '3f'],
            reps=1,
            intPlot=False,
            use_gui=False):
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

        if isinstance(pixel_DACs, basestring):
            pixel_DACs = [pixel_DACs]

        # Loop over pixel_DAC values
        for pixel_DAC in pixel_DACs:
            countsDict[pixel_DAC] = {}
            print('Set pixel DACs to %s' % pixel_DAC)

            # Set pixel DAC values
            if len(pixel_DAC) > 2:
                pixelCode = pixel_DAC
            else:
                pixelCode = pixel_DAC * 256
            self.DPX_write_pixel_DAC_command(slot, pixelCode, file=False)

            '''
            resp = ''
            while resp != pixelCode:
                resp = self.DPX_read_pixel_DAC_command(slot)
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
                self.DPX_write_periphery_DAC_command(
                    slot, self.peripherys[slot - 1] + ('%04x' % int(THL)))
                self.DPX_data_reset_command(slot)
                time.sleep(0.001)

                # Read ToT values into matrix
                countsList.append(
                    self.DPXReadToTDatakVpModeCommand(slot).flatten())

            countsList = np.asarray(countsList).T
            THLRangeFast = [THLRangeFast[item[0][0]] if np.any(item) else np.nan for item in [
                np.argwhere(counts > 3) for counts in countsList]]

            # Precise loop
            if use_gui:
                yield {'DAC': pixel_DAC}

            THLRangeSlow = np.around(THLRange[np.logical_and(THLRange >= (
                np.nanmin(THLRangeFast) - 10), THLRange <= np.nanmax(THLRangeFast))])

            NTHL = len(THLRangeSlow)
            # Do not use tqdm with GUI
            if use_gui:
                loop_range = THLRangeSlow
            else:
                loop_range = tqdm(THLRangeSlow)
            for cnt, THL in enumerate(loop_range):
                # Repeat multiple times since data is noisy
                counts = np.zeros((16, 16))
                for lp in range(reps):
                    self.DPX_write_periphery_DAC_command(
                        slot, self.peripherys[slot - 1] + ('%04x' % int(THL)))
                    self.DPX_data_reset_command(slot)
                    time.sleep(0.001)

                    # Read ToT values into matrix
                    counts += self.DPXReadToTDatakVpModeCommand(slot)

                    if intPlot:
                        im.set_data(counts)
                        ax.set_title('THL: ' + hex(THL))
                        fig.canvas.draw()

                counts /= float(reps)
                countsDict[pixel_DAC][int(THL)] = counts

                # Return status as generator when using GUI
                if use_gui:
                    yield {'status': float(cnt) / len(loop_range)}
            print()
        if use_gui:
            yield {'countsDict': countsDict}
        else:
            return countsDict

    def getNoiseLevel(
            self,
            countsDict,
            THLRange,
            pixel_DACs=[
                '00',
                '3f'],
            noiseLimit=3):
        if isinstance(pixel_DACs, basestring):
            pixel_DACs = [pixel_DACs]

        # Get noise THL for each pixel
        noiseTHL = {key: np.zeros((16, 16)) for key in pixel_DACs}

        gaussDict, gaussSmallDict, gaussLargeDict = {
            key: [] for key in pixel_DACs}, {
            key: [] for key in pixel_DACs}, {
            key: [] for key in pixel_DACs}

        # Loop over each pixel in countsDict
        for pixel_DAC in pixel_DACs:
            for pixelX in range(16):
                for pixelY in range(16):
                    for idx, THL in enumerate(THLRange):
                        if THL not in countsDict[pixel_DAC].keys():
                            continue

                        if countsDict[pixel_DAC][THL][pixelX,
                                                     pixelY] >= noiseLimit and noiseTHL[pixel_DAC][pixelX,
                                                                                                  pixelY] == 0:
                            noiseTHL[pixel_DAC][pixelX, pixelY] = THL

                            gaussDict[pixel_DAC].append(THL)
                            if pixelY in [0, 1, 14, 15]:
                                gaussSmallDict[pixel_DAC].append(THL)
                            else:
                                gaussLargeDict[pixel_DAC].append(THL)

        return gaussDict, noiseTHL
