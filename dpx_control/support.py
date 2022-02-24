from __future__ import print_function
import json
import numpy as np
import textwrap
import serial
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.special


class Support(object):
    @classmethod
    def normal(cls, x, A, mu, sigma):
        return A * np.exp(-(x - mu) / (2 * sigma**2))

    @classmethod
    def getColor(cls, c, N, idx):
        cmap = mpl.cm.get_cmap(c)
        norm = mpl.colors.Normalize(vmin=0.0, vmax=N - 1)
        return cmap(norm(idx))

    @classmethod
    def getColorBar(cls, ax, cbMin, cbMax, N=20, label=None, rotation=90):
        # Plot colorbar
        cmap = mpl.cm.get_cmap('viridis', N)
        norm = mpl.colors.Normalize(vmin=cbMin, vmax=cbMax)
        cBar = mpl.colorbar.ColorbarBase(
            ax, cmap=cmap, norm=norm, orientation='vertical')

        # cBar.ax.invert_yaxis()
        cBar.formatter.set_powerlimits((0, 0))
        cBar.ax.yaxis.set_offset_position('right')
        cBar.update_ticks()

        labels = np.linspace(cbMin, cbMax, N + 1)
        locLabels = np.linspace(cbMin, cbMax, N)
        loc = labels + abs(labels[1] - labels[0]) * .5
        cBar.set_ticks(loc)
        cBar.ax.set_yticklabels(
            ['%.1f' % loc for loc in locLabels], rotation=rotation, verticalalignment='center')
        cBar.outline.set_visible(False)
        cBar.set_label(label)

    # https://stackoverflow.com/questions/12090503/listing-available-com-ports-with-python
    @classmethod
    def getSerialPorts(cls):
        import glob
        """ Lists serial port names

            :raises EnvironmentError:
                On unsupported or unknown platforms
            :returns:
                A list of the serial ports available on the system
        """
        if sys.platform.startswith('win'):
            ports = ['COM%s' % (i + 1) for i in range(256)]
        elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
            # this excludes your current terminal "/dev/tty"
            ports = glob.glob('/dev/tty[A-Za-z]*')
        elif sys.platform.startswith('darwin'):
            ports = glob.glob('/dev/tty.*')
        else:
            raise EnvironmentError('Unsupported platform')

        result = []
        for port in ports:
            try:
                s = serial.Serial(port)
                s.close()
                result.append(port)
            except (OSError, serial.SerialException):
                pass
        return result

    @classmethod
    def getDerivative(cls, x, y):
        deltaX = x[1] - x[0]
        der = np.diff(y) / deltaX
        return x[:-1] + deltaX, der

    @classmethod
    def erfFit(cls, x, a, b, c):
        return a * (0.5 * scipy.special.erf((x - b) / c) + 0.5)

    @classmethod
    def normalErf(cls, x, a, b, c):
        return 0.56419 * a * np.exp(-(x - b)**2 / (c**2)) + 0.5

    @classmethod
    def linear_fit(cls, x, m, t):
        return m * x + t

    @classmethod
    def erfStdFit(cls, x, a, b, c, d):
        return a * (scipy.special.erf((x - b) / c) + 1) + d

    @classmethod
    def meanSigmaLinePlot(cls, fn, pIpixeldac, pPixelDAC):
        d = json.load(open(fn, 'rb'))
        meanMatrix, sigmaMatrix = d['mean'], d['sigma']
        meanMatrix, sigmaMatrix = meanMatrix, sigmaMatrix

        # Ipixeldac params
        minIpixeldac, maxIpixeldac, NIpixeldac = pIpixeldac
        # pixel_DAC params
        minDAC, maxDAC, NDAC = pPixelDAC

        # Create x-axis
        pixel_DAC = np.linspace(minDAC, maxDAC, NDAC)

        figMean = plt.figure(figsize=(7, 7))
        axMean = figMean.add_axes([0.1, 0.14, 0.7, 0.8])
        axCBarMean = figMean.add_axes([0.85, 0.1, 0.05, 0.8])
        figSigma = plt.figure(figsize=(7, 7))
        axSigma = figSigma.add_axes([0.1, 0.14, 0.7, 0.8])
        axCBarSigma = figSigma.add_axes([0.85, 0.1, 0.05, 0.8])

        for row in range(len(meanMatrix)):
            axMean.plot(
                pixel_DAC,
                meanMatrix[row],
                color=cls.getColor(
                    'viridis',
                    len(meanMatrix),
                    row))
            axSigma.plot(
                pixel_DAC,
                sigmaMatrix[row],
                color=cls.getColor(
                    'viridis',
                    len(meanMatrix),
                    row))

        cls.getColorBar(
            axCBarMean,
            minIpixeldac * 100,
            maxIpixeldac * 100,
            NIpixeldac,
            label=r'$I_{\mathrm{pixel_DAC}}$ (%)',
            rotation=0)
        cls.getColorBar(
            axCBarSigma,
            minIpixeldac * 100,
            maxIpixeldac * 100,
            NIpixeldac,
            label=r'$I_{\mathrm{pixel_DAC}}$ (%)',
            rotation=0)

        # Labels
        axMean.set_xlabel('PixelDAC')
        axMean.set_ylabel(r'$\mu_{\mathrm{THL}}$ (THL)')

        axSigma.set_xlabel('PixelDAC')
        axSigma.set_ylabel(r'$\sigma_{\mathrm{THL}}$ (THL)')

        figMean.show()
        figSigma.show()

        # Init plot
        # fig = plt.figure(figsize=(7, 4))
        # ax = fig.add_axes([0.1, 0.14, 0.7, 0.8])
        # Add colorbar later
        # axCBar = fig.add_axes([0.85, 0.1, 0.05, 0.8])

        # ax.hist(gaussCorrDict[pixel_DAC], label=str(I_pixeldac), bins=np.arange(0, 1500, 5), alpha=0.5, color=self.getColor('viridis', len(I_pixeldacList), idx))

        # Add colorbar
        # self.getColorBar(axCBar, I_pixeldacList[0]*100, I_pixeldacList[-1]*100, N=len(I_pixeldacList), label=r'I_pixel_DAC (\%)')

        # Standard deviation plot
        # figStd, axStd = plt.subplots()
        # axStd.plot(I_pixeldacList, stdList)

        # figStd.show()
        raw_input('')

        return

    @classmethod
    def meanSigmaMatrixPlot(cls, fn, pIpixeldac, pPixelDAC):
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        d = json.load(open(fn, 'rb'))
        meanMatrix = d['mean']
        print(meanMatrix)
        sigmaMatrix = d['sigma']

        figMean, axMean = plt.subplots()
        figSigma, axSigma = plt.subplots()

        # Ipixeldac params
        minIpixeldac, maxIpixeldac, NIpixeldac = pIpixeldac
        # pixel_DAC params
        minDAC, maxDAC, NDAC = pPixelDAC

        # Calculate bin widths
        wIpixeldac = float(maxIpixeldac - minIpixeldac) / NIpixeldac
        wDAC = float(maxDAC - minDAC) / NDAC

        # Sigma plot
        im = axSigma.imshow(
            sigmaMatrix.T,
            extent=[
                minIpixeldac -
                0.5 *
                wIpixeldac,
                maxIpixeldac +
                0.5 *
                wIpixeldac,
                minDAC -
                0.5 *
                wDAC,
                maxDAC +
                0.5 *
                wDAC],
            aspect='auto')
        divider = make_axes_locatable(axSigma)
        caxSigma = divider.append_axes('right', size='5%', pad=0.1)
        figSigma.colorbar(im, cax=caxSigma)
        axSigma.set_xlabel('I_pixeldac'), axSigma.set_ylabel('pixel_DAC')
        figSigma.show()

        # Mean plot
        im = axMean.imshow(
            meanMatrix.T,
            extent=[
                minIpixeldac -
                0.5 *
                wIpixeldac,
                maxIpixeldac +
                0.5 *
                wIpixeldac,
                minDAC -
                0.5 *
                wDAC,
                maxDAC +
                0.5 *
                wDAC],
            aspect='auto')
        divider = make_axes_locatable(axMean)
        caxMean = divider.append_axes('right', size='5%', pad=0.1)
        figMean.colorbar(im, cax=caxMean)
        axMean.set_xlabel('I_pixeldac'), axMean.set_ylabel('pixel_DAC')
        figMean.show()

        raw_input('')

    @classmethod
    def isLarge(cls, pixel):
        if ((pixel - 1) %
            16 == 0) or (pixel %
                         16 == 0) or ((pixel + 1) %
                                      16 == 0) or ((pixel + 2) %
                                                   16 == 0):
            return False
        return True

    @classmethod
    def energyToToTAtan(cls, x, a, b, c, t):
        return np.where(x > b, a * (x - b) + c *
                        np.arctan((x - b) / t), np.nan)

    @classmethod
    def energyToToT(cls, x, a, b, c, t, atan=False):
        if atan:
            return EnergyToToTAtan(x, a, b, c, t)
        else:
            return np.where(x > t, a * x + b + float(c) / (x - t), np.nan)

    @classmethod
    def convertToDecimal(cls, res, length=4):
        # Convert 4 hexadecimal characters each to int
        hexNumList = np.asarray(
            [int(hexNum, 16) for hexNum in textwrap.wrap(res.decode("utf-8"), length)])
        return hexNumList.reshape((16, 16))

    # Infinite for loop
    @classmethod
    def infinite_for(cls):
        x = 0
        while True:
            yield x
            x += 1

    # Megalix control
    @classmethod
    def megalix_connect(cls, port):
        import serial
        return serial.Serial(port, stopbits=2)

    @classmethod
    def megalix_set_kvpmA(cls, mlx, voltage, current, dt=1):
        import time
        focus_size = 's'
        cmd = 'ei %s %s %d %s\n' % (voltage, current, dt, focus_size)
        mlx.write(cmd.encode('utf-8'))
        print('Sent command %s to megalix' % cmd)
        time.sleep(1)

    @classmethod
    def megalix_xray_on(cls, mlx):
        mlx.write('PREP 1\n'.encode('utf-8'))
        mlx.write('SWRON\n'.encode('utf-8'))

    @classmethod
    def megalix_xray_off(cls, mlx):
        mlx.write('SWROFF\n'.encode('utf-8'))
        mlx.write('PREP 0\n'.encode('utf-8'))
