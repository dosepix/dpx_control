#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import cPickle
from scipy.optimize import fsolve, root
import scipy.signal
import scipy.special
import seaborn as sns
import peakutils

INFILE = 'THL_1394/ToTMeasurement_Am.p'
CALIBFILE = 'ToTtoTHLParams.p'
SLOT = 1
SIMPLE = True
PLOT = False

def main():
    # Load conversion parameters
    calib = cPickle.load(open(CALIBFILE, 'rb'))

    # Load spectrum
    data = np.asarray( cPickle.load(open(INFILE, 'rb'))['Slot%d' % SLOT] ).T

    paramsDict = THLConversion(data, calib, plot=PLOT)

    # Store parameter dictionary to file
    cPickle.dump(paramsDict, open('ToTtoEnergy.p', 'wb'))

def THLConversion(data, calib, use_hist=False, plot=False):
    # Store results in dictionary
    paramsDict = {}

    for pixel in range(256):
        if not isBig(pixel + 1):
            continue

        if plot:
            # Create new figure
            fig, ax = plt.subplots(4, 1, figsize=(10, 15))

        if not use_hist:
            # Get data for current pixel, remove zero entries
            pixelData = np.asarray(data[pixel], dtype=float)
            pixelData = pixelData[pixelData > 0]
            if not len(pixelData):
                continue

            # Calculate mean and std of ToT spectrum
            mean = np.mean(pixelData)
            sig = np.std(pixelData)

            # Get rid of outliers
            # pixelData = pixelData[pixelData < (mean + sig)]
            hist, bins = np.histogram(pixelData, bins=int(max(pixelData) - min(pixelData)))

        else:
            hist, bins = np.asarray(data['hist'][pixel], dtype=float), np.asarray(data['bins'][pixel], dtype=float)
            # Remove zeros
            if bins[0] == 0:
                bins = bins[1:]
                hist = hist[1:]
                
            cond = (hist > 0)
            cond = np.append(cond, True)
            bins = bins[cond]
            hist = hist[hist > 0]
                
            mean = 1./np.sum(hist) * np.dot(hist, bins[:-1])
            sig = 0.1 * mean # 1./np.sum(hist) * np.dot(hist, np.square(bins[:-1] - mean))
            
        if plot:
            # Plot ToT spectrum
            ax[0].step(bins[:-1], hist, where='post')
            ax[0].axvline(x=mean, ls='-')
            ax[0].axvline(x=mean-sig, ls='--')
            ax[0].axvline(x=mean+sig, ls='--')
            ax[0].set_xlabel('ToT')
            ax[0].set_ylabel('Counts')

        # Get calibration parameters for current pixel
        a, b, c, t = calib[pixel]['a'], calib[pixel]['b'], calib[pixel]['c'], calib[pixel]['t']

        # Transform ToT bins to THL bins
        bins = ToTtoEnergy(bins, a, b, c, t)
        
        try:
            # Get rid of negative values
            hist = hist[bins[:-1] > 0]
            bins = bins[bins > 0]
            bins = np.append(bins, bins[-1])
            # print bins
        except:
            paramsDict[pixel] = {'a': np.nan, 'b': np.nan, 'c': np.nan, 't': np.nan, 'h': np.nan, 'k': np.nan}
            continue

        if plot:
            # Plot THL spectrum
            ax[1].step(bins[:-1], hist, where='post')
            ax[1].axvline(x=ToTtoEnergy(np.asarray([mean]), a, b, c, t), ls='-')
            ax[1].axvline(x=ToTtoEnergy(np.asarray([mean-sig]), a, b, c, t), ls='--')
            ax[1].axvline(x=ToTtoEnergy(np.asarray([mean+sig]), a, b, c, t), ls='--')
            ax[1].set_xlabel(r'$\mathrm{THL}_\mathrm{corr}$')
            ax[1].set_ylabel('Counts')

        # Fit to peaks
        # 60 keV peak located at maximum
        muList = [1650, 1900] # [bins[:-1][np.argmax(hist)], 2200]
        sigmaList = [20, 20, 100] # ToTtoEnergy([mean+sig], a, b, c, t)[0] - mu
        # energyList = [26.3446, 59.5409]
        # energyList = [26.167, 58.3] # [33, 59.5409] 
        energyList = [16., 26.167, 58.3]
        THLList = []

        # Filter the data
        try:
            hist_filt = scipy.signal.savgol_filter(hist, 11, 3)
        except:
            paramsDict[pixel] = {'a': np.nan, 'b': np.nan, 'c': np.nan, 't': np.nan, 'h': np.nan, 'k': np.nan}
            continue

        # Find peaks
        peakIdx = peakutils.indexes(hist_filt, thres=0.11, min_dist=15)
        xPeak = bins[:-1][peakIdx]
        yPeak = hist_filt[peakIdx]
        
        if plot:
            ax[2].step(bins[:-1], hist_filt, where='post')
            ax[2].plot(xPeak, yPeak, marker='x', ls='', markersize=20, color='k')
            ax[2].set_title('Filter and find peaks')
            ax[2].set_xlabel('')
            ax[2].set_ylabel('Counts')

        # Concatenate coordinates and get two largest peaks
        # peakList = np.argsort(yPeak)
        # xPeak, yPeak = xPeak[peakList[-2:]], yPeak[peakList[-2:]]
        try:
            xPeak, yPeak = [xPeak[0], xPeak[-3], xPeak[-1]], [yPeak[0], yPeak[-3], yPeak[-1]]
            print xPeak, yPeak
        except:
            paramsDict[pixel] = {'a': np.nan, 'b': np.nan, 'c': np.nan, 't': np.nan, 'h': np.nan, 'k': np.nan}
            continue
           
        for k in range(len(xPeak)):
            mu, sigma = xPeak[k], sigmaList[k]
            p0 = (mu, 10., 600., 300., 100.)

            for i in range(2):
                try:
                    x = bins[:-1][abs(bins[:-1] - mu) < 4*sigma]
                    y = hist[abs(bins[:-1] - mu) < 4*sigma]
                
                    popt, pcov = scipy.optimize.curve_fit(normalTotal, x, y, p0=p0)
                except:
                    popt = p0
                # print popt

                p0 = popt
                mu, sigma, a, b, c = popt
                
            THLList.append( mu )

            if plot:
                # Show fit in plot
                xFit = np.linspace(min(x), max(x), 1000)
                ax[1].plot(xFit, normalTotal(xFit, *popt))
                
        # Linear conversion of THL to energy
        slope = (energyList[0] - energyList[1]) / (THLList[0] - THLList[1])
        offset = energyList[0] - slope * THLList[0]
        if len(energyList) > 2:
            p0 = (slope, offset) 
            popt, pcov = scipy.optimize.curve_fit(linear, THLList, energyList)
            slope, offset = popt
            '''
            fig_, ax_ = plt.subplots()
            THLList = np.asarray( THLList )
            ax_.plot(THLList, energyList, marker='x', ls='', color='C0')
            ax_.plot(THLList, linear(THLList, *popt), ls='-', color='C0')
            print np.sqrt(np.diag(pcov))
            plt.show()
            plt.close(fig_)
            '''
               
        # Convert bins
        bins = slope*np.asarray(bins) + offset
        if plot:
            ax[3].step(bins[:-1], hist, where='post')
            ax[3].set_xlabel('Energy (keV)')
            ax[3].set_ylabel('Counts')

        # Store resulting parameters in dictionary
        paramsDict[pixel] = {'a': calib[pixel]['a'], 'b': calib[pixel]['b'], 'c': calib[pixel]['c'], 't': calib[pixel]['t'], 'h': slope, 'k': offset}

        if plot:
            # Show plot
            ax[0].set_yscale("log", nonposy='clip')
            ax[1].set_yscale("log", nonposy='clip')
            ax[3].set_yscale("log", nonposy='clip')
            plt.tight_layout()
            plt.show()
            plt.close(fig)

    return paramsDict

# === SUPPORT FIT FUNCTIONS ===
def linear(x, m, t):
    return m*x + t

def normal(x, mu, sigma, A, off):
	return A * np.exp(-(x - mu)**2/(2*sigma**2)) + off

# Erf-function is used to model the background
def normalShift(a, b, mu, sigma):
	return np.sqrt(2./np.pi)*float(b)/a*sigma

def normalBack(x, mu, sigma, a, b):
	return b * scipy.special.erf((x+normalShift(a, b, mu, sigma) - mu) / (np.sqrt(2) * sigma)) + abs(b)

def normalTotal(x, mu, sigma, a, b, c):
	return normal(x+normalShift(a, b, mu, sigma), mu, sigma, a, c) + normalBack(x, mu, sigma, a, b)

def isBig(pixel):
	smallIdx = [1, 2]
	bigIdx = []

	while max(smallIdx) < 256:
		maxSmall = max(smallIdx)
		bigIdx += range(maxSmall+1, maxSmall+13)
		maxBig = max(bigIdx)
		smallIdx += range(maxBig+1, maxBig+5)

	if pixel in bigIdx:
		return True
	else:
		return False

def ToTtoEnergy(x, a, b, c, t):
	if SIMPLE:
		return ToTtoEnergySimple(x, a, b, c, t)
	else:
		f = lambda xVal : a*(xVal - b) + c*np.arctan((xVal - b)/t) - x
		return root(f, np.full(len(x), 30, dtype=float), method='broyden1')['x']

def ToTtoEnergySimple(x, a, b, c, t):
	return b + 1./(4 * a) * (2*x + np.pi*c + np.sqrt(16*a*c*t + (2*x + np.pi*c)**2))

if __name__ == '__main__':
	main()

