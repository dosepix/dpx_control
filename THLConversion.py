#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import cPickle
from scipy.optimize import fsolve, root
import scipy.special

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

	# Store results in dictionary
	paramsDict = {}

	for pixel in range(256):
		if not isBig(pixel + 1):
			continue

		if PLOT:
			# Create new figure
			fig, ax = plt.subplots(3, 1)

		# Get data for current pixel, remove zero entries
		pixelData = np.asarray(data[pixel], dtype=float)
		pixelData = pixelData[pixelData > 0]

		# Calculate mean and std of ToT spectrum
		mean = np.mean(pixelData)
		sig = np.std(pixelData)

		# Get rid of outliers
		pixelData = pixelData[pixelData < (mean + sig)]
		hist, bins = np.histogram(pixelData, bins=int(max(pixelData) - min(pixelData)))

		if PLOT:
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
		print bins

		if PLOT:
			# Plot THL spectrum
			ax[1].step(bins[:-1], hist, where='post')
			ax[1].axvline(x=ToTtoEnergy(np.asarray([mean]), a, b, c, t), ls='-')
			ax[1].axvline(x=ToTtoEnergy(np.asarray([mean-sig]), a, b, c, t), ls='--')
			ax[1].axvline(x=ToTtoEnergy(np.asarray([mean+sig]), a, b, c, t), ls='--')
			ax[1].set_xlabel(r'$\mathrm{THL}_\mathrm{corr}$')
			ax[1].set_ylabel('Counts')

		# Fit to peaks
		# 60 keV peak located at maximum
		muList = [bins[:-1][np.argmax(hist)], 2200]
		sigmaList = [20, 20] # ToTtoEnergy([mean+sig], a, b, c, t)[0] - mu

		energyList = [59.5409, 26.3446]
		THLList = []

		try:
			for k in range(len(muList)):
				mu, sigma = muList[k], sigmaList[k]
				print mu, sigma

				p0 = (mu, 10., 600., 300., 100.)
				for i in range(2):
					x = bins[:-1][abs(bins[:-1] - mu) < 4*sigma]
					y = hist[abs(bins[:-1] - mu) < 4*sigma]

					try:
						popt, pcov = scipy.optimize.curve_fit(normalTotal, x, y, p0=p0)
					except:
						popt = p0
					print popt

					p0 = popt
					mu, sigma, a, b, c = popt

				THLList.append( mu )

				if PLOT:
					# Show fit in plot
					xFit = np.linspace(min(x), max(x), 1000)
					ax[1].plot(xFit, normalTotal(xFit, *popt))
		except:
			continue

		# Linear conversion of THL to energy
		slope = (energyList[0] - energyList[1]) / (THLList[0] - THLList[1])
		offset = energyList[0] - slope * THLList[0]

		# Convert bins
		bins = slope*np.asarray(bins) + offset
		if PLOT:
			ax[2].step(bins[:-1], hist, where='post')
			ax[2].set_xlabel('Energy (keV)')
			ax[2].set_ylabel('Counts')

		# Store resulting parameters in dictionary
		paramsDict[pixel] = {'a': calib[pixel]['a'], 'b': calib[pixel]['b'], 'c': calib[pixel]['c'], 't': calib[pixel]['t'], 'h': slope, 'k': offset}

		if PLOT:
			# Show plot
			ax[0].set_yscale("log", nonposy='clip')
			ax[1].set_yscale("log", nonposy='clip')
			ax[2].set_yscale("log", nonposy='clip')
			plt.tight_layout()
			plt.show()

	# Store parameter dictionary to file
	cPickle.dump(paramsDict, open('ToTtoEnergy.p', 'wb'))

# === SUPPORT FIT FUNCTIONS ===
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

