#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import cPickle
import argparse
from scipy.optimize import fsolve
import scipy.optimize
import peakutils

def main():
	ap = argparse.ArgumentParser(description='Process some integers.')
	ap.add_argument('-fn', '--filename', type=str, help='File to plot', required=True)
	ap.add_argument('-sl', '--slot', type=int, help='Slot to plot', required=False)

	args = ap.parse_args()
	fn = args.filename
	if args.slot:
		slot = args.slot
	else:
		slot = 1

	# Load file
	print fn
	x = cPickle.load( open(fn, 'r') )

	# plotAnim(x['Slot%d' % slot])
	# return

	data = np.asarray( x['Slot%d' % slot] )

	plt.hist(data.flatten(), bins=np.arange(1000))
	plt.show()

	plotCorrected(data)
	return
	print getPixelShift(data)

def plotCorrected(data):
	d = cPickle.load(open('testPulseParams.p', 'r'))
	aList, bList, cList, tList = d['a'], d['b'], d['c'], d['t']

	# Transpose to get pixel info
	data = np.asarray( data ).T

	peakCorrection = []
	binsEnergyList = []
	histList = []
	peak = []
	pixelList = []

	for pixel in range(256):
		if not isLarge(pixel):
			continue

		pixelData = data[pixel]
		pixelData = pixelData[pixelData > 0]

		a, b, c, t = aList[pixel], bList[pixel], cList[pixel], tList[pixel]

		# Make histogram
		hist, bins = np.histogram(pixelData, bins=np.arange(0, 4093, 1))

		# Convert to energy
		binsEnergy = ToTtoEnergy(bins, a, b, c, t)
		binsEnergyList.append( binsEnergy )
		histList.append( hist )

		# Find all peaks
		peakIdx = peakutils.indexes(hist, thres=.005, min_dist=40)
		peakCorrection.append( binsEnergy[peakIdx][0] )

		if pixel == 6:
			peak = binsEnergy[peakIdx]

		# plt.plot(np.diff( binsEnergy[peakIdx] ))
		# plt.show()

		# plt.plot(binsEnergy[peakIdx], hist[peakIdx], marker='x', ls='')
		# plt.step(binsEnergy[:-1], hist)
		# plt.show()

		pixelList.append( pixel )

	plt.semilogy(binsEnergyList[0][:-1], np.sum(histList, axis=0))
	plt.show()

	peakCorrectionFirst = np.mean( peakCorrection ) - np.asarray( peakCorrection )
	peak = np.asarray( peak ) - peak[0] + np.mean(peakCorrection)
	eventListTotal = []

	for i, binsEnergy in enumerate( binsEnergyList ):
		binsEnergy = np.asarray( binsEnergy[:-1] ) + peakCorrectionFirst[i]

		# Find all peaks
		peakIdx = peakutils.indexes(histList[i], thres=.005, min_dist=40)
		binsDiff = binsEnergy[peakIdx]

		# plt.plot(peak[:len(peakIdx)], binsEnergy[peakIdx][:len(peak)] / peak[:len(peakIdx)] ) 
		popt, pcov = scipy.optimize.curve_fit(linear, binsEnergy[peakIdx][:len(peak)], peak[:len(peakIdx)] / binsEnergy[peakIdx][:len(peak)] )
		perr = np.sqrt( np.diag(pcov) )
		# plt.plot( binsEnergy[peakIdx], linear(binsEnergy[peakIdx], *popt))

		pixelData = data[pixelList[i]]
		pixelData = pixelData[pixelData > 0]

		a, b, c, t = aList[pixelList[i]], bList[pixelList[i]], cList[pixelList[i]], tList[pixelList[i]]

		eventList = []
		for event in pixelData:
			eventList.append( ToTtoEnergy([event], a, b, c, t) )

		eventList = np.asarray( eventList ) 
		eventList += peakCorrectionFirst[i]
		eventList *= linear(eventList, *popt)

		eventListTotal += list( eventList )

	# Make histogram
	hist, bins = np.histogram(eventListTotal, bins=np.arange(0, 4095, 0.5))
	plt.semilogy(bins[:-1], hist)

	plt.show()

def linear(x, m, t):
	return m*x + t

def getPixelShift(data):
	pixelData = []
	muListTotal = []
	peakListTotal = []

	slopeDict = {'pixel%d' % i: {} for i in range(256)}
	for pixel in range(256):
		if not isLarge(pixel):
			continue
		pixelData = list( data[:,pixel].flatten() )
		
		# Get slot from data
		# fig, ax = plt.subplots()
		# ax.set_yscale("log", nonposy='clip')
		hist, bins = np.histogram(pixelData, bins=np.arange(0, 500, 1))

		# Do FFT of signal
		# getFFT(bins[:-1], hist, bins[1] - bins[0])

		# Find all peaks
		peakIdx = peakutils.indexes(hist, thres=.001, min_dist=40)
		print peakIdx
		# plt.plot(bins[:-1], hist)
		# plt.plot(bins[peakIdx], hist[peakIdx], marker='x', ls='')
		# plt.show()
		# plt.clf()

		xPeak = bins[peakIdx]
		muList = []
		sigmaList = []
		muErrList = []
		sigmaErrList = []
		peakList = []
		peakCnt = 0
		for xP in xPeak:
			# Filter data
			x_ = bins[:-1][abs(bins[:-1] - xP) <= 25]
			y_ = hist[abs(bins[:-1] - xP) <= 25]
 
 			p0 = (max(y_), xP, 10., 0.1)
			try:
				popt, pcov = scipy.optimize.curve_fit(normal, x_, y_, p0=p0)
				perr = np.sqrt( np.diag(pcov) )

				A, mu, sigma, off = popt
				if abs(sigma) > 20:
					continue

				muList.append( mu )
				sigmaList.append( np.abs(sigma) )
				muErrList.append( perr[1] )
				sigmaErrList.append( perr[2] )

				# plt.plot(x_, normal(x_, *popt))
				# plt.plot(x_, y_)
				# plt.show()
			except: 
				'''
				plt.plot(x_, normal(x_, *p0))
				plt.plot(x_, y_)
				plt.show()
				'''

				muList.append( np.nan )
				pass

			peakList.append( peakCnt )
			peakCnt += 1

		muListTotal.append( muList )
		peakListTotal.append( peakList )

		# plt.step(bins[:-1], hist, where='post')
		# plt.show() 

		# sigma vs. Index
		'''
		plt.errorbar(peakList, sigmaList, yerr=sigmaErrList, marker='x', ls='', color='cornflowerblue')
		popt, pcov = scipy.optimize.curve_fit(lambda x, m, t: m*x + t, peakList, sigmaList)
		plt.plot(peakList, popt[0]*np.asarray(peakList) + popt[1], ls='-', color='cornflowerblue')
		plt.xlabel('Index')
		plt.ylabel(r'$\sigma$ (ToT)')
		plt.grid()
		plt.show()
		'''

		# mu vs. Index
		print peakList
		print muList
		print
		
		peakList = np.asarray(peakList)
		muList = np.asarray(muList)
		# Remove nans
		peakList = peakList[~np.isnan(muList)]
		muList = muList[~np.isnan(muList)]

		plt.errorbar(peakList, muList, yerr=muErrList, marker='x', ls='', color=getColor('Blues', 256, pixel))
		try:
			popt, pcov = scipy.optimize.curve_fit(lambda x, m, t: m*x + t, peakList, muList)
			slopeDict['pixel%d' % pixel]['m'] = popt[0]
			slopeDict['pixel%d' % pixel]['t'] = popt[1]

		except:
			continue

		print 'Diff'
		print np.diff(muList)

		plt.plot(peakList, popt[0]*np.asarray(peakList) + popt[1], ls='-', color=getColor('Blues', 256, pixel))
		plt.xlabel('Index')
		plt.ylabel(r'$\mu$ (ToT)')
		plt.grid()
	# getFFT(bins[:-1], hist, bins[1] - bins[0])

	# Mean mu
	max_len = np.max([len(a) for a in muListTotal])
	muListTotal = np.asarray([np.pad(a, (0, max_len - len(a)), 'constant', constant_values=np.nan) for a in muListTotal])
	print muListTotal

	muTotal = np.nanmean(np.asarray(muListTotal), axis=0)
	plt.plot(np.arange(len(muTotal)), muTotal, ls='--', c='k')
	plt.show()

	# Shift spectra onto first peak of mean
	muListTotal = np.asarray([ np.asarray(muListTotal[i])- (muListTotal - muTotal)[:, 0][i] for i in range(len(muListTotal)) ])

	# Calculate mean correction
	'''
	for pixel in range(len(muListTotal)):
		print 'Pixel %d: ' % pixel,
		muDiff = muTotal/muListTotal[pixel]
		print muListTotal[pixel]
		plt.plot(np.arange(muDiff.size), muDiff)
		plt.show()
	'''

	

	return slopeDict

def Idontknow():
	for i in range(2, 15):
		x_ = np.asarray(x['Slot%d' % slot])[:,i].flatten()

		# Remove zeros
		x_ = x_[x_ > 0]
		print len(x_)

		a = 2.7501806945155627
		b = 8.482372245673544
		c = 74.51948122705083
		t = 2.609729787968528

		hist, bins = np.histogram(x_, bins=np.linspace(0, 2000, 1000))
		ax.step(bins[:-1], hist, where='post')
	plt.show()
	return

	bins_ = ToTtoEnergy(bins, a, b, c, t)
	# plt.plot(hist, hist_, marker='x', ls='')
	plt.step(bins_[:-1], hist, where='post')

	ax.set_xlabel('ToT')
	ax.set_ylabel('Counts')
	ax.set_xlim(0, 1200)
	plt.show()

def getFFT(x, y, deltaX, plot=True):
	# Calculate FFT
	freq = np.fft.rfftfreq(len(x), deltaX)
	freqAmplitude = np.fft.rfft(y)

	if plot:
		plt.plot(freq, freqAmplitude / max(freqAmplitude))
		plt.show()

	freqAmplitude[freqAmplitude < 0.2] = 0

	# Inverse FFT
	amplitude = np.fft.irfft( freqAmplitude )
	
	if plot:
		plt.plot(np.arange(len(amplitude)), amplitude)
		plt.show()
	
	return amplitude

# p: Period
def multiGauss(x, p):
	return

def normal(x, A, mu, sigma, off):
	return A*np.exp(-(x-mu)**2 / (2*sigma**2) ) + off

def EnergyToToT(x, a, b, c, t):
	return a*(x - b) + c*np.arctan((x - b)/t)

def ToTtoEnergy(x, a, b, c, t):
	f = lambda xVal : a*(xVal - b) + c*np.arctan((xVal - b)/t) - x
	return fsolve(f, np.full(len(x), 19))

def isLarge(pixel):
	if ( (pixel - 1) % 16 == 0 ) or ( pixel % 16 == 0 ) or ( (pixel + 1) % 16 == 0 ) or ( (pixel + 2) % 16 == 0 ):
		return False
	return True

def getColor(c, N, idx):
	import matplotlib as mpl
	cmap = mpl.cm.get_cmap(c)
	norm = mpl.colors.Normalize(vmin=0.0, vmax=N - 1)
	return cmap(norm(idx))

def plotAnim(x, window=100, step=50):
	bins = np.linspace(0, 2000, 1000)
	print len(x), len(x) - window
	for i in range(0, len(x) - window, step):
		print i, i + window
		xFlat = np.asarray(x[i:i + window]).flatten()

		# Remove zeros
		xFlat = xFlat[xFlat > 0]

		plt.clf()
		plt.hist(xFlat, bins=bins)
		plt.pause(0.1)

if __name__ == '__main__':
	main()

