#!/usr/bin/env python
import numpy as np
import cPickle
import matplotlib.pyplot as plt
import scipy.optimize 
import scipy.odr

ENERGYLIST = [25, 50, 75, 100, 125] #[25, 30, 40, 50, 75, 100, 125, 148]
INFILELIST = ['temperatures_no_sensor/temperatureToT_col0_%d.p' % energy for energy in ENERGYLIST] # ['temperatureToT_col0_%d.p' % energy for energy in ENERGYLIST] 
OUTDIR = 'plotTemperatureToT'
CUTTEMP = 0
OFFSETTEMP = 1570
PLOT = True
SAVE = True

def main():
	offsetList, offsetErrList = [], []
	slopeList, slopeErrList = [], []

	for INFILE in INFILELIST:
		time, temp, tempErr, ToT, ToTErr = getData(INFILE)

		# = Plot =
		if PLOT:
			fig, ax = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw = {'width_ratios': [1, 1]}, sharey=True)

			# ax0 label
			ax[0].set_xlabel('Time (s)')
			ax[0].set_ylabel('Temperature (DAC)')

			# ax1 label
			ax[1].set_xlabel(r'$\mu_\mathrm{ToT}$')
			# ax[1].set_ylabel('Temperature (DAC)')

			# Temperature vs time
			ax[0].errorbar(time, temp, yerr=tempErr, marker='x', ls='', color='cornflowerblue')
			ax[0].axhline(y=OFFSETTEMP, ls='--')
		# Fit
		try:
			popt, pcov = scipy.optimize.curve_fit(heating, time, temp, sigma=tempErr, p0=(600, 200, 1590, 1., 1., 1540, 1550))
			timeFit = np.linspace(min(time), max(time), 1000)
			# if PLOT:
			#	ax[0].plot(timeFit, heating(timeFit, *popt), color='cornflowerblue', alpha=.7)
		except:
			pass

		# Plot for each pixel
		offset, slope = [], []
		offsetErr, slopeErr = [], []

		for i in range(len(ToT)):
			print len(ToT[i]), ToT[i]
			print len(temp), temp

			# Fit
			fitModel = scipy.odr.Model(linear)
			fitData = scipy.odr.RealData(ToT[i], temp, sx=ToTErr[i], sy=tempErr)
			odr = scipy.odr.ODR(fitData, fitModel, beta0=[1., 1.])
			out = odr.run()
			popt, perr = out.beta, out.sd_beta
			print popt

			m, t = popt
			mErr, tErr = perr
			offset.append( (OFFSETTEMP - t) / m )
			offsetErr.append( 0 ) # np.sqrt((tErr/m)**2 + ((OFFSETTEMP - t)/m**2 * mErr)**2) )
			slope.append( m )
			slopeErr.append( 0 ) # mErr )

			ToTFit = np.asarray( [np.min(ToT), np.max(ToT)] )
			if PLOT:
				ax[1].errorbar(ToT[i], temp, xerr=ToTErr[i], yerr=tempErr, color=getColor('tab20', len(ToT), i), marker='x', ls='')
				ax[1].plot(ToTFit, linear(popt, ToTFit), color=getColor('tab20', len(ToT), i))

		if PLOT:
			ax[1].axhline(y=OFFSETTEMP, ls='--')
		offsetList.append( offset ), offsetErrList.append( offsetErr )
		slopeList.append( slope ), slopeErrList.append( slopeErr )

		if PLOT:
			ax[1].set_ylim(0.99 * min(temp), 1.01 * max(temp))

			title = INFILE.split('.')[0]

			plt.title(title)
			plt.tight_layout()
			if SAVE:
				plt.savefig(OUTDIR + '/%s.pdf' % title)
			plt.show()

	offsetList, slopeList = np.asarray( offsetList ).T, np.asarray( slopeList ).T
	offsetErrList, slopeErrList = np.asarray( offsetErrList ).T, np.asarray( slopeErrList ).T
	print offsetList
	print slopeList

	# Slope plot
	# fig, ax = plt.subplots(2, 1, figsize=(5, 8), sharex=True)

	calibSlopeList, calibOffsetList = [], []
	for i in range(len(offsetList)):
		# ax[0].plot(ENERGYLIST, offsetList[i], color=getColor('tab20', len(ToT), i))
		# ax[1].plot(ENERGYLIST, slopeList[i], color=getColor('tab20', len(ToT), i))
		plt.errorbar(offsetList[i], 1./np.asarray(slopeList[i]), xerr=offsetErrList[i], yerr=np.asarray(slopeErrList[i]), color=getColor('tab20', len(ToT), i), marker='x', ls='')

		# Fit
		popt, perr = scipy.optimize.curve_fit(lambda x, m, t: m*x + t, np.asarray(offsetList[i]), 1./np.asarray(slopeList[i]))
		print popt
		plt.plot(offsetList[i], popt[0]*np.asarray(offsetList[i]) + popt[1], color=getColor('tab20', len(ToT), i))

		calibSlopeList.append( popt[0] )
		calibOffsetList.append( popt[1] )

	plt.xlabel('Offset (ToT)')
	plt.ylabel('Slope (ToT/DAC)')
	plt.tight_layout()
	plt.show()

	# Check goodness of fit
	meanListTotal, stdListTotal = [], []
	realMeanListTotal, realStdListTotal = [], []
	for INFILE in INFILELIST:
		time, temp, tempErr, ToT, ToTErr = getData(INFILE)
		
		# fig, ax = plt.subplots()

		# Loop over pixels
		meanList, stdList = [], []
		realMeanList, realStdList = [], []
		for i in range(16):
			realToT = getRealToT(ToT[i], np.asarray(temp), OFFSETTEMP, calibSlopeList[i], calibOffsetList[i])
			print realToT

			# ax.hist(realToT, color=getColor('tab20', 16, i))
			# ax.plot(ToT[i], temp, color=getColor('tab20', 16, i))
			# ax.plot(realToT, temp, color=getColor('tab20', 16, i))

			# Calculate mean and std
			realMeanList.append( np.mean(realToT) ), realStdList.append( np.std(realToT) )
			meanList.append( np.mean(ToT[i]) ), stdList.append( np.std(ToT[i]) )
		meanListTotal.append( meanList )
		realMeanListTotal.append( realMeanList )
		stdListTotal.append( stdList )
		realStdListTotal.append( realStdList )

		# plt.show()
		# plt.clf()

	# Plot mean and std
	meanListTotal, stdListTotal = np.asarray(meanListTotal).T, np.asarray(stdListTotal).T
	realMeanListTotal, realStdListTotal = np.asarray(realMeanListTotal).T, np.asarray(realStdListTotal).T
	offsetList = np.asarray(offsetList)
	print offsetList

	for i in range(16):
		figMeanStd, axMeanStd = plt.subplots(2, 1, figsize=(5, 5), gridspec_kw={'height_ratios': (3, 1)}, sharex=True)

		axMeanStd[0].plot(offsetList[i], realStdListTotal[i], marker='x')
		axMeanStd[0].plot(offsetList[i], stdListTotal[i], marker='x')

		axMeanStd[1].plot(offsetList[i], (stdListTotal[i] - realStdListTotal[i]) / stdListTotal[i], marker='x')

		axMeanStd[1].set_xlabel(r'ToT$_\mathrm{offset}$')
		axMeanStd[0].set_ylabel(r'$\sigma_\mathrm{ToT}$')
		axMeanStd[1].set_ylabel(r'$(\sigma_\mathrm{ToT} - \sigma_\mathrm{ToT, real}) / \sigma_\mathrm{ToT}$')
		axMeanStd[1].axhline(y=0, color='k', ls='--', lw='.9', alpha=.5)

		axMeanStd[0].set_title('Pixel #%d' % i)
		plt.tight_layout()
		if SAVE:
			plt.savefig(OUTDIR + '/pixel%s.pdf' % i)
		plt.show()
	
def getData(fn):
	# Load dict from file
	d = cPickle.load( open(fn, 'rb') )

	# Get data
	temp, tempErr = np.asarray(d['temp'][1:]), np.asarray(d['tempErr'][1:])
	ToT, ToTErr = np.asarray(d['ToT']), np.asarray(d['ToTErr'])
	print ToT
	time = np.asarray(d['time'][1:])

	# Get minimum dimension
	dim = min(len(temp), len(ToT), len(time))

	# Cut on temperature
	ToT, ToTErr = ToT[:dim][temp > CUTTEMP].T, ToTErr[:dim][temp > CUTTEMP].T
	time = time[:dim][temp > CUTTEMP]
	tempErr, temp = tempErr[:dim][temp > CUTTEMP], temp[:dim][temp > CUTTEMP]

	return time, temp, tempErr, ToT, ToTErr

def linear(p, x):
	m, t = p
	return m*x + t

def heating(x, tmax, toff, Tmax, tau1, tau2, offset1, offset2):
	return np.where(x <= toff, offset1, np.where(x <= tmax, (Tmax - offset1)*(1 - np.exp(-tau1*(x - toff))) + offset1, (Tmax - offset2)*np.exp(-tau2*(x - tmax)) + offset2))

def getRealToT(x, T, Toff, m, t):
	return -(t*(Toff - T) + x) / (m*(Toff - T) - 1)

def getColor(c, N, idx):
	import matplotlib as mpl
	cmap = mpl.cm.get_cmap(c)
	norm = mpl.colors.Normalize(vmin=0.0, vmax=N - 1)
	return cmap(norm(idx))

if __name__ == '__main__':
	main()

