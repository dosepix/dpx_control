#!/usr/bin/env python
import numpy as np
import cPickle
import matplotlib.pyplot as plt
import argparse
import scipy.optimize

ATAN = False
SIMPLE = True

def main():
	# Get filename from argument parser
	ap = argparse.ArgumentParser(description='Process some integers.')
	ap.add_argument('-fn', '--filename', type=str, help='File to plot', required=True)
	args = ap.parse_args()

	# Load data from dict
	d = cPickle.load( open(args.filename, 'rb') )

	# Results dict
	resDict = {}

	for col in d.keys():
		THL, THLErr = np.asarray(d[col]['THL']), np.asarray(d[col]['THLErr'])
		ToT, ToTErr = np.asarray(d[col]['ToT']), np.asarray(d[col]['ToTErr'])

		THLErr[THLErr > 100] = np.nan
		ToTErr[ToTErr > 30] = np.nan
		print THL
		print ToT

		for pixel in range(16):
			# Sort data by THL
			THL_, THLErr_, ToT_, ToTErr_ = zip(*sorted(zip(THL[pixel], THLErr[pixel], ToT[pixel], ToTErr[pixel])))

			# Fit
			if ATAN:
				p0 = (-0.4, max(THL_), -30, 50) 
				bounds = ((-0.6, -np.inf, -50, 10), (-0.2, np.inf, -10, 200))
			elif SIMPLE:
				p0 = (-0.4, max(THL_), -30, 50) 
				bounds = ((-0.6, -np.inf, -50, 10), (-0.2, np.inf, -10, 200))
			else:
				p0 = (-0.4, 1000, 30, max(THL_))
				bounds = (4*[-np.inf], 4*[np.inf])

			popt, pcov = scipy.optimize.curve_fit(EnergyToToT, THL_, ToT_, p0=p0, bounds=bounds)
			a, b, c, t = popt
			paramDict = {'a': a, 'b': b, 'c': c, 't': t}
			resDict[col*16 + pixel] = paramDict
			print popt, np.sqrt(np.diag(pcov))
			print
			THLFit = np.linspace(min(THL_), max(THL_), 1000)
			ToTFit = EnergyToToT(THLFit, *popt)

			# Plot
			plt.errorbar(THL_, ToT_, xerr=THLErr_, yerr=ToTErr_, marker='x', color=getColor('Blues', 16 + 5, pixel + 5), ls='')
			plt.plot(THLFit, ToTFit, color=getColor('Blues', 16 + 5, pixel + 5))
			plt.xlabel('THL')
			plt.ylabel('ToT')

		plt.show()

	print resDict
	cPickle.dump(resDict, open('ToTtoTHLParams.p', 'wb'))

def EnergyToToTAtan(x, a, b, c, t):
	return np.where(x < b, a*(x - b) + c*np.arctan((x - b)/t), 0)

def EnergyToToTSimple(x, a, b, c, t):
	# return np.where(abs((b - x) / t) < 1, a*(x - b) + c * ((x - b) / t), a*(x - b) - (np.pi*c*(x - b)) / (2 * (x - b)) - c*(t / (x - b)))
	res = np.where(x < b, a*(x - b) - c * (np.pi / 2 + t / (x - b)), 0)
	res[res < 0] = 0
	return res

def EnergyToToT(x, a, b, c, t):
	if ATAN:
		return EnergyToToTAtan(x, a, b, c, t)
	elif SIMPLE:
		return EnergyToToTSimple(x, a, b, c, t)
	else:
		return np.where(x < t, a*x + b + float(c)/(x - t), 0)

def getColor(c, N, idx):
	import matplotlib as mpl
	cmap = mpl.cm.get_cmap(c)
	norm = mpl.colors.Normalize(vmin=0.0, vmax=N - 1)
	return cmap(norm(idx))

if __name__ == '__main__':
	main()

