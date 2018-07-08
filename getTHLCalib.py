#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.special
import cPickle

def main():
	d = cPickle.load(open('THLCalib.p', 'rb'))
	volt = d['Volt']
	thl = d['ADC']
	print d.keys()

	# plt.plot(np.arange(len(thl)), volt)
	# plt.plot(thl, volt)
	# plt.show()

	# Sort by THL
	thl, volt = zip(*sorted(zip(thl, volt)))
	plt.plot(thl, volt)
	plt.show()

	plt.plot(thl[:-1], abs(np.diff(volt)))
	diff = abs(np.diff(volt))
	edges = np.argwhere(diff > 100).flatten() + 1
	print edges

	yList = []

	x = np.asarray( thl[:edges[0]] )
	y = np.asarray( volt[:edges[0]] )
	popt, pcov = scipy.optimize.curve_fit(erf, x, y)
	yList += list( y )
	plt.plot(x, y)
	plt.plot(x, erf(x, *popt))

	THLEdgesLow = [0]
	THLEdgesHigh = [x[-1]]
	mList, tList = [], []
	for i in range(len(edges) - 1):
		x_, y_ = x, y
		x = np.asarray( thl[edges[i]:edges[i+1]] )
		y = np.asarray( volt[edges[i]:edges[i+1]] )

		x = x[y > max(y_)]
		y = y[y > max(y_)]

		THLEdgesLow.append( x[0] )
		THLEdgesHigh.append( x[-1] )

		plt.plot(x, y)
		yList += list(y)

		popt, pcov = scipy.optimize.curve_fit(linear, x, y)
		m, t = popt
		mList.append( m ), tList.append( t )
		plt.plot(x, linear(x, *popt))

	x_ = np.asarray( thl[edges[-1]:] )
	y_ = np.asarray( volt[edges[-1]:] )
	x_ = x_[y_ > max(y)]
	y_ = y_[y_ > max(y)]
	THLEdgesLow.append( x_[0] )
	THLEdgesHigh.append( x_[-1] )

	print THLEdgesLow
	print THLEdgesHigh

	popt, pcov = scipy.optimize.curve_fit(linear, x_, y_)
	plt.plot(x_, y_)
	plt.plot(x_, linear(x_, *popt))
	yList += list( y_ )
	plt.show()

	thl = np.asarray(thl)
	volt = np.asarray(volt)

	plt.plot(yList)
	plt.show()

	return 

	plt.plot(volt, [getTHL(V, volt, thl) for V in volt])
	plt.show()

	popt, pcov = scipy.optimize.curve_fit(sawtoothSlope, thl, volt, p0=(1., 250., 500.))
	print popt

	plt.plot(thl, sawtoothSlope(thl, *popt))
	plt.show()

def getTHL(V, volt, THL):
	return THL[abs(V - volt).argmin()]

def linear(x, m, t):
	return m*(x - t)

def erf(x, a, b, c, d):
	return a * (scipy.special.erf((x - b)/c) + 1) + d

def sawtooth(x, a, p):
	return -2*a/np.pi * np.arctan(1./np.tan(x*np.pi/p))

def sawtoothSlope(x, A, a, p):
	return A*x + sawtooth(x, a, p)

if __name__ == '__main__':
	main()

