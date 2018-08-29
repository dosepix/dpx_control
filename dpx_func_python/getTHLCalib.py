#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.special
import scipy.integrate
import seaborn as sns
import cPickle

OUTDIR = 'THLCalibPlots/'
SAVE = False

def main():
	d = cPickle.load(open('../../dpx_data/THLCalibration/THLCalib_6.p', 'rb'))
	volt = d['Volt']
	thl = d['ADC']
	print d.keys()

	# Sort by THL
	thl, volt = zip(*sorted(zip(thl, volt)))
	plt.plot(thl, volt)

        plt.xlabel('THL')
        plt.ylabel('V_ThA')
        plt.grid()

        # Remove spines
        sns.despine(top=True, right=True, offset=False, trim=True)
        if SAVE:
                plt.savefig(OUTDIR + 'THLCurve.pdf')
        plt.tight_layout()
	plt.show()

	plt.plot(thl[:-1], abs(np.diff(volt)))
	diff = abs(np.diff(volt))
	edges = np.argwhere(diff > 100).flatten() + 1
	print edges

        # TEST
        edges = list(edges)
        edges.insert(0, 0)
        edges.append( 8190 )

        THLEdgesLow, THLEdgesHigh = [0], []

        x1 = np.asarray( thl[edges[0]:edges[1]] )
        y1 = np.asarray( volt[edges[0]:edges[1]] )
        plt.plot(x1, y1)
        popt1, pcov1 = scipy.optimize.curve_fit(erf, x1, y1)
        plt.plot(x1, erf(x1, *popt1))

        for i in range(1, len(edges) - 2):
                # Succeeding section
                x2 = np.asarray( thl[edges[i]:edges[i+1]] )
                y2 = np.asarray( volt[edges[i]:edges[i+1]] )

                # Plot sections
                plt.plot(x2, y2)

                popt2, pcov2 = scipy.optimize.curve_fit(linear, x2, y2)
                m1, m2, t1, t2 = popt1[0], popt2[0], popt1[1], popt2[1]
                plt.plot(x2, linear(x2, *popt2))

                # Get central position
                # Calculate intersection to get edges
                if i == 1:
                        Vcenter = 0.5*(erf(edges[i], *popt1) + linear(edges[i], m2, t2))
                        THLEdgesHigh.append( scipy.optimize.fsolve(lambda x: erf(x, *popt1) - Vcenter, 100)[0] )
                else:
                        Vcenter = 1./(m1 + m2) * (2*edges[i]*m1*m2 + t1*m1 + t2*m2)
                        THLEdgesHigh.append( (Vcenter - t1)/m1 )

                plt.axhline(y=Vcenter, ls='--', lw=.9, color='k')
                THLEdgesLow.append( (Vcenter - t2)/m2 )

                popt1, pcov1 = popt2, pcov2

        THLEdgesHigh.append( 8190 )

        print np.asarray(THLEdgesLow, dtype=int)
        print np.asarray(THLEdgesHigh, dtype=int)
        plt.xlabel('THL')
        plt.ylabel('V_ThA')
        sns.despine(top=True, right=True, offset=False, trim=True)
        if SAVE:
                plt.savefig(OUTDIR + 'THLCurveSegments.pdf')
        plt.tight_layout()
        plt.show()

        return

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

	print np.asarray(THLEdgesLow, dtype=int)
	print np.asarray(THLEdgesHigh, dtype=int)

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

def getLinearIntegral(x, m, edge, pos=False):
        return 0.5 * m*(x**2 - edge**2) - edge*x + edge**2

def linear(x, m, t):
	return m*x + t

def erf(x, a, b, c, d):
	return a * (scipy.special.erf((x - b)/c) + 1) + d

def sawtooth(x, a, p):
	return -2*a/np.pi * np.arctan(1./np.tan(x*np.pi/p))

def sawtoothSlope(x, A, a, p):
	return A*x + sawtooth(x, a, p)

if __name__ == '__main__':
	main()

