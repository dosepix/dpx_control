#!/usr/bin/env python
import numpy as np
import cPickle
import matplotlib.pyplot as plt
import argparse
import scipy.optimize
import seaborn as sns

ATAN = False
SIMPLE = True
ENERGY_CONV = 'ToTtoEnergy.p'

def main():
    # Get filename from argument parser
    ap = argparse.ArgumentParser(description='Process some integers.')
    ap.add_argument('-fn', '--filename', type=str, help='File to plot', required=True)
    args = ap.parse_args()

    # Load data from dict
    d = cPickle.load( open(args.filename, 'rb') )

    if ENERGY_CONV:
        convDict = cPickle.load( open(ENERGY_CONV, 'rb') )
    else:
        convDict = None

    resDict = plotToTtoTHL(d, convDict=convDict)
    cPickle.dump(resDict, open('ToTtoTHLParams.p', 'wb'))

def plotToTtoTHL(d, convDict=None, save=None):
    # Results dict
    resDict = {}

    for col in d.keys():
        THL, THLErr = np.asarray(d[col]['THL']), np.asarray(d[col]['THLErr'])
        ToT, ToTErr = np.asarray(d[col]['ToT']), np.asarray(d[col]['ToTErr'])

        THLErr[THLErr > 100] = np.nan
        ToTErr[ToTErr > 30] = np.nan
        # print THL
        # print ToT

        fig, ax = plt.subplots()
        if convDict:
            figEn, axEn = plt.subplots()
            figSig, axSig = plt.subplots()

        # Fit
        if ATAN:
            p0 = [-0.4, 0, -30, 50]
            bounds = ((-0.6, -np.inf, -50, 10), (-0.2, np.inf, -10, 200))
        elif SIMPLE:
            p0 = [-0.4, 0, -30, 50] 
            bounds = [[-np.inf]*4, [np.inf]*4] 
            # bounds = ((-0.6, -np.inf, -50, 10), (-0.2, np.inf, -10, 200))
        else:
            p0 = [-0.4, 1000, 30, 0]
            bounds = (4*[-np.inf], 4*[np.inf])

        for pixel in range(16):
            # Sort data by THL
            THL_, THLErr_, ToT_, ToTErr_ = zip(*sorted(zip(THL[pixel], THLErr[pixel], ToT[pixel], ToTErr[pixel])))
            if not ATAN and not SIMPLE:
                p0[-1] = np.nanmax(THL_)
            else:
                p0[1] = np.nanmax(THL_)
            
            # Filter outliers
            meanTHL, stdTHL = np.nanmean(THL_), np.nanstd(THL_)
            # print meanTHL, stdTHL
            THL_, THLErr_, ToT_, ToTErr_ = np.asarray(THL_), np.asarray(THLErr_), np.asarray(ToT_), np.asarray(ToTErr_)
            filtCond = THL_ <= 3000 # abs(THL_ - meanTHL) <= 3 * stdTHL
            THLErr_, ToT_, ToTErr_ = THLErr_[filtCond], ToT_[filtCond], ToTErr_[filtCond]
            THL_ = THL_[filtCond]
            
            try:
                popt, pcov = scipy.optimize.curve_fit(EnergyToToT, THL_, ToT_, p0=p0, bounds=bounds)
            except:
                popt = p0
            a, b, c, t = popt
            paramDict = {'a': a, 'b': b, 'c': c, 't': t}
            resDict[col*16 + pixel] = paramDict
            # print popt, np.sqrt(np.diag(pcov))
            # print
            THLFit = np.linspace(min(THL_), max(THL_), 1000)
            ToTFit = EnergyToToT(THLFit, *popt)

            # Plot
            ax.errorbar(THL_, ToT_, xerr=THLErr_, yerr=ToTErr_, marker='x', color=getColor('Blues', 16 + 5, pixel + 5), ls='')
            ax.plot(THLFit, ToTFit, color=getColor('Blues', 16 + 5, pixel + 5))
            ax.set_xlabel(r'$\mathrm{THL}_\mathrm{corr}$')
            ax.set_ylabel('ToT')

            # Energy plot
            if convDict:
                try:
                    # ToT vs. Energy
                    params = convDict[col + pixel]
                    h, k = params['h'], params['k']
                    axEn.errorbar(np.asarray(THL_)*h + k, ToT_, xerr=np.asarray(THLErr_)*h, yerr=ToTErr_, marker='x', color=getColor('Blues', 16 + 5, pixel + 5), ls='')
                    axEn.plot(THLFit*h + k, ToTFit, color=getColor('Blues', 16 + 5, pixel + 5))
                    axEn.set_xlabel('Energy (keV)')
                    axEn.set_ylabel('ToT')

                    # Std vs. Energy
                    axSig.plot(np.asarray(THL_)*h + k, np.asarray(ToTErr_), color=getColor('Blues', 16 + 5, pixel + 5))
                    axSig.set_xlabel('Energy (keV)')
                    axSig.set_ylabel(r'$\Delta$ToT')
                except:
                    continue

        fig.suptitle('Column #%d' % col)
        sns.despine(fig=fig, top=True, right=True, offset=False, trim=True)
        if convDict:
            sns.despine(fig=figEn, top=True, right=True, offset=False, trim=True)
            figEn.show()
            sns.despine(fig=figSig, top=True, right=True, offset=False, trim=True)
            figSig.show()

        if save is not None:
            outFn = ''
            if '.' in save:
                outFn = save.split('.')[0]
            else:
                outFn = save
            outFn += '_col%d.svg' % col
            fig.savefig(outFn)
        plt.show()
            
        plt.close(fig)
        if convDict:
            plt.close(figEn)
            plt.close(figSig)

    # print resDict
    return resDict
    
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

