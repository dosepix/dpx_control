#!/usr/bin/env python
import numpy as np
import cPickle, hickle
import matplotlib.pyplot as plt
import scipy.optimize 
import scipy.odr

# ENERGYLIST = [25, 50, 75, 100, 125] #[25, 30, 40, 50, 75, 100, 125, 148]
# INFILELIST = ['temperatures_no_sensor/temperatureToT_col0_%d.p' % energy for energy in ENERGYLIST] # ['temperatureToT_col0_%d.p' % energy for energy in ENERGYLIST] 
INFILELIST = ['temperatureToT_16.p']
OUTDIR = 'plotTemperatureToT'
CUTTEMP = 0
OFFSETTEMP = 1570
PLOT = True

def main():
    for INFILE in INFILELIST:
        plotTemperature(INFILE)
    
def plotTemperature(tempDict, offsettemp=1570, plot=False, outdir=None):
    offsetList, offsetErrList = [], []
    slopeList, slopeErrList = [], []

    # Load data from file
    time_, energy_, temp_, tempErr_, ToT_, ToTErr_ = getData(tempDict)
    
    # Loop over energies
    for energy in sorted(list(set(energy_))):
        # Filter data by energies
        energyCond = (energy_ == energy)
        time, temp, tempErr, ToT, ToTErr = time_[energyCond], temp_[energyCond], tempErr_.T[energyCond].T, ToT_.T[energyCond].T, ToTErr_.T[energyCond].T
        
        # = Plot =
        if plot:
            figList, axList = [], []
            for i in range(16):
                fig, ax = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw = {'width_ratios': [1, 1]}, sharey=True)

                # ax0 label
                ax[0].set_xlabel('Time (s)')
                ax[0].set_ylabel('Temperature (DAC)')

                # ax1 label
                ax[1].set_xlabel(r'$\mu_\mathrm{ToT}$')
                # ax[1].set_ylabel('Temperature (DAC)')

                # Temperature vs time
                ax[0].errorbar(time, temp, yerr=tempErr, marker='x', ls='', color='cornflowerblue')
                ax[0].axhline(y=offsettemp, ls='--')
                
                figList.append( fig ), axList.append( ax )

        # Fit
        try:
            popt, pcov = scipy.optimize.curve_fit(heating, time, temp, sigma=tempErr, p0=(600, 200, 1590, 1., 1., 1540, 1550))
            timeFit = np.linspace(min(time), max(time), 1000)
            # if PLOT:
            #   ax[0].plot(timeFit, heating(timeFit, *popt), color='cornflowerblue', alpha=.7)
        except:
            pass

        # Plot for each pixel
        offset, slope = [], []
        offsetErr, slopeErr = [], []

        ToTFitList, poptList = [], []
        for i in range(len(ToT)):
            # print len(ToT[i]), ToT[i]
            # print len(temp), temp

            # Fit
            fitModel = scipy.odr.Model(linear)
            fitData = scipy.odr.RealData(ToT[i], temp, sx=ToTErr[i], sy=tempErr)
            odr = scipy.odr.ODR(fitData, fitModel, beta0=[1., 1.])
            out = odr.run()
            popt, perr = out.beta, out.sd_beta
            poptList.append( popt )
            
            ToTFit = np.asarray( [np.min(ToT), np.max(ToT)] )
            ToTFitList.append( ToTFit )

            # if any(np.isnan(popt)):
            #    continue
            # print 'popt', popt

            m, t = popt
            mErr, tErr = perr
            offset.append( (offsettemp - t) / m )
            offsetErr.append( 0 ) # np.sqrt((tErr/m)**2 + ((offsettemp - t)/m**2 * mErr)**2) )
            slope.append( m )
            slopeErr.append( 0 ) # mErr )
              
        offsetList.append( offset ), offsetErrList.append( offsetErr )
        slopeList.append( slope ), slopeErrList.append( slopeErr )
                
        if plot:
            for idx in range(len(axList)):
                ax, fig = axList[idx], figList[idx]
                
                minList, maxList = [], []
                for i in range(16):
                    ax[1].errorbar(ToT[idx * 16 + i], temp, xerr=ToTErr[idx * 16 + i], yerr=tempErr, color=getColor('tab20', len(ToT) // 16, i % 16), marker='x', ls='')
                    ax[1].plot(ToTFitList[idx + i], linear(poptList[idx * 16 + i], ToTFitList[idx * 16 + i]), color=getColor('tab20', len(ToT) // 16, i % 16), label=str(i % 16))
                    
                    minList.append( min(ToT[idx * 16 + i]) ), maxList.append( max(ToT[idx * 16 + i]) )
                plt.legend()

                ax[1].axhline(y=offsettemp, ls='--')
                ax[1].set_ylim(0.99 * min(temp), 1.01 * max(temp))
                ax[1].set_xlim(0.95 * min(minList), 1.05 * max(maxList))

                title = 'Column: %d, Energy: %.2f keV' % (idx, energy * 1.e-3) # INFILE.split('.')[0]
                fig.suptitle(title)
                # plt.tight_layout()
                if outdir:
                    plt.savefig(outdir + '/%s.pdf' % title)
            plt.show()
            for fig in figList:
                plt.close(fig)
            
    offsetList, slopeList = np.asarray( offsetList ).T, np.asarray( slopeList ).T
    offsetErrList, slopeErrList = np.asarray( offsetErrList ).T, np.asarray( slopeErrList ).T
    # print slopeList

    # Slope plot
    # fig, ax = plt.subplots(2, 1, figsize=(5, 8), sharex=True)

    calibSlopeList, calibOffsetList = [], []
    for i in range(len(offsetList)):
        # ax[0].plot(ENERGYLIST, offsetList[i], color=getColor('tab20', len(ToT), i))
        # ax[1].plot(ENERGYLIST, slopeList[i], color=getColor('tab20', len(ToT), i))
        if plot:
            plt.errorbar(offsetList[i], 1./np.asarray(slopeList[i]), xerr=offsetErrList[i], yerr=np.asarray(slopeErrList[i]), color=getColor('tab20', len(ToT), i), marker='x', ls='')

        # Fit
        offsetList_ = np.asarray(offsetList[i])
        offsetList_ = offsetList_[~np.isnan(offsetList_)]
        slopeList_ = np.asarray(slopeList[i])
        slopeList_ = slopeList_[~np.isnan(slopeList_)]

        popt, perr = scipy.optimize.curve_fit(lambda x, m, t: m*x + t, offsetList_, 1./slopeList_)
        try:
            if plot:
                plt.plot(offsetList[i], popt[0]*offsetList_ + popt[1], color=getColor('tab20', len(ToT), i))
        except:
            pass
            
        calibSlopeList.append( popt[0] )
        calibOffsetList.append( popt[1] )

    if plot:
        plt.xlabel('Offset (ToT)')
        plt.ylabel('Slope (ToT/DAC)')
        plt.tight_layout()
        plt.show()
        
        # Histogram of slope and offset distribution
        plt.hist(calibSlopeList, bins=50)
        plt.xlabel('Slope (ToT/DAC)')
        plt.ylabel('Counts')
        plt.show()
        
        plt.hist(calibOffsetList, bins=50)
        plt.xlabel('Offset (ToT)')
        plt.ylabel('Counts')

    # Check goodness of fit
    meanListTotal, stdListTotal = [], []
    realMeanListTotal, realStdListTotal = [], []
    for energy in sorted(list(set(energy_))):
        # Filter data by energies
        energyCond = (energy_ == energy)
        time, temp, tempErr, ToT, ToTErr = time_[energyCond], temp_[energyCond], tempErr_.T[energyCond].T, ToT_.T[energyCond].T, ToTErr_.T[energyCond].T
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 3), sharey=True)

        # Loop over pixels
        meanList, stdList = [], []
        realMeanList, realStdList = [], []
        
        # print ToT, calibSlopeList, calibOffsetList
        for i in range(16): # len(ToT)):
            realToT = getRealToT(ToT[i], np.asarray(temp), offsettemp, calibSlopeList[i], calibOffsetList[i])
            # print realToT

            # ax.hist(realToT, color=getColor('tab20', 16, i))
            ax[0].plot(ToT[i], temp, color=getColor('tab20', 16, i), alpha=.5)
            ax[1].plot(realToT, temp, color=getColor('tab20', 16, i))

            # Calculate mean and std
            realMeanList.append( np.mean(realToT) ), realStdList.append( np.std(realToT) )
            meanList.append( np.mean(ToT[i]) ), stdList.append( np.std(ToT[i]) )
        meanListTotal.append( meanList )
        realMeanListTotal.append( realMeanList )
        stdListTotal.append( stdList )
        realStdListTotal.append( realStdList )

        plt.show()
        plt.clf()

    # Plot mean and std
    meanListTotal, stdListTotal = np.asarray(meanListTotal).T, np.asarray(stdListTotal).T
    realMeanListTotal, realStdListTotal = np.asarray(realMeanListTotal).T, np.asarray(realStdListTotal).T
    offsetList = np.asarray(offsetList)
    # print offsetList

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
        if outdir:
            plt.savefig(outdir + '/pixel%s.pdf' % i)
        plt.show()
        
    # Return calibrated slopes and offsets for all pixels. Included is the offset temperature
    # for which the calibration was performed. By inserting these values into getRealToT, a 
    # transformation from a measured ToT value at a certain temperature to the corresponding
    # ToT value at the offset temperature is provided.
    outDict = {'slope': calibSlopeList, 'offset': calibOffsetList, 'Toffset': offsettemp}
    return outDict
    
def getData(d, cuttemp=0):
    # Get data
    temp, tempErr = np.asarray(d['temp']), np.asarray(d['tempErr'])
    ToT, ToTErr = np.asarray(d['ToT']), np.asarray(d['ToTErr'])
    # print ToT
    time = np.asarray(d['time'])
    
    # Old files do not have the energy key
    try:
        energy = np.asarray(d['energy'])
    except:
        energy = np.zeros(len(time))
        energy.fill(np.nan)

    # Get minimum dimension
    dim = min(len(temp), len(ToT), len(time))

    # Cut on temperature
    temp = temp[:dim]
    ToT, ToTErr = ToT[:dim][temp > cuttemp].T, ToTErr[:dim][temp > cuttemp].T
    time, energy = time[:dim][temp > cuttemp], energy[:dim][temp > cuttemp]
    tempErr, temp = tempErr[:dim][temp > cuttemp], temp[:dim][temp > cuttemp]

    return time, energy, temp, tempErr, ToT, ToTErr

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

