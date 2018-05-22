#!/usr/bin/env python
import numpy as np
import time
import serial
import textwrap
from collections import namedtuple
import os
import sys
import configparser 

import matplotlib.pyplot as plt
import scipy.signal
import scipy.optimize
import scipy.constants

# Global Flags
DEBUG = False

def main():
	# Create object of class and establish connection
	dpx = Dosepix('/dev/ttyUSB0', 2e6, 'DPXConfigNew.conf')

	# while True:
	#	dpx.DPXReadPixelDACCommand(1)

	dpx.energySpectrumDAC(1, THLhigh=int(dpx.THLs[0], 16), THLlow=1000, THLstep=25, timestep=1, intPlot=True)
	# dpx.testPulse(1, 10)
	# dpx.measureToT()
	# dpx.measureDose(10)
	# dpx.thresholdEqualization(1, reps=1, intPlot=False, resPlot=True)

	# dpx.thresholdEqualizationConfig('DPXConfigNew.conf', reps=1, intPlot=False, resPlot=False)

	# Close connection
	dpx.close()
	return

class Dosepix:
	# === CONSTANTS ===
	__startOfTransmission 	= unichr(0x02)
	__endOfTransmission 	= unichr(0x03) 

	# = Receiver =
	__receiverDPX1 = '01'	# Dosepix
	__receiverDPX2 = '02'
	__receiverDPX3 = '03'
	__receiverMC = '04'		# Microcontroller
	__receiverVC = '09'		# Voltage Controller
	__receiverHV = '11'		# High Voltage

	# = Subreceiber =
	__subReceiverNone = '000'

	# = Sender =
	__senderPC = '10'

	# = Commands =
	__commandNone = ''
	__commandNoneLength = '0' * 6

	# = HV Commands =
	__HVenable = '000'
	__HVdisable = '001'
	__HVisEnabled = '002'
	__HVsetDAC = '003'
	__HVgetDAC = '004'

	# = VC Commands =
	__VCset3V3 = '000'
	__VCset1V8 = '001'
	__VCgetVoltage = '002'

	# = MC Commands =
	__MCgetVersion = '000'
	__MCLEDenable = '001'
	__MCLEDdisable = '002'
	__MCgetADCvalue = '003'
	__MCsetSPIclock1 = '004'
	__MCsetSPIclock2 = '005'
	__MCsetSPIclock3 = '006'

	# = DPX Commands =
	__DPXwriteOMRCommand = '001' 
	__DPXwriteConfigurationCommand = '002'
	__DPXwriteSingleThresholdCommand = '003'
	__DPXwritePixelDACCommand = '004'
	__DPXwritePeripheryDACCommand = '005'
	__DPXwriteColSelCommand = '006'
	__DPXburnSingleFuseCommand = '007'
	__DPXreadOMRCommand = '008'
	__DPXreadConfigurationCommand = '009'
	__DPXreadDigitalThresholdsCommand = '010'
	__DPXreadPixelDACCommand = '011'
	__DPXreadPeripheryDACCommand = '012'
	__DPXreadColumnTestPulseCommand = '013'
	__DPXreadColSelCommand = '014'
	__DPXreadChipIdCommand = '015'

	__DPXglobalResetCommand = '020'
	__DPXdataResetCommand = '021'

	__DPXreadToTDataDosiModeCommand = '050'
	__DPXreadBinDataDosiModeCommand = '051'
	__DPXreadToTDatakVpModeCommand = '053'
	__DPXreadToTDataIntegrationModeCommand = '054'

	__DPXgeneralTestPulse = '057'
	__DPXreadToTDataInkVpModeWithFixedFrameSizeCommand = '066'
	__DPXgeneralMultiTestPulse = '067'

	__DPXstartStreamingReadout = '068'
	__DPXstopStreamingReadout ='069'

	# = CRC =
	__CRC = 'FFFF'

	# = OMR =
	OMROperationModeType = namedtuple("OMROperationMode", "DosiMode TestWakeUp PCMode IntegrationMode")
	__OMROperationMode = OMROperationModeType(
		DosiMode = 0b00 << 22,
		TestWakeUp = 0b01 << 22,
		PCMode = 0b10 << 22,
		IntegrationMode = 0b11 << 22)

	OMRGlobalShutterType = namedtuple("OMRGlobalShutter", "ClosedShutter OpenShutter")
	__OMRGlobalShutter = OMRGlobalShutterType(
		ClosedShutter = 0b0 << 21,
		OpenShutter = 0b1 << 21)

	# Do not use 200 MHz!
	OMRPLLType = namedtuple("OMRPLL", "Direct f16_6MHz f20MHz f25MHz f33_2MHz f50MHz f100MHz")
	__OMRPLL = OMRPLLType(Direct = 0b000 << 18,
		f16_6MHz = 0b001 << 18,
		f20MHz = 0b010 << 18,
		f25MHz = 0b011 << 18,
		f33_2MHz = 0b100 << 18,
		f50MHz = 0b101 << 18,
		f100MHz = 0b110 << 18)

	OMRPolarityType = namedtuple("OMRPolarity", "electron hole")
	__OMRPolarity = OMRPolarityType(
		hole = 0b0 << 17,
		electron = 0b1 << 17)

	OMRAnalogOutSelType = namedtuple("OMRAnalogOutSel", "V_ThA V_TPref_fine V_casc_preamp V_fbk V_TPref_coarse V_gnd I_preamp I_disc1 I_disc2 V_TPbufout V_TPbufin I_krum I_dac_pixel V_bandgap V_casc_krum Temperature V_per_bias V_cascode_bias High_Z")
	__OMRAnalogOutSel = OMRAnalogOutSelType(
		V_ThA = 0b00001 << 12,
		V_TPref_fine = 0b00010 << 12, 
		V_casc_preamp = 0b00011 << 12,
		V_fbk = 0b00100 << 12,
		V_TPref_coarse = 0b00101 << 12,
		V_gnd = 0b00110 << 12,
		I_preamp = 0b00111 << 12,
		I_disc1 = 0b01000 << 12,
		I_disc2 = 0b01001 << 12,
		V_TPbufout = 0b01010 << 12,
		V_TPbufin = 0b01011 << 12,
		I_krum = 0b01100 << 12,
		I_dac_pixel = 0b01101 << 12,
		V_bandgap = 0b01110 << 12,
		V_casc_krum = 0b01111 << 12,
		Temperature = 0b11011 << 12,
		V_per_bias = 0b11100 << 12,
		V_cascode_bias = 0b11101 << 12,
		High_Z = 0b11111 << 12)

	OMRAnalogInSelType = namedtuple("OMRAnalogOutSel", "V_ThA V_TPref_fine V_casc_preamp V_fbk V_TPref_coarse V_gnd I_preamp I_disc1 I_disc2 V_TPbufout V_TPbufin I_krum I_dac_pixel V_bandgap V_casc_krum Temperature V_per_bias V_cascode_bias V_no")
	__OMRAnalogInSel = OMRAnalogInSelType(
		V_ThA = 0b00001 << 7,
		V_TPref_fine = 0b00010 << 7, 
		V_casc_preamp = 0b00011 << 7,
		V_fbk = 0b00100 << 7,
		V_TPref_coarse = 0b00101 << 7,
		V_gnd = 0b00110 << 7,
		I_preamp = 0b00111 << 7,
		I_disc1 = 0b01000 << 7,
		I_disc2 = 0b01001 << 7,
		V_TPbufout = 0b01010 << 7,
		V_TPbufin = 0b01011 << 7,
		I_krum = 0b01100 << 7,
		I_dac_pixel = 0b01101 << 7,
		V_bandgap = 0b01110 << 7,
		V_casc_krum = 0b01111 << 7,
		Temperature = 0b11011 << 7,
		V_per_bias = 0b11100 << 7,
		V_cascode_bias = 0b11101 << 7,
		V_no = 0b11111 << 7)

	OMRDisableColClkGateType = namedtuple("OMRDisableColClkGate", "Enabled Disabled")
	__OMRDisableColClkGate = OMRDisableColClkGateType(
		Enabled = 0b0 << 6,
		Disabled = 0b1 << 6)

	# = ConfBits =
	ConfBitsType = namedtuple("ConfBits", "MaskBit TestBit_Analog TestBit_Digital")
	__ConfBits = ConfBitsType(
		MaskBit = 0b1 << 2,
		TestBit_Analog = 0b1 << 1,
		TestBit_Digital = 0b1 << 0)

	def __init__(self, portName, baudRate, configFn):
		# Read config
		self.peripherys = ''
		self.OMR = ''
		self.THLs = [[]] * 3
		self.confBits = [[]] * 3
		self.pixelDAC = [[]] * 3
		self.binEdges = [[]] * 3
		self.confBits = [[]] * 3
		self.readConfig(configFn)

		self.ser = serial.Serial(portName, baudRate)
		#self.ser.set_buffer_size(rx_size=4096, tx_size=4096)

		assert self.ser.is_open, 'Error: Could not establish serial connection!'

		self.initDPX()

	# === MAIN FUNCTIONS ===
	def initDPX(self):
		# Start HV
		self.HVSetDac('0000')

		self.HVActivate()
		# Check if HV is enabled
		print 'Check if HV is activated...',
		for i in range(5):
			if self.HVGetState():
				print 'done!'
				break
			else:
				self.HVActivate()
		else:
			assert 'HV could not be activated'

		# Set voltage to 3.3V
		self.VCVoltageSet3V3()

		# Disable LED
		self.MCLEDdisable()

		# Wait
		time.sleep(0.5)

		# Global reset
		for i in range(1, 3 + 1):
			# Do three times
			for j in range(3):
				self.DPXGlobalReset(i)
		time.sleep(0.5)

		# = Write Settings =
		for i in range(1, 3 + 1):
			self.DPXWriteConfigurationCommand(i, self.confBits[i-1])
			self.DPXWriteOMRCommand(i, self.OMR)
			# self.DPXWriteOMRCommand(1, ['DosiMode', 'OpenShutter', 'f100MHz', 'electron', 'Temperature', 'V_no', 'Disabled'])

			# Merge peripheryDACcode and THL value
			self.DPXWritePeripheryDACCommand(i, self.peripherys + self.THLs[i-1])
			print 'Periphery DAC on Slot %d set to: %s' % (i, self.DPXReadPeripheryDACCommand(i))

			self.DPXWritePixelDACCommand(i, self.pixelDAC[i-1])
			print 'Pixel DAC on Slot %d set to: %s' % (i, self.DPXReadPixelDACCommand(i))
		print
		time.sleep(0.5)

		# = Data Reset =
		for i in range(1, 3 + 1):
			self.DPXDataResetCommand(i)

		# = Dummy Readout =
		for i in range(1, 3 + 1):
			self.DPXReadToTDataDosiModeCommand(i)

		# = Bin Edges =
		for i in range(1, 3 + 1):
			for binNr in range(16):
				self.DPXWriteSingleThresholdCommand(i, self.binEdges[i-1])
		time.sleep(1)

		# = Empty Bins =
		for i in range(1, 3 + 1):
			# Loop over bins
			for col in range(1, 16 + 1):
				self.DPXWriteColSelCommand(i, 16 - col)
				# Dummy readout
				self.DPXReadBinDataDosiModeCommand(i)

	def measureDose(self, measurement_time=120):
		# Set Dosi Mode in OMR
		# If OMR code is list
		if not isinstance(self.OMR, basestring):
			OMRCode = self.OMR
			OMRCode[0] = 'DosiMode'
			self.DPXWriteOMRCommand(slot, OMRCode)
		else:
			self.DPXWriteOMRCommand(slot, (int(self.OMR, 16) & ~((0x11) << 22)))

		# = START MEASUREMENT =
		print 'Measuring the dose!'
		time.sleep(measurement_time)

		# = Readout =
		outSlotList = []
		for i in range(1, 3 + 1):
			outList = []
			
			# Loop over bins
			for col in range(1, 16 + 1):
				self.DPXWriteColSelCommand(i, 16 - col)
				out = self.DPXReadBinDataDosiModeCommand(i)
				outList.append( out )

			outSlotList.append( outList )

		for k, outSlot in enumerate(outSlotList):
			print 'Slot%d' % k
			dataMatrix = np.rec.fromarrays( outSlot )
			for i in range(len(dataMatrix)):
				print np.asarray([list(entry) for entry in dataMatrix[i]])
				plt.imshow(np.asarray([list(entry) for entry in dataMatrix[i]]))
				plt.show()

	def measureToT(self, cnt=100):
		# Set Dosi Mode in OMR
		# If OMR code is list
		if not isinstance(self.OMR, basestring):
			OMRCode = self.OMR
			OMRCode[0] = 'DosiMode'
			self.DPXWriteOMRCommand(slot, OMRCode)
		else:
			self.DPXWriteOMRCommand(slot, (int(self.OMR, 16) & ~((0x11) << 22)))

		ToTList = []
		for c in range(cnt):
			for slot in range(1, 3 + 1):
				for i in range(3):
					self.DPXDataResetCommand(slot)

				data = self.DPXReadToTDataDosiModeCommand(slot)
				data = data[data > 0]
				ToTList += list(data.flatten())

	def testPulse(self, slot, length, column=0):
		# Set Polarity to hole and Photon Counting Mode in OMR
		OMRCode = self.OMR
		if not isinstance(self.OMR, basestring):
			OMRCode[3] = 'hole'
		else:
			OMRCode = (int(self.OMR, 16) & ~(1 << 17))

		if not isinstance(self.OMR, basestring):
			OMRCode[0] = 'DosiMode'
		else:
			OMRCode = (int(self.OMR, 16) & ~((0x11) << 22)) | (0x10 << 22)
		
		self.DPXWriteOMRCommand(slot, '%024x' % OMRCode)

		# Set ConfBits to use test charge on preamp input if pixel is enabled
		# Only select one column at max
		confBits = np.zeros((16, 16))
		confBits.fill(getattr(self.__ConfBits, 'MaskBit'))
		confBits[column] = [getattr(self.__ConfBits, 'TestBit_Analog')] * 16
		print confBits

		# confBits = np.asarray( [int(num, 16) for num in textwrap.wrap(self.confBits[slot-1], 2)] )
		# confBits[confBits != getattr(self.__ConfBits, 'MaskBit')] = getattr(self.__ConfBits, 'TestBit_Analog')
		self.DPXWriteConfigurationCommand(slot, ''.join(['%02x' % conf for conf in confBits.flatten()]))

		# Measure multiple test pulses and return the average ToT
		energyRange = np.linspace(0, 100e3, 200)
		ToTValues = []
		for energy in energyRange:
			self.DPXWritePeripheryDACCommand(slot, self.getTestPulseVoltageDAC(slot, energy))

			data = np.zeros((16, 16))
			for i in range(10):
				self.DPXDataResetCommand(slot)
				self.DPXGeneralTestPulse(slot, 40)
				data += self.DPXReadToTDataDosiModeCommand(slot)
			data = data / 10

			# Filter zero entries
			# data = data[data != 0]

			ToTValues.append( data[column] )

		for row in range(16):
			y = []
			for ToTValue in ToTValues:
				# print ToTValue
				y.append( ToTValue[row])
			plt.plot(energyRange/1000, y)

		plt.xlabel('Energy (keV)')
		plt.ylabel('ToT')
		plt.grid()
		plt.show()

		# Reset ConfBits
		self.DPXWriteConfigurationCommand(slot, self.confBits[slot-1])

		# Reset peripheryDACs
		self.DPXWritePeripheryDACCommand(slot, self.peripherys + self.THLs[slot-1])
		return

	def energySpectrumDAC(self, slot, THLhigh=int(self.THLs[slot-1], 16), THLlow=1000, THLstep=25, timestep=1, intPlot=False):
		# Description: Measure cumulative energy spectrum 
		#              by performing a threshold scan. The
		#              derivative of this spectrum resembles
		#              the energy spectrum.

		assert DACHigh <= 8191, "energySpectrumDAC: DACHigh value set too high!"

		# Take data in integration mode: 
		#     Sum of deposited energy in ToT
		if not isinstance(self.OMR, basestring):
			OMRCode[0] = 'IntegrationMode'
		else:
			OMRCode = (int(self.OMR, 16) | ((0x11) << 22))

		self.DPXWriteOMRCommand(slot, '%024x' % OMRCode)

		if intPlot:
			plt.ion()
			fig, ax = plt.subplots()

			# Empty plot
			lineCum, = ax.plot(np.nan, np.nan, label='Cumulative')
			limeDer = ax.plot(np.nan, np.nan, label='Derivative')

			# Settings
			plt.xlabel('THL (DAC)')
			plt.ylabel('Deposited energy (ToT)')

			plt.grid()

		# Loop over DAC values
		THLList = []
		dataList = []
		for THL in np.linspace(THLlow, THLhigh, THLstep):
			THLList.append( THL )

			# Set threshold
			self.DPXWritePeripheryDACCommand(slot, self.peripherys + THL)

			self.DPXDataResetCommand(slot)
			time.sleep(timestep)
			data = self.DPXReadToTDataIntegrationModeCommand(slot).flatten()
			dataList.append( np.mean(data) )

			# Update plot
			if intPot:
				# Cumulative
				lineCum.set_xdata( THLList )
				lineCum.set_ydata( dataList )
				
				# Derivative
				lineDer.set_xdata( np.asarray(THLList[:-1]) + 0.5*THLstep)
				lineDer.set_ydata( diff(dataList) / float(THLstep) )

				fig.canvas.draw()

		# Reset OMR and peripherys
		self.DPXWriteOMRCommand(slot, self.OMR)
		self.DPXWritePeripheryDACCommand(slot, self.peripherys + THLs[slot-1])

	# If cnt = 0, loop infinitely
	def ADCWatch(self, slot, OMRAnalogOutList, cnt=0):
		# Interactive plot
		plt.ion()
		fig, ax = plt.subplots()

		# Plot settings
		plt.xlabel('Time (s)')

		# Init plot
		OMRAnalogOutDict = {}

		lineDict = {}	# Lines of plots
		startDict = {}	# Used to normalize the ADC values

		startTime = time.time()

		for OMRAnalogOut in OMRAnalogOutList:
			dataDict = {}

			# Select AnalogOut
			print 'OMR Manipulation:'
			OMRCode_ = int(self.OMR, 16)
			OMRCode_ &= ~(0b11111 << 12)
			OMRCode_ |= getattr(self.__OMRAnalogOutSel, OMRAnalogOut)
			self.DPXWriteOMRCommand(slot, hex(OMRCode_).split('0x')[-1])

			# Data
			# Take mean of 10 samples for first point
			meanList = []
			for i in range(10):
				val = float(int(self.MCGetADCvalue(), 16))
				meanList.append( val )
			print meanList

			startVal = float(np.mean(meanList))
			startDict[OMRAnalogOut] = startVal
			dataDict['data'] = [1.]

			# Time
			currTime = time.time() - startTime
			dataDict['time'] = [currTime]

			OMRAnalogOutDict[OMRAnalogOut] = dataDict

			line, = ax.plot([currTime], [1.], label=OMRAnalogOut)
			lineDict[OMRAnalogOut] = line

		plt.legend()
		plt.grid()
		fig.canvas.draw()

		print OMRAnalogOutDict

		# Run measurement forever if cnt == 0
		loopCnt = 0
		while (True if cnt == 0 else loopCnt < cnt):
			try:
				for i, OMRAnalogOut in enumerate(OMRAnalogOutList):

					# Select AnalogOut
					OMRCode_ = int(self.OMR, 16) & ~(0b11111 << 12)
					OMRCode_ |= getattr(self.__OMRAnalogOutSel, OMRAnalogOut)
					print hex(OMRCode_).split('0x')[-1]
					self.DPXWriteOMRCommand(slot, hex(OMRCode_).split('0x')[-1])

					# Time 
					currTime = time.time() - startTime

					# Data
					currVal = float(int(self.MCGetADCvalue(), 16))
					print currVal, startDict[OMRAnalogOut]

					# Set values in dict
					OMRAnalogOutDict[OMRAnalogOut]['time'].append(currTime)

					OMRAnalogOutDict[OMRAnalogOut]['data'].append(currVal/startDict[OMRAnalogOut])

					# Refresh line
					lineDict[OMRAnalogOut].set_xdata(OMRAnalogOutDict[OMRAnalogOut]['time'])

					if len(OMRAnalogOutDict[OMRAnalogOut]['data']) > 21:
						lineDict[OMRAnalogOut].set_ydata(scipy.signal.savgol_filter( OMRAnalogOutDict[OMRAnalogOut]['data'], 21, 5))
					else: 
						lineDict[OMRAnalogOut].set_ydata(OMRAnalogOutDict[OMRAnalogOut]['data'])

				# Scale plot axes
				plt.xlim(min([min(OMRAnalogOutDict[key]['time']) for key in OMRAnalogOutDict.keys()]), max([max(OMRAnalogOutDict[key]['time']) for key in OMRAnalogOutDict.keys()]))

				plt.ylim(min([min(OMRAnalogOutDict[key]['data']) for key in OMRAnalogOutDict.keys()]), max([max(OMRAnalogOutDict[key]['data']) for key in OMRAnalogOutDict.keys()]))

				fig.canvas.draw()
				loopCnt += 1

			except (KeyboardInterrupt, SystemExit):
				# Store data and plot in files
				print 'KeyboardInterrupt'
				raise

	def thresholdEqualizationConfig(self, configFn, reps=1, intPlot=False, resPlot=True):
		for i in range(1, 3 + 1):
			pixelDAC, THL, confMask = self.thresholdEqualization(i, reps, intPlot, resPlot)

			# Set values
			self.pixelDAC[i-1] = pixelDAC
			self.THLs[i-1] = THL
			self.confBits[i-1] = confMask

		self.writeConfig(configFn)

	def thresholdEqualization(self, slot, reps=1, intPlot=False, resPlot=True):
		THLlow, THLhigh = 5150, 5630
		NTHL = THLhigh - THLlow
		THLstep = 1
		noiseLimit = 3
		spacing = 2
		existAdjust = True

		print '== Threshold equalization of detector %d ==' % slot

		# Set PC Mode in OMR in order to read kVp values
		# If OMR code is list
		if not isinstance(self.OMR, basestring):
			OMRCode = self.OMR
			OMRCode[0] = 'PCMode'
			self.DPXWriteOMRCommand(slot, OMRCode)
		else:
			self.DPXWriteOMRCommand(slot, (int(self.OMR, 16) & ~((0x11) << 22)) | (0x10 << 22))

		pixelDACs = ['00', '3f']
		# pixelDACs = ['%02x' % num for num in np.arange(0, 64, 9)]

		countsDict = self.getTHLLevel(slot, THLlow, THLhigh, THLstep, pixelDACs, reps, intPlot)

		gaussDict, noiseTHL = self.getNoiseLevel(countsDict, THLlow, THLhigh, THLstep, pixelDACs, noiseLimit)

		# Get mean values of Gaussians
		meanDict = {}
		for pixelDAC in pixelDACs:
			meanDict[pixelDAC] = np.mean( np.ma.masked_equal(gaussDict[pixelDAC], 0) )

		if len(pixelDACs) > 2:
			def slopeFit(x, m, t):
				return m*x + t
			slope = np.zeros((16, 16))
			offset = np.zeros((16, 16))

		else: 
			slope = (noiseTHL['00'] - noiseTHL['3f']) / 63.
			offset = noiseTHL['00']

		x = [int(key, 16) for key in pixelDACs]
		for pixelX in range(16):
			for pixelY in range(16):
				y = []
				for pixelDAC in pixelDACs:
					y.append(noiseTHL[pixelDAC][pixelX, pixelY])

				if len(pixelDACs) > 2:
					popt, pcov = scipy.optimize.curve_fit(slopeFit, x, y)
					slope[pixelX, pixelY] = abs(popt[0])
					offset[pixelX, pixelY] = popt[1]

					print popt

				# plt.plot(adjust[pixelX, pixelY], mean, marker='x')
				if resPlot:
					plt.plot(x, y, alpha=.5)

		mean = 0.5 * (meanDict['00'] + meanDict['3f'])
		print meanDict['00'], meanDict['3f'], mean

		if resPlot:
			plt.xlabel('DAC')
			plt.ylabel('THL')
			plt.axhline(y=mean, ls='--')
			plt.grid()
			plt.show()

		# Get adjustment value for each pixel
		adjust = (offset - mean) / slope + 0.5
		
		# Consider extreme values
		adjust[adjust > 63] = 63
		adjust[adjust < 0] = 0

		# Convert to integer
		adjust = adjust.astype(dtype=int)

		# Set new pixelDAC values
		pixelDACNew = ''.join(['%02x' % entry for entry in adjust.flatten()])

		# Repeat procedure to get noise levels
		countsDictNew = self.getTHLLevel(slot, THLlow, THLhigh, THLstep, pixelDACNew, reps, intPlot)

		gaussDictNew, noiseTHLNew = self.getNoiseLevel(countsDictNew, THLlow, THLhigh, THLstep, pixelDACNew, noiseLimit)

		# Plot the results of the equalization
		if resPlot:
			bins = np.linspace(min(gaussDict['3f']), max(gaussDict['00']), 100)

			for pixelDAC in ['00', '3f']:
				plt.hist(gaussDict[pixelDAC], bins=bins, label='%s' % pixelDAC, alpha=0.5)

			plt.hist(gaussDictNew[pixelDACNew], bins=bins, label='After equalization', alpha=0.5)

			plt.legend()

			plt.xlabel('THL')
			plt.ylabel('Counts')

			plt.show()

		# Create confBits
		confMask = np.zeros((16, 16)).astype(str)
		confMask.fill('00')
		confMask[abs(noiseTHLNew[pixelDACNew] - mean) > 10] = '%04x' % getattr(self.__ConfBits, 'MaskBit')
		confMask = ''.join(confMask.flatten())

		print
		print 'Summary:'
		print 'pixelDACs:', pixelDACNew
		print 'confMask:', confMask
		print 'THL:', '%04x' % (mean - 20)

		# Restore OMR values
		self.DPXWriteOMRCommand(slot, self.OMR)

		# Subtract value from THLMin to guarantee robustness
		return pixelDACNew, '%04x' % (mean - 20), confMask

	def getTHLLevel(self, slot, THLlow, THLhigh, THLstep=1, pixelDACs=['00', '3f'], reps=1, intPlot=False):
		NTHL = THLhigh - THLlow
		countsDict = {}

		# Interactive plot showing pixel noise
		if intPlot:
			plt.ion()
			fig, ax = plt.subplots()

			plt.xlabel('x (pixel)')
			plt.ylabel('y (pixel)')
			im = ax.imshow(np.zeros((16, 16)), vmin=0, vmax=255)

		if isinstance(pixelDACs, basestring):
			pixelDACs = [pixelDACs]

		# Loop over pixelDAC values
		for pixelDAC in pixelDACs:
			countsDict[pixelDAC] = {}
			print 'Set pixel DACs to %s' % pixelDAC
			
			# Set pixel DAC values
			if len(pixelDAC) > 2:
				pixelCode = pixelDAC
			else:
				pixelCode = pixelDAC*256
			self.DPXWritePixelDACCommand(slot, pixelCode, file=False)

			resp = ''
			while resp != pixelCode:
				resp = self.DPXReadPixelDACCommand(slot)

			# Dummy readout
			for j in range(3):
				self.DPXReadToTDatakVpModeCommand(slot)
				time.sleep(0.2)

			# Noise measurement
			# Loop over THL values
			print 'Loop over THLs'
			for cnt, THL in enumerate( range(THLlow, THLhigh, THLstep) ):
				self.statusBar(float(cnt)/NTHL * 100 + 1)

				# Repeat multiple times since data is noisy
				counts = np.zeros((16, 16))
				for lp in range(reps):
					self.DPXWritePeripheryDACCommand(slot, self.peripherys + ('%04x' % THL))
					self.DPXDataResetCommand(slot)
					time.sleep(0.001)

					# Read ToT values into matrix
					counts += self.DPXReadToTDatakVpModeCommand(slot)
					
					if intPlot:
						im.set_data(counts)
						ax.set_title('THL: ' + hex(THL))
						fig.canvas.draw()

				counts /= reps
				countsDict[pixelDAC][THL] = counts
			print 
			print

		return countsDict

	def getNoiseLevel(self, countsDict, THLlow, THLhigh, THLstep=1, pixelDACs=['00', '3f'], noiseLimit=3):
		if isinstance(pixelDACs, basestring):
			pixelDACs = [pixelDACs]

		# Get noise THL for each pixel
		noiseTHL = {key: np.zeros((16, 16)) for key in pixelDACs}

		gaussDict, gaussSmallDict, gaussLargeDict = {key: [] for key in pixelDACs}, {key: [] for key in pixelDACs}, {key: [] for key in pixelDACs}

		# Loop over each pixel in countsDict
		for pixelDAC in pixelDACs:
			for pixelX in range(16):
				for pixelY in range(16):
					for THL in range(THLlow, THLhigh, THLstep):
						if countsDict[pixelDAC][THL][pixelX, pixelY] >= noiseLimit and noiseTHL[pixelDAC][pixelX, pixelY] == 0:
								noiseTHL[pixelDAC][pixelX, pixelY] = THL

								gaussDict[pixelDAC].append(THL)
								if pixelY in [0, 1, 14, 15]:
									gaussSmallDict[pixelDAC].append(THL)
								else:
									gaussLargeDict[pixelDAC].append(THL)

		return gaussDict, noiseTHL

	def close(self):
		# = Shut down =
		self.HVDeactivate()
		print 'Check if HV is deactivated...',
		for i in range(5):
			if not self.HVGetState():
				print 'done!'
				break
			else:
				self.HVDeactivate()
		else:
			assert 'HV could not be deactivated'

		print 'Measurement finished.'

		self.ser.close()

	# === initializeDPX ===
	# Establish serial connection to board
	def initializeDPX(self, portName, baudRate):
		# TODO: Add autodetection of correct com-port
		return 

	# === HV SECTION ===
	def HVSetDac(self, DAC):
		assert len(DAC) == 4, 'Error: DAC command has to be of size 4!'
 
		self.sendCmd([self.__receiverHV, self.__subReceiverNone, self.__senderPC, self.__HVsetDAC, '%06d' % len(DAC), DAC, self.__CRC])

	def HVGetDac(self):
		self.sendCmd([self.__receiverHV, self.__subReceiverNone, self.__senderPC, self.__HVgetDAC, '%06d' % len(DAC), DAC, self.__CRC])

		return self.getDPXResponse()

	def HVActivate(self):
		self.sendCmd([self.__receiverHV, self.__subReceiverNone, self.__senderPC, self.__HVenable, self.__commandNoneLength, self.__commandNone, self.__CRC])

	def HVDeactivate(self):
		self.sendCmd([self.__receiverHV, self.__subReceiverNone, self.__senderPC, self.__HVdisable, self.__commandNoneLength, self.__commandNone, self.__CRC])

	def HVGetState(self):
		self.sendCmd([self.__receiverHV, self.__subReceiverNone, self.__senderPC, self.__HVisEnabled, self.__commandNoneLength, self.__CRC])

		res = int(self.getDPXResponse())

		if res:
			return True
		else:
			return False

	# === VC SECTION ===
	def VCVoltageSet3V3(self):
		self.sendCmd([self.__receiverVC, self.__subReceiverNone, self.__senderPC, self.__VCset3V3, self.__commandNoneLength, self.__commandNone, self.__CRC])

	def VCVoltageSet3V3(self):
		self.sendCmd([self.__receiverVC, self.__subReceiverNone, self.__senderPC, self.__VCset1V8, self.__commandNoneLength, self.__commandNone, self.__CRC])

	def VCGetVoltage(self):
		self.sendCmd([self.__receiverVC, self.__subReceiverNone, self.__senderPC, self.__VCgetVoltage, self.__commandNoneLength, self.__commandNone, self.__CRC])

		res = int(self.getDPXResponse())
		if res:
			return '3.3V'
		else:
			return '1.8V'

	# === MC SECTION ===
	def MCLEDenable(self):
		self.sendCmd([self.__receiverMC, self.__subReceiverNone, self.__senderPC, self.__MCLEDenable, self.__commandNoneLength, self.__commandNone, self.__CRC])

	def MCLEDdisable(self):
		self.sendCmd([self.__receiverMC, self.__subReceiverNone, self.__senderPC, self.__MCLEDdisable, self.__commandNoneLength, self.__commandNone, self.__CRC])

	def MCGetADCvalue(self):
		self.sendCmd([self.__receiverMC, self.__subReceiverNone, self.__senderPC, self.__MCgetADCvalue, self.__commandNoneLength, self.__commandNone, self.__CRC])

		return self.getDPXResponse()

	def MCGetFirmwareVersion(self):
		self.sendCmd([self.receiverMC, self.__subReceiverNone, self.__senderPC, self.__MCgetVersion, self.__commandNoneLength, self.__commandNone, self.__CRC])

		return self.getDPXResponse()

	# === DPX SECTION ===
	def DPXWriteOMRCommand(self, slot, OMRCode):
		if type(OMRCode) is list:
			OMRCodeList = OMRCode
			OMRTypeList = [self.__OMROperationMode,
			self.__OMRGlobalShutter,
			self.__OMRPLL,
			self.__OMRPolarity,
			self.__OMRAnalogOutSel,
			self.__OMRAnalogInSel,
			self.__OMRDisableColClkGate]

			OMRCode = 0x000000
			for i, OMR in enumerate(OMRCodeList):
				OMRCode |= getattr(OMRTypeList[i], OMR)

			OMRCode = hex(OMRCode).split('0x')[-1]

		self.sendCmd([self.getReceiverFromSlot(slot), self.__subReceiverNone, self.__senderPC, self.__DPXwriteOMRCommand, '%06d' % len(OMRCode), OMRCode, self.__CRC])

		return self.getDPXResponse()

	def DPXReadOMRCommand(self, slot):
		self.sendCmd([self.getReceiverFromSlot(slot), self.__subReceiverNone, self.__senderPC, self.__commandNoneLength, self.__commandNone, self.__CRC])

		return self.getDPXResponse()

	def DPXGlobalReset(self, slot):
		self.sendCmd([self.getReceiverFromSlot(slot), self.__subReceiverNone, self.__senderPC, self.__DPXglobalResetCommand, self.__commandNoneLength, self.__commandNone, self.__CRC])

		return self.getDPXResponse()

	def DPXWriteConfigurationCommand(self, slot, confBitsFn, file=False):
		if file:
			with open(confBitsFn, 'r') as f:
				confBits = f.read()
			confBits = confBits.split('\n')
			assert len(confBits) == 1 or (len(confBits) == 2 and confBits[1] == ''), "Conf-Bits file must contain only one line!"
			confBits = confBits[0]
		else:
			confBits = confBitsFn

		self.sendCmd([self.getReceiverFromSlot(slot), self.__subReceiverNone, self.__senderPC, self.__DPXwriteConfigurationCommand, '%06d' % len(confBits), confBits, self.__CRC])

		return self.getDPXResponse()

	def DPXWriteSingleThresholdCommand(self, slot, THFn, file=False):
		if file:
			with open(THFn, 'r') as f:
				TH = f.read()
		else:
			TH = THFn

		self.sendCmd([self.getReceiverFromSlot(slot), self.__subReceiverNone, self.__senderPC, self.__DPXwriteSingleThresholdCommand, '%06d' % len(TH), TH, self.__CRC])

		return self.getDPXResponse()

	def DPXWriteColSelCommand(self, slot, col):
		colCode = '%02x' % col
		self.sendCmd([self.getReceiverFromSlot(slot), self.__subReceiverNone, self.__senderPC, self.__DPXwriteColSelCommand, '%06d' % len(colCode), colCode, self.__CRC])

		return self.getDPXResponse()

	def DPXWritePeripheryDACCommand(self, slot, code):
		self.sendCmd([self.getReceiverFromSlot(slot), self.__subReceiverNone, self.__senderPC, self.__DPXwritePeripheryDACCommand, '%06d' % len(code), code, self.__CRC])

		return self.getDPXResponse()

	def DPXReadPeripheryDACCommand(self, slot):
		self.sendCmd([self.getReceiverFromSlot(slot), self.__subReceiverNone, self.__senderPC, self.__DPXreadPeripheryDACCommand, self.__commandNoneLength, self.__commandNone, self.__CRC])

		return self.getDPXResponse()

	def DPXWritePixelDACCommand(self, slot, code, file=False):
		if file:
			with open(code, 'r') as f:
				code = f.read().split('\n')[0]

		# else: use code string

		self.sendCmd([self.getReceiverFromSlot(slot), self.__subReceiverNone, self.__senderPC, self.__DPXwritePixelDACCommand, '%06d' % len(code), code, self.__CRC])

		return self.getDPXResponse()

	def DPXReadPixelDACCommand(self, slot):
		self.sendCmd([self.getReceiverFromSlot(slot), self.__subReceiverNone, self.__senderPC, self.__DPXreadPixelDACCommand, self.__commandNoneLength, self.__commandNone, self.__CRC])

		return self.getDPXResponse()

	def DPXDataResetCommand(self, slot):
		self.sendCmd([self.getReceiverFromSlot(slot), self.__subReceiverNone, self.__senderPC, self.__DPXdataResetCommand, self.__commandNoneLength, self.__commandNone, self.__CRC])

		return self.getDPXResponse()

	def DPXReadBinDataDosiModeCommand(self, slot):
		self.sendCmd([self.getReceiverFromSlot(slot), self.__subReceiverNone, self.__senderPC, self.__DPXreadBinDataDosiModeCommand, self.__commandNoneLength, self.__commandNone, self.__CRC])

		return self.convertToDecimal(self.getDPXResponse())

	def DPXReadToTDataDosiModeCommand(self, slot):
		self.sendCmd([self.getReceiverFromSlot(slot), self.__subReceiverNone, self.__senderPC, self.__DPXreadToTDataDosiModeCommand, self.__commandNoneLength, self.__commandNone, self.__CRC])

		return self.convertToDecimal(self.getDPXResponse())

	def DPXReadToTDataIntegrationModeCommand(self, slot):
		self.sendCmd([self.getReceiverFromSlot(slot), self.__subReceiverNone, self.__senderPC, self.__DPXreadToTDataIntegrationModeCommand, self.__commandNoneLength, self.__commandNone, self.__CRC])

		return self.convertToDecimal(self.getDPXResponse())

	def DPXReadToTDatakVpModeCommand(self, slot):
		self.sendCmd([self.getReceiverFromSlot(slot), self.__subReceiverNone, self.__senderPC, self.__DPXreadToTDatakVpModeCommand, self.__commandNoneLength, self.__commandNone, self.__CRC])

		return self.convertToDecimal(self.getDPXResponse(), 2)

	def DPXGeneralTestPulse(self, slot, length):
		lengthHex = '%04x' % length

		self.sendCmd([self.getReceiverFromSlot(slot), self.__subReceiverNone, self.__senderPC, self.__DPXgeneralTestPulse, '%06d' % len(lengthHex), lengthHex, self.__CRC])

		return self.getDPXResponse()

	def DPXGeneralMultiTestPulse(self, slot, length):
		lengthHex = '%04x' % length

		self.sendCmd([self.getReceiverFromSlot(slot), self.__subReceiverNone, self.__senderPC, self.__DPXgeneralMultiTestPulse, '%06d' % len(lengthHex), lengthHex, self.__CRC])

		return self.getDPXResponse()

	# == SUPPORT ==
	def readConfig(self, configFn):
		config = configparser.ConfigParser()
		config.read(configFn)

		# Mandatory sections
		sectionList = ['General', 'OMR', 'Slot1', 'Slot2', 'Slot3']

		# Check if set, else throw error
		for section in config.sections():
			assert section in sectionList, 'Config: %s is a mandatory section and has to be specified' % section

		# Read General
		if 'peripheryDAC' in config['General']:
			self.peripherys = config['General']['peripheryDAC']
		else:
			for i in range(1, 3 + 1):
				assert 'peripheryDAC' in config['Slot%d' % i], 'Config: peripheryDAC has to be set in either General or the Slots!'

		# Read OMR
		if 'code' in config['OMR']:
			self.OMR = config['OMR']['code']
		else:
			OMRList = []

			OMRCodeList = ['OperationMode', 'GlobalShutter', 'PLL', 'Polarity', 'AnalogOutSel', 'AnalogInSel', 'OMRDisableColClkGate']
			for OMRCode in OMRCodeList:
				assert OMRCode in config['OMR'], 'Config: %s has to be specified in OMR section!' % OMRCode

				OMRList.append(config['OMR'][OMRCode])

			self.OMR = OMRList

		# Read slot specific data
		for i in range(1, 3 + 1):
			assert 'Slot%d' % i in config.sections(), 'Config: Slot %d is a mandatory section!' % i

			# THL
			assert 'THL' in config['Slot%d' % i], 'Config: THL has to be specified in Slot%d section!' % i
			self.THLs[i-1] = config['Slot%d' % i]['THL']

			# confBits - optional field
			if 'confBits' in config['Slot%d' % i]:
				self.confBits[i-1] = config['Slot%d' % i]['confBits']
			else:
				# Use all pixels
				self.confBits[i-1] = '00' * 256

			# pixelDAC
			assert 'pixelDAC' in config['Slot%d' % i], 'Config: pixelDAC has to be specified in Slot%d section!' % i
			self.pixelDAC[i-1] = config['Slot%d' % i]['pixelDAC']

			# binEdges
			assert 'binEdges' in config['Slot%d' % i], 'Config: binEdges has to be specified in Slot%d section!' % i
			self.binEdges[i-1] = config['Slot%d' % i]['binEdges']

		return

	def writeConfig(self, configFn):
		config = configparser.ConfigParser()
		config['General'] = {'peripheryDAC': self.peripherys}

		if not isinstance(self.OMR, basestring):
			OMRCodeList = ['OperationMode', 'GlobalShutter', 'PLL', 'Polarity', 'AnalogOutSel', 'AnalogInSel', 'OMRDisableColClkGate']
			config['OMR'] = {OMRCode: self.OMR[i] for i, OMRCode in enumerate(OMRCodeList)}
		else:
			config['OMR'] = {'code': self.OMR}

		for i in range(1, 3 + 1):
			config['Slot%d' % i] = {'pixelDAC': self.pixelDAC[i-1], 'binEdges': self.binEdges[i-1], 'confBits': self.confBits[i-1], 'binEdges': self.binEdges[i-1], 'THL': self.THLs[i-1]}

		with open(configFn, 'w') as configFile:
			config.write(configFile)

	def sendCmd(self, cmdList):
		# Typical command string:
		# RRrrrSSSssCCCllllllcccc
		# R - receiver
		# r - subreceiver
		# s - sender
		# C - command
		# l - command length
		# c - CRC (unused, usually set to FFFF)

		self.ser.write(self.__startOfTransmission.encode())

		if DEBUG:
			print self.__startOfTransmission.encode(),
		for cmd in cmdList:
			if not cmd:
				continue
			for c in cmd:
				if DEBUG:
					print unichr(ord(c)),
				self.ser.write(unichr(ord(c)).encode())
			if DEBUG:
				print ' ',

		if DEBUG:
			print self.__endOfTransmission.encode()

		self.ser.write(self.__endOfTransmission.encode())

	def getResponse(self):
		res = self.ser.readline()
		if DEBUG:
			print res
		return res

	def getDPXResponse(self):
		res = self.getResponse()

		if DEBUG:
			print 'Length:', res[11:17]
		cmdLength = int( res[11:17] )

		if DEBUG:
			print 'CmdData:', res[17:17+cmdLength]
		cmdData = res[17:17+cmdLength]

		return cmdData

	def getReceiverFromSlot(self, slot):
		if slot == 1:
			receiver = self.__receiverDPX1
		elif slot == 2:
			receiver = self.__receiverDPX2
		elif slot == 3:
			receiver = self.__receiverDPX3
		else:
			assert 'Error: Function needs to access one of the three slots.'

		return receiver

	def convertToDecimal(self, res, length=4):
		# Convert 4 hexadecimal characters each to int
		hexNumList = np.asarray( [int(hexNum, 16) for hexNum in textwrap.wrap(res, length)] )
		return hexNumList.reshape((16, 16))

	def statusBar(self, perc, width='screen'):
		if width == 'screen':
			width = int(os.popen('tput cols', 'r').read()) - 8

		p = int(perc * float(width)/100)
		sys.stdout.write('\r')
		sys.stdout.write('[%-*s] %d%%' % (int(width), '='*p, perc))
		sys.stdout.flush()

	def getTestPulseVoltageDAC(self, slot, energy):
		# Set coarse and fine voltage of test pulse
		peripheryDACcode = int(self.peripherys + self.THLs[slot-1], 16)

		# Use nominal value of test capacitor
		C = 5.14e-15

		deltaV = energy * scipy.constants.e / (C * 3.64)

		assert deltaV < 1.275, "TestPulse Voltage: The energy of the test pulse was set too high!"

		# Set coarse voltage to 0
		voltageDiv = 2.5e-3
		DACVal = int((1.275 - deltaV) / voltageDiv)

		# Delete current values
		peripheryDACcode &= ~(0xff << 32)	# coarse
		peripheryDACcode &= ~(0x1ff << 16)	# fine

		# Adjust only fine voltage
		peripheryDACcode |= (DACVal << 16)
		peripheryDACcode |= (0xff << 32)
		print '%032x' % peripheryDACcode
		print DACVal

		return '%032x' % peripheryDACcode

if __name__ == '__main__':
	main()
