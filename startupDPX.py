#!/usr/bin/env python
import numpy as np
import time
import serial
import textwrap
from collections import namedtuple
import os
import os.path
import sys
import configparser 
import yaml

import matplotlib.pyplot as plt
import scipy.signal
import scipy.optimize
import scipy.constants
import scipy.special
import scipy.interpolate
import cPickle

# Global Flags
DEBUG = False

def main():
	# Create object of class and establish connection
	dpx = Dosepix('/dev/ttyUSB0', 2e6, 'DPXConfigNew.conf')

	'''
	# Read peripheryDAC values from config
	d = dpx.splitPerihperyDACs(dpx.peripherys+ dpx.THLs[0], perc=True)

	# Change pixelDAC step value
	d['I_pixeldac'] = 0.2
	code = dpx.periheryDACsDictToCode(d, perc=True)

	# Set peripheryDACs
	dpx.peripherys = code
	for i in range(1, 3 + 1):
		dpx.DPXWritePeripheryDACCommand(i, code)
	'''

	'''
	while True:
		for item in np.reshape( list( dpx.DPXReadDigitalThresholdsCommand(1) ), (256, 16*4) ):
			print ''.join(item)
		print
	'''

	#while True:
		# print dpx.DPXReadPixelDACCommand(1)
		# dpx.DPXWriteOMRCommand(1, '0000')
		# print dpx.DPXReadPeripheryDACCommand(1)
		# print dpx.DPXReadOMRCommand(1)

	# dpx.ADCWatch(1, ['Temperature', 'I_preamp'], cnt=0)

	# dpx.energySpectrumTHL(1, THLhigh=8000, THLlow=int(dpx.THLs[0], 16), THLstep=25, timestep=1, intPlot=True)
	
	# dpx.measureTHL(1)

	# dpx.ToTtoTHL(1)
	dpx.energySpectrumTHL(1)
	#dpx.testPulseDosi(1)
	# dpx.testPulseSigma(1)
	# dpx.testPulseToT(1, 10)

	# dpx.measureToT(1)
	# dpx.measureToT(slot=1, outFn='ToTMeasurement.p', cnt=3000, intPlot=False)
	# dpx.measureDose(measurement_time=120)
	# dpx.thresholdEqualization(1, reps=1, intPlot=False, resPlot=True)

	# dpx.thresholdEqualizationConfig('DPXConfigNew_temp.conf', reps=1, intPlot=False, resPlot=True)

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

		# Load THL calibration data
		if os.path.isfile('THLCalib.p'):
			d = cPickle.load(open('THLCalib.p', 'rb'))
			self.voltCalib = np.asarray(d['Volt']) / max(d['Volt'])
			self.THLCalib = np.asarray(d['THL'])
		else:
			self.voltCalib = None
			self.THLCalib = None

		self.THLEdgesLow = [0, 804, 1313, 1827, 2342, 2833, 3360, 3872, 4388, 4889, 5410, 5918, 6446, 6961, 7464, 7992]
		self.THLEdgesHigh = [511, 1023, 1535, 2047, 2559, 3071, 3583, 4095, 4607, 5119, 5631, 6143, 6655, 7167, 7679, 8190]
		self.THLEdges = []
		for i in range(len(self.THLEdgesLow)):
			self.THLEdges += list( np.arange(self.THLEdgesLow[i], self.THLEdgesHigh[i] + 1) )

	# === MAIN FUNCTIONS ===
	def initDPX(self):
		# Start HV
		self.HVSetDac('0000')
		print 'HV DAC set to %s' % self.HVGetDac()

		self.HVActivate()
		# Set voltage to 3.3V
		self.VCVoltageSet3V3()

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

		print 'Voltage set to %s' % self.VCGetVoltage()

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
		gray = [0, 1, 3, 2, 6, 4, 5, 7, 15, 13, 12, 14, 10, 11, 9, 8]
		for i in range(1, 3 + 1):
			# self.DPXWriteSingleThresholdCommand(i, self.binEdges[i-1])
			# TODO: Workaround!
			for binEdge in range(16):
				gc = gray[binEdge]
				
				'''
				if gc == 0:
					binEdges = ('%01x' % gc + '%03x' % 0x000)*256 
				else:
					binEdges = ('%01x' % gc + '%03x' % 0xfff)*256 
				'''

				binEdges = ('%01x' % gc + '%03x' % (20*binEdge + 15)) * 256
				print 1000. / 16. * binEdge + 15 
				#print binEdges
				self.DPXWriteSingleThresholdCommand(i, binEdges)

		# = Empty Bins =
		for i in range(1, 3 + 1):
			# Loop over bins
			for col in range(1, 16 + 1):
				self.DPXWriteColSelCommand(i, 16 - col)
				# Dummy readout
				self.DPXReadBinDataDosiModeCommand(i)

	def measureDose(self, measurement_time=120, outFn='doseMeasurement.p'):
		# Set Dosi Mode in OMR
		# If OMR code is list
		OMRCode = self.OMR
		if not isinstance(OMRCode, basestring):
			OMRCode[0] = 'DosiMode'
		else:
			OMRCode = int(OMRCode, 16) & ~((0x11) << 22)

		for slot in range(1, 3 + 1):
			self.DPXWriteOMRCommand(slot, OMRCode)

		# = START MEASUREMENT =
		print 'Measuring the dose!'

		self.clearBins()
		time.sleep(measurement_time)

		# = Readout =
		outSlotList = []
		for i in range(1, 3 + 1):
			outList = []

			# Loop over bins
			for col in range(1, 16 + 1):
				self.DPXWriteColSelCommand(i, 16 - col)
				out = self.DPXReadBinDataDosiModeCommand(i)

				# Bug: discard strange values in matrix
				print np.mean(out), np.std(out)
				out[abs(out - np.mean(out)) > np.std(out)] = 0

				outList.append( out )
			# print outList

			outSlotList.append( outList )

		for k, outSlot in enumerate(outSlotList):
			print 'Slot%d' % k
			dataMatrix = np.rec.fromarrays( outSlot )
			for i in range(len(dataMatrix)):
				print np.asarray([list(entry) for entry in dataMatrix[i]])
				plt.imshow(np.asarray([list(entry) for entry in dataMatrix[i]]))
				plt.show()

		# Reset OMR
		for slot in range(1, 3 + 1):
			self.DPXWriteOMRCommand(slot, self.OMR)

		# Store data to file
		outDict = {'Slot%d' % i: [] for i in range(1, 3 + 1)}
		for i, outSlot in enumerate(outSlotList):
			print outSlot
			outDict['Slot%d' % (i + 1)] = np.asarray([np.asarray(outSlotBin).flatten() for outSlotBin in outSlot]).T

		self.pickleDump(outDict, outFn)

	def clearBins(self):
		# Clear bins
		for i in range(1, 3 + 1):
			for k in range(3):
				self.DPXDataResetCommand(i)
				self.DPXReadToTDataDosiModeCommand(i)

			for col in range(16):
				self.DPXWriteColSelCommand(i, 16 - col)
				self.DPXReadBinDataDosiModeCommand(i)

	# If cnt is set to 0, perform endless loop
	# Keyboard Interrupts are caught in order to store data afterwards
	def measureToT(self, slot=1, outFn='ToTMeasurement.p', cnt=1000000, intPlot=True):
		# Set Dosi Mode in OMR
		# If OMR code is list
		OMRCode = self.OMR
		if not isinstance(OMRCode, basestring):
			OMRCode[0] = 'DosiMode'
		else:
			OMRCode = int(OMRCode, 16) & ~((0x11) << 22)

		# Check which slots to read out
		if isinstance(slot, int):
			slotList = [slot]
		elif not isinstance(slot, basestring):
			slotList = slot

		# Set mode in slots
		for slot in slotList:
			self.DPXWriteOMRCommand(slot, OMRCode)

		# Init plot
		if intPlot:
			plt.ion()

			fig, ax = plt.subplots()

			# Create empty axis
			line, = ax.plot(np.nan, np.nan, color='k') # , where='post')
			ax.set_xlabel('ToT')
			ax.set_ylabel('Counts')
			# ax.set_yscale("log", nonposy='clip')
			plt.grid()

			# Init bins and histogram data
			bins = np.arange(0, 1000, 1)
			histData = np.zeros(len(bins)-1)

			ax.set_xlim(min(bins), max(bins))

		# For KeyboardInterrupt exception
		print 'Starting ToT Measurement!'
		print '========================='
		measStart = time.time()
		try:
			ToTDict = {'Slot%d' % slot: [] for slot in slotList}

			c = 0
			startTime = time.time()
			if intPlot:
				dataPlot = []

			while True if (cnt == 0) else (c <= cnt):
				for slot in slotList:
					# Reset data registers
					self.DPXDataResetCommand(slot)

					# Read data
					data = self.DPXReadToTDataDosiModeCommand(slot)

					data = data.flatten()
					# Remove overflow
					data[data >= 4096] -= 4096
					if intPlot:
						data_ = data
						data_ = data_[[self.isLarge(pixel) for pixel in range(256)]]
						dataPlot += data.tolist()

					ToTDict['Slot%d' % slot].append( data.tolist() )
				
				if c > 0 and not c % 100:
					print '%.2f Hz' % (100./(time.time() - startTime))
					startTime = time.time()

				# Increment loop counter
				c += 1

				# Update plot every 100 iterations
				if intPlot:
					if not c % 10:
						dataPlot = np.asarray(dataPlot)
						# Remove empty entries
						dataPlot = dataPlot[dataPlot > 0]

						hist, bins_ = np.histogram(dataPlot, bins=bins)
						histData += hist
						dataPlot = []

						line.set_xdata(bins_[:-1])
						line.set_ydata(histData)

						# Update plot scale
						ax.set_ylim(1, 1.1 * max(histData))
						fig.canvas.draw()

			# Loop finished
			self.pickleDump(ToTDict, outFn)
			print 'Measurement time: %.2f min' % ((time.time() - measStart) / 60.)
			for key in ToTDict.keys():
				print 'Slot%d: %d events' % (slot, len(np.asarray(ToTDict[key]).flatten()) / 256.)

		except (KeyboardInterrupt, SystemExit):
			# Store data and plot in files
			print 'KeyboardInterrupt-Exception: Storing data!'
			print 'Measurement time: %.2f min' % ((time.time() - measStart) / 60.)
			for key in ToTDict.keys():
				print 'Slot%d: %d events' % (slot, len(np.asarray(ToTDict[key]).flatten()) / 256.)

			self.pickleDump(ToTDict, outFn)
			raise

		# Reset OMR
		for slot in slotList:
			self.DPXWriteOMRCommand(slot, self.OMR)

	def testPulseInit(self, slot, column=0):
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
		
		self.DPXWriteOMRCommand(slot, OMRCode)

		if column == 'all':
			columnRange = range(16)
		elif not isinstance(column, basestring):
			if isinstance(column, int):
				columnRange = [column]
			else:
				columnRange = column

		return columnRange

	def testPulseClose(self, slot):
		# Reset ConfBits
		self.DPXWriteConfigurationCommand(slot, self.confBits[slot-1])

		# Reset peripheryDACs
		self.DPXWritePeripheryDACCommand(slot, self.peripherys + self.THLs[slot-1])

	def maskBitsColumn(self, slot, column):
		# Set ConfBits to use test charge on preamp input if pixel is enabled
		# Only select one column at max
		confBits = np.zeros((16, 16))
		confBits.fill(getattr(self.__ConfBits, 'MaskBit'))
		confBits[column] = [getattr(self.__ConfBits, 'TestBit_Analog')] * 16
		# print confBits

		# confBits = np.asarray( [int(num, 16) for num in textwrap.wrap(self.confBits[slot-1], 2)] )
		# confBits[confBits != getattr(self.__ConfBits, 'MaskBit')] = getattr(self.__ConfBits, 'TestBit_Analog')

		self.DPXWriteConfigurationCommand(slot, ''.join(['%02x' % conf for conf in confBits.flatten()]))

	def testPulseSigma(self, slot, column=[0, 1, 2], N=100):
		columnRange = self.testPulseInit(slot, column=column)

		fig, ax = plt.subplots()

		energyRange = np.linspace(5e3, 100e3, 50)
		meanListEnergy, sigmaListEnergy = [], []

		for energy in energyRange:
			DACval = self.getTestPulseVoltageDAC(slot, energy, True)
			print 'Energy DAC:', DACVal
			self.DPXWritePeripheryDACCommand(slot, DACval)

			meanList, sigmaList = [], []
			print 'E = %.2f keV' % (energy / 1000.)
			for k, column in enumerate(columnRange):
				self.maskBitsColumn(slot, column)

				# Record ToT spectrum of pulse
				dataList = []
				for n in range(N):
					self.statusBar(float(k*N + n)/(len(columnRange) * N) * 100 + 1)
					self.DPXDataResetCommand(slot)
					self.DPXGeneralTestPulse(slot, 100)
					dataList.append( self.DPXReadToTDataDosiModeCommand(slot)[column] )

				dataList = np.asarray(dataList)
				for pixel in range(16):
					# ax.hist(dataList[:,pixel], bins=30)
					meanList.append( np.mean(dataList[:,pixel]) )
					sigmaList.append( np.std(dataList[:,pixel]) )
					# plt.show()

			print
			print meanList
			meanListEnergy.append( meanList )
			sigmaListEnergy.append( sigmaList )

		meanListEnergy = np.asarray(meanListEnergy)
		sigmaListEnergy = np.asarray(sigmaListEnergy)
		# for pixel in range(len(meanListEnergy[0])):

		plt.errorbar(energyRange/1000., [np.mean(item) for item in sigmaListEnergy], xerr=energyRange/1000.*0.18, yerr=np.asarray([np.std(item) for item in sigmaListEnergy])/np.sqrt(len(columnRange)*16), marker='x', ls='')

		# Fit curve
		#popt, pcov = scipy.optimize.curve_fit(self.energyToToTFitAtan, energyRange/1000., [np.mean(item) for item in meanListEnergy], p0=(3, 8, 50, 1))
		#perr = np.sqrt( np.diag(pcov) )
		#print popt, perr/popt*100

		#plt.plot(energyRange/float(1000), self.energyToToTFitAtan(energyRange/float(1000), *popt))

		plt.xlabel('Energy (keV)')
		plt.ylabel('Standard deviation (ToT)')
		plt.show()

		self.testPulseClose(slot)

	def testPulseToT(self, slot, length, column='all', paramOutFn='testPulseParams.p'):
		columnRange = self.testPulseInit(slot, column=column)

		# Measure multiple test pulses and return the average ToT
		energyRange = np.asarray(list(np.linspace(5e3, 20e3, 30)) + list(np.linspace(20e3, 100e3, 20)))
		energyRangeFit = np.linspace(5e3, 100e3, 1000)
		# energyRange = np.arange(509, 0, -10)

		paramOutDict = {'a': [], 'b': [], 'c': [], 't': []}
		for column in  columnRange:
			self.maskBitsColumn(slot, column)

			ToTValues = []
			for energy in energyRange:
				# DACval = self.getTestPulseVoltageDAC(slot, energy)
				DACval = self.getTestPulseVoltageDAC(slot, energy, True)
				self.DPXWritePeripheryDACCommand(slot, DACval)

				data = np.zeros((16, 16))
				for i in range(10):
					self.DPXDataResetCommand(slot)
					self.DPXGeneralTestPulse(slot, 1000)
					data += self.DPXReadToTDataDosiModeCommand(slot)
				data = data / 10

				# Filter zero entries
				# data = data[data != 0]

				ToTValues.append( data[column] )

			x = energyRange/float(1000)
			for row in range(16):
				y = []
				for ToTValue in ToTValues:
					# print ToTValue
					y.append( ToTValue[row])

				# Fit curve
				popt, pcov = scipy.optimize.curve_fit(self.energyToToTFitAtan, x, y, p0=(3, 8, 50, 1))
				perr = np.sqrt( np.diag(pcov) )

				# Store in dictionary
				a, b, c, t = popt
				paramOutDict['a'].append( a )
				paramOutDict['b'].append( b )
				paramOutDict['c'].append( c )
				paramOutDict['t'].append( t )

				print popt, perr/popt*100

				'''
				plt.plot(x, y, marker='x')
				plt.plot(energyRangeFit/float(1000), self.energyToToTFitAtan(energyRangeFit/float(1000), *popt))
				plt.show()
				'''

		# Dump to file
		cPickle.dump(paramOutDict, open(paramOutFn, 'wb'))

		# Plot
		plt.xlabel('Energy (keV)')
		plt.ylabel('ToT')
		plt.grid()
		plt.show()

		self.testPulseClose(slot)
		return

	def testPulseDosi(self, slot=1, column='all'):
		column = 5
		columnRange = self.testPulseInit(slot, column=column)

		# Select DosiMode
		OMRCode = self.OMR
		if not isinstance(OMRCode, basestring):
			OMRCode[0] = 'DosiMode'
		else:
			OMRCode = int(OMRCode, 16) & ~((0x11) << 22)

		self.DPXWriteOMRCommand(slot, OMRCode)

		# Specify energy range
		energyRange = list(np.linspace(20e3, 100e3, 10))

		ToTValues = []
		# Loop over energies
		for energy in energyRange:
			DACval = self.getTestPulseVoltageDAC(slot, energy, True)
			self.DPXWritePeripheryDACCommand(slot, DACval)

			# Clear data
			self.clearBins()

			# Loop over columns
			for column in columnRange:
				self.maskBitsColumn(slot, column)
				self.DPXWriteColSelCommand(slot, 16 - 5 - 1)
				# Dummy readouts
				self.DPXReadBinDataDosiModeCommand(slot)
				self.DPXReadBinDataDosiModeCommand(slot)
				self.DPXReadBinDataDosiModeCommand(slot)

				# Generate pulses
				for i in range(10):
					time.sleep(1)
					self.DPXGeneralTestPulse(1, 1000)

				print self.DPXReadBinDataDosiModeCommand(slot)

			# = Readout =
			# Loop over bins
			outList = []
			# for col in range(1, 16 + 1):
			self.DPXWriteColSelCommand(slot, 16 - 5 - 1)
			# self.DPXWriteColSelCommand(slot, 16 - col)
			# out = self.DPXReadToTDatakVpModeCommand(slot)
			# out = np.sum( self.DPXReadBinDataDosiModeCommand(slot), axis=1 )
			out = self.DPXReadBinDataDosiModeCommand(slot)

			# Bug: discard strange values in column
			print np.mean(out), np.std(out)
			print out 
			# out[abs(out - np.mean(out)) > np.std(out)] = 0

			outList.append( out )
			print 

			'''
			dataMatrix = np.rec.fromarrays( outList )
			for i in range(len(dataMatrix)):
				print np.asarray([list(entry) for entry in dataMatrix[i]])
				plt.imshow(np.asarray([list(entry) for entry in dataMatrix[i]]))
				plt.show()
			'''

	def energyToToTFitAtan(self, x, a, b, c, d):
		return np.where(x > b, a*(x - b) + c*np.arctan((x - b)/d), 0)

	def energyToToTFitHyp(self, x, a, b, c, d):
		return np.where(x > d, a*x + b + float(c)/(x - d), 0)

	def ToTtoTHL(self, slot=1, column=0, THLstep=1, valueLow=0, valueHigh=460, valueCount=460, energy=False, plot=False, outFn='ToTtoTHL.p'):
		# Description: Generate test pulses and measure their ToT
		#              values. Afterwards, do a THL-scan in order to
		#              find the corresponding THL value. Repeat 
		#              multiple times to find the correlation between
		#              ToT and THL
		# Parameters: 
		#   - energy: if set, iteration is performed over energies.
		#             Else, Test Pulse DAC values are modified directly.

		# Set low level to zero and high level to noise limit
		# for the coarse measurement
		THLlow = 0
		THLhigh = int(self.THLs[slot-1], 16)

		# Set AnalogOut to V_ThA
		OMRCode = self.OMR
		if not isinstance(OMRCode, basestring):
			OMRCode[4] = 'V_ThA'
		else:
			OMRCode &= ~(0b11111 << 12)
			OMRCode |= getattr(self.__OMRAnalogOutSel, 'V_ThA')
		self.DPXWriteOMRCommand(slot, OMRCode)

		# Select only one column
		self.maskBitsColumn(slot, column)

		# Store results in lists
		ToTListTotal, ToTErrListTotal = [], []
		THLListTotal, THLErrListTotal = [], []

		valueRange = np.linspace(valueLow, valueHigh, valueCount)

		# Number of test pulses for ToT measurement
		NToT = 100

		# Loop over test pulse energies
		for val in valueRange:
			# Activates DosiMode
			columnRange = self.testPulseInit(slot, column=column)

			# Set test pulse energy
			if energy:
				print 'Energy: %.2f keV' % (val/1000.)
			else:
				val = int(val)
				print 'DAC: %04x' % val

			DACval = self.getTestPulseVoltageDAC(slot, val, energy)
			self.DPXWritePeripheryDACCommand(slot, DACval)

			# Generate test pulses
			dataList = []
			for i in range(NToT):
				self.DPXDataResetCommand(slot)
				self.DPXGeneralTestPulse(slot, 1000)
				data = self.DPXReadToTDataDosiModeCommand(slot)[column]
				dataList.append( data )

			# Determine corresponding ToT value
			testPulseToT = np.mean( dataList , axis=0)
			testPulseToTErr = testPulseToT / np.sqrt( NToT )

			# Store in lists
			ToTListTotal.append( testPulseToT )
			ToTErrListTotal.append( testPulseToTErr )

			# Set PCMode
			# WARNING: 8bit counter only!
			if not isinstance(OMRCode, basestring):
				OMRCode[0] = 'PCMode'
			else:
				OMRCode = (int(OMRCode, 16) | ((0x11) << 22))
			self.DPXWriteOMRCommand(slot, OMRCode)

			# Loop over THLs
			# Init result lists
			lastTHL = 0
			THLList = []
			dataList = []
			THLMeasList = []

			# Dummy readout of AnalogOut
			# self.MCGetADCvalue()
			# THLRange = np.arange(THLlow, THLhigh, THLstep)
			# THLRange = self.THLCalib[np.logical_and(self.THLCalib > THLlow, self.THLCalib < THLhigh)]
			THLRange = np.asarray(self.THLEdges)
			THLRange = THLRange[np.logical_and(THLRange > THLlow, THLRange < THLhigh)]

			# Do a fast measurement to estimate the position of the edge for each pixel
			THLFastList = []
			dataFastList = []
			THLFastStep = 100
			NPulses = 30		# Has to be less than 256

			for cnt, THL in enumerate(THLRange[::THLFastStep]):
				# Set new THL
				self.DPXWritePeripheryDACCommand(slot, DACval[:-4] + '%04x' % THL)

				# Start measurement
				self.DPXDataResetCommand(slot)
				for i in range(NPulses):
					self.DPXGeneralTestPulse(slot, THLFastStep)
				data = self.DPXReadToTDatakVpModeCommand(slot)[column]
				data[data > NPulses] = NPulses

				# Store data in lists
				THLFastList.append( THL )
				dataFastList.append( data )

			print THLFastList
			print np.asarray(dataFastList).T

			# Check if array is empty
			if not np.count_nonzero(dataFastList):
				# If so, energy is set too low
				valueCount -= 1
				ToTListTotal.pop()
				ToTErrListTotal.pop()
				continue

			# Get derivative and find maximum for every pixel
			xFastDer = np.asarray(THLFastList[:-1]) + THLFastStep / 2.
			dataFastDer = [xFastDer[np.argmax(np.diff(data) / float(THLFastStep))] for data in np.asarray(dataFastList).T if np.count_nonzero(data)]

			print dataFastDer

			# Get mean THL value
			meanTHL = np.mean( dataFastDer )

			# Set THL range to new estimated region
			THLRange = THLRange[abs(THLRange - meanTHL) < 100]

			# Do a slow but precise measurement with small THL steps
			for cnt, THL in enumerate(THLRange[::THLstep]):
				self.statusBar(float(cnt)/len(THLRange[::THLstep]) * 100 + 1)
				# for V in np.linspace(0.34, 0.36, 1000):
				# THL = self.getTHLfromVolt(V)
				# THLmeas = np.mean( [float(int(self.MCGetADCvalue(), 16)) for i in range(3) ] )
				# THLMeasList.append( THLmeas )
				# print THL, '%04x' % THL

				# Set new THL
				self.DPXWritePeripheryDACCommand(slot, DACval[:-4] + '%04x' % THL)
				# print DACval[:-4] + '%04x' % THL
				# print self.peripherys + '%04x' % THL

				# Generate test pulse and measure ToT in integrationMode
				# dataTemp = np.zeros(16)
				self.DPXDataResetCommand(slot)
				for i in range(30):
					self.DPXGeneralTestPulse(slot, 1000)
				data = self.DPXReadToTDatakVpModeCommand(slot)[column]
				# print data

				# Store data in lists
				THLList.append( THL )
				dataList.append( data )

				# lastTHL = THLmeas
			print

			# Calculate derivative of THL spectrum
			xDer = np.asarray(THLList[:-1]) + 0.5*THLstep
			dataDer = [np.diff(data) / float(THLstep) for data in np.asarray(dataList).T]

			peakList = []
			peakErrList = []
			# Transpose to access data of each pixel
			for data in np.asarray(dataList).T:
				THLListFit = np.linspace(min(THLList), max(THLList), 1000)

				# Perform erf-Fit
				p0 = [100., THLList[int(len(THLList) / 2.)], 3.]
				try:
					popt, pcov = scipy.optimize.curve_fit(self.erfFit, THLList, data, p0=p0)
					perr = np.sqrt(np.diag(pcov))
					print popt
				except:
					popt = p0
					perr = len(p0) * [0]
					print 'Fit failed!'
					pass

				# Return fit parameters
				# a: Amplitude
				# b: x-offset
				# c: scale
				a, b, c = popt
				peakList.append( b )
				peakErrList.append( perr[2] / np.sqrt(2) )

				# Savitzky-Golay filter the data
				# Ensure odd window length
				windowLength = int(len(THLRange[::THLstep]) / 10.)
				if not windowLength % 2:
					windowLength += 1

				# Plots
				if plot:
					try:
						dataFilt = scipy.signal.savgol_filter(data, windowLength, 3)
						plt.plot(THLList, dataFilt)
						plt.plot(*self.getDerivative(THLList, dataFilt))
					except:
						pass

					plt.plot(THLListFit, self.erfFit(THLListFit, *popt), ls='-')
					plt.plot(THLListFit, self.normalErf(THLListFit, *popt), ls='-')
					plt.plot(THLList, data, marker='x', ls='')
					plt.show()

			THLListTotal.append( peakList )
			THLErrListTotal.append( peakErrList )
			print

		# Transform to arrays
		THLListTotal = np.asarray(THLListTotal).T
		THLErrListTotal = np.asarray(THLErrListTotal).T
		ToTListTotal = np.asarray(ToTListTotal).T
		ToTErrListTotal = np.asarray(ToTErrListTotal).T

		print THLListTotal
		print THLErrListTotal
		print ToTListTotal
		print ToTErrListTotal

		# Show results in plot
		fig, ax = plt.subplots()
		for i in range(len(THLListTotal)):
			ax.errorbar(THLListTotal[i], ToTListTotal[i], xerr=THLErrListTotal[i], yerr=ToTErrListTotal[i], color=self.getColor('Blues', len(ToTListTotal), i), marker='x')

		ax.grid()
		plt.xlabel('THL (DAC)')
		plt.ylabel('ToT')

		# TODO: Perform fit
		plt.show()

		# Save to pickle file
		if outFn:
			# Create dictionary
			d = {'THL': THLListTotal, 'THLErr': THLErrListTotal, 'ToT': ToTListTotal, 'ToTErr': ToTErrListTotal}

			self.pickleDump(d, outFn)

		return

	def getDerivative(self, x, y):
		deltaX = x[1] - x[0]
		der = np.diff(y) / deltaX
		return x[:-1] + deltaX, der

	def erfFit(self, x, a, b, c):
		return a*(0.5 * scipy.special.erf((x - b)/c) + 0.5)

	def normalErf(self, x, a, b, c):
		return 0.56419*a*np.exp(-(x-b)**2/(c**2)) + 0.5

	def energySpectrumTHL(self, slot=1, THLhigh=4975, THLlow=5000, THLstep=1, timestep=1, intPlot=True):
		# Description: Measure cumulative energy spectrum 
		#              by performing a threshold scan. The
		#              derivative of this spectrum resembles
		#              the energy spectrum.

		THLhigh = int(self.THLs[slot-1], 16)	

		assert THLhigh <= 8191, "energySpectrumDAC: DACHigh value set too high!"

		# Take data in integration mode: 
		#     Sum of deposited energy in ToT
		OMRCode = self.OMR
		if not isinstance(OMRCode, basestring):
			OMRCode[0] = 'DosiMode'
			# Select AnalogOut
			OMRCode[4] = 'V_ThA'

		else:
			OMRCode = (int(OMRCode, 16) | ((0x11) << 22))

			# Select AnalogOut
			OMRCode &= ~(0b11111 << 12)
			OMRCode |= getattr(self.__OMRAnalogOutSel, 'V_ThA')

		self.DPXWriteOMRCommand(slot, OMRCode)
		self.DPXWriteColSelCommand(slot, 0)

		lastTHL = 0
		if intPlot:
			plt.ion()
			fig, ax = plt.subplots()
			axDer = ax.twinx()

			# Empty plot
			lineCum, = ax.plot(np.nan, np.nan, label='Cumulative', color='cornflowerblue')
			lineDer, = axDer.plot(np.nan, np.nan, label='Derivative', color='crimson')
			# lineTHL, = axTHL.plot(np.nan, np.nan, label='THL', color='orange')

			# Settings
			plt.xlabel('THL (DAC)')
			ax.set_ylabel('Counts / s')
			axDer.set_ylabel('Derivative (a.u.)')
			# axTHL.set_ylabel('THL (V)')

			plt.grid()

		# Loop over DAC values
		THLList = []
		THLVList = []
		dataList = []

		THLRange = np.asarray(self.THLEdges)
		THLRange = THLRange[np.logical_and(THLRange > THLlow, THLRange < THLhigh)]

		# Savitzky-Golay filter the data
		# Ensure odd window length
		windowLength = int(len(THLRange[::THLstep]) / 10.)
		if not windowLength % 2:
			windowLength += 1

		# Empty bins
		self.DPXDataResetCommand(slot)
		for col in range(16):
			self.DPXWriteColSelCommand(slot, col)
			self.DPXReadBinDataDosiModeCommand(slot)

		# Measure time
		startTime = time.time()
		for THL in THLRange[::THLstep]:
			# Set threshold
			self.DPXWritePeripheryDACCommand(slot, self.peripherys + '%04x' % THL)

			THLList.append( THL )

			# Start frame 
			# OMRCode[1] = 'ClosedShutter'
			# self.DPXWriteOMRCommand(slot, OMRCode)
			
			# Wait to accumulate data
			time.sleep(timestep)

			pixelData = []
			# Loop over pixel columns
			for col in range(16):
				self.DPXWriteColSelCommand(slot, col) 
				data = np.sum(self.DPXReadBinDataDosiModeCommand(slot), axis=1)

				# Reject small pixels
				data = data[2:-2]

				# Gather in pixel matrix
				pixelData.append( data )

			# Scale data with readout time
			pixelData = np.asarray(pixelData) / float(time.time() - startTime)

			# After readout, new measurement begins
			startTime = time.time()

			# Flatten and sum data to get total number of events
			data = pixelData.flatten()[-55]
			print data
			dataList.append( np.sum(data) )

			# OMRCode[1] = 'OpenShutter'
			# self.DPXWriteOMRCommand(slot, OMRCode)
			# End frame

			# Update plot
			if intPlot:
				if len(THLList[1:]) < windowLength:
					continue
					
				print windowLength
				print dataList

				dataFilt = scipy.signal.savgol_filter(dataList, windowLength, 3)

				THLList_ = np.arange(len(THLList[1:]))

				# Cumulative
				lineCum.set_xdata( THLList_ )
				lineCum.set_ydata( dataFilt[1:] )
				
				# Derivative
				dataDer = np.diff(dataFilt[1:])
				lineDer.set_xdata( np.asarray(THLList_[:-1]) + 0.5)
				lineDer.set_ydata( dataDer )

				# Measure threshold voltage
				# lineTHL.set_xdata( THLList )
				# lineTHL.set_ydata( THLVList )

				ax.set_xlim(min(THLList_), max(THLList_))
				ax.set_ylim(0.9*min(dataFilt[1:]), 1.1*max(dataFilt[1:]))

				# axTHL.set_xlim(min(THLList), max(THLList))
				# axTHL.set_ylim(0.9*min(THLVList), 1.1*max(THLVList))

				if dataDer.size:
					dataDer *= max(dataFilt)/max(dataDer)
					axDer.set_xlim(min(THLList_), max(THLList_))
					axDer.set_ylim(0.9*min(dataDer[1:]), 1.1*max(dataDer[1:]))

				fig.canvas.draw()

		raw_input('')

		# Reset OMR and peripherys
		self.DPXWriteOMRCommand(slot, self.OMR)
		self.DPXWritePeripheryDACCommand(slot, self.peripherys + self.THLs[slot-1])

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
			if type(self.OMR) is list:
				OMRCode_ = self.OMRListToHex(self.OMR)
			OMRCode_ = int(OMRCode_, 16)

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
					if type(self.OMR) is list:
						OMRCode_ = self.OMRListToHex(self.OMR)
					OMRCode_ = int(OMRCode_, 16)
					OMRCode_ = OMRCode_ & ~(0b11111 << 12)
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

	def measureTHL(self, slot, THLhigh=8191, THLlow=0, THLstep=1, N=10):
		# Display execution time at the end
		startTime = time.time()

		# Select V_ThA for AnalogOut
		if type(self.OMR) is list:
			OMRCode_ = self.OMRListToHex(self.OMR)
		OMRCode_ = int(OMRCode_, 16)

		OMRCode_ &= ~(0b11111 << 12)
		OMRCode_ |= getattr(self.__OMRAnalogOutSel, 'V_ThA')
		self.DPXWriteOMRCommand(slot, hex(OMRCode_).split('0x')[-1])

		THLList = np.arange(THLlow, THLhigh, THLstep)
		THLVoltMean = []
		THLVoltErr = []
		print 'Measuring THL voltage!'
		for cnt, THL in enumerate(THLList):
			self.statusBar(float(cnt)/len(THLList) * 100 + 1)
			
			# Set threshold
			self.DPXWritePeripheryDACCommand(slot, self.peripherys + '%04x' % THL)

			# Measure multiple times
			ADCValList = []
			for i in range(N):
				ADCVal = float(int(self.MCGetADCvalue(), 16))
				ADCValList.append( ADCVal )

			THLVoltMean.append( np.mean(ADCValList) )
			THLVoltErr.append( np.std(ADCValList)/np.sqrt(N) )

		plt.errorbar(THLList, THLVoltMean, yerr=THLVoltErr, marker='x')
		plt.show()

		# Sort lists
		THLVoltMeanSort, THLListSort = zip(*sorted(zip(THLVoltMean, THLList)))
		print THLListSort
		plt.plot(THLVoltMeanSort, THLListSort)
		plt.show()

		cPickle.dump({'Volt': THLVoltMeanSort, 'THL': THLListSort}, open('THLCalib.p', 'wb'))

		print 'Execution time: %.2f min' % ((time.time() - startTime)/60.)

	def thresholdEqualizationConfig(self, configFn, reps=1, intPlot=False, resPlot=True):
		for i in range(1, 3 + 1):
			pixelDAC, THL, confMask = self.thresholdEqualization(i, reps, intPlot, resPlot)

			# Set values
			self.pixelDAC[i-1] = pixelDAC
			self.THLs[i-1] = THL
			self.confBits[i-1] = confMask

		self.writeConfig(configFn)

	def thresholdEqualization(self, slot, reps=1, intPlot=False, resPlot=True):
		THLlow, THLhigh = 5150, 5450
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

			'''
			resp = ''
			while resp != pixelCode:
				resp = self.DPXReadPixelDACCommand(slot)
			'''	

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
		self.sendCmd([self.__receiverHV, self.__subReceiverNone, self.__senderPC, self.__HVgetDAC, self.__commandNoneLength, self.__commandNone, self.__CRC])

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

	def VCVoltageSet1V8(self):
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
	def OMRListToHex(self, OMRCode):
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

		return OMRCode

	def DPXWriteOMRCommand(self, slot, OMRCode):
		if type(OMRCode) is list:
			OMRCode = self.OMRListToHex(OMRCode)

		self.sendCmd([self.getReceiverFromSlot(slot), self.__subReceiverNone, self.__senderPC, self.__DPXwriteOMRCommand, '%06d' % len(OMRCode), OMRCode, self.__CRC])

		return self.getDPXResponse()

	def DPXReadOMRCommand(self, slot):
		self.sendCmd([self.getReceiverFromSlot(slot), self.__subReceiverNone, self.__senderPC, self.__DPXreadOMRCommand, self.__commandNoneLength, self.__commandNone, self.__CRC])

		return self.getDPXResponse()

	def DPXReadDigitalThresholdsCommand(self, slot):
		self.sendCmd([self.getReceiverFromSlot(slot), self.__subReceiverNone, self.__senderPC, self.__DPXreadDigitalThresholdsCommand, self.__commandNoneLength, self.__commandNone, self.__CRC])

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

		return self.convertToDecimal(self.getDPXResponse(), 6)

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

	def getBinEdges(self, slot, energyDict, paramDict, transposePixelMatrix=False):
		a, b, c, t = paramDict['a'], paramDict['b'], paramDict['c'], paramDict['t']
		grayCode = [0, 1, 3, 2, 6, 4, 5, 7, 15, 13, 12, 14, 10, 11, 9, 8]

		if slot == 0:
			energyType = 'free'
		elif slot == 1:
			energyType = 'Al'
		else:
			energyType = 'Sn'

		binEdgeString = ''
		for pixel in range(256):
			if self.isBig(pixel):
				energyList = energyDict['large'][energyType]
			else:
				energyList = energyDict['large'][energyType]

			energyList = np.asarray()

			# Convert to ToT
			ToTList = self.energyToToT(energyList, a[pixel], b[pixel], c[pixel], d[pixel])

			for binEdge in range(16):
				grayC = grayCode[binEdge]
				ToT = int( ToTList[binEdge] )

				binEdgeString += ('%04x' % grayC)
				binEdgeString += ('%012x' % ToT)

		return binEdgeString

	def isLarge(self, pixel):
		if ( (pixel - 1) % 16 == 0 ) or ( pixel % 16 == 0 ) or ( (pixel + 1) % 16 == 0 ) or ( (pixel + 2) % 16 == 0 ):
			return False
		return True

	def energyToToTAtan(self, x, a, b, c, t):
		return np.where(x > b, a*(x - b) + c*np.arctan((x - b)/t), np.nan)

	def energyToToT(self, x, a, b, c, t, atan=False):
		if atan:
			return EnergyToToTAtan(x, a, b, c, t)
		else:
			return np.where(x > t, a*x + b + float(c)/(x - t), np.nan)

	def getResponse(self):
		res = self.ser.readline()
		while res[0] != '\x02':
			res = self.ser.readline()

		if DEBUG:
			print res
		return res

	def getDPXResponse(self):
		# res = ''
		# while not res:
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

	def pickleDump(self, outDict, outFn):
		# Check if file already exists
		while os.path.isfile(outFn):
			outFnFront = outFn.split('.')[0]
			outFnFrontSplit = outFnFront.split('_')
			if len(outFnFrontSplit) >= 2:
				if outFnFrontSplit[-1].isdigit():
					fnNum = int( outFnFrontSplit[-1] ) + 1
					outFn = ''.join(outFnFrontSplit[:-1]) + '_' + str(fnNum) + '.p'
				else:
					outFn = outFnFront + '_1.p'
			else:
				outFn = outFnFront + '_1.p'


		with open(outFn, 'w') as f:
			cPickle.dump(outDict, f)

	def getTestPulseVoltageDAC(self, slot, DACVal, energy=False):
		# Set coarse and fine voltage of test pulse
		peripheryDACcode = int(self.peripherys + self.THLs[slot-1], 16)

		if energy:
			# Use nominal value of test capacitor
			C = 5.14e-15

			deltaV = DACVal * scipy.constants.e / (C * 3.62)

			assert deltaV < 1.275, "TestPulse Voltage: The energy of the test pulse was set too high! Has to be less than or equal 148 keV."

			# Set coarse voltage to 0
			voltageDiv = 2.5e-3
			DACVal = int((1.275 - deltaV) / voltageDiv)
		else:
			assert DACVal >= 0, 'Minimum THL value must be at least 0'
			assert DACVal <= 0x1ff, 'Maximum THL value mustn\'t be greater than %d' % 0x1ff

		# Delete current values
		peripheryDACcode &= ~(0xff << 32)	# coarse
		peripheryDACcode &= ~(0x1ff << 16)	# fine

		# Adjust only fine voltage
		peripheryDACcode |= (DACVal << 16)
		peripheryDACcode |= (0xff << 32)
		print '%032x' % peripheryDACcode
		print DACVal

		return '%032x' % peripheryDACcode

	def splitPerihperyDACs(self, code, perc=True):
		if perc:
			percEightBit = float( 2**8 )
			percNineBit = float( 2**9 )
			percThirteenBit = float( 2**13 )
		else:
			percEightBit, percNineBit, percThirteenBit = 1, 1, 1


		code = bin(int(code, 16)).split('0b')[-1]
		print code
		d = {'V_ThA': int(code[115:], 2) / percThirteenBit,
			'V_tpref_fine': int(code[103:112], 2) / percNineBit,
			'V_tpref_coarse': int(code[88:96], 2) / percEightBit, 
			'I_tpbufout': int(code[80:88], 2) / percEightBit, 
			'I_tpbufin': int(code[72:80], 2) / percEightBit, 
			'I_disc2': int(code[64:72], 2) / percEightBit, 
			'I_disc1': int(code[56:64], 2) / percEightBit, 
			'V_casc_preamp': int(code[48:56], 2) / percEightBit, 
			'V_gnd': int(code[40:48], 2) / percEightBit, 
			'I_preamp': int(code[32:40], 2) / percEightBit, 
			'V_fbk': int(code[24:32], 2) / percEightBit, 
			'I_krum': int(code[16:24], 2) / percEightBit, 
			'I_pixeldac': int(code[8:16], 2) / percEightBit, 
			'V_casc_reset': int(code[:8], 2) / percEightBit}

		print 'PeripheryDAC values in',
		if perc:
			print 'percent:'
		else:
			print 'DAC:'
		print yaml.dump(d, indent=4, default_flow_style = False)
		print 

		return d

	def periheryDACsDictToCode(self, d, perc=True):
		if perc:
			percEightBit = float( 2**8 )
			percNineBit = float( 2**9 )
			percThirteenBit = float( 2**13 )
		else:
			percEightBit, percNineBit, percThirteenBit = 1, 1, 1

		code = 0
		code |= int(d['V_ThA'] * percThirteenBit) 
		code |= (int(d['V_tpref_fine'] * percNineBit) << 25 - 9)
		code |= (int(d['V_tpref_coarse'] * percEightBit) << 40 - 8)
		code |= (int(d['I_tpbufout'] * percEightBit) << 48 - 8)
		code |= (int(d['I_tpbufin'] * percEightBit) << 56 - 8)
		code |= (int(d['I_disc2'] * percEightBit) << 64 - 8)
		code |= (int(d['I_disc1'] * percEightBit) << 72 - 8)
		code |= (int(d['V_casc_preamp'] * percEightBit) << 80 - 8)
		code |= (int(d['V_gnd'] * percEightBit) << 88 - 8)
		code |= (int(d['I_preamp'] * percEightBit) << 96 - 8)
		code |= (int(d['V_fbk'] * percEightBit) << 104 - 8)
		code |= (int(d['I_krum'] * percEightBit) << 112 - 8)
		code |= (int(d['I_pixeldac'] * percEightBit) << 120 - 8)
		code |= (int(d['V_casc_reset'] * percEightBit) << 128 - 8)

		return hex(code).split('0x')[-1]

	def getTHLfromVolt(self, V):
		return self.THLCalib[abs(V - self.voltCalib).argmin()]

	def getVoltFromTHL(self, THL):
		return self.voltCalib[np.argwhere(self.THLCalib == THL)]

	def normal(self, x, A, mu, sigma):
		return A * np.exp(-(x - mu)/(2 * sigma**2))

	def getColor(self, c, N, idx):
		import matplotlib as mpl
		cmap = mpl.cm.get_cmap(c)
		norm = mpl.colors.Normalize(vmin=0.0, vmax=N - 1)
		return cmap(norm(idx))

if __name__ == '__main__':
	main()
