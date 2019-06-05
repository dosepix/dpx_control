import numpy as np

import dpx_settings as ds

class DPX_test_pulse(object):
    def testPulseInit(self, slot, column=0):
        # Set Polarity to hole and Photon Counting Mode in OMR
        OMRCode = self.OMR
        if not isinstance(self.OMR, basestring):
            OMRCode[3] = 'hole'
        else:
            OMRCode = '%04x' % (int(self.OMR, 16) & ~(1 << 17))

        if not isinstance(self.OMR, basestring):
            OMRCode[0] = 'DosiMode'
        else:
            OMRCode = '%04x' % ((int(self.OMR, 16) & ~((0b11) << 22)) | (0b10 << 22))
        
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
        confBits.fill(getattr(ds._ConfBits, 'MaskBit'))
        confBits[column] = [getattr(ds._ConfBits, 'TestBit_Analog')] * 16
        # print confBits

        # confBits = np.asarray( [int(num, 16) for num in textwrap.wrap(self.confBits[slot-1], 2)] )
        # confBits[confBits != getattr(ds._ConfBits, 'MaskBit')] = getattr(ds._ConfBits, 'TestBit_Analog')

        self.DPXWriteConfigurationCommand(slot, ''.join(['%02x' % conf for conf in confBits.flatten()]))

    def getTestPulseVoltageDAC(self, slot, DACVal, energy=False):
        # Set coarse and fine voltage of test pulse
        peripheryDACcode = int(self.peripherys + self.THLs[slot-1], 16)

        if energy:
            # Use nominal value of test capacitor
            # and DACVal as energy
            C = 5.14e-15

            deltaV = DACVal * scipy.constants.e / (C * 3.62)

            assert deltaV < 1.275, "TestPulse Voltage: The energy of the test pulse was set too high! Has to be less than or equal to 148 keV."

            # Set coarse voltage to 0
            voltageDiv = 2.5e-3
            DACVal = int((1.275 - deltaV) / voltageDiv)
        else:
            assert DACVal >= 0, 'Minimum THL value must be at least 0'
            assert DACVal <= 0x1ff, 'Maximum THL value mustn\'t be greater than %d' % 0x1ff

        # Delete current values
        peripheryDACcode &= ~(0xff << 32)   # coarse
        peripheryDACcode &= ~(0x1ff << 16)  # fine

        # Adjust fine voltage only
        peripheryDACcode |= (DACVal << 16)
        peripheryDACcode |= (0xff << 32)
        print '%032x' % peripheryDACcode
        print DACVal

        return '%032x' % peripheryDACcode

