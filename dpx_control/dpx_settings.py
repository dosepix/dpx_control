from __future__ import print_function

from collections import namedtuple
# === CONSTANTS ===
_startOfTransmission   = b"\x02"
_endOfTransmission     = b"\x03"

# = Receiver =
_receiverDPX1 = '01'   # Dosepix
_receiverDPX2 = '02'
_receiverDPX3 = '03'
_receiverMC = '04'     # Microcontroller
_receiverVC = '09'     # Voltage Controller
_receiverHV = '11'     # High Voltage

# = Subreceiber =
_subReceiverNone = '000'

# = Sender =
_senderPC = '10'

# = Commands =
_commandNone = ''
_commandNoneLength = '0' * 6

# = HV Commands =
_HVenable = '000'
_HVdisable = '001'
_HVisEnabled = '002'
_HVsetDAC = '003'
_HVgetDAC = '004'

# = VC Commands =
_VCset3V3 = '000'
_VCset1V8 = '001'
_VCgetVoltage = '002'

# = MC Commands =
_MCgetVersion = '000'
_MCLEDenable = '001'
_MCLEDdisable = '002'
_MCgetADCvalue = '003'
_MCsetSPIclock1 = '004'
_MCsetSPIclock2 = '005'
_MCsetSPIclock3 = '006'

# = DPX Commands =
_DPXwriteOMRCommand = '001' 
_DPXwriteConfigurationCommand = '002'
_DPXwriteSingleThresholdCommand = '003'
_DPXwritePixelDACCommand = '004'
_DPXwritePeripheryDACCommand = '005'
_DPXwriteColSelCommand = '006'
_DPXburnSingleFuseCommand = '007'
_DPXreadOMRCommand = '008'
_DPXreadConfigurationCommand = '009'
_DPXreadDigitalThresholdsCommand = '010'
_DPXreadPixelDACCommand = '011'
_DPXreadPeripheryDACCommand = '012'
_DPXreadColumnTestPulseCommand = '013'
_DPXreadColSelCommand = '014'
_DPXreadChipIdCommand = '015'

_DPXglobalResetCommand = '020'
_DPXdataResetCommand = '021'

_DPXreadToTDataDosiModeCommand = '050'
_DPXreadBinDataDosiModeCommand = '051'
_DPXreadToTDatakVpModeCommand = '053'
_DPXreadToTDataIntegrationModeCommand = '054'

_DPXgeneralTestPulse = '057'
_DPXreadToTDataInkVpModeWithFixedFrameSizeCommand = '066'
_DPXgeneralMultiTestPulse = '067'

_DPXstartStreamingReadout = '068'
_DPXstopStreamingReadout ='069'

# = Custom Functions =
_DPXreadToTDataDosiModeMultiCommand = '090'

# = CRC =
_CRC = 'FFFF'

# = OMR =
OMROperationModeType = namedtuple("OMROperationMode", "DosiMode TestWakeUp PCMode IntegrationMode")
_OMROperationMode = OMROperationModeType(
    DosiMode = 0b00 << 22,
    TestWakeUp = 0b01 << 22,
    PCMode = 0b10 << 22,
    IntegrationMode = 0b11 << 22)

OMRGlobalShutterType = namedtuple("OMRGlobalShutter", "ClosedShutter OpenShutter")
_OMRGlobalShutter = OMRGlobalShutterType(
    ClosedShutter = 0b0 << 21,
    OpenShutter = 0b1 << 21)

# Do not use 200 MHz!
OMRPLLType = namedtuple("OMRPLL", "Direct f16_6MHz f20MHz f25MHz f33_2MHz f50MHz f100MHz")
_OMRPLL = OMRPLLType(Direct = 0b000 << 18,
    f16_6MHz = 0b001 << 18,
    f20MHz = 0b010 << 18,
    f25MHz = 0b011 << 18,
    f33_2MHz = 0b100 << 18,
    f50MHz = 0b101 << 18,
    f100MHz = 0b110 << 18)

OMRPolarityType = namedtuple("OMRPolarity", "electron hole")
_OMRPolarity = OMRPolarityType(
    hole = 0b0 << 17,
    electron = 0b1 << 17)

OMRAnalogOutSelType = namedtuple("OMRAnalogOutSel", "V_ThA V_TPref_fine V_casc_preamp V_fbk V_TPref_coarse V_gnd I_preamp I_disc1 I_disc2 V_TPbufout V_TPbufin I_krum I_dac_pixel V_bandgap V_casc_krum Temperature V_per_bias V_cascode_bias High_Z")
_OMRAnalogOutSel = OMRAnalogOutSelType(
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

OMRAnalogInSelType = namedtuple("OMRAnalogInSel", "V_ThA V_TPref_fine V_casc_preamp V_fbk V_TPref_coarse V_gnd I_preamp I_disc1 I_disc2 V_TPbufout V_TPbufin I_krum I_dac_pixel V_bandgap V_casc_krum Temperature V_per_bias V_cascode_bias V_no")
_OMRAnalogInSel = OMRAnalogInSelType(
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
_OMRDisableColClkGate = OMRDisableColClkGateType(
    Enabled = 0b0 << 6,
    Disabled = 0b1 << 6)

# = ConfBits =
ConfBitsType = namedtuple("ConfBits", "MaskBit TestBit_Analog TestBit_Digital")
_ConfBits = ConfBitsType(
    MaskBit = 0b1 << 2,
    TestBit_Analog = 0b1 << 1,
    TestBit_Digital = 0b1 << 0)
