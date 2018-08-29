#!/usr/bin/env python
import curses
import curses_gui
import sys
import dpx_func_python

options = ['THL Equalization', 'Measure ToT', 'Measure Dose', 'Measure Integration', 'Measure THL', 'Measure Temperature', 'Exit']
functions = []

dpx = dpx_func_python.Dosepix('/dev/ttyUSB0', 2e6, 'Configurations/DPXConfig_22.conf')

class DPXGUI(object):
    def __init__(self, stdscreen):
        self.screen = stdscreen
        curses.curs_set(0)

        submenu_items = [
                ('THL Equalization', dpx.thresholdEqualizationConfig('DPXConfig.conf', I_pixeldac=None, reps=1, intPlot=False, resPlot=True)),
                ('flash', curses.flash)
                ]
        submenu = curses_gui.GUI(submenu_items, self.screen)

        main_menu_items = [
                ('beep', curses.beep),
                ('flash', curses.flash),
                ('submenu', submenu.display)
                ]
        main_menu = curses_gui.GUI(main_menu_items, self.screen)
        main_menu.display()

curses.wrapper(DPXGUI)
dpx.close()
sys.exit()

screen = curses_gui.GUI(options)
test = screen.serial_ports()
# sys.exit()
# selection = screen.selectOption()
screen.display()
screen.close()

# print
print test


if selection == len(options):
    screen.close()
    # dpx.close()
    sys.exit()

sys.exit()

# dpx.ADCWatch(1, ['V_ThA', 'V_TPref_fine', 'V_casc_preamp', 'V_fbk', 'V_TPref_coarse', 'V_gnd', 'I_preamp', 'I_disc1', 'I_disc2', 'V_TPbufout', 'V_TPbufin', 'I_krum', 'I_dac_pixel', 'V_bandgap', 'V_casc_krum', 'V_per_bias', 'V_cascode_bias', 'Temperature', 'I_preamp'], cnt=0)
# dpx.energySpectrumTHL(1, THLhigh=8000, THLlow=int(dpx.THLs[0], 16), THLstep=25, timestep=1, intPlot=True)
# dpx.ToTtoTHL(slot=1, column='all', THLstep=1, valueLow=1.5e3, valueHigh=30e3, valueCount=20, energy=True, plot=False, outFn='ToTtoTHL.p')

# dpx.energySpectrumTHL(1)
dpx.measureToT(1, intPlot=False)
# dpx.testPulseDosi(1)
# dpx.testPulseSigma(1)
# dpx.testPulseToT(1, 10)

# dpx.measureDose(measurement_time=120)
# dpx.measureIntegration()
# dpx.temperatureWatch(slot=1, column='all', frames=1000, energyRange=(25.e3, 25.e3), fn='TemperatureToT_DPX22_125keV.p', intplot=True)

# dpx.measureTHL(1, fn='THLCalib_test.p', plot=False)
# dpx.thresholdEqualizationConfig('Configurations/DPXConfig_12.conf', I_pixeldac=None, reps=1, intPlot=False, resPlot=True)

# Close connection
dpx.close()