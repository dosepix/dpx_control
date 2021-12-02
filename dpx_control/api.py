#!/usr/bin/env python
from __future__ import print_function
import dpx_control as dpx
import sys
import zerorpc
import numpy as np
import json
import pandas as pd
import time
import gevent
import serial.tools.list_ports

def main():
    addr = 'tcp://127.0.0.1:' + parse_port()
    s = zerorpc.Server(DPXapi())
    s.bind(addr)
    print('start running on {}'.format(addr))
    s.run()

def parse_port():
    port = 4242
    try:
        port = int(sys.argv[1])
    except Exception as e:
        pass
    return '{}'.format(port)

class DPXapi(object):
    # Function to check if server is ready
    def echo(self, text):
        """echo any text"""
        return text

    def connectDPX(self):
        self.dpxObj = dpx.Dosepix('/dev/ttyUSB0', 2e6, 'DPXConfig.conf')
        self.dpxObj.setGUI()
        return str('Success!')

    def disconnectDPX(self):
        del self.dpxObj


    def findDPX(self):
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            if 'FT232R USB UART' in p:
                return True
        else:
            return False

    def test(self):
        # return 'Test'
        return [1, 2, 3]

    @zerorpc.stream
    def measureToT(self, out, cnt):
        # Function yields histogram data
        rand = []
        bins = np.arange(100)

        cnt = 100

        for f in np.arange(int(cnt)):
            gevent.sleep(0)
            d = np.random.normal(30, 10, 1000)
            rand += list( d )
            h, b = np.histogram(rand, bins=bins)

            data = json.dumps({'frame': f / 100., 'data': [{'ToT': b[:-1][i], 'Counts': h[i]} for i in range(len(h))]})
            yield data
            time.sleep(0.1)
        # return str('Not implemented!')
        return

        # Settings
        slot = 1
        cnt = int( cnt )    # Ensure cnt is given as int
        # cnt = 10000
        # out = 'ToTMeasurement/'

        # Get generator object
        ToTgen = self.dpxObj.measureToT(slot=slot, outDir=out, cnt=cnt, storeEmpty=False, logTemp=False, paramsDict=None, intPlot=False)
        for b, h in ToTgen:
            gevent.sleep(0)     # Required for heartbeats
            data = json.dumps({'frame': 1., 'data': [{'ToT': b[:-1][i], 'Counts': h[i]} for i in range(len(h))]})
            yield data
        return

    @zerorpc.stream
    def testPlot(self):
        # Processing example

        '''
        for x in np.arange(100):
            # Do some calculations
            yield pd.DataFrame({'status': x}).to_json()
        '''

        xStart = np.arange(10000)
        for x_ in xStart:
            gevent.sleep(0)     # Required for heartbeats
            x = np.linspace(x_, x_+10, 100).tolist()
            y = np.sin(x).tolist()
            # data = pd.DataFrame({'status': x_, 'data': {'x': x.tolist(), 'y': y.tolist()}})
            data = json.dumps({'status': x_ / 10000., 'data': [{'x': x[i], 'y': y[i]} for i in range(len(x))]})
            # print( data )
            yield data # pd.DataFrame({'x': x.tolist(), 'y': y.tolist()}).to_json(orient='records')
            time.sleep(.1)

if __name__ == '__main__':
    main()

