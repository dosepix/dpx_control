#!/usr/bin/env python
from __future__ import print_function
import dpx_func_python as dpx
import sys
import zerorpc
import numpy as np
import json
import pandas as pd
import time

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

    def test(self):
        # return 'Test'
        return [1, 2, 3]

    def measureToT():
        return

    @zerorpc.stream
    def testPlot(self):
        # Processing example

        '''
        for x in np.arange(100):
            # Do some calculations
            yield pd.DataFrame({'status': x}).to_json()
        '''

        xStart = np.arange(100)
        for x_ in xStart:
            x = np.linspace(x_, x_+10, 100).tolist()
            y = np.sin(x).tolist()
            # data = pd.DataFrame({'status': x_, 'data': {'x': x.tolist(), 'y': y.tolist()}})
            data = json.dumps({'status': x_, 'data': [{'x': x[i], 'y': y[i]} for i in range(len(x))]})
            # print( data )
            yield data # pd.DataFrame({'x': x.tolist(), 'y': y.tolist()}).to_json(orient='records')
            time.sleep(.1)

if __name__ == '__main__':
    main()
