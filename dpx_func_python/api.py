#!/usr/bin/env python
from __future__ import print_function
import startupDPX as dpx
import sys
import zerorpc

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
    def createObject(self):
        self.dpxObj = dpx.Dosepix('/dev/ttyUSB0', 2e6, 'DPXConfig.conf')
        return str('Success!')

    def test(self):
        return 'Test'

if __name__ == '__main__':
    main()
