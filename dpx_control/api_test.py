#!/usr/bin/env python
import api

def main():
	dpxapi = api.DPXapi()
	print( dpxapi.findDPX() )

	'''
	res = dpxapi.measureToT('test', 100)
	for r in res:
		print( r )
	'''

if __name__ == '__main__':
	main()