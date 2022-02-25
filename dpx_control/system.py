from __future__ import print_function
import os
import os.path
import sys
import numpy as np
import json

class System(object):
    def pickleDump(self, outDict, outFn, overwrite=False, method='json'):
        if not '.' in outFn:
            ending = '.' + method
        else:
            ending = '.' + outFn.split('.')[-1]

        # Check if file already exists
        if not overwrite:
            while os.path.isfile(outFn):
                # Split dir and file
                outFnSplit = outFn.split('/')
                if len(outFnSplit) >= 2:
                    directory, outFn = outFnSplit[:-1], outFnSplit[-1]
                else:
                    directory = None

                outFnFront = outFn.split('.')[0]
                outFnFrontSplit = outFnFront.split('_')
                if len(outFnFrontSplit) >= 2:
                    if outFnFrontSplit[-1].isdigit():
                        fnNum = int( outFnFrontSplit[-1] ) + 1
                        outFn = ''.join(outFnFrontSplit[:-1]) + '_' + str(fnNum) + ending
                    else:
                        outFn = outFnFront + '_1' + ending
                else:
                    outFn = outFnFront + '_1' + ending

                # Reattach directory
                if directory:
                    outFn = '/'.join(directory + [outFn])

        with open(outFn, 'w') as f:
            json.dump(outDict, f, cls=self.NumpyEncoder)

    # Convert every array in dictionary to list
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    def makeDirectory(self, directory):
        while os.path.isdir(directory):
            dir_front = directory.split('/')[0]
            dir_front_split = dir_front.split('_')
            if len(dir_front_split) >= 2:
                if dir_front_split[-1].isdigit():
                    dir_num = int(dir_front_split[-1]) + 1
                    directory = ''.join(dir_front_split[:-1]) + '_' + str(dir_num) + '/'
                else:
                    directory = dir_front + '_1/'
            else:
                directory = dir_front + '_1/'

        os.makedirs(directory)
        return directory
