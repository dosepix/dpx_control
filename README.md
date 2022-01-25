# Dosepix Control Software for python3

Module name: dpx\_control  
Author: Sebastian Schmidt  
E-Mail: schm.seb@gmail.com  

## Installation

There are multiple ways to install the module. The easiest one is to use a virtual environment. More experiened users might consider to install the module directly. Please refer to the instructions below and ensure that python 3 is used.  

### Virtual Environment Installation

First,  a directory for the virtual environment has to be created. To provide an example, it is called `dpx_venv` in the following.  Afterwards, the environment is created via

```bash
python3 -m venv dpx_venv
```

Activate the virtual environment by executing

```bash
source dpx_virtenv/bin/activate
```

If everything worked correctly, the name of your virtual environment should appear in parentheses in front of your command prompt. Finally, proceed like described in the "Direct Installation"-section below.

### Direct Installation

`sudo` might be needed in order to provide installation privileges. This won't be necessary when installing in an virtual environment.  

#### via pip

If no administrator access is possible, add the parameter `--user` right behind `install`.

```bash
python3 -m pip install /path/to/package
```

If you want to modify the code later on, use  

```bash
python3 -m pip install -e /path/to/package
```

instead.

##### via `setup.py`

Execute in the module's main directory:

```bash
python3 setup.py install
```

If you want to modify the code later on, use  

```bash
python3 setup.py develop
```

instead.

## Examples

### Dosepix initialization

First, import the module.

```python
import dpx_control
```

The connection to the Dosepix test board is established via:

```python
dpx = dpx_control.Dosepix(portName, baudRate=2e6, configFN=None, thl_calib_files=None, params_file=None, bin_edges_file=None)
```

This creates an object `dpx` of the class `Dosepix`.  
Important parameters are:  

| Parameter | Function |
| :-------- | :------- |
| `portName`           | Name of the used com-port of the PC. For Linux, it usually is `/dev/ttyUSB0`. For Windows, the port name has the form of 'COMX'. |
| `baudRate`           | Used baud rate of the connection between DPX test board and PC. This is set to 2e6 in the board's current firmware and shouldn't be modified here. |
| `configFn`           | Configuration file containing important parameters of the used Dosepix detectors. |
| `thl_calib_files`    | The DAC value and corresponding voltage of the threshold (THL) show a dependency of a sloped sawtooth. By measuring this dependency, a corrected threshold value can be used. Only important for certain tasks like threshold equalization or threshold scan measurements. |
| `params_file`        | File containing the calibration curve parameters (a, b, c, t) for each detector and pixel. Only needed for dose measurements as it is used to specify the bin edges in DosiMode. |
| `bin_edges_file`     | File containing the bin edges used in DosiMode. If `params_file` is set, the file should contain the bin edges in energy. Else, it should contain the bin edges in ToT. |
| `eye_lens`           | Set to `True` if hardware for eye lens dosimetry is used, as it only utilizes a single slot. Standard value is `False` |

A measurement can be started by using the `dpx` object. For example a ToT-measurement:

```python
dpx.measureToT(slot=[1, 2, 3], intPlot=True, cnt=10000, storeEmpty=True, logTemp=True)
```

See documentation for more info.

### Equalization

See the [equalization-script](examples/equalization.py).

First, important parameters are defined:

```python
PORT = '/dev/ttyUSB0'
CONFIG_FN = 'DPXConfig.conf'
CONFIG_DIR = 'config/'
CHIP_NUMS = [22, 6, 109]
CALIB_THL = False
```

`CONFIG_FN` specifies the file in which the configuration of the current setup is stored. This file will be created in the directory specified in `CONFIG_DIR`.  If the configuration directory does not exist, the program will create the folder by itself.  
`CHIP_NUMS` are the identification numbers of the used detectors, usually written on the backside of the COB.  

This equalization procedure includes THL measurementes which are optional. They are performed if the flag `CALIB_THL` is set to `True`. Afterwards, a THL-calibration file is created for each detector. This improves the equalization procedure but is not a necessity. At the current revision of the DPX test board, the measurement of the relation between THL DAC and THL voltage is only possible at Slot 1. Therefore, only one detector can be measured at a time.  
**IMPORTANT:** the board has to be disconnected from power when switching detectors!  

Afterwards, the command  

```python
dpx.thresholdEqualizationConfig(CONFIG_DIR + '/' + CONFIG_FN, I_pixeldac=None, reps=1, intPlot=False, resPlot=True)
```

performs the threshold equalization and stores the results in the specified configuration file. If `intPlot` is set to `True`, equalization results are shown for each detector once the equalization is done.
