from setuptools import setup

setup(name='dpx_func_python',
	version='0.1',
	description='DPX control software',
	author='Sebastian Schmidt',
	author_email='sebastian.seb.schmidt@fau.de',
	license='MIT',
	packages=['dpx_func_python'],
	entry_points={
		'console_scripts' : [
			'dpx_func_python = dpx_func_python.dpx_func_python:main',
		]
	},
	install_requires=[
                'matplotlib',
		'pytest',
		'hickle',
		'pandas',
		'numpy',
		'scipy',
		'pyserial',
		'pyyaml',
		'configparser',
		# 'sphinx',
		# 'sphinx_rtd_theme'
	])
