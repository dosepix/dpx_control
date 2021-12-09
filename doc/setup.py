from setuptools import setup

setup(name='dpx_control',
    version='0.2',
    description='DPX control software',
    author='Sebastian Schmidt',
    author_email='schm.seb@gmail.com',
    license='MIT',
    packages=['dpx_control'],
    entry_points={
        'console_scripts' : [
            'dpx_control = dpx_control.dpx_control:main',
        ]
    },
    install_requires=[
        'matplotlib',
        'numpy',
        'scipy',
        'pyserial',
        'configparser',
        'tqdm'
    ])
