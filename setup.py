import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(name='dpx_control',
    version='0.3.3',
    description='DPX control software',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Sebastian Schmidt',
    author_email='schm.seb@gmail.com',
    url="https://github.com/dosepix/dpx_control",
    project_urls={
        "Bug Tracker": "https://github.com/dosepix/dpx_control/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    license='GNU GPLv3',
    entry_points={
        'console_scripts' : [
            'dpx_control = dpx_control.dpx_control:main',
        ]
    },
    packages=["dpx_control"],
    install_requires=[
        'matplotlib',
        'numpy',
        'scipy',
        'pyserial',
        'configparser',
        'tqdm'
    ]
)
