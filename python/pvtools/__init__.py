""" This file causes this directory to be seen as an importable python module."""
import os
import importlib

# for the purpose of trying to run by the command line in some use cases where python package installation is repeat and onerous,
# import and install of all required packages is done automatically

externalPackageList=['numpy','scipy','matplotlib','networkx']
# import knownPackages

for packageName in externalPackageList:
    try:
        importlib.import_module(packageName)
    except ModuleNotFoundError:
        print(f"The package named {packageName} is not installed with this python installation. Will attempt to automatically install with pip.\n\nTo manually install this use:\npip install {packageName}\n\t-or-\n[your python executable] -m pip install {packageName}")
        os.system(f'python -m pip install {packageName}')    

from .readpvpfile import readpvpfile
from .readpvpheader import readpvpheader
from .writepvpfile import writepvpfile
from .display import view
from .pvpFile import pvpOpen
from .arrangedictionary import arrangedictionary
from .readenergyprobe import readenergyprobe
from .readlayerprobe import readlayerprobe
