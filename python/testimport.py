knownPackages='os'
knownPackages = 'numpy'
knownPackageList=['numpy','scipy','matplotlib','networkx']
# import knownPackages

import os
import importlib
#importlib.import_module(knownPackages)
import pvtools

for packageName in knownPackageList:
    try:
        importlib.import_module(packageName)
    except ModuleNotFoundError:
        print(f"The package named {packageName} is not installed with this python installation. Will attempt to automatically install with pip.\n\nTo manually install this use:\npip install {packageName}\n\t-or-\n[your python executable] -m pip install {packageName}")
        os.system(f'python -m pip install {packageName}')
