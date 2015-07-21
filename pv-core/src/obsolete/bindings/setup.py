from distutils.core import setup, Extension

module1 = Extension('pypv', sources = ['pypv.cpp'],
							libraries = ['pv'],
							library_dirs = ['..'])

setup (name = 'PyPv', version = '1.0',
	   description = 'This is a Python interface to Petavision package',
	   ext_modules = [module1])

