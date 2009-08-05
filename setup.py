#!/usr/bin/env python
"""setup module for python version of phasesym code"""
from distutils.core import setup

setup(name='Distutils',
      version='1.0',
      description='Python implementation of phasesym program',
      author='Abhijit Bendale',
      author_email='bendale@mit.edu',
      url='http://www.python.org/sigs/distutils-sig/',
      packages=['main_phasesym','test_reg'],
     )
