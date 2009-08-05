# Test module

import sys
import os

import unittest
import nose
import random
import optparse
import Image
import glob

import scipy as sp
import numpy as np
from numpy.testing import *

from scipy.io import *
from mainPhasesym import *

matfiles = glob.glob('*.mat') 

def compare_py_mat(matfile):

    """Main module to compare the output generated from
    matlab and output generated from python program for phasesym"""

    # Here you could have **kwargs as dicts so that you don't have to type
    # all the variables. You can just reference the variables

    matVars = sp.io.loadmat(matfile)
    p_phaseSym, p_orientation = phasesym(matVars['image'], 
                                         matVars['scale'], 
                                         matVars['orient'],
                                         matVars['minWaveLength'], 
                                         matVars['mult'], 
                                         matVars['sigmaOnf'], 
                                         matVars['dThetaOnSigma'], 
                                         matVars['k'], 
                                         matVars['polarity'])

    # have kwargs
    pouts = p_phaseSym
    mouts = matVars['phaseSym']
    assert_array_almost_equal(pouts, mouts)

def test_generator():

    """ Generate tests as long as  *.mat files (stored in matfiles variable) 
    exist in directory"""
    
    for matfile in matfiles:
        yield compare_py_mat, matfile # Do we need **kwargs here 
    

        


