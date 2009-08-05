########################################################################
# This module is regression test module for python version of phasesym #
# This module compares between matlab outputs and python outputs for   #
# same inputs as a part of regression tests                            #
#                                                                      #
# This module is part of pyphasesym package                            #
########################################################################
""" This is a regressio"""

import sys
import os

import unittest
import nose
import optparse
import Image
import glob

import scipy as sp
import numpy as np
from numpy.testing import *

from scipy.io import *
from mainPhasesym import *

# Get all the mat files for regression test
matfiles = glob.glob('matfiles/*.mat') 

#-------------------------------------------------------------------------------
def compare_py_mat(matfile):

    """Regression test function

    Compare output of matlab and python for same inputs
    Inputs obtained from stored mat files 
    Output for matlab obtained from stored mat files
    Output for python obtained by invoking phasesym function from 
    mainPhasesym.py module"""

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
    assert_array_almost_equal(p_phaseSym, matVars['phaseSym'])
    assert_array_almost_equal(p_orientation, matVars['orientation'])

#-------------------------------------------------------------------------------
def test_generator():

    """ Generate tests as long mat files exists files 

    mat files stored in matfiles/ directory of the package"""
    
    for matfile in matfiles:
        yield compare_py_mat, matfile # Do we need **kwargs here 
    

        


