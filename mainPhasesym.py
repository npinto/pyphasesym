#######################################################################################
# This function calculates the phase symmetry of points in an image.	              #
# This is a contrast invariant measure of symmetry.  This function can be             #
# used as a line and blob detector.  The greyscale 'polarity' of the lines            #
# that you want to find can be specified. The convolutions are done with FFT          #
# usage <input_filename> <output_filename>                                            #
#######################################################################################

import os
import sys
import math
import optparse
import Image
import ImageOps

import numpy as np
import scipy as sp
from numpy.fft import * 
from scipy.io import *

from mlabwrap import mlab

MODULE_DTYPE = "float64"

#Default values. These are used if user does not specify these values
DEFAULT_NSCALE = 5.
DEFAULT_ORIENT = 6.
DEFAULT_MWAVELENGTH = 3.
DEFAULT_MULT = 2.1
DEFAULT_SIGMAONF = 0.55
DEFAULT_DTHETASEGMA = 1.2
DEFAULT_NSTD = 2.
DEFAULT_POLARITY = 0
DEFAULT_OVERWRITE = False

#--------------------------------------------------------------
def filterIntialization(rows, cols):
    """Filter initialization components """
    
    # Set up X and Y matrices with ranges normalised to +/- 0.5
    # The following code adjusts things appropriately for odd and even values
    # of rows and columns.

    if np.mod(cols,2):
        xr = np.arange(-(cols - 1)/2,(cols - 1)/2 + 1, dtype = "float32" )/(cols - 1)
    else:
        xr= np.arange(-cols/2, (cols/2 - 1) + 1, dtype = "float32")/cols
        
    if np.mod(rows,2):
        yr = np.arange( -(rows - 1)/2, (rows - 1)/2 + 1, dtype="float32")/(rows-1)
    else:
        yr = np.arange( -rows/2, (rows/2 - 1) +1, dtype = "float32")/rows

    x,y = np.meshgrid(xr,yr)
    
    # Matrix values contain *normalised* radius from centre.
    radius = np.sqrt(x**2 + y**2)        

    # Matrix values contain polar angle.
    theta = np.arctan2(-y,x)

    # Quadrant shift radius and theta so that filters	       
    # are constructed with 0 frequency at the corners.       
    # Get rid of the 0 radius value at the 0		       
    # frequency point (now at top-left corner)	       
    # so that taking the log of the radius will 	       
    # not cause trouble.				       

    radius = np.fft.ifftshift(radius)
    theta = np.fft.ifftshift(theta)
    radius[0][0] = 1

    sintheta = np.sin(theta)
    costheta = np.cos(theta)

    return sintheta, costheta, radius

#--------------------------------------------------------------
def getLowPassFilter(rows, cols, cutoff, n):

    if cutoff < 0 or cutoff > 0.5:
        print "ERROR:cutoff frequency must be between 0 and 0.5"
        #TODO: check function controls here

    if n < 1.:
        print "ERROR: n must be an integer >= 1"
        #TODO: check function controls here

    if np.mod(cols,2):
        xr = np.arange(-(cols-1)/2, (cols - 1)/2 + 1, dtype="float32")/(cols - 1)
    else:
        xr = np.arange(-cols/2, (cols/2 - 1) + 1, dtype="float32")/cols

    if np.mod(rows,2):
        yr = np.arange(-(rows-1)/2, (rows - 1)/2 + 1, dtype="float32")/(rows-1)
    else:
        yr = np.arange(-rows/2, (rows/2 - 1) + 1, dtype="float32")/rows

    x,y = np.meshgrid(xr,yr)
    radius = np.sqrt(x**2 + y**2)
    f = np.fft.ifftshift( 1./ (1. + (radius/ cutoff) ** (2*n) ))

    return f

#--------------------------------------------------------------
def getGabor(radius, lp, nscale, minWaveLength, mult, sigmaOnf):
    """ Get gabors with logarithmic transfer fucntions """
    
    logGabor = []

    for s in range(int(nscale)):
        wavelength =  minWaveLength * (mult ** s)
        fo = 1./wavelength                                   # Filter Frequence

        # Apply low pass filter
        logGabor += [(np.exp((-(np.log(radius/fo))**2)/(2 * np.log(sigmaOnf) **2)))*lp]

        # Set the value at the 0 frequency point of the filter
        # back to zero (undo the radius fudge)
        logGabor[s][0][0] = 0

    return logGabor

#--------------------------------------------------------------
def getSpread(sintheta, costheta, norient, dThetaSigma):

    """ Compute angular components of the filter"""

    spread = []
    
    # Calculate the standard deviation of the angular Gaussian function
    # used to construct filters in the frequency plane.     
    thetaSigma = np.pi/norient/dThetaSigma

    # For each orientation
    for o in range(int(norient)):
        angl = (o * np.pi)/ norient                           # Filter angle

        # For each point in the filter matrix calculate the angular distance from
        # the specified filter orientation.  To overcome the angular wrap-around
        # problem sine difference and cosine difference values are first computed
        # and then the atan2 function is used to determine angular distance.
        
        ds = sintheta * np.cos(angl) - costheta * np.sin(angl)  # Difference in sine
        dc = costheta * np.cos(angl) + sintheta * np.sin(angl)  # Difference in cosine
        dtheta = abs(np.arctan2(ds,dc))                         # Abs angular distance
        spread += [np.exp((-dtheta**2)/(2*thetaSigma**2))]      # Angular filter component

    return spread

#--------------------------------------------------------------
def getOrientationEnergy(orientationEnergy, EO, o, polarity, nscale):

    """Calculate the phase symmetry measure based on polartiy"""
    #look for 'white' and 'black' spots
    if polarity == 0: 
        for s in range(int(nscale)):
            orientationEnergy = orientationEnergy + abs(np.real(EO[s][o])) - abs(np.imag(EO[s][o]))

    #Just look for 'white' spots
    elif polarity == 1:
        for s in range(int(nscale)):
            orientationEnergy = orientationEnergy + np.real(EO[s][o]) - abs(np.imag(EO[s][o]))

    #Just look for 'black' spots
    elif polarity == -1:
        for s in range(int(nscale)):
            orientationEnergy = orientationEnergy - np.real(EO[s][o]) - abs(np.imag(EO[s][o]))

    return orientationEnergy

#--------------------------------------------------------------
def getphasesym(rows, cols, imfft, logGabor, spread, nscale, norient, k, polarity):

    """ The main loop of phasesym"""

    # Array initialization
    EO = np.ndarray((nscale, norient, rows, cols),dtype="complex_")
    ifftFilterArray = np.ndarray((nscale,rows,cols), dtype="float64")
    zero = np.zeros((rows,cols), dtype="float32")
    totalSumAmp = np.zeros((rows,cols), dtype="float32")
    totalEnergy = np.zeros((rows,cols), dtype="float32")
    orientation = np.zeros((rows,cols), dtype="float32")
    epsilon = 0.0001

    
    for o in range(int(norient)):  #for each orientation

        # Array initialization
        sAmpThisOrient = np.zeros((rows,cols), dtype="float32")
        orientationEnergy = np.zeros((rows,cols), dtype="float32")


        for s in range(int(nscale)):  # for each scale
            
            #Multiply radial and angular components to get filter
            filter = logGabor[s] * spread[o]  
            ifftFilterArray[s] = np.real(np.fft.ifft2(filter))*np.sqrt(rows*cols)
            
            #Convolve image with even and odd filters returning the result in EO
            EO[s][o] = np.fft.ifft2(imfft * filter)
            Amp = abs(EO[s][o]) #Amplitude response
            sAmpThisOrient = sAmpThisOrient + Amp
            
            # Record mean squared filter value at smallest
            # scale. This si used for noise estimation
            if s == 0:
                EM_n = sum(sum(filter**2))
        
        # Now Calulate phase symmetry measure
        orientationEnergy = getOrientationEnergy(orientationEnergy,EO, o, polarity, nscale)

        # Noise Compensation
        # We estimate the noise power from the energy squared response at the
        # smallest scale.  If the noise is Gaussian the energy squared will
        # have a Chi-squared 2DOF pdf.  We calculate the median energy squared
        # response as this is a robust statistic.  From this we estimate the
        # mean.  The estimate of noise power is obtained by dividing the mean
        # squared energy value by the mean squared filter value

        medianE2n = np.median((abs(EO[0][o])** 2).ravel())
        meanE2n = -medianE2n/np.log(0.5)

        noisePower = meanE2n/EM_n           # Estimate noise power

        # Now estimate the total energy^2 due to noise
        # Estimate for sum(An^2) + sum(Ai.*Aj.*(cphi.*cphj + sphi.*sphj))

        EstSumAn2 = np.zeros((rows,cols), dtype="float32")
        for s in range(int(nscale)):
            EstSumAn2 = EstSumAn2 + ifftFilterArray[s]**2

        EstSumAiAj = np.zeros((rows,cols), dtype="float32")
        for si in range(int(nscale - 1)):
            for sj in range(si+1, int(nscale)):
                EstSumAiAj = EstSumAiAj + ifftFilterArray[si] * ifftFilterArray[sj]
        
        EstNoiseEnergy2 = 2*noisePower*sum(sum(EstSumAn2)) + 4*noisePower*sum(sum(EstSumAiAj))

        tau = np.sqrt(EstNoiseEnergy2/2)         # Rayleigh parameter
        EstNoiseEnergy = tau*np.sqrt(np.pi/2)    # Expected value of noise energy
        EstNoiseEnergySigma = np.sqrt((2 - np.pi/2)* tau**2)
        T =  EstNoiseEnergy + k*EstNoiseEnergySigma  # Noise threshold

        # The estimated noise effect calculated above is only valid for the PC_1
        # measure.  The PC_2 measure does not lend itself readily to the same
        # analysis.  However empirically it seems that the noise effect is
        # overestimated roughly by a factor of 1.7 for the filter parameters
        # used here.
        T = T/1.7

        # Apply noise threshold
        orientationEnergy = np.maximum(orientationEnergy - T, zero)

        # Update accumulator matrix for sumAn and totalEnergy
        totalSumAmp = totalSumAmp + sAmpThisOrient
        totalEnergy = totalEnergy + orientationEnergy

        # Update orientation matrix by finding image points where the energy in
        # this orientation is greater than in any previous orientation (the
        # change matrix) and then replacing these elements in the orientation
        # matrix with the current orientation number.

        if o == 0:
            maxEnergy = orientationEnergy
        else:
            change = orientationEnergy > maxEnergy
            orientation = o * change + orientation * np.logical_not(change)
            maxEnergy = np.maximum(maxEnergy, orientationEnergy)

    # Normalize totalEnergy by the totalSumAmp to obtain phase symmetry
    # epsilon is used to avoid division by 000
    phasesym = totalEnergy/(totalSumAmp + epsilon)

    # Convert orientation matrix values to degrees
    orientation = orientation * (180/norient)

    return phasesym, orientation

#--------------------------------------------------------------
def phasesym_fromarray(input_filename):
    """mlabwrap calls to matlab and save check-variables at each checkpoint"""

    image = Image.open(input_filename)       
    image = ImageOps.grayscale(image)  
    image = np.asarray(image)          
    mlab.phasesym(image)
    cp2 = sp.io.loadmat('cp2.mat')
    sintheta = cp2['sintheta']
    costheta = cp2['costheta']
    radius = cp2['radius']
    return sintheta, costheta, radius

# def phasesym(input_array,
#              params...
#              ):

    """python entry of phasesym"""
    
#     image = Image.open(input_filename)       
#     image = ImageOps.grayscale(image)
#     cols,rows = image.size
    
    #image fft
    imfft = np.fft.fft2(input_array)

    # Filter initializations
    sintheta, costheta, radius = filterIntialization(rows, cols)

    # Filters are constructed in terms of two components.
    # 1) The radial component, which controls the frequency band that the filter
    #    responds to
    # 2) The angular component, which controls the orientation that the filter
    #    responds to.
    # The two components are multiplied together to construct the overall filter.

    # Construct filter radial components
    
    # First construct a low-pass filter that is as large as possible, yet falls
    # away to zero at the boundaries.  All log Gabor filters are multiplied by
    # this to ensure no extra frequencies at the 'corners' of the FFT are
    # incorporated as this seems to upset the normalisation process when
    # calculating phase congrunecy.
    
    # Construct Low pass filter
    lp = getLowPassFilter(rows, cols, 0.4, 10.)

    # Radial Component
    logGabor = getGabor(radius, 
                        lp,
                        nscale = DEFAULT_NSCALE,
                        minWaveLength = DEFAULT_MWAVELENGTH,
                        mult = DEFAULT_MULT,
                        sigmaOnf = DEFAULT_SIGMAONF)
    
    # Construct the angular filter components
    spread = getSpread(sintheta,
                       costheta,
                       norient = DEFAULT_ORIENT,
                       dThetaSigma = DEFAULT_DTHETASEGMA)

    # Get phase symmetry and orientation of image
    phaseSym, orientation = getphasesym(rows, 
                           cols, 
                           imfft,
                           logGabor,
                           spread,
                           nscale = DEFAULT_NSCALE,
                           norient = DEFAULT_ORIENT,
                           k = DEFAULT_NSTD,
                           polarity = DEFAULT_POLARITY)

    return phasesym, orientation

#--------------------------------------------------------------
def phasesym_fromfilename(
    input_filename,
    output_filename,
    # --
    nscale = DEFAULT_NSCALE,
    # 
    norient = DEFAULT_ORIENT,
    minWaveLength = DEFAULT_MWAVELENGTH,
    mult = DEFAULT_MULT,
    sigmaOnf = DEFAULT_SIGMAONF,
    dThetaOnSigma = DEFAULT_DTHETASEGMA,
    k = DEFAULT_NSTD,
    polarity = DEFAULT_POLARITY,
    overwrite = DEFAULT_OVERWRITE
    ):
    
    """Fire matlab and python calls and perform value check"""
    # can I overwrite ?
    
    # if not print error, and quit

    # if ok: open image

    # get array
#    output_array = phasesym_fromarray(input_array,
#                                      XXX,
#                                      XXX)

#    sintheta_m, costheta_m, radius_m = phasesym_fromarray(input_filename)
    phasesym_p, orientation = phasesym(input_filename)
    
#    mlab.imshow(phasesym_p)
    
#--------------------------------------------------------------
def main():
    """ optparsing and function call to matlab--python comparison module """

    usage = "usage: %prog [options] <input_filename> <output_filename>"

    parser = optparse.OptionParser(usage=usage)

    parser.add_option("--nscale", "-s",
                      default=DEFAULT_NSCALE,
                      type="float", 
                      metavar="FLOAT",
                      help="[default=%default]")
    
    parser.add_option("--norient", "-o",
                      default=DEFAULT_ORIENT,
                      type="float", 
                      metavar="FLOAT",
                      help="[default=%default]")

    parser.add_option("--minWaveLength", "-w",
                      default=DEFAULT_MWAVELENGTH,
                      type="float", 
                      metavar="FLOAT",
                      help="[default=%default]")

    parser.add_option("--mult", "-m",
                      default=DEFAULT_MULT,
                      type="float", 
                      metavar="FLOAT",
                      help="[default=%default]")

    parser.add_option("--sigmaOnf", "-g",
                      default=DEFAULT_SIGMAONF,
                      type="float", 
                      metavar="FLOAT",
                      help="[default=%default]")

    parser.add_option("--dThetaOnSigma", "-d",
                      default=DEFAULT_DTHETASEGMA,
                      type="float", 
                      metavar="FLOAT",
                      help="[default=%default]")

    # XXX: careful here (better name than "k")
    parser.add_option("--nstdeviations", "-k",
                      default=DEFAULT_NSTD,
                      type="float", 
                      metavar="FLOAT",
                      help="[default=%default]")

    parser.add_option("--polarity", "-p",
                      default=DEFAULT_POLARITY,
                      type="int", 
                      metavar="INT",
                      help="[default=%default]")

    parser.add_option("--overwrite", 
                      default=DEFAULT_OVERWRITE,
                      help="[default=%default]")

    opts, args = parser.parse_args()


    if len(args) < 1:
        print "ERROR: Supply Image"
        parser.print_help()
    else:
        input_filename = args[0]
        output_filename = args[1]
        
        phasesym_fromfilename(input_filename,
                              output_filename,
                              # -- 
                              nscale = opts.nscale,
                              norient = opts.norient,
                              minWaveLength = opts.minWaveLength,
                              mult = opts.mult,
                              sigmaOnf = opts.sigmaOnf,
                              dThetaOnSigma = opts.dThetaOnSigma,
                              k = opts.k,
                              polarity = opts.polarity,
                              overwrite = opts.overwrite
                              )


if __name__ == "__main__":
    main()
