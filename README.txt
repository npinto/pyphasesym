###############################################################################
PyPhasesym
###############################################################################

We are pleased to present here a python implementation for calculating phase-
symmetry for a given image. Phasesym is described in detail in Kovesi et al 
[Kovesi97]_ and [Kovesi99]_. Matlab version of the code can be found can be
found on Peter Kovesi's website.

The python implementation is more modularized than original phasesym code. A
detailed tutorial on how to use the code can be found in tutorial section of 
this documentation. This code follows pylint naming scheme. Regression tests
and unit tests are provided for sanity check of the code. 

A detailed description of functioning and usage of each function in the
package is given in API.

This python implementation of phasesym follows MIT License which can be found 
in license.txt.  For original license for phasesym, the user is advised to 
refer Peter Kovesi's original code.

.. module:: pyphasesym
   :synopsis: A python implementation of phasesym version 1.01

Documentation Contents:
*******************************************************************************

.. toctree::
   :maxdepth: 2
   
   tutorial
   license

.. [Kovesi97] Peter Kovesi, "Symmetry and Asymmetry From Local Phase" AI'97, 
Tenth Australian Joint Conference on Artificial Intelligence. 2 - 4 December 
1997. http://www.cs.uwa.edu.au/pub/robvis/papers/pk/ai97.ps.gz.

.. [Kovesi99] Peter Kovesi, "Image Features From Phase Congruency". Videre: 
A Journal of Computer Vision Research. MIT Press. Volume 1, Number 3, 
Summer 1999 http://mitpress.mit.edu/e-journals/Videre/001/v13.html
