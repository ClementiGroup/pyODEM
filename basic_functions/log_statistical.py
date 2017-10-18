""" Contains negative logarithmic versions of functions in statistical.py

Normalization is not required, nor are values bounded between 0 and 1.

"""

def harmonic(r,r0,width):
    V = ((r - r0)**2)/(2.*(width**2))
    return V

def derivative_harmonic(r, r0, width):
    dV = (r - r0) / (width**2)
    return dV

def wrapped_harmonic(r0, width):
    def new_harmonic(r):
        return harmonic(r,r0,width)
    return new_harmonic

def wrapped_derivative_harmonic(r0, width):
    def new_derivative_harmonic(r):
        return derivative_harmonic(r, r0, width)
    return new_derivative_harmonic
