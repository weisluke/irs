import numpy as np
import struct

def read_params(fname):
    '''
    Read in a whitespace delimited text file of (key value)
    lines and return a dictionary mapping keys to values
    '''
    params = {}
    with open(fname) as f:
        for line in f:
            (key, val) = line.split()
            try:
                params[key] = float(val)
                if params[key] == int(params[key]):
                    params[key] = int(params[key])
            except:
                params[key] = val
    return params

def read_stars(fname, is_double = False):
    '''
    Read in a binary file of star information

    fname - name of the file to read
    is_double - bool, whether the stars are single or double precision
                default: False
    '''
    if is_double:
        dtype = np.float64
    else:
        dtype = np.float32

    with open(fname) as f:
        nstars = np.fromfile(f, dtype=np.int32, count=1)[0]
        rectangular = np.fromfile(f, dtype=np.int32, count=1)[0]
        corner = np.fromfile(f, dtype=dtype, count=2)
        theta_e = np.fromfile(f, dtype=dtype, count=1)[0]
        stars = np.fromfile(f, dtype=dtype)
        stars = stars.reshape(nstars, 3)

    return stars

def write_stars(fname, nstars, rectangular, corner, theta_star, stars, is_double = False):
    '''
    Write a binary file of star information

    fname - name of the file to write
    nstars - number of stars
    rectangular - whether the star field is rectangular (1) or circular (0)
    corner - array of 2 numbers representing the (x1, x2) corner of the star field
             in the case of rectangular star fields
             if the star field is circular, this should be (rad, 0)
    theta_star - einstein radius of a unit mass point lens in arbitary units
                 typically 1
    stars - array of length nstars in the form (x1, x2, mass)
    is_double - bool, whether the stars are single or double precision
                default: False
    '''
    if is_double:
        dtype = 'd'
    else:
        dtype = 'f'

    with open(fname, 'wb') as f:
        s = struct.pack('i', nstars)
        f.write(s)
        s = struct.pack('i', rectangular)
        f.write(s)
        s = struct.pack(dtype * 2, *corner)
        f.write(s)
        s = struct.pack(dtype, theta_star)
        f.write(s)
        s = struct.pack(dtype * len(stars.ravel()), *stars.ravel())
        f.write(s)
        f.close()

def read_array(fname, dtype, is_complex = False):
    '''
    Read in a binary file of a 2d array of numbers

    fname - name of the file to read
    dtype - type for the array
    is_complex - bool, whether the numbers are complex
                 or not
                 default: False
    '''
    with open(fname) as f:
        nrows, ncols = np.fromfile(f, dtype=np.int32, count=2)
        dat = np.fromfile(f, dtype=dtype)
        if is_complex:
            dat = dat.reshape(nrows, ncols, 2)
        else:
            dat = dat.reshape(nrows, ncols)

    return dat

def read_map(fname, is_ipm = True, is_double = False):
    '''
    Read in a binary file of map information

    fname - name of the file to read
    is_ipm - whether the map was created from inverse polygon
             mapping (True) or inverse ray shooting (False)
             Default: True
    is_double - bool, whether an ipm map is in single or
                double precision
                default: False
    '''
    if is_ipm:
        if is_double:
            return read_array(fname, np.float64)
        else:
            return read_array(fname, np.float32)
    else:
        return read_array(fname, np.int32)

def read_hist(fname):
    '''
    Read in a whitespace delimited text file of
    (value, num_pixels) lines
    '''
    return np.loadtxt(fname, dtype=np.int32)

