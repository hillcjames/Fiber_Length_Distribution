import math
import datetime
import random
from PIL import Image, ImageDraw
import cython
cimport numpy as np

import string
import random
from numpy.core.numeric import ndarray

from scipy import ndimage, misc
from matplotlib.pyplot import *




def horizEdgeArray( w, thickness ):
    try:
        # with fiber width w
        a = np.zeros((2 + thickness, (int)(1.5 * w)))
         
        # x-location of the edge of the fiber
        fiberEdge = (int)(0.25 * w)
         
        # width of the central all-light region
        fiberCenterWidth = (int)(0.75*w)
         
        # x-location of the edge of the central all-light region
        fiberCenterEdge = (int)((len(a[0]) - fiberCenterWidth) / 2)
         
         
        edgePixelValue = 1
        centerPixelVlaue = 3
         
        fiberTotalValue = fiberCenterWidth * centerPixelVlaue + (w - fiberCenterWidth) * edgePixelValue
         
        outsidePixelValue = -1 * (int)(fiberTotalValue / (2 * fiberEdge))
         
#         raise Exception(fiberEdge, fiberCenterEdge, edgePixelValue, centerPixelVlaue)
        for t in range(0, thickness):
            for i in range(0, fiberEdge):
                a[(1 + t, i)] = outsidePixelValue
                a[ (1 + t, len(a[0]) - 1 - i)] = outsidePixelValue
             
            for i in range(fiberEdge, fiberCenterEdge):
                a[(1 + t, i)] = edgePixelValue
                a[ (1 + t, len(a[0]) - 1 - i)] = edgePixelValue
             
            for i in range(fiberCenterEdge, fiberCenterEdge + fiberCenterWidth):
                a[(1 + t, i)] = centerPixelVlaue
                
    except Exception:
        print("Width value too low.")
        raise
    return a

def vertEdgeArray( w, thickness ):
    try:
        # with fiber width w
        a = np.zeros(((int)(1.5 * w), 2 + thickness))
        
        # x-location of the edge of the fiber
        fiberEdge = (int)(0.25 * w)
        
        # width of the central all-light region
        fiberCenterWidth = (int)(0.75*w)
        
        # x-location of the edge of the central all-light region
        fiberCenterEdge = (int)((len(a) - fiberCenterWidth) / 2)
        
        
        edgePixelValue = 1
        centerPixelVlaue = 3
        
        fiberTotalValue = fiberCenterWidth * centerPixelVlaue + (w - fiberCenterWidth) * edgePixelValue
        
        outsidePixelValue = -1 * (int)(fiberTotalValue / (2 * fiberEdge))
    
#         print(fiberEdge, fiberCenterEdge)
        for t in range(0, thickness):
            for i in range(0, fiberEdge):
                a[(i, 1 + t)] = outsidePixelValue
                a[ (len(a) - 1 - i), 1 + t] = outsidePixelValue
            
            for i in range(fiberEdge, fiberCenterEdge):
                a[(i, 1)] = edgePixelValue
                a[ (len(a) - 1 - i), 1 + t] = edgePixelValue
            
            for i in range(fiberCenterEdge, fiberCenterEdge + fiberCenterWidth):
                a[(i, 1 + t)] = centerPixelVlaue
    
    except Exception:
        print("Width value too low.")
        
    return a

def crossEdgeArray( w ):
    # with fiber width w
    a = np.zeros(((int)(1.5 * w), (int)(1.5 * w)))
    
    # x-location of the edge of the fiber
    fiberEdge = (int)(0.25 * w)
    
    # width of the central all-light region
    fiberCenterWidth = (int)(0.75*w)
    
    # x-location of the edge of the central all-light region
    fiberCenterEdge = (int)((len(a) - fiberCenterWidth) / 2)
    
    
    edgePixelValue = 1
    centerPixelVlaue = 3
    
    fiberTotalValue = fiberCenterWidth * centerPixelVlaue + (w - fiberCenterWidth) * edgePixelValue
    
    outsidePixelValue = -1 * (int)(fiberTotalValue / (2 * fiberEdge))
    
#     print(fiberEdge, fiberCenterEdge)
    for i in range(0, fiberEdge):
        a[i, len(a)//2] = outsidePixelValue
        a[len(a) - 1 - i, len(a)//2] = outsidePixelValue
        
        a[len(a)//2, i] = outsidePixelValue
        a[len(a)//2, len(a) - 1 - i] = outsidePixelValue
    
    for i in range(fiberEdge, fiberCenterEdge):
        a[(i, len(a)//2)] = edgePixelValue
        a[len(a) - 1 - i, len(a)//2] = edgePixelValue
        
        a[len(a)//2, i] = edgePixelValue
        a[len(a)//2, len(a) - 1 - i] = edgePixelValue
    
    
    for i in range(fiberCenterEdge, fiberCenterEdge + fiberCenterWidth):
        a[(i, len(a)//2)] = centerPixelVlaue
        a[len(a)//2, i] = centerPixelVlaue
    
    return a

def fiberBox( size, c, t, fiberW ):
    # returns a numpy 2d array of width w and height h with 
    #     a fiber-representation going through it at angle t.
    # The fiber has width fiberW.
    # using equation of line ax + by = c
    # and distance formula d = |Ax + By + C| / sqrt(A**2 + B**2), where B = 1

    w, h = size[:]
    box = 64 * np.ones((h,w))
    
    m = math.tan(t)
    b = c[1] - m * c[0] #since it passes through the centerpoint
    
    A = -m
    C = -b
    
    denom = 1/math.sqrt(A**2 + 1)
    
    for x in range(0, w):
        for y in range(0, h):
            distToLine = abs(A * x + y + C) * denom
            if distToLine < fiberW / 2:
                box[y][x] = (1 - distToLine/(fiberW/2)) * 172 + 64

            
    return box
    

def getFilteredPixVal(im, x0, y0, flt):
    # p is the center point, so adjust it
    x0 -= len(flt)//2
    y0 -= len(flt[0])//2
    
    sum = 0
    pixelsVisted = 0

    for x in range(0, len(flt)):
        for y in range(0, len(flt[0])):
            try:
                sum += flt[ x ][ y ] * im[ x0 + x ][ y0 + y ]
                pixelsVisted += 1
            except IndexError:
                ()
#     print(sum)
#     try:
    return sum
#     except ZeroDivisionError:
#         return 0
                
def mergeIms( im1, im2 ):
    if len(im1) != len(im2) or len(im1[0]) != len(im2[0]):
        raise ValueError('Dimension mismatch')
    
    tempIm = ndarray((len(im1), len(im1[0])))

    for x in range(0, len(tempIm[0])):
        for y in range(0, len(tempIm)):
            tempIm[y][x] = im1[y][x] if (im1[y][x] > im2[y][x]) else im2[y][x]
    
    return tempIm
         
def pickyConvolvement( im, f1, f2 ):
#     return im
    
    im1 = ndimage.convolve(im, f1)
    im2 = ndimage.convolve(im, f2)
    
    tempIm = mergeIms(im1, im2)
    
    return 255 * tempIm / tempIm.max()

# im = crossEdgeArray(7)




