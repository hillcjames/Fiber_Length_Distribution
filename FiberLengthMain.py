'''
Created on Jul 13, 2015

@author: Christopher Hill

This will count fibers in an image and measure their lengths.

It requires the installation of python3, PIL, image_slicer, cython, matplotlib, scikit-image,
and maybe not iamge_slicer
on linux, you can just do:
    sudo apt-get install python3
    look up how to install pip, AND install it using python3, not python
    use pip to install PIL
    sudo -H python3 -m pip install image_slicer
    sudo -H pip3 install cython
    sudo -H pip3 install scikit-image

Assumes:
    A grayscale image with relatively few large bundles of overlapping fibers 
    
Possible algorithm:
    look for a light (but not red) square (same width as fiber), and look in a circle until you find another light 
    square.
    remember the first square, and travel in that direction, making course adjustments as you go, but always moving
    in a straight line away from the initial square. Once you hit black, record that point, and do the same thing but
    going in the other direction away from the original point,  until you hit black again. Record the two end points 
    as a fiber and color the whole line red.
    When tracing the line, you can't start on red, but red shouldn't stop the trace, to allow for overlapping fibers.
    Don't just search for light squares; only allow ones with dark on either side. 
    Do a binary search at each initial point, but let the angle take care of it when actually tracing a fiber.
    Like once the starting point is pointed mostly the right direction, start moving, and make course adjustments
    as you go, and if it's a straight thin line, then the dark will automatically be on either side of the light, 
    perpendicularly to the angle that the light square is facing. So I need to use rotated squares. 
    Use a center point and a rotation angle.
    
    Actually, don't keep track of a master angle, only the previous angle, and allow only small deviations in that
        angle - no sharp bends.
        
    Also, I have discovered that pixels = im.load() works faster than np.array(im). 7.6s as opposed to 10.0
    And that it's twice as fast to rotate just the corners and interpolate, than to generate a grid and rotate
        every point
        
        
    Thinking of an algorithm that mimics how a human would draw a line. Look to what would cause a human to 
    make a mistake, and emulate that. All the different cases a human would miss, and what they would look for.
    Maybe look at the average blocks around the current block, if there's only one white spot then just find the
    best angle and go in that direction, but if the white spot it finds is too wide to be a single fiber, look
    closer for boundaries - a square which has lightness in the center and grey on either side, as close to the 
    current direction as possible. Don't find the best square, just find all the squares which match the criteria
    and then pick the one which deviates the least. If there are no squares that do so, but there are some which 
    are light, then just keep going in the current direction until you hit blackness or a square which you can
    definitively determine to be on a fiber.

    
'''
from PIL import Image, ImageDraw
from colorSpread import getStats
import numpy as np
from scipy import dot, array
from math import cos, sin, pi, sqrt, ceil, atan
import datetime
import os
from json.decoder import NaN
import timeit


from scipy import ndimage, misc
from matplotlib.pyplot import *
from matplotlib.lines import Line2D
from skimage.feature import peak_local_max
from skimage.draw import line_aa

from numpy.core.numeric import ndarray
from subprocess import call
from random import randint

# im = Image.open("Images/midSizedTest.jpg")
# # im = Image.open("Images/smallTest.jpg")
# pixels = im.load()
# print(pixels[(0,0)])
# print(1/0)

class BigImage:
    def __init__(self, data):
        self.imDir = data[0]
        self.imName = data[1]
        self.num_columns = data[2]
        self.num_rows = data[3]
        self.avg = data[4]
        self.stdev = data[5]
        self.names = data[6]
        self.imgs = data[7]
        self.pxls = data[8]
        self.extlessName = data[9]
        self.sliceW = data[10]
        self.sliceH = data[11]
#         self.w, self.h = data[0][:]
        
        
#     def __init__(self, imDir, imName, numTiles):
#         self.imDir = imDir
#         self.imName = imName
#         self.num_columns = int(ceil(sqrt(numTiles)))
#         self.num_rows = int(ceil(numTiles / float(self.num_columns)))
#         self.avg = 0
#         self.stdev = 0
#         self.names = []
#         self.imgs = []
#         self.pxls = []
#         self.extlessName = ""
#         self.sliceW, self.sliceH = 0, 0
#         self.w, self.h = 0, 0
#         
#         index = 0
#         while imName[len(imName) - 1 - index] != '.':
#             index += 1
#         index += 1
#         
#         # create a new folder with that image's name, minus the extension
#         folder = imName[:len(imName)-index]
#         self.extlessName = folder
#         slicesExist = False
#         try:
#             os.makedirs(os.path.join(imDir, folder))
#         except Exception:
#             try:
#                 os.rmdir(os.path.join(imDir, folder))
#                 os.makedirs(os.path.join(imDir, folder))
#             except Exception:
#                 print("Folder to hold image slices already exists and isn't empty, with name.", folder)
#                 print("Assuming it to be valid, if fileNotFound errors are given later, \
#                         \nor if anything inside the BigImage __init__() method has been changed, \
#                         \ntry deleting that folder and re-running program.\n")
#                 slicesExist = True
#         if not slicesExist:
#             import image_slicer
#             try:
#                 image_slicer.slice(os.path.join(imDir, imName), numTiles)
#             except Exception as e:
#                 print(e)
#             print("Finished slicing.")
#             
#             for j in range(0, self.num_rows):
#                 for i in range(0,self.num_columns):
#                     # folder is the name of the image minus its extension
#                     sliceName = folder + "_0"+str(j+1) + "_0"+str(i+1) + ".png"
#                     newName = os.path.join(imDir, folder, sliceName)
#                     os.rename(os.path.join(imDir, sliceName), newName)
#                 
#         for j in range(0,self.num_rows):
#             for i in range(0,self.num_columns):
#                 sliceName = folder + "_0"+str(j+1) + "_0"+str(i+1) + ".png"
#                 newName = os.path.join(imDir, folder, sliceName)
#                 self.names.append(newName)
#                 self.imgs.append(Image.open(newName))
#                 self.imgs[len(self.imgs)-1].convert("RGB")
#         
#         for x in range(0, len(self.imgs)):
#             self.pxls.append(self.imgs[x].load())
#         self.sliceW, self.sliceH = self.imgs[0].size
#         
#         self.avg, self.stdev = getStats(self.pixels, self.sliceW, self.sliceH)
    
    @classmethod
    def fromFile(cls, imDir, imName, numTiles):
        imDir = imDir
        imName = imName
        num_columns = int(ceil(sqrt(numTiles)))
        num_rows = int(ceil(numTiles / float(num_columns)))
#         avg = 0
#         stdev = 0
        names = []
        imgs = []
        pxls = []
#         extlessName = ""
#         sliceW, sliceH = 0, 0
#         w, h = 0, 0
        
        index = 0
        while imName[len(imName) - 1 - index] != '.':
            index += 1
        index += 1
        
        # create a new folder with that image's name, minus the extension
        folder = imName[:len(imName)-index]
        extlessName = folder
        slicesExist = False
        try:
            os.makedirs(os.path.join(imDir, folder))
        except Exception:
            try:
                os.rmdir(os.path.join(imDir, folder))
                os.makedirs(os.path.join(imDir, folder))
            except Exception:
                print("Folder to hold image slices already exists and isn't empty, with name.", folder)
                print("Assuming it to be valid, if fileNotFound errors are given later, \
                        \nor if anything inside the BigImage __init__() method has been changed, \
                        \ntry deleting that folder and re-running program.\n")
                slicesExist = True
        if not slicesExist:
            import image_slicer
            try:
                image_slicer.slice(os.path.join(imDir, imName), numTiles)
            except Exception as e:
                print(e)
            print("Finished slicing.")
            
            for j in range(0, num_rows):
                for i in range(0,num_columns):
#                     folder is the name of the image minus its extension
                    sliceName = folder + "_0"+str(j+1) + "_0"+str(i+1) + ".png"
                    newName = os.path.join(imDir, folder, sliceName)
                    os.rename(os.path.join(imDir, sliceName), newName)
        
        
        for j in range(0,num_rows):
            for i in range(0,num_columns):
                sliceName = folder + "_0"+str(j+1) + "_0"+str(i+1) + ".png"
                newName = os.path.join(imDir, folder, sliceName)
                names.append(newName)
                imgs.append(Image.open(newName))
                imgs[len(imgs)-1].convert("RGB")
#                 w, h = imgs[len(imgs)-1].size
#                 
#                 imgs[len(imgs)-1] = imgs[len(imgs)-1].resize((w//2, h//2))
        
        for x in range(0, len(imgs)):
            pxls.append(imgs[x].load())
        sliceW, sliceH = imgs[0].size
        
        avg, stdev = 0, 0
        
        data = []
        data.append(imDir)
        data.append(imName)
        data.append(num_columns)
        data.append(num_rows)
        data.append(avg)
        data.append(stdev)
        data.append(names)
        data.append(imgs)
        data.append(pxls)
        data.append(extlessName)
        data.append(sliceW)
        data.append(sliceH)
        
        obj = cls(data)
        obj. avg, obj.stdev =  getStats(obj.pixels, obj.sliceW, obj.sliceH)
        return obj
    
    @staticmethod
    def copy(cls, original):
        
        imDir = original.imDir
        imName = original.imName
        num_columns = original.num_columns
        num_rows = original.num_rows
        avg = original.avg
        stdev = original.stdev
        extlessName = original.extlessName
        names = []
        imgs = []
        pxls = []
        
        for i in range(0, num_columns * num_rows):
#             for y in range(0, num_rows):
            names.append( original.names[i] )
            imgs.append(  original.imgs[i].copy()  )
            pxls.append(  imgs[len(imgs)-1].load()  )

        sliceW, sliceH = original.sliceW, original.sliceH
        
        data = []
        data.append(imDir)
        data.append(imName)
        data.append(num_columns)
        data.append(num_rows)
        data.append(avg)
        data.append(stdev)
        data.append(names)
        data.append(imgs)
        data.append(pxls)
        data.append(extlessName)
        data.append(sliceW)
        data.append(sliceH)
        
        return cls(data)
        
        
    def pixels(self, p):
#         if p[0] is float:
#             return self.floatPixels(p)
        # if the slices are arranged on a grid, the x-coord on that grid
        x = p[0] // self.sliceW
        # if the slices are arranged on a grid, the y-coord on that grid
        y = p[1] // self.sliceH
        
        p0 = (p[0] % self.sliceW, p[1] % self.sliceH)
        
        return self.pxls[ x + y*self.num_columns ][p0]
        
        
    def floatPixels(self, p):
        # if the slices are arranged on a grid, the x-coord on that grid
        x = int(p[0] / self.sliceW)
        # if the slices are arranged on a grid, the y-coord on that grid
        y = int(p[1] / self.sliceH)
        
#         if (p[0] + 1 >= self.) or (p[1] + 1 >= self.h):
#             raise IndexError
        
        p00 = ( (int(p[0]) % self.sliceW), (int(p[1]) % self.sliceH))
        p01 = ( (int(p[0]) % self.sliceW), (int(p[1]) % self.sliceH) + 1)
        p10 = ( (int(p[0]) % self.sliceW) + 1, (int(p[1]) % self.sliceH))
        p11 = ( (int(p[0]) % self.sliceW) + 1, (int(p[1]) % self.sliceH) + 1)
        

        decPart = np.array(p) - np.array(p00)
        
        pxls = self.pxls[ x + y*self.num_columns ]
        
        total = (1-decPart[0])*(1-decPart[1]) * pxls[p00]
        total += (1-decPart[0]) * decPart[1] * pxls[p01]
        total += decPart[0]*(1-decPart[1]) * pxls[p10]
        total += decPart[0]*decPart[1] * pxls[p11]
        
        return total
    
    
    def size(self):
        return self.sliceW*self.num_columns, self.sliceH*self.num_rows
    
    def show(self, i):
        try:
            self.imgs[i].show()
        except IndexError:
            print("Index", i, "is out of range in list of length", len(self.imgs))
    
    def putpixel(self, p, c):
        # if the slices are arranged on a grid, the x-coord on that grid
        x = p[0] // self.sliceW
        # if the slices are arranged on a grid, the y-coord on that grid
        y = p[1] // self.sliceH

        p1 = tuple((p[0]-x*self.sliceW, p[1]-y*self.sliceH))

        self.imgs[x+y*self.num_columns].putpixel(p1,c)
        

    # This is dumb and probably slow but I'll fix it later.
    # Instead of only printing the fiber on the tiles the fiber touches, it just tries to print it on all of them.
    def drawFiber(self, fiber, c):
        for i in range(0, len(self.imgs)):
            offset = ( (i%self.num_columns)*self.sliceW, (i//self.num_columns)*self.sliceH )
#             print("**********",offset, self.sliceW, self.sliceH, i, self.num_columns, self.num_rows)
            fiber.draw(self.imgs[i], c, offset)
    
    def saveAll(self):
        for i in range(0, len(self.imgs)):
            image = self.imgs[i]
            name = "{}_done_{}_{}.bmp".format( self.extlessName, i%self.num_columns, i/self.num_columns)
            path = os.path.join(self.imDir, self.extlessName, name)
            image.save(path)
#             print(path)


    
# im = Image.new('L', (64, 64), 0)
# pxls = im.load()
# im.putpixel((11, 11), 100)
# im.putpixel((11, 12), 100)
# 
# p = floatPixels(pxls, (11, 10.5))
# print(p)
# print(1/0)

def sqrDist( p1, p2 ):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

class Fiber:
    def __init__(self, points, fiberW):
        self.pnts = points
        self.w = fiberW
        self.calcLineProperties()
    
    def calcLineProperties(self):
        self.calcAngle()
        self.calcLength()
        
        p1, p2 = self.pnts[0], self.pnts[len(self.pnts)-1]
        
        
        self.a = (p2[1] - p1[1])/(p1[0]*p2[1] - p2[0]*p1[1] + 0.01)
        self.b = (p2[0] - p1[0])/(p1[1]*p2[0] - p2[1]*p1[0] + 0.01)
    
    def calcAngle(self):
        p0, p1 = self.pnts[0], self.pnts[len(self.pnts)-1]
        self.angle = atan((p1[1] - p0[1])/(p1[0] - p0[0]+0.1))
    
    def calcLength(self):
        self.length = 0
        for i in range(0, len(self.pnts)-1):
            self.length += sqrt(sqrDist(self.pnts[i], self.pnts[i+1]))
    
#     def getAvgPnt(self):
#         xSum = 0
#         ySum = 0
#         for p in self.pnts:
#             xSum += p[0]
#             ySum += p[1]
#         return (xSum//len(self.pnts), ySum//len(self.pnts))
    
    def draw(self, im, c, offset = (0, 0)):
        draw = ImageDraw.Draw(im)
        for i in range(0, len(self.pnts)-1):
            p1 = (self.pnts[i][0] - offset[0], self.pnts[i][1] - offset[1])
            p2 = (self.pnts[i+1][0] - offset[0], self.pnts[i+1][1] - offset[1])
            draw.ellipse([(p1[0]-self.w/2+1, p1[1]-self.w/2+1),(p1[0]+self.w/2-1, p1[1]+self.w/2-1)], fill = c)
#             draw.ellipse([(p1[0]-offset[0]-self.w/2+1, p1[1]-offset[1]-self.w/2+1),(p1[0]-offset[0]+self.w/2-1, p1[1]-offset[1]+self.w/2-1)], fill = c)
            draw.line([p1,p2], width = (self.w*6)//5, fill = c)
        p0 = (self.pnts[len(self.pnts)-1][0] - offset[0], self.pnts[len(self.pnts)-1][1] - offset[1])
        draw.ellipse([(p0[0]-self.w/2+1, p0[1]-self.w/2+1),(p0[0]+self.w/2-1, p0[1]+self.w/2-1)], fill = c)
        
    def getEndPointsStr(self):
        return str(self.pnts[0][0]) + " " + str(self.pnts[0][1]) + " " + str(self.pnts[len(self.pnts)-1][0]) + " " + str(self.pnts[len(self.pnts)-1][1])
        
    def getEndPoints(self):
        return self.pnts[0], self.pnts[len(self.pnts)-1]

def intTup( t ):
    return (int(t[0]), int(t[1]))
  
#tests drawFiber
# im = Image.new('L', (300,300), 0)
# l = [(30,100),(50,170),(70,180),(100,220)]
# f = Fiber(l, 10)
# f.drawFiber(im)
# im.show()
# print(1/0)
# p0, p1 = (0,0), (-1.1,1)
#  
# print(atan((p1[1] - p0[1])/(p1[0] - p0[0]))*180/pi)
# print(1/0)

# def rotate(pts, t, center = array([0,0])):
# #     print()
#     return dot(pts - center, array([[cos(t),sin(t)],[-sin(t),cos(t)]])) + center
def rotate(pts, t, center = [0,0]):
#     print()

#     t1 = pts - center
#     print(t)
#     t2 = np.array([[cos(t),sin(t)],[-sin(t),cos(t)]])
#     t3 = dot( t1, t2)
#     t4 = t3 + center
#     return t4
    return dot(pts - center, array([[cos(t),sin(t)],[-sin(t),cos(t)]])) + center

# #this returns an angle measured from the x-axis. White is pi below this angle, black is pi above.
# #it works analogously to a binary search. If the value it calculates is negative, it turns one way.
# # if it's positive, it turns the other. The amount it turns is half of the amount it turned the previous time.
# # it tries to find the angle that splits the amount of color equally on each side, and returns that angle + pi/2.
# # It assumes that the angle of no contrast is 90 degrees off of the angle of highest contrast. I think it's true
# # by definition. 
# def getBestAngle(strip):
#     
#     halfPrevMovement = pi
#     usedTheta = - pi / 2
#     for i in range(0, 8):
#         diff = getLightnessDiffference(sqr, usedTheta)
#         halfPrevMovement /= 2.0
#         if diff > 0:
#             usedTheta -= halfPrevMovement
#         else:
#             usedTheta += halfPrevMovement
#         i+=0
# 
#     return usedTheta + pi / 2

def showBigger(im, pixels):
    oW, oH = im.size
    im1 = Image.new('L', (2*oW, 2*oH), 0)
    bigPix = im1.load()
    for x in range(0, oW):
        for y in range(0, oH):
            bigPix[2*x,2*y] = pixels[x,y]
            bigPix[2*x+1,2*y] = pixels[x,y]
            bigPix[2*x,2*y+1] = pixels[x,y]
            bigPix[2*x+1,2*y+1] = pixels[x,y]
    return im1

def getBestInRegion(im, p, pAvg, fiberW):
    
    best = pAvg
    bestP = p
    for t in range(0,360,30):
        theta = t/180*pi
        p1 = (int(p[0]+fiberW/2*cos(theta)), int(p[1]+fiberW/2*sin(theta)))
        try:
            newAvg = getAvgDotWhiteness(im, p1, fiberW)
        except IndexError:
            continue
        if newAvg > best:
            best = newAvg
            bestP = p1
            
    return bestP#, best
    
    
def getBestStartAngle(im, p, fiberW, stripL, spacing):
    # swing a box around the given point like a clock hand, looking for the angle which covers the most white.
    
    #corners are numbered in clockwise order, with the corner opposite c1 skipped 
    bestT = 0
    bestVal = 0
#     bestP = 0
    for theta in range(0,360,5):
        t = theta / 180 * pi
        # this rectangle starts pointing upwards, towards -y.
#         c1 = np.array(rotate( np.array([p[0]-fiberW/2, p[1]]), t, (p[0], p[1])))
#         c2 = np.array(rotate( np.array([p[0]+fiberW/2, p[1]]), t, (p[0], p[1])))
#         c3 = np.array(rotate( np.array([p[0]-fiberW/2, p[1]+stripL]), t, (p[0], p[1])))
#         v12hat = np.array([(c2[0]-c1[0])/fiberW, (c2[1]-c1[1])/fiberW])
#         v13hat = np.array([(c3[0]-c1[0])/stripL, (c3[1]-c1[1])/stripL])

        # this rectangle starts pointing sideways, towards +x.
        c1 = np.array(rotate( np.array([p[0], p[1]-fiberW/2]), t, (p[0], p[1])))
        c2 = np.array(rotate( np.array([p[0], p[1]+fiberW/2]), t, (p[0], p[1])))
        c3 = np.array(rotate( np.array([p[0]+stripL, p[1]-fiberW/2]), t, (p[0], p[1])))
        v12hat = np.array([(c2[0]-c1[0])/fiberW, (c2[1]-c1[1])/fiberW])
        v13hat = np.array([(c3[0]-c1[0])/stripL, (c3[1]-c1[1])/stripL])
    
        # total whiteness of strip 
        whiteness = 0
        sqrCounter = 0
        p0 = 0
        # go through each square in the strip
        for x in range(0, fiberW, spacing):
            for y in range(0, stripL, spacing):
                p0 = c1 + x * v12hat + y*v13hat
                try:
                    whiteness += im.pixels( intTup(p0) )
                except IndexError:
#                     print(p, p0)
                    raise
                    
                sqrCounter += 1
        whiteness/=sqrCounter
#         p2 = c1 + fiberW * v12hat + stripL*v13hat
#         p2 = (int(p2[0]), int(p2[1]))
#         print(p, whiteness, t/pi*180, p2)
#         for x in range(0, stripW, spacing):
#             for y in range(0, stripL, spacing):
#                 p0 = c1 + x * v12hat + y*v13hat
#                 p0 = (int(p0[0]), int(p0[1]))
#                 im.putpixel(p0, int(whiteness))
        
        if whiteness > bestVal:
            bestVal = whiteness
            bestT = t
#             p3 = c1 + fiberW * v12hat + stripL*v13hat
#             p3 = c1 + stripL*v13hat
#             bestP = (int(p3[0]), int(p3[1]))
#             p3 = (int(p3[0]), int(p3[1]))
#             print(p3, (p3[0]-p[0],p3[1]-p[1]), t/pi*180, sin(bestT), cos(bestT))
#         break

#     return bestT, bestVal
    return bestT

def getStripCentroid(im, t, p, fiberW, spacing, avg):
    stripCenX, stripCenY = p[:]
    stripW = int(3 * fiberW)
    stripL = 4 * fiberW 
     
    #corners are numbered in clockwise order, with the corner opposite c1 skipped 
#     t = angle / 180 * pi
    c1 = np.array(rotate( np.array([stripCenX-stripW/2, stripCenY-stripL/2]), t, (stripCenX, stripCenY)))
    c2 = np.array(rotate( np.array([stripCenX+stripW/2, stripCenY-stripL/2]), t, (stripCenX, stripCenY)))
    c3 = np.array(rotate( np.array([stripCenX-stripW/2, stripCenY+stripL/2]), t, (stripCenX, stripCenY)))
#     v12 = np.array([(c2[0]-c1[0]), (c2[1]-c1[1])])
#     v13 = np.array([(c3[0]-c1[0]), (c3[1]-c1[1])])
    v12hat = np.array([(c2[0]-c1[0])/stripW, (c2[1]-c1[1])/stripW])
    v13hat = np.array([(c3[0]-c1[0])/stripL, (c3[1]-c1[1])/stripL])
 
    # total whiteness of strip 
    total = 0
     
    # x-center of 'mass' (whiteness)
    xCOM = 0
     
    # y-center of 'mass' (whiteness)
    yCOM = 0
     
    # go through each square in the strip
    for x in range(0, stripW, spacing):
        for y in range(0, stripL, spacing):
            p = c1 + x * v12hat + y*v13hat
            
            #gives a value 0-1 based on distance from centerline, where points along the centerline are highest
            distFromCenter = (1 - abs(0.1+((x-stripW/2)/(0.5*stripW))))
            
            pInt = intTup(p)
            val = (im.pixels(pInt) - avg) * distFromCenter
#             val = (im.pixels(p) - avg) * distFromCenter
            xCOM += p[0] * val*val
            yCOM += p[1] * val*val
            total += val*val
    c = (int(xCOM/total), int(yCOM/total))
    
    return c

# def checkPoint_coarse( pixels, boxW, x, y ):
#     lBound = int(0 - boxW/2)
#     hBound = boxW + lBound
#     for i in range(lBound, hBound):
#         for j in range(lBound, hBound):
#             pixels[i + x, j + y]
#     return

# def checkPoint_strip(image, pixels, spacing, stripDimensions):
#     angle = 27 #getBestAngle()
#     colorList = checkStrip(pixels, angle, stripDimensions, spacing)
#     #
#     # stuff to say which way to turn in order to get a better angle
#     #
# #     image.show()
#     showBigger(image, pixels).show()

def getAvgDotWhiteness(im, p, r):
    # use a full box and just iterate over the parts you need
    whiteness = 0
    rSqr = r**2
    start = -r - 1
    stop = r + 1
    numDots = 0
    for x in range(start,stop):
        y = - sqrt(abs(rSqr - x*x))
        while( (y)*(y) < rSqr - x*x+0.1 ):
#             if im.pixels( (p[0]+int(x), p[1]+int(y)) ) == 255:
#                 y+=1
#                 continue
            whiteness += im.pixels( (p[0]+int(x), p[1]+int(y)) )
#             im.putpixel((p[0]+int(x), p[1]+int(y)),255)
            y+=1
            numDots += 1
#     im.imgs[0].show()
    return whiteness/numDots


def getAvgSqrWhiteness(im, p, r):
#     if r != int(r):
#         raise Exception("input r must be an integer")
    x0, y0 = p[:]
    w = 2*r + 1
#     sqr = np.array([[0]*w]*w)
    sum1 = 0
    try:
        for x in range(0, w):
            for y in range(0, w):
    #             sqr[x][y] = matrix[x0-r+x][y0-r+y]
                sum1 += im.pixels(intTup((x0-r+x,y0-r+y)))
        sum1 /= w**2
    except IndexError:
        return 0
    return sum1

def pointsAreGood(im, avg, stdev, fPoints):
    if len(fPoints) < 3:
        return 0
    badPoints = 0
    for i in range(0, len(fPoints)-1):
        p = fPoints[i]
        pMid = ( (fPoints[i+1][0]+p[0])//2, (fPoints[i+1][1]+p[1])//2 )
        
        if im.pixels(p) < avg + stdev/2:
            badPoints += 1
        if im.pixels(pMid) < avg + stdev/2:
            badPoints += 1
    if im.pixels(fPoints[len(fPoints)-1]) < avg + stdev/2:
            badPoints += 1
    return badPoints/(len(fPoints)*2-1) < 0.3


def getStraightFiber(im, fiber):
    '''
    This returns a straightened version of the input fiber, if said fiber is accurate.
    Otherwise, returns 0.
    '''
#     from scipy.optimize import curve_fit
    from scipy import stats
    xd = [p[0] for p in fiber.pnts]
    yd = [p[1] for p in fiber.pnts]
#     m, b = curve_fit(straightFit, xd, yd, p0 = (0, 0))[0]
    m, b = stats.linregress(xd,yd)[:2]
#     print(m,b)
#     pMid = fiber.getAvgPnt()
    newList = []
    
    if abs(m) > 2:
        # this adjusts and adds all the points
#         for i in range(len(fiber.pnts)):
#             newList.append( (int((fiber.pnts[i][1] - b)/m), fiber.pnts[i][1]) )
        # this only adjusts and saves the end points, since that's all we need.
        newList.append( ( (int(fiber.pnts[0][1] - b)/m), fiber.pnts[0][1]) )
        newList.append( ( (int(fiber.pnts[len(fiber.pnts)-1][1] - b)/m), fiber.pnts[len(fiber.pnts)-1][1]) )
        
    elif m != NaN:
        # this adjusts and adds all the points
#         for i in range(len(fiber.pnts)):
#             newList.append( (fiber.pnts[i][0], int(m*fiber.pnts[i][0] + b)) )
        # this only adjusts and saves the end points, since that's all we need.
        newList.append( (fiber.pnts[0][0], int(m*fiber.pnts[0][0] + b)) )
        newList.append( (fiber.pnts[len(fiber.pnts)-1][0], int(m*fiber.pnts[len(fiber.pnts)-1][0] + b)) )
    else:
        # fiber is perfectly vertical, can't really calculate its slope
#         newList.append( fiber.pnts[0] )
#         newList.append( fiber.pnts[len(fiber.pnts)-1] )
#         return Fiber(newList, fiber.w);
        return 0
    
    p1 = np.array(newList[0])
    p2 = np.array(newList[1])

    vec = p2 - p1
    length = sqrt(sqrDist(p1, p2))
    numSections = int(length/fiber.w) 
    vec /= numSections
    
    goodPoints = 0
    
    for i in range(0, numSections + 2):
        p = p1 + vec * i
        p = intTup(p)
        try:
#             if getAvgDotWhiteness(im, p, fiber.w//2) > im.avg + im.stdev/2:
#                 goodPoints += 1
            goodPoints += checkPoint(im, p, fiber.w//2)[0]
        except Exception:
            ()
    
    if goodPoints / (numSections + 1) > 0.70:
        return Fiber(newList, fiber.w)
    else:
        return 0


def straightFit(x, A, B): # this is your 'straight line' y=f(x)
    return A*x + B


def fixPoint(im, t, p, fiberW):
    '''
    This should check whether the next strip along the t direction is capable of being on a fiber,
    and if so, return the best point with the smallest change to t.
    '''
    stripCenX, stripCenY = p[:]
    stripW = 2 * fiberW
    stripL = 1 * fiberW 
      
    #corners are numbered in clockwise order, with the corner opposite c1 skipped 
#     t = angle / 180 * pi
    c1 = np.array(rotate( np.array([stripCenX-stripW/2, stripCenY-stripL/2]), t, (stripCenX, stripCenY)))
    c2 = np.array(rotate( np.array([stripCenX+stripW/2, stripCenY-stripL/2]), t, (stripCenX, stripCenY)))
    c3 = np.array(rotate( np.array([stripCenX-stripW/2, stripCenY+stripL/2]), t, (stripCenX, stripCenY)))
    v12hat = np.array([(c2[0]-c1[0])/stripW, (c2[1]-c1[1])/stripW])
    v13hat = np.array([(c3[0]-c1[0])/stripL, (c3[1]-c1[1])/stripL])
    
#     spacing = 1
    
#     divisions = 7
    
    strip = np.zeros(stripW)
#     stripAvgs = np.zeros(stripW//2 + 1)
#     
#     xCOM = 0
#     total = 0
    
    # go through each column in the strip
    for x in range(0, stripW, 1):
        colSum = 0
        for y in range(0, stripL, 1):
            p = c1 + x * v12hat + y*v13hat
            p = intTup(p)
            colSum += im.pixels(p) - im.avg
            
#         val = colSum*colSum
        distFromCenter = (1 - abs(0.1+((x-stripW/2)/(0.5*stripW))))
        if colSum > 0:
            strip[x] = colSum *colSum * distFromCenter
        else:
            strip[x] = colSum * colSum
        
#         xCOM += p[0] * strip[x/2]
#         total += strip[x/2]

#     bestI = 0
#     max0 = -100;
#     for i1 in range(0, len(stripAvgs)):
#         total = 0
#         stripAvg = 0
# #         stripAvgs[i1] = 0
#         for i2 in range(0, stripW//2):
#             t = strip[i1 + i2]
# #             stripAvgs[i1] += t
#             stripAvg += t
#             total += t
# #         stripAvgs[i1] /= total
#         stripAvg /= total
#         if stripAvg > max0:
#             bestI = i1
#             max0 = stripAvg
#             
# 
#     c = (int(p[0]+bestI), int( (c1[1] + (v12hat*stripW + v13hat*stripL)[1]/2) ))

    bestAvg = 0
    bestI = len(strip)//2
    for i in range(0, len(strip)):
        if strip[i] > bestAvg:
            bestAvg = strip[i]
            bestI = i
                
    c = (int(c1[0] + (v12hat*bestI + v13hat*stripL/2)[0]),
         int( (c1[1] + (v12hat*bestI + v13hat*stripL/2)[1]) ))
    
    return c


def traceFiber2(im, fiberW, initialP):
    jumpDist = 1
    # just in case, add something to break it if it goes into a loop.
    # Maybe if like the user doesn't blot out the background well and you get a thin white ring
    # going around the image, or a severely bent fiber. Don't want that to loop.
    # keep track of angle adjustments
    
    # first find best angle of lightness starting at p, have checkstrip do so like a clock hand;
    # check a strip in each direction and save the best angle
    # then move in that direction until you hit black.
    # count blue as white.
    # once done, color fiber in blue.
    # if it fails for some reason, return 0.
    

    initialP = getBestInRegion(im, initialP, getAvgSqrWhiteness(im, initialP, fiberW//2), fiberW)
    
    fiberPoints = []
    fiberPoints.append(initialP)
    
    try:
        t = getBestStartAngle(im, initialP, fiberW//4, 4*fiberW, 1)
    except IndexError:
        return 0

    atEnd = False
    firstDirectionCompleted = False
    prevP = initialP
    
    while not atEnd:
        nextP = (prevP[0]+int(fiberW*jumpDist*cos(t)),prevP[1]+int(fiberW*jumpDist*sin(t)))
        
        '''####'''
        stripCenX, stripCenY = nextP[:]
        stripW = 1 * fiberW
        stripL = 1 * fiberW
#         field = np.zeros((stripW, stripL))
        
        #corners are numbered in clockwise order, with the corner opposite c1 skipped 
        c1 = np.array(rotate( np.array([stripCenX-stripW/2, stripCenY-stripL/2]), t, (stripCenX, stripCenY)))
        c2 = np.array(rotate( np.array([stripCenX+stripW/2, stripCenY-stripL/2]), t, (stripCenX, stripCenY)))
        c3 = np.array(rotate( np.array([stripCenX-stripW/2, stripCenY+stripL/2]), t, (stripCenX, stripCenY)))
        v12hat = np.array([(c2[0]-c1[0])/stripW, (c2[1]-c1[1])/stripW])
        v13hat = np.array([(c3[0]-c1[0])/stripL, (c3[1]-c1[1])/stripL])
        
        avg = 0
        for x in range(0, stripW):
            colSum = 0
            for y in range(0, stripL):
                p = c1 + x * v12hat + y*v13hat
                colSum += im.pixels(p)
            avg += colSum
        avg /= stripL * stripW
        
        adjustedP = 0/1
    #     spacing = 1
        
    #     divisions = 7
        
        
#         # go through each column in the strip
#         for x in range(0, stripW, 1):
#             colSum = 0
#             for y in range(0, stripL, 1):
#                 p = c1 + x * v12hat + y*v13hat
#     #             pInt = intTup(p)
#                 colSum += im.pixels(p) - im.avg
#                 
#             distFromCenter = (1 - abs(0.1+((x-stripW/2)/(0.5*stripW))))
#             if colSum > 0:
#                 strip[x] = colSum *colSum * distFromCenter
#             else:
#                 strip[x] = colSum * colSum
        
        '''####'''
        
        
            
        try:
            newT = atan((adjustedP[1]-prevP[1])/(adjustedP[0]-prevP[0]))
        except ZeroDivisionError:
            newT = pi/2+0.001

        while newT - t > pi/2:
            newT -= pi
        while newT - t < -pi/2:
            newT += pi
        if abs(newT - t) > pi/8:
            #if it has a huge bend for some or any reason
            break
        t = newT

        if getAvgSqrWhiteness(im, adjustedP, fiberW//2) > im.avg + im.stdev/2:
            fiberPoints.append(adjustedP)
            prevP = adjustedP
            
        else:
            atEnd = True
            
        if atEnd and not firstDirectionCompleted:
#             print("In second thing")
            atEnd = False
            firstDirectionCompleted = True
            fiberPoints.reverse()
            prevP = initialP
            t -= pi
    

    if not pointsAreGood(im, im.avg, im.stdev, fiberPoints):
        return 0
    fiber = Fiber(fiberPoints, fiberW)
#     straightFiber = fiber
    straightFiber = getStraightFiber(im, fiber)
    # unnecessary, but otherwise I'll forget it can be 0 too
    if straightFiber == 0:
        return 0

    '''###'''

def traceFiber(im, fiberW, initialP):
    jumpDist = 4
    # just in case, add something to break it if it goes into a loop.
    # Maybe if like the user doesn't blot out the background well and you get a thin white ring
    # going around the image, or a severely bent fiber. Don't want that to loop.
    # keep track of angle adjustments
    
    # first find best angle of lightness starting at p, have checkstrip do so like a clock hand;
    # check a strip in each direction and save the best angle
    # then move in that direction until you hit black.
    # count blue as white.
    # once done, color fiber in blue.
    # if it fails for some reason, return 0.
    
#     initialP = (210, 196)
#     initialP = (205, 416)
#     im.putpixel(initialP, 180)
#     im.imgs[0].show()
#     return
    initialP = getBestInRegion(im, initialP, getAvgSqrWhiteness(im, initialP, fiberW//2), fiberW)
#     im.putpixel(initialP, 255)
#     im.imgs[0].show()
#     return
#     p = (208, 196)
#     p = (170, 145)
#     p = (40, 367)
# #     p = (185, 138)
#     pAvg = getAvgDotWhiteness(im, p, fiberW//2)
#     p5 = getBestInRegion(im, p, pAvg, fiberW)
#     p7 = getBestInRegion(im, p5, pAvg, fiberW)
#     t, lightness = getBestAngle(im, p, fiberW//4, 4*fiberW, 1)
# #     t+=pi
#     p3 = (p[0]+int(fiberW*4*cos(t)),p[1]+int(fiberW*4*sin(t)))
# 
#     p6 = getStripCentroid(im, t, p, fiberW, fiberW//4, avg)
#     im.putpixel(p, 70)
#     im.putpixel(p3, 100)
#     im.putpixel(p6, 180)
#     print(p,p6)
#     im.imgs[0].show()
#     return
# 
#     t, lightness = getBestAngle(im, p, fiberW//4, 4*fiberW, 1)
# #     t -= pi
#     p3 = (p[0]+int(fiberW*4*cos(t)),p[1]+int(fiberW*4*sin(t)))
#     t2 = atan(p3[1]/p3[0])
# #     print("***",(p2[0]-p[0], p2[1]-p[1]),t,t2, t2-t)
#     im.putpixel(p, 180)
# #     im.putpixel(p2, 255)
#     im.putpixel(p3, 255)
#     im.imgs[0].show()
#     print(t, lightness)
    
    # if any of the three dots it checks for the next point are blue, have it try again
    # treat blue as dark
    # If I treat blue as light:
    #    the code will not be able to cross intersections accurately
    # if I treat it as dark:
    #    it might cause some fibers to be cut in half, but only if they're intersected at a shallow angle.
    #    Assuming that if the next dot is dark and blue, it tries looking ahead in a straight line until that's no longer true.
    #    Maybe add some spread too.
    
    fiberPoints = []
    fiberPoints.append(initialP)
    
    try:
        t = getBestStartAngle(im, initialP, fiberW//4, 4*fiberW, 1)
    except IndexError:
        return 0
#     t+=pi
    atEnd = False
    firstDirectionCompleted = False
    prevP = initialP
    while not atEnd:
        nextP = (prevP[0]+int(fiberW*jumpDist*cos(t)),prevP[1]+int(fiberW*jumpDist*sin(t)))
        try:
#             adjustedP = getStripCentroid(im, t, nextP, fiberW, 1, im.avg)
            adjustedP = fixPoint(im, t, nextP, fiberW)
        except IndexError:
            adjustedP = nextP
            
        try:
            newT = atan((adjustedP[1]-prevP[1])/(adjustedP[0]-prevP[0]))
        except ZeroDivisionError:
#             print(prevP, nextP, adjustedP)
            newT = pi/2+0.001
#             raise
#         if initialP == (35, 325):
#             print("t =", t, newT, abs(newT - t) > pi/8, prevP, nextP, adjustedP)
#         print(t, newT)
        while newT - t > pi/2:
            newT -= pi
        while newT - t < -pi/2:
            newT += pi
#         if initialP == (35, 325):
#             print("t =", t, newT, abs(newT - t) > pi/8, prevP, nextP, adjustedP)
        if abs(newT - t) > pi/8:
            #if it has a huge bend for some or any reason
            break
        t = newT
#         print("**",t, newT)

        if getAvgSqrWhiteness(im, adjustedP, fiberW//2) > im.avg + im.stdev/2:
            fiberPoints.append(adjustedP)
            prevP = adjustedP
#             if len(fiberPoints) > 7:
#                 print("break", fiberPoints)
#                 break
        else:
            atEnd = True
            
#             print(adjustedP, getAvgSqrWhiteness(im, adjustedP, fiberW//2))
#             try:
#                 im.putpixel(adjustedP, 40)
#             except Exception:
#                 ()
            
        if atEnd and not firstDirectionCompleted:
#             print("In second thing")
            atEnd = False
            firstDirectionCompleted = True
            fiberPoints.reverse()
            prevP = initialP
            t -= pi
    
#     for p in fiberPoints:
#         im.putpixel(p, 255)
#     im.imgs[0].show()
#     print(len(fiberPoints))

    if not pointsAreGood(im, im.avg, im.stdev, fiberPoints):
        return 0
    fiber = Fiber(fiberPoints, fiberW)
#     straightFiber = fiber
    straightFiber = getStraightFiber(im, fiber)
    # unnecessary, but otherwise I'll forget it can be 0 too
    if straightFiber == 0:
        return 0
    
#     if fiber.getEndPointsStr() == "81 269 81 269":
#         print(initialP)
#         print(fiberPoints)
#         print(1/0)
#       
#     im.drawFiber(fiber, avg)
#     im.drawFiber(fiber, avg-stdev/2)
#     im.putpixel(initialP, 255)
#     try:
#         im.putpixel(fiberPoints[len(fiberPoints)-1], 150)
#     except Exception:
#         ()

#         print(fiberPoints[len(fiberPoints)-1])
#         raise
    return straightFiber 

def checkPoint(im, p, r):
    x,y = p[:]
    sidesBiggerThan = 0
    # getAvgSqrWhiteness(im, (x,y), r) takes about 5/6 of the time that avgDot takes, but consistently (as expected) reports lower values
    # for light regions than avgDot does, and identical values for dark regions. So it kinda skews the graph a little, and would
    # require a lower tolerance of color variation among the fibers.
    # I'll wait and see if the speed is needed.
    currentAvg = getAvgDotWhiteness(im, p, r)
    if currentAvg < im.avg + im.stdev/2:
        return 0,0
    for i in range(-1,2):
        for j in range(-1,2):
            try:
                if getAvgDotWhiteness(im, (x+r*i,y+r*j), r) < currentAvg:
                    sidesBiggerThan += 1
            except Exception:
                print("bad values were: ",(x+r*i,y+r*j), p, x, y )
                raise
    if sidesBiggerThan >= 4:
#         im0.putpixel((x,y), (0,255,0))
        return 1, currentAvg
    return 0, 0

def fixBrokenFibers(fiberList, fiberW):
    print("Hitching together broken fibers")
    i1 = 0
    while(i1 < len(fiberList)):
        f1 = fiberList[i1]
        i2 = i1+1
        while(i2 < len(fiberList)):
            f2 = fiberList[i2]
            
            dist, newLength, far1, far2 = getOrderedEndPoints(f1, f2)
#             if near1 == (87, 119) and near2 == (89, 114):
#                 print(dist, abs(f1.angle - f2.angle), pi/24)
#                 print(1/0)
                
            endpointsNear = (dist < 16*(fiberW**2))
            sameSlope = abs(f1.angle - f2.angle) < pi/24
            if endpointsNear and sameSlope and (newLength > f1.length) and (newLength > f2.length):
                fiberList.remove(f2)
                f1 = Fiber([far1, far2], fiberW)
                fiberList[i1] = f1
                i2 -= 1
            i2 += 1
        i1 += 1
    return fiberList


# this returns 2 distances and 2 points, in order: 
#   min, max, 2 farthest points (min, max, p3, p4)
def getOrderedEndPoints(f1, f2):
    p1, p2 = f1.getEndPoints()
    p3, p4 = f2.getEndPoints()
    dist13 = sqrDist(p1, p3)
    dist14 = sqrDist(p1, p4)
    dist23 = sqrDist(p2, p3)
    dist24 = sqrDist(p2, p4)
    
    l = [(dist13, p1, p3), (dist14, p1, p4), (dist23, p2, p3), (dist24, p2, p4)]
    
    minDist = (100000000000, 0)
    maxDist = (-5, 0)
    for i1 in range(0, 4):
        if l[i1][0] >= maxDist[0]:
            maxDist = l[i1]
        if l[i1][0] <= minDist[0]:
            minDist = l[i1]
    
    p1, p2 = minDist[1:]
    p3, p4 = maxDist[1:]
    return minDist[0], maxDist[0], p3, p4

# f1 = Fiber([(300, 300),(400, 430)], 10)
# f2 = Fiber([(100, 100),(290, 290)], 10)
# f1 = Fiber([(556, 331),(676, 327)], 10)
# f2 = Fiber([(688, 330),(780, 336)], 10)
# im = Image.new("RGB", (800,800), (0,0,0))
# im.putpixel(f1.getEndPoints()[0], (255, 0, 0))
# im.putpixel(f1.getEndPoints()[1], (255, 255, 0))
# im.putpixel(f2.getEndPoints()[0], (0, 255, 0))
# im.putpixel(f2.getEndPoints()[1], (0, 255, 255))
# im.show()
# print(f1.angle, f2.angle)
# print(abs(f1.angle - f2.angle))
# print( pi/35 )
# raise Exception("Look up/think about how to compare slopes")
# '''
# It appears that the ratio of slopes is much more stable near 1/1 than it is at extrema; therefore a different
# tolerance would be required for firbers with slope ~1 than for fibers with slope ~0.05
# '''
# 
# fL = [f1, f2]
# print(fL[0].getEndPointsStr(), fL[1].getEndPointsStr())
# fixBrokenFibers(fL, 10)
# print(fL[0].getEndPointsStr())
#    
# print(1/0)

def printReport(fiberList):
    print(len(fiberList), "fibers were found.")
    
    total = 0
    for i in range(0, len(fiberList)):
        total += fiberList[i].length
    avg = total/len(fiberList)
    
    print("Total length of fibers:", total)
    print("Average length of a fiber:", avg)
    
    sum0 = 0
    for i in range(0, len(fiberList)):
        dif = (fiberList[i].length - avg)
        sum0 += dif*dif
    stdev = sqrt(sum0/len(fiberList))
    
    print("Standard deviation of fiber length:", stdev)

def getSqrAvg(im, c1, v12hat, v13hat, sqrW):
    avg = 0
    for x in range(0, sqrW):
        for y in range(0, sqrW):
            p = intTup(c1 + x * v12hat + y*v13hat)
            avg += im.pixels(p)
    return avg/(sqrW**2)


def getFiberW(im, p, t, fiberW):
    # this gets the width of a (theoretically) solitary fiber
    tl = t - pi/2
    tr = t + pi/2
    vL_hat = 0
    
    ld = 0
    rd = 0
    
    pAvg = getAvgDotWhiteness(im, p, 2)
    
    for i in range(0, fiberW//2):
        x1 = 2
        

def testThingy(im, p0, fiberW):
    stripCenX, stripCenY = p0[:]
    stripW = 3 * fiberW
    stripL = 3 * fiberW
    t = getBestStartAngle(im, p0, fiberW, stripL, 1)
   
#     field = np.zeros((stripW, stripL))
        
    #corners are numbered in clockwise order, with the corner opposite c1 skipped. W/ X+ right, Y+ down. 
    c1 = np.array(rotate( np.array([stripCenX-stripW/2, stripCenY-stripL/2]), t, (stripCenX, stripCenY)))
    c2 = np.array(rotate( np.array([stripCenX+stripW/2, stripCenY-stripL/2]), t, (stripCenX, stripCenY)))
    c3 = np.array(rotate( np.array([stripCenX-stripW/2, stripCenY+stripL/2]), t, (stripCenX, stripCenY)))
    v12hat = np.array([(c2[0]-c1[0])/stripW, (c2[1]-c1[1])/stripW])
    v13hat = np.array([(c3[0]-c1[0])/stripL, (c3[1]-c1[1])/stripL])
#     localAvg = getSqrAvg(im, c1 + v13hat * stripW/2 + v12hat * stripW/2, v12hat, v13hat, fiberW)
#     print(localAvg, im.avg - im.stdev/4)
#     im.putpixel( intTup(p0), 100)
#     im.show(0)
#     return
    
#     sqrAvg = getSqrAvg(im, p0, v12hat, v13hat, fiberW//2)
    sqrAvg = getSqrAvg(im, c1 + v12hat/4 + v13hat/4, v12hat, v13hat, fiberW//2)
#     sqrAvg = getSqrAvg(im, c1, v12hat, v13hat, fiberW)

    totalBoxes = 0
    lightBoxes = 0
    #this iterates around the perimeter of a square box of points centered on p0
#     list = []
    
    for x in range(0, stripW):
        for y in range(0, stripL):
            p = intTup(c1 + v13hat * x + v12hat * y)
            totalBoxes += 1
            if im.pixels(p) > sqrAvg - im.stdev/3:
                lightBoxes += 1
    
#     for s in range(0, 2):
#         for w in range(0,2):
#             for l in range(1, stripL):
#                 cen = 0
#                 if w == 0:
#                     cen = c1 + s * v13hat * stripW + v12hat * l
#                 else:
#                     cen = c1 + s * v12hat * stripW + v13hat * l
# 
# #                 list.append((intTup(cen), int(l/stripL * 255)))
#                 localAvg = getSqrAvg(im, cen, v12hat, v13hat, fiberW)
#                 totalBoxes += 1
#                 if localAvg > im.avg - im.stdev/4:
#                     lightBoxes += 1

#     for p in list:
#         im.putpixel(p[0], p[1])
#     im.putpixel( intTup(p0), 100)
#     im.putpixel( intTup(p0 + 10*v12hat), 255)
#     im.show(0)
    prcntGood = (lightBoxes/totalBoxes)
#     print("\n  ", prcntGood, " ", t/pi * 180, "\n")
    return prcntGood

def main(fiberW, imDir, imName):
    
#     im = Image.new('RGB', (500,500), (0,0,0))
#     for i1 in range(0,500):
#         for i2 in range(0,500):
#             im.putpixel((i1,i2), (int(255*i1/500),int(255*i2/500),int(255*(1-i1/500)*(1-i2/500))))
#     im = Image.open("testImgs/singleStraightGlass2.jpg") # has avg of 16 and stdev of 6
#     im = Image.open("tempSmaller.jpg") # has avg of 16.5 and stdev of 7
#     pixels = im.load()
    im = BigImage.fromFile(imDir, imName, 4)
    imW, imH = im.size()
    print("Image size:", imW, "x", imH)
    
    print("Average and Stdev:",im.avg, im.stdev)
    im2 = BigImage.copy(BigImage, im)
    # single
#     p = (320, 140)
# #     p = (293, 75)
# #     p = (123, 75)
#     # crossing
# #     p = (205, 73)
# #     p = (103, 103)
# #     p = (257, 174)
#     
# #     print(testThingy(im, p, fiberW), getAvgSqrWhiteness(im, p, fiberW//2))
# #     im2.putpixel(p, 255)
# #     im2.show(0)
# # #     im3 = im2.imgs[0].resize((200, 120))
# # # #     im3.show()
# # #     im4 = im3.resize((imW//2, imH//2))
# # #     im4.show()
# #     return
# 
# #     for x in range(fiberW*3 + int(imW/8) - 80, 80+int(5*imW/16)-fiberW*3, 100):
# #         for y in range(fiberW*3 + int(imH/8) - 80, 80+int(5*imH/16)-fiberW*3, 100):
# #     Random.next
#     imCol = im2.imgs[0].copy().convert('RGB')
#     end = 5000
#     i = 0
#     while i < end:
#         x = randint(40, 420)
#         y = randint(30, 270)
#         prcntGood = testThingy(im, (x, y), fiberW)
#         avg = getAvgDotWhiteness(im, (x, y), fiberW//2)
#         if ( avg > im.avg + im.stdev):
#             if (prcntGood > 0.96):
# #             im2.putpixel((x, y), 255)
#                 imCol.putpixel((x, y), (255, 0, 0))
#                 print(i, end, avg, im.avg + im.stdev/4)
#                 i+=1
#             else:
#                 imCol.putpixel((x, y), (0, 255, 0))
#     imCol.show()
# #     im2.imgs[0].show()
# #     im2.imgs[1].show()
# #     im2.imgs[2].show()
# #     im2.imgs[3].show()
#     
# #     im2.saveAll()
#     return
     
#     for x in range(2, imW-2):
#         for y in range(2, imH-2):
#             if abs(im.pixels((x,y)) - avg) < stdev:
#                 c = 0
#                 for x1 in range(-1, 2):
#                     for y1 in range(-1, 2):
#                         if (x1 == 0) and (y1 == 0):
#                             continue 
#                         c += im.pixels((x+x1,y+y1))
#                 c /= 8
#                 im.putpixel((x,y), c)
#     im.imgs[0].save("colorAdjustedSmallTest.tif")
#     im.imgs[0].show()
#     return8
#     
#     #this is the number of lines to separate the strip into - must be ~ 3*fiberWidth
# #     stripW = fiberW*3
#     
#     # this accounts for slight luminosity deviations along the fiber, will need a bunch of pixels to do so.
#     # must be short enough though to assume all fibers look straight inside the strip though.
#     # ~60 maybe
# #     stripH = 60
# #     
# #     stripCenX = 50
# #     stripCenY = 50
# 
#     
    fiberList = []
#     fullFiberList = []
    
    skipSize = 10
#     p = (467,170)
#     print(getAvgDotWhiteness(im, p, 10))
#     im.putpixel(p, 255)
#     im.imgs[0].show()
#     return

    start = int(fiberW*1.5)
    stop = imW - int(fiberW*1.5)

    for x in range(start, stop, skipSize):
        for y in range(int(fiberW*1.5), imH - int(fiberW*1.5), skipSize):
            p = (x,y)

#             pAvg = getAvgSqrWhiteness(im, p, fiberW//2)
#             pointIsValid = pAvg > avg + stdev/2
            
            pointIsValid, pAvg = checkPoint(im, p, fiberW//2)
# 
            if pointIsValid:
                p = getBestInRegion(im, p, pAvg, fiberW)

                fiber = traceFiber(im, fiberW, p)

                if fiber != 0:
                    fiberList.append(fiber)
                    im.drawFiber(fiber, im.avg-im.stdev/8)
#                     try:
#                         im.putpixel(fiber.pnts[len(fiber.pnts)-1], 150)
#                         im.putpixel(fiber.pnts[0], 255)
#                     except Exception:
#                         ()
#                     if len(fiberList) > 1:
#                         im.imgs[0].show()
#                         return
        print("{}% completed".format(int(1000*x/(stop-start))/10))
#     im.imgs[0].show()
#     im.imgs[1].show()
#     im.imgs[2].show()
#     im.imgs[3].show()
#     
    
    fiberList = fixBrokenFibers(fiberList, fiberW)
    
    
    printReport(fiberList)
    
    
    f = open(os.path.join( im.imDir, im.extlessName, (im.extlessName + "_data_output_file.txt")), "w")
    
    for i in range(0, len(fiberList)):
#         im.drawFiber(fiberList[i], 80+int(i/len(fiberList)*(255-2*80)))
        im.drawFiber(fiberList[i], 80+int(i/len(fiberList)*(255-2*80)))
        try:
            im.putpixel(fiberList[i].pnts[0], 255)
            im.putpixel(fiberList[i].pnts[len(fiberList[i].pnts)-1], 255)
            f.write(fiberList[i].getEndPointsStr())
            if i < len(fiberList)-1:
                f.write("\n")
        except Exception:
            ()
        
        
#         im.drawFiber(fiberList[i], 20+2*int(i%115))
#         print(fiberList[i].length)
#     im.drawFiber(fiberList[5], 200)
#     im.imgs[int(len(im.imgs)/2)].show()
#     im.imgs[int((len(im.imgs)-1)/2)].show()
    im.imgs[0].show()
    im.imgs[1].show()
    im.imgs[2].show()
    im.imgs[3].show()
    
    im.saveAll()
    
    # http://stackoverflow.com/questions/403421
#     fiberList.sort()

def plotIm( im ):
    imshow(im, cmap='Greys_r', interpolation='none', origin='lower', vmin=0, vmax=255,
            extent=(0, len(im[0]), 0, len(im))
           )


def displayPlots( ims):
    fig = figure()
    
    if len(ims) == 0:
        show()
        return
    
    w = int( sqrt(2*len(ims)) )
    h = int(len(ims) / w)
    while w*h < len(ims):
        if min([w, h]) == w:
            w += 1
        else:
            h += 1
    
    i = 0
    for im in ims:
        fig.add_subplot(h, w, i + 1)
        plotIm(im)
        i += 1
        
    show()
    
    
def checkIfOnFiber(im, boolFilter, x0, y0):
    # boolFilter is something like [false false true true true rue false false]
    # and so his checks if there is a threshold below all/most of the true values and below all/most of the false ones
    # naive implementation:
    #     create list of false values and true values
    #     compare lowest of trues to highest of false
    #         maybe how many trues are less than the highest false, and vice versa
    #     if enough values are good, say the point is good.
    # 
    # 
    
    # x0 and y0 are at the center
    x0 -= len(boolFilter) / 2
    y0 -= len(boolFilter[0]) / 2
    
    if x0 < 0 or y0 < 0 or x0 + len(boolFilter) >= len(im) or y0 + len(boolFilter) >= len(im[0]):
        return False
    
    section = im[x0:x0+len(boolFilter), y0:y0+len(boolFilter[0])]
    upper = section * boolFilter
#     upper -= ndarray.min(upper)
#     upper /= (ndarray.max(upper)/255)
#     upper = upper.astype(int)
    
    lower = section * -1 * (boolFilter)
#     lower -= ndarray.min(lower)
#     lower /= (ndarray.max(lower)/255)
#     lower = lower.astype(int)
    print(upper)
    print(lower)
    print(boolFilter)
    print(-1 * (boolFilter))
    return upper, lower


def findPath(im, p1, p2):
    return True
    
    pathExists = False
    
    x_coords, y_coords = zip(*[p1, p2])
    A = np.vstack([x_coords, np.ones(len(x_coords))]).T
    m, b = np.lstsq(A, y_coords)[0]
    
    return pathExists


 
# def drawLine(im, p1, p2):
     
 
# def findFibersWeb(im):
#      
#              
#     '''
#     im is a binary image
#     '''
#     im = im.copy()
#      
#      
#     l0 = np.array([(48, 124), (56, 103), (65, 121), (73, 140), (79, 117)])
#      
#     l = []
#     for p in l0:
#         l.append(Point(tuple(p[:])))
#         im[ tuple(p[:]) ] = [1, 0, 0]
#      
#     for p1 in l:
#          
#         for p2 in l:
#             if findPath(im, p1, p2):
#                 Point.link(p1, p2)
#         if 
#          
#          
#      
# #     im[ im > 0 ] = 1
#     displayPlots([255*im])
#     return 
    
# def getDistribution(im):
#     points = []
#     
#     
#     return points


def getCol(depth):
    if depth % 3 == 0:
        return [1, 0, 0]
    elif depth % 3 == 1:
        return [0, 1, 0]
    else:
        return [0, 0, 1]
        
    
def splitDraw(im, out, x1, x2, y1, y2, depth):
#     print(((2 - depth)*"\t"), "Split,", x1, x2, y1, y2, depth)
    if (depth == 0) or (y1 == y2) or (x1 == x2):
        return
    depth -= 1
    region = im[y1:y2, x1:x2]
#     print(region)
#     print(len(im), len(im[0]), y2-y1, x2-x1, ndarray.mean(region))
    uniformity = np.abs(ndarray.mean(region)*2 - 1)
    print("U:", uniformity)
    if uniformity > 0.97:
        return
    yC, xC = np.array(ndimage.measurements.center_of_mass(region)).astype(int)
#     print(yC, xC)
    for x in range(x1, x2):
        out[y1 + yC][x] = getCol(depth)
    for y in range(y1, y2):
        out[y][x1 + xC] = getCol(depth)
#     print((1 - depth)*"\t_", x1, xC, x2, y1, yC, y2)
    splitDraw(im, out, x1, x1 + xC, y1, y1 + yC, depth )
    splitDraw(im, out, x1 + xC, x2, y1, y1 + yC, depth )
    splitDraw(im, out, x1, x1 + xC, y1 + yC, y2, depth )
    splitDraw(im, out, x1 + xC, x2, y1 + yC, y2, depth )
    return  


def split(im, out, x1, x2, y1, y2, depth):
#     print(((2 - depth)*"\t"), "Split,", x1, x2, y1, y2, depth)
    if depth == 0:
#         out[(y1 + y2)/2, (x1 + x2)/2] = getCol(depth)
        return
    if (y1 == y2) or (x1 == x2):
        return
    depth -= 1
    region = im[y1:y2, x1:x2]
#     print(len(im), len(im[0]), y2-y1, x2-x1, ndarray.mean(region))
    uniformity = np.abs(ndarray.mean(region)*2 - 1)
    print("U:", uniformity)
    if uniformity > 0.95:
        return
    yC, xC = np.array(ndimage.measurements.center_of_mass(region)).astype(int)
    print((1 - depth)*"\t_", x1, xC, x2, y1, yC, y2)
    out[y1 + yC, x1 + xC] = getCol(depth)
    split(im, out, x1, x1 + xC, y1, y1 + yC, depth )
    split(im, out, x1 + xC, x2, y1, y1 + yC, depth )
    split(im, out, x1, x1 + xC, y1 + yC, y2, depth )
    split(im, out, x1 + xC, x2, y1 + yC, y2, depth )
    return  

class Graph(dict):
    
    class Point:
        def __init__(self, p0):
            self.p = p0
            self.links = []
            self.end = False
            self.visited = False
             
        def getStatus(self):
            return len(self.links)
        
        def printLinks(self, prefix):
            for p0 in self.links:
                print(prefix, p0.p)
             
        @staticmethod
        def linked(p1, p2):
            return p2 in p1.links
             
        @staticmethod
        def link(p1, p2):
            if not Graph.Point.linked(p1, p2):
                p1.links.append(p2)
            if not Graph.Point.linked(p2, p1):
                p2.links.append(p1)
             
        @staticmethod
        def unlink(p1, p2):
            p1.links.remove(p2)
            p2.links.remove(p1)
             
        @staticmethod
        def cutOut(p):
            if len(p.links) != 2:
                # happens when an isolated cycle is reduced down from 3 points to 2
                # because removing one doesn't link the remaining two together again,
                # because they're already linked. SO they're now linked just to each
                # other - and therefore only have one link each.
                # should rarely happen, since linked points must be close to co-linear 
                return
            
            # to be called on points with only two links, in a line
            p1 = p.links[0]
            p2 = p.links[1]
            Graph.Point.unlink(p1, p)
            Graph.Point.unlink(p, p2)
            Graph.Point.link(p1, p2)
            
#         def combine(self, p1, p2):
#             # combines the links in p2 into p1
#             for p in p2.links:
#                 if not p in p1.links:
#                     Graph.Point.link(p1, p)
#                 Graph.Point.unlink(p, p2)
    
    def __init__(self, im, points, maxDist, fiberW):
        # for each point, look in a circle until you hit another point or reach the maximum
        # link distance. 
        radiusList = np.arange(1.0 , maxDist, 0.4)
        thetaList0 = np.arange(0 , 2*pi, pi/60)

        for p in points:
            self[p] = Graph.Point(p)
            
        self.im = im
        self.fw = fiberW
        self.endPoints = {}
        self.invSqrRt2 = 1/sqrt(2) 
#         raise Exception("test on tiny image with 45 degree line to see why it links so many together")
    
        i = 0
        for p0 in self:
            if i%100 == 0:
                print(i, len(points), "building graph")
            i+=1
#             if i == 1000:
#                 break
#             for entry in self.items():
#                 print(entry[0], [link.p for link in entry[1].links])
#             for t0 in range(0, 360, 6):
#                 t = t0*pi/180
            thetaList = thetaList0.copy()
            for t in thetaList:
                if t == -1:
                    continue
                for r in radiusList:
                    p = (int(p0[0] + (r * np.array([cos(t), sin(t)]))[0]), 
                         int(p0[1] + (r * np.array([cos(t), sin(t)]))[1]) )
                    if (p != p0) and (p in self):
#                         print("\t", (int)(t*100)/100, r, p, p0)
                        if not (Graph.Point.linked(self[p], self[p0])):
#                             print("\t\t", (int)(t*100)/100, r, p, p0)
                            notCycle = True
                              
                            for p1 in self[p0].links:
                                # if the points are adjacent, l/r/u/d.
#                                 print(p0, p, p1.p, abs(p[0] - p1.p[0]) + abs(p[1] - p1.p[1]))
                                if abs(p[0] - p1.p[0]) + abs(p[1] - p1.p[1]) == 1:
                                    notCycle = False
                            
                            if findPath(im, p0, p) and notCycle:
#                                 print("\t\t\t", (int)(t*100)/100, r, p, p0)
                                Graph.Point.link(self[p], self[p0])
                                self.removeArc(thetaList, t, r)
                                break
                        else:
                            break


#             for t0 in range(0, 360, 6):
#                 t = t0*pi/180
#                 for r in radiusList:
#                     p = (int(p0[0] + (r * np.array([cos(t), sin(t)]))[0]), 
#                          int(p0[1] + (r * np.array([cos(t), sin(t)]))[1]) )
#     #                 print("\t", p)
#                     if Graph.temp1(im, p, p0, points, self):
#                         break
            

    #             p = (int(p0[0] + (radiusList[-1] * np.array([cos(t), sin(t)]))[0]), 
    #                  int(p0[1] + (radiusList[-1] * np.array([cos(t), sin(t)]))[1]) )
                    
    #                 im[p] = 255
    #         displayPlots([im])
    #         return
        
        
        # you need two, since I broke small cycles in the init,
        # so what would have been
        #     1-{2}, 2-{1, 3, 4}, 3-{2, 4}, 4-{2, 3, 5}, 5-{4)
        # is now
        #     1-{2}, 2-{1, 3}, 3-{2, 4}, 4-{2, 3, 5}, 5-{4)
        # which yields after the first prune
        #     1-{2}, 2-{1, 3}, 3-{2, 4}, 4-{3, 5}, 5-{4)
        # which yields
        #     1-{2}, 2-{1, 3}, 3-{2, 4}, 4-{3, 5}, 5-{4), then
        #     1-{2}, 2-{1, 3}, 4-{2, 5}, 5-{4), then
        #     1-{5}, 5-{1)
        print("About to prune")
        self.prune()
        print("About to prune again")
        self.prune()
        print("...Done")
        
        self.getEndpoints()
    
#     @staticmethod
#     def temp1(im, p, p0, points, g):
#         if (p != p0) and (p in g):
#             if not (Graph.Point.linked(g[p], g[p0])):
#                 
#                 notCycle = Graph.temp2(p, p0, g)         
#                  
#                 if findPath(im, p0, p) and notCycle:
#                     Graph.Point.link(g[p], g[p0])
#             else:
#                 return True
#     
#     @staticmethod
#     def temp2(p, p0, g):
#         notCycle = True
#                  
#         for p1 in g[p0].links:
#             # if the points are adjacent, l/r/u/d.
# #                                 print(p0, p, p1.p, abs(p[0] - p1.p[0]) + abs(p[1] - p1.p[1]))
#             if abs(p[0] - p1.p[0]) + abs(p[1] - p1.p[1]) == 1:
#                 notCycle = False
#         return notCycle
    
    def removeArc(self, angleList, angle, dist):
        # times 4 to lessen likelihood of cycles
        arcAngle = 4* atan( self.invSqrRt2 / dist )
        lBnd = angle - arcAngle
        rBnd = angle + arcAngle
        if lBnd < 0:
            lBnd += 2*pi
        if rBnd > 2*pi:
            rBnd -= 2*pi

        for i in range(0, len(angleList)):
            if angleList[i] == -1:
                continue
            if angleList[i] > lBnd and angleList[i] < lBnd + 2 * arcAngle:
                angleList[i] = -1
            elif angleList[i] < rBnd and angleList[i] > rBnd - 2 * arcAngle:
                angleList[i] = -1
            
    
    @staticmethod
    def toPointDict(l):
        d = {}
        for p in l:
            d[p] = Graph.Point(p)
        return d
    
    def getEndpoints(self):
        self.endPoints = {}
        for p in self:
            if len(self[p].links) == 1:
                self.endPoints[p] = self[p]
    
    def traceFibers(self, minL):
        
        for p in self.copy():
            if not (p in self):
                continue
            
            fiberGood = False
            l = []
            
            raise NotImplemented
        
            l.append(self[p])
            
            f = Fiber(l, self.fw)
            
            if fiberGood:
                # this seems slow, having to make a mini graph every fiber
                fGraph = Graph(f.pnts)
                self.unlinkChain( l )
                fGraph.prune()
            else:
                ()
                # Maybe?
    #             unlinkFiber( l )
    
    def prune(self):
        pToDelete = []
        pToCut = []
        for p0 in self:
            p = self[p0]
            if len(p.links) < 1:
                pToDelete.append(p.p)
                
            inLine = True
            if (len(p.links) == 2) and inLine:
                pToCut.append(p.p)
                
        for p in pToDelete:
            self[p].visited = True
            if self[p] in self.endPoints:
                self.endPoints.remove(self[p])
            del self[p]
            
        for p in pToCut:
            if (len(self[p].links) == 2):
#                 print("....", p, len(self[p].links))
#                 self[p].printLinks(",,,,")
                Graph.Point.cutOut(self[p])
                del self[p]
            
    def drawGraph(self, im):
    #     for p in g:
    #         g[p].visited = False
        
        for p1 in self:
            for p2 in self[p1].links:
                rr, cc, val = line_aa(*p1 + p2.p)
                im[rr, cc] = 55  + randint(0, 1)
    #             if not g[p2].visited:
    #                 
    #     for p in g:
    #         g[p].visited = False
    
    def getNextPoint(self, node, prev, angleTol):
        # angleTol is the tolerance on either side. So half of total range.
        t0 = self.getAngle(prev.p, node.p)
        best = (0, angleTol)
        for x in node.links:
            if x == prev:
                continue
            t = self.getAngle(node.p, x.p)
            err = t - t0
            while err > pi:
                err -= 2 * pi 
            while err < -pi:
                err += 2 * pi
            
            if abs(err) < best[1]:
                best = (x, abs(err))
        
        return best[0]
            
    
    def getAngle(self, p1, p2):
        return atan((p2[1] - p1[1])/(p2[0] - p1[0]+0.1))
    
    def findFibers(self):
        fiberList = []
        
#         print("fibers")
        for e in self.endPoints.copy():
#             print("e: ", e)
            if not e in self.endPoints or not e in self:
                continue
            
            chain = []
            
            prv = self[e]
            nxt = self[e].links[0]
            chain.append(self[e])
            
            while True:
#                 print(nxt.p)
                temp = -1
#                 print(nxt.p, len(nxt.links))
                if len(nxt.links) == 2:
                    if nxt.links[0] != prv:
                        temp = nxt.links[0]
                    else: 
                        temp = nxt.links[1]
                elif  len(nxt.links) == 1:
                    chain.append(nxt)
                    break
                else: #intersection
                    nxP = self.getNextPoint(nxt, prv, pi/8)
                    if nxP != 0:
                        temp = nxP
                        
                if temp == -1:
                    chain.append(nxt)
                    break
                
                prv = nxt
                nxt = temp
#                 print(nxt)
                chain.append(nxt)
                
                if nxt.p not in self.endPoints:
                    break
                
            chain0 = []
            for p in chain:
                chain0.append(p.p)
#             print(chain0)
            Graph.unlinkChain(chain)
            self.prune()
            #don't prune the whole thing, just prune the points that had been touched
#             self.pruneChain()
            
            
            f = Fiber(chain0, self.fw)
            if f.length > 4:
                fiberList.append(f)
        return fiberList
    
    @staticmethod
    def unlinkChain( l ):
        for i in range(0, len(l) - 1):
            p1 = l[i]
            p2 = l[i + 1]
            if Graph.Point.linked(p1, p2):
                Graph.Point.unlink(p1, p2)
    
    def save(self, fileName):
        file = open(fileName, mode='w')
        for p in self:
            string = " ".join(str(x) for x in p) + " " + " ".join(" ".join(str(x) for x in p0.p) for p0 in self[p].links) + "\n"
            print(string)
            file.write(string)
    
    @staticmethod
    def load(fileName):
        file = open(fileName, mode='r')
        file.read
        graph = Graph()
        print("here")
        for l in file:
            print("__")
            l = l[:-1].split(" ")
            for i in range(0, len(l)):
                l[i] = int(l[i])
            print(l)
            p = (l[0], l[1])
            l = l[2:]
            print(l)
            point = Graph.Point(p)
            for i in range(0, len(l), 2):
                point.links.append((l[i], l[i+1]))
                
        for p in graph:
            for i in range(0, len(graph[p].links)):
                if graph[p].links[i] in graph:
                    graph[p].links[i] = graph[graph[p].links[i]]
                else:
                    print(graph[p].links[i])


def getLineCoeffs(p1, p2):
    print(p1, p2, p1[1]*p2[0], p2[1]*p1[0])
    a = (p2[1] - p1[1])/(p1[0]*p2[1] - p2[0]*p1[1] + 0.01)
    b = (p2[0] - p1[0])/(p1[1]*p2[0] - p2[1]*p1[0] + 0.01)
    return a, b
#     print(a, b)
#     if abs(a) < abs(b):
#         for x in range(p1[0], p2[0] + 1):
#             y = 1/b - (a/b)*x
#     else:
#         for y in range(p1[1], p2[1] + 1):
#             x = 1/a - (b/a)*y
#     return (y, x)

def binFibers(fiberSegments, imW, imH):
    '''
    This will store bins in a dictionary, allowing for range queries by dividing out the
    lower numbers of the keys of access.
    The bin will be found using the bin dictionary, and then that bin's count will be incremented.
    At the end, the counts will be used to find local maximums of line likelihood.
    The line will be traced from left to right until it gets close to a segment, and will continue
    looking ahead until it goes off-screen or stops being some distance close to a segment.
    Segments in that bin, that is.
    Remember to only look ahead for a lack of a segment, don't adjust the new endpoint until you've
    found a segment to extend the line to.
    '''
    class Bin:
        def __init__(self, a, b):
            # ax + by = 1
            self.a = a
            self.b = b
            self.count = 0
    
    numBins = 500
    bins = {}
    
    for f in fiberSegments:
        a, b = getLineCoeffs(f.pnts[0], f.pnts[len(f.pnts) - 1])
        A = int(a*numBins)
        B = int(b*numBins)
        bin = Bin(a, b)
        if not (A, B) in bins:
            bins[(A, B)] = bin
        bins[(A, B)].count =+ 1
        print(a, b)

    


def tempGraphStuff(im, l, fw):
#     fw = 2
#     l = [(4, 4), (5, 5), (13, 5), (5, 15), (15, 15), (10, 10), (20, 20), (5, 28)]
#     l = [(4, 4), (6, 4), (6, 5), (3, 2), (8, 5), (9, 7)]
#     im = np.zeros((10, 10))
    im0 = im.copy()
    print("About to create graph")
    g = Graph(im, l, 1.5*sqrt(5), fw)
#     g = Graph(im, l, 6, fw) # find how to fix cycles
    print("...Done")
#     return
    
    
#     g.drawGraph(im0)
#     for p in g:
#         im0[p] = 255
#         print(g[p].p, [p0.p for p0 in g[p].links], im0[p])
        

#     g.save("1_50_may_5.txt")
#     
#     im1 = im.copy()
#     newG = Graph.load("1_50_may_5.txt")
    
    
#     file = open(fileName, mode='w')
#     string = " ".join(str(p) for p in g.endPoints)
#     print(string)
#         file.write(string)
    
    im1 = im.copy()
    
    g.drawGraph(im)
    for p in g:
        im[p] = 255
    
    
    fibers = g.findFibers()
    
    binFibers(fibers, len(im), len(im[0]))
    
    for f in fibers:
        print(f, f.pnts[0], f.pnts[len(f.pnts) - 1])
        p1 = f.pnts[0]
        p2 = f.pnts[len(f.pnts) - 1]
        
        rr, cc, val = line_aa(*p1 + p2)
        im1[rr, cc] = 55  + randint(0, 1)
        
        im1[p1] = 255
        im1[p2] = 255
    
#     newG.drawGraph(im1)
#     for p in g:
#         im[p] = 255
#         print(g[p].p, [p0.p for p0 in g[p].links])
     
    displayPlots([im, im1])
    
    
    return

# when traversing, give nodes a value that is unique for every fiber trace.
# to act as a 'visited' flag, without needing to reset all flags between fiber traces.


def filterImage( im0, fw):
#     im0 = ndimage.gaussian_filter(im.copy(), sigma=2)
    blankIm = np.zeros((len(im0), len(im0[0])))

    fH = 5 * horizEdgeArray(fw, 1)
    fV = 5 * vertEdgeArray(fw, 1)
     
    res0 = pickyConvolvement(im0.astype(int), fH, fV)
    res0[res0 < 0] = 0
    res0[res0 > 255] = 255
    res1 = 256*toBinImg(res0, 120)
     
    im = toBinImg(res1, 200)
    
    distance = ndimage.distance_transform_edt(im)
    
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=im)
    
    f0 = np.zeros((3,3))
    f0[0][0] = 1
    f0[2][0] = 1
    f0[1][2] = 1
    f1 = np.rot90(f0)
    f2 = np.rot90(f1)
    f3 = np.rot90(f2)
    f4 = ndimage.generate_binary_structure(2, 1).astype(int)
    f4[1,1] = 0
    f4[1] *= 2

    temp0 = toBinImg(local_maxi - ndimage.convolve(local_maxi, f0)//2)
    temp1 = toBinImg(temp0 - ndimage.convolve(temp0, f1)//2)
    temp2 = toBinImg(temp1 - ndimage.convolve(temp1, f2)//2)
    temp3 = toBinImg(temp2 - ndimage.convolve(temp2, f3)//2)
    final = toBinImg(temp3 - ((ndimage.convolve(temp3, f4)%4) + 1 )//4)
    ims = [ im0, local_maxi * 64, temp3*64, final*64 ]
#     displayPlots(ims)
#     return
    maxPoints = np.nonzero(final)
     
    maxPoints = list(zip(maxPoints[0], maxPoints[1]))
    
    tempGraphStuff(blankIm, maxPoints, fw)
    return final

from quickFuncs import *
def cPythonStuff():
    
#     call(["python3", "setup.py", "build_ext", "--inplace"])
#     im0 = fiberBox((20, 20), (10, 10), 3*pi/4, 5)
#     im0 = mergeIms( fiberBox((40, 20), (10, 15), 4*pi/7, 5), fiberBox((40, 20), (30, 15), 8*pi/7, 5) )
#     im1 = mergeIms( fiberBox((40, 20), (30, 15), 2*pi/7, 5), fiberBox((40, 20), (20, 15), 7*pi/9, 5) )
#     im0 = mergeIms(im0, im1)
#     im0 = fiberBox((40, 20), (10, 15), 4*pi/7, 5)
#     im0 = mergeIms( fiberBox((400, 200), (100, 150), 4*pi/7, 10), fiberBox((400, 200), (300, 150), 3*pi/9, 10) )
#     filterImage(im0, 5)
    
#     im = np.zeros((20,20))
# 
#     p1 = (2, 3)
#     p2 = (7, 10)
#     
#     a = (p2[1] - p1[1])/(p1[0]*p2[1] - p2[0]*p1[1])
#     b = (p2[0] - p1[0])/(p1[1]*p2[0] - p2[1]*p1[0])
#     print(a, b)
#     if abs(a) < abs(b):
#         for x in range(p1[0], p2[0] + 1):
#             y = 1/b - (a/b)*x
#             im[(int(y), int(x))] = 255
#     else:
#         for y in range(p1[1], p2[1] + 1):
#             x = 1/a - (b/a)*y
#             im[(int(y), int(x))] = 255
#     displayPlots([im])
    
    im0 = np.array( Image.open("Images/smallTest2.jpg") )
    im0 = ndimage.gaussian_filter(im0.copy(), sigma=3)
  
    filterImage(im0, 10)
    
    return
#     x, y = np.indices((5, 5))
#     x1, y1, = 2, 2
#     r1 = 1.2
#     mask_circle = (x - x1) ** 2 + (y - y1) ** 2 < r1 ** 2
#     print(mask_circle)
#     local_maxi = peak_local_max(distance, indices=False, footprint=mask_circle, labels=im)
    
#     im1 = toBinImg(distance.copy(), 2)

#     findFibersWeb(im)
    
    
#     l = [im0, 255*im, 255*(distance/ndarray.max(distance)), 255*(im1/ndarray.max(im1))]
#     displayPlots(l[1::2])
#     displayPlots([im0])
#     pyplot.imshow(255*out1)
#     show()
    
#     print(out)
    return
    
    
#     im1 = mergeIms(fiberBox(dimensions, (40, 80), 7*pi/8, 10), fiberBox(dimensions, (180, 130), pi/4, 10))
#     im2 = mergeIms(fiberBox(dimensions, (150, 70), 0*pi/8, 10), fiberBox(dimensions, (20, 180), 5*pi/8, 10))
#     im3 = mergeIms(im1, fiberBox(dimensions, (150, 70), 4*pi/8, 10))
#     im = mergeIms(im2, im3)
#     temp = np.zeros(5)
#     temp[1] = 1
#     temp[2] = 2
#     temp[3] = 1
#     indx = temp >= 1 
#     print(temp, ndarray.sum(indx))
#     raise Exception("\nsome way to determine if, when a threshhold is applied to a given square \
# as it is examined,\n the resulting boolean matrix is close enough to what it \
# should be.")
#     
#     return

    im = np.array( Image.open("Images/smallTest2.jpg") )
    im0 = im.copy()

    im = ndimage.gaussian_filter(im.copy(), sigma=2)
    
    
    fw = 10
    fH = 5 * horizEdgeArray(fw, 1)
    fV = 5 * vertEdgeArray(fw, 1)
    
    res0 = pickyConvolvement(im.astype(int), fH, fV)
    res0[res0 < 0] = 0
    res0[res0 > 255] = 255
    res0 = 256*toBinImg(res0, 120)
    res1 = ndimage.median_filter(res0, size=8)
#     findFibersWeb()
    displayPlots([res0, res1])
    return
    
      
#     sx = ndimage.sobel(im, axis=0, mode='constant')
#     sy = ndimage.sobel(im, axis=1, mode='constant')
#     sob = np.hypot(sx, sy)
#       
#     bin0 = toBinImg(im0.copy(), 31)
#     bin1 = toBinImg(res2.copy(), 131)
#       
#     close_bin0 = ndimage.binary_closing(bin0)
#     close_bin1 = ndimage.binary_closing(bin1)
#       
#     close_open_bin0 = ndimage.binary_opening(close_bin0)
#     close_open_bin1 = ndimage.binary_opening(close_bin1)
#     subIms = [bin0, bin1]
#     displayPlots( subIms )


'''
First off, get clean binary image.
Then create a network of connected white spots.
    Overlay a larger grid. At each intersection, check for whiteness (in binary img)
    if white, check immediate surrounding area (on grid) for whiteness.
    Repeat, unless point has been processed already.
    If found, create add link to the second spot, to the first one.
    So points now hold links to all the white spots near them.
    Remove all points with exactly two links in a relative line through the center.
        like A-B-C-D --> A-D
    So now have a list of endpoints and intersections.
    Go through endpoints.
        Follow links, in the straightest line you can, until you either hit another endpoint
        or there are no more links close enough to being in front of you.
        Save that thing as a line.
        Remove all links used by the fiber.
        If what was an intersection now has only two links, remove it like before.
        If it only has one, add it to the endpoints list.
        If it has zero, remove it entirely.

'''

    
if __name__ == "__main__":
#     import cProfile
#     cProfile.run('cPythonStuff()')
    cPythonStuff()
    
#     d1 = datetime.datetime.now()

#     im = BigImage("","rainbow.jpg",9)
#     im = BigImage('Images/smallTest.jpg')
#     main(10, "Images/","colorAdjustedSmallTest.tif")
#     test()
#     main(10, "Images/","smallTest.jpg")

#     main(8, "Images/","smallTest2.jpg")

#     main(10, "Images", "midSizedTest.jpg")
#     main(10, 'Images/CarbonFiber/', 'GM_LCF_EGP_23wt%_Middle_FLD1(circleLess).tif')

#     d2 = datetime.datetime.now()
#     print('Running time:', d2-d1)



    