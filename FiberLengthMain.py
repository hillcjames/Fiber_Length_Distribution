'''
Created on Jul 13, 2015

@author: Christopher Hill

This will count fibers in an image and measure their lengths.

It requires the installation of python3, PIL, and image_slicer
on linux, you can just do:
    sudo apt-get install python3
    look up how to install pip, AND install it using python3, not python
    use pip to install PIL
    sudo python3 -m pip install image_slicer

Assumes:
    
    
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
        self.sliceW, self.sliceH = data[10][:]
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
        avg = 0
        stdev = 0
        names = []
        imgs = []
        pxls = []
        extlessName = ""
        sliceW, sliceH = 0, 0
        w, h = 0, 0
        
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
        
        for x in range(0, len(imgs)):
            pxls.append(imgs[x].load())
        sliceW, sliceH = imgs[0].size
        
        avg, stdev = getStats(cls.pixels, sliceW, sliceH)
        
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
        data.append(sliceW, sliceH)
        
        return cls(data)
    
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
            names.append( original.names[i].copy() )
            imgs.append(  original.imgs[i].copy()  )
            pxls.append(  original.pxls[i].copy()  )

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
        data.append(sliceW, sliceH)
        
        return cls(data)
        
        
    def pixels(self, p):
#         if p[0] is float:
#             return self.floatPixels(p)
        # if the slices are arranged on a grid, the x-coord on that grid
        x = int(p[0] / self.sliceW)
        # if the slices are arranged on a grid, the y-coord on that grid
        y = int(p[1] / self.sliceH)
        
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
        x = int(p[0] / self.sliceW)
        # if the slices are arranged on a grid, the y-coord on that grid
        y = int(p[1] / self.sliceH)
#         
#         p0 = (p[0] % self.sliceW, p[1] % self.sliceH)
#         
        p1 = tuple((p[0]-x*self.sliceW, p[1]-y*self.sliceH))
#         print(p, p1)
        self.imgs[x+y*self.num_columns].putpixel(p1,c)
        
#         self.pxls[ x + y*self.num_columns ][p0] = c
#     def copy(self):
#         copy = BigImage()
#         self.num_columns
#         self.num_rows
#         self.sliceW, self.sliceH
#         self.names
#         self.imgs
#         self.pxls

    # This is dumb and probably slow but I'll fix it later.
    # Instead of only printing the fiber on the tiles the fiber touches, it just tries to print it on all of them.
    def drawFiber(self, fiber, c):
        for i in range(0, len(self.imgs)):
            offset = ( (i%self.num_columns)*self.sliceW, int(i/self.num_columns)*self.sliceH )
#             print("**********",offset, self.sliceW, self.sliceH, i, self.num_columns, self.num_rows)
            fiber.draw(self.imgs[i], offset, c)
    
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
        self.calcAngle()
        self.calcLength()
    
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
#         return (int(xSum/len(self.pnts)), int(ySum/len(self.pnts)))
    
    def draw(self, im, offset, c):
        draw = ImageDraw.Draw(im)
        for i in range(0, len(self.pnts)-1):
            p1 = (self.pnts[i][0] - offset[0], self.pnts[i][1] - offset[1])
            p2 = (self.pnts[i+1][0] - offset[0], self.pnts[i+1][1] - offset[1])
            draw.ellipse([(p1[0]-self.w/2+1, p1[1]-self.w/2+1),(p1[0]+self.w/2-1, p1[1]+self.w/2-1)], fill = c)
#             draw.ellipse([(p1[0]-offset[0]-self.w/2+1, p1[1]-offset[1]-self.w/2+1),(p1[0]-offset[0]+self.w/2-1, p1[1]-offset[1]+self.w/2-1)], fill = c)
            draw.line([p1,p2], width = int(self.w*1.2), fill = c)
        p0 = (self.pnts[len(self.pnts)-1][0] - offset[0], self.pnts[len(self.pnts)-1][1] - offset[1])
        draw.ellipse([(p0[0]-self.w/2+1, p0[1]-self.w/2+1),(p0[0]+self.w/2-1, p0[1]+self.w/2-1)], fill = c)
        
    def getEndPointsStr(self):
        return str(self.pnts[0][0]) + " " + str(self.pnts[0][1]) + " " + str(self.pnts[len(self.pnts)-1][0]) + " " + str(self.pnts[len(self.pnts)-1][1])
        
    def getEndPoints(self):
        return self.pnts[0], self.pnts[len(self.pnts)-1]
        
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

def rotate(pts, t, center = [0,0]):
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
    
    
# swing a box around the given point like a clock hand, looking for the angle which covers the most white.
def getBestStartAngle(im, p, fiberW, stripL, spacing):
    
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
                    whiteness += im.pixels( (int(p0[0]), int(p0[1])) )
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
    
    
    return bestT, bestVal

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
            pInt = (int(p[0]), int(p[1]))
            #gives a value 0-1 based on distance from centerline, where points along the centerline are highest
            distFromCenter = (1 - abs(0.1+((x-stripW/2)/(0.5*stripW))))
#             val = (im.pixels(pInt) - avg) * distFromCenter
            val = (im.floatPixels(p) - avg) * distFromCenter
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
                sum1 += im.pixels((x0-r+x,y0-r+y))
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
        pMid = ( int((fPoints[i+1][0]+p[0])/2), int((fPoints[i+1][1]+p[1])/2) )
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
        newList.append( (int((fiber.pnts[0][1] - b)/m), fiber.pnts[0][1]) )
        newList.append( (int((fiber.pnts[len(fiber.pnts)-1][1] - b)/m), fiber.pnts[len(fiber.pnts)-1][1]) )
        
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
        p = (int(p[0]), int(p[1]))
        try:
#             if getAvgDotWhiteness(im, p, int(fiber.w/2)) > im.avg + im.stdev/2:
#                 goodPoints += 1
            goodPoints += checkPoint(im, p, int(fiber.w/2))[0]
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
    stripAvgs = np.zeros(int(stripW/2 + 1))
    
    xCOM = 0
    total = 0
    
    # go through each column in the strip
    for x in range(0, stripW, 1):
        colSum = 0
        for y in range(0, stripL, 1):
            p = c1 + x * v12hat + y*v13hat
#             pInt = (int(p[0]), int(p[1]))
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
#         for i2 in range(0, int(stripW/2)):
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
    bestI =int(len(strip)/2)
    for i in range(0, len(strip)):
        if strip    [i] > bestAvg:
            bestAvg = strip[i]
            bestI = i
                
    c = (int(c1[0] + (v12hat*stripW + v13hat*stripL)[0]/2),
         int( (c1[1] + (v12hat*stripW + v13hat*stripL)[1]/2) ))
    
    return c

def traceFiber(im, fiberW, initialP):
    jumpDist = 2
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
    initialP = getBestInRegion(im, initialP, getAvgSqrWhiteness(im, initialP, int(fiberW/2)), fiberW)
#     im.putpixel(initialP, 255)
#     im.imgs[0].show()
#     return
#     p = (208, 196)
#     p = (170, 145)
#     p = (40, 367)
# #     p = (185, 138)
#     pAvg = getAvgDotWhiteness(im, p, int(fiberW/2))
#     p5 = getBestInRegion(im, p, pAvg, fiberW)
#     p7 = getBestInRegion(im, p5, pAvg, fiberW)
#     t, lightness = getBestAngle(im, p, int(fiberW/4), 4*fiberW, 1)
# #     t+=pi
#     p3 = (p[0]+int(fiberW*4*cos(t)),p[1]+int(fiberW*4*sin(t)))
# 
#     p6 = getStripCentroid(im, t, p, fiberW, int(fiberW/4), avg)
#     im.putpixel(p, 70)
#     im.putpixel(p3, 100)
#     im.putpixel(p6, 180)
#     print(p,p6)
#     im.imgs[0].show()
#     return
# 
#     t, lightness = getBestAngle(im, p, int(fiberW/4), 4*fiberW, 1)
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
        t = getBestStartAngle(im, initialP, int(fiberW/4), 4*fiberW, 1)[0]
    except IndexError:
        return 0
#     t+=pi
    atEnd = False
    firstDirectionCompleted = False
    prevP = initialP
    while not atEnd:
        nextP = (prevP[0]+int(fiberW*jumpDist*cos(t)),prevP[1]+int(fiberW*jumpDist*sin(t)))
        try:
            adjustedP = getStripCentroid(im, t, nextP, fiberW, 1, im.avg)
#             adjustedP = fixPoint(im, t, nextP, fiberW)
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

        if getAvgSqrWhiteness(im, adjustedP, int(fiberW/2)) > im.avg + im.stdev/2:
            fiberPoints.append(adjustedP)
            prevP = adjustedP
#             if len(fiberPoints) > 7:
#                 print("break", fiberPoints)
#                 break
        else:
            atEnd = True
            
#             print(adjustedP, getAvgSqrWhiteness(im, adjustedP, int(fiberW/2)))
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
    im.show(1)
    return
    
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

#             pAvg = getAvgSqrWhiteness(im, p, int(fiberW/2))
#             pointIsValid = pAvg > avg + stdev/2
            
            pointIsValid, pAvg = checkPoint(im, p, int(fiberW/2))
# 
            if pointIsValid:
                p = getBestInRegion(im, p, pAvg, fiberW)
#                 im.putpixel(p,255)
# #     for p in l1:
# #         im.putpixel(p, 180)
# #     for p in l2:
# #         im.putpixel(p, 255)
# #     im.imgs[0].show()
# #     im.imgs[1].show()
# #     return
                fiber = traceFiber(im, fiberW, p)
#                 im.putpixel(p,90)
#                 return
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
    



# def test():
#     fiberW = 9
#     im = BigImage("Images/","colorAdjustedSmallTest.tif", 2)
#     imW, imH = im.size()
#     print("Image size:", imW, "x", imH)
#     
#     stdev, avg = im.getStats()
#     print("Average and Stdev:",avg,stdev)
#     fiberList = []
#     
#     map = np.zeros(shape=(int(imW/3),int(imH/3)))
#     
#     skipSize = 10
# 
#     start = int(fiberW*0.5)
#     stop = imW - int(fiberW*0.5)
#     
#     for x in range(start, stop, 3):
#         for y in range(int(fiberW*0.5), imH - int(fiberW*0.5), 3):
#             p = (x,y)
#             map[x/3][y/3] = getAvgDotWhiteness(im, p, int(fiberW/2))
#     
#     mpIm = Image.new("L",(int(imW/3),int(imH/3)), 0)
#     
#     for x in range(0, int(imW/3)):
#         for y in range(0, int(imH/3)):
#             if map[x][y] > avg:
#                 
#             mpIm.putpixel((x,y), map[x][y])
#     
#     mpIm.show()
    
if __name__ == "__main__":
    d1 = datetime.datetime.now()
#     im = BigImage("","rainbow.jpg",9)
#     im = BigImage('Images/smallTest.jpg')
#     main(10, "Images/","colorAdjustedSmallTest.tif")
#     test()
#     main(10, "Images/","smallTest.jpg")
    main(8, "Images/","smallTest2.jpg")
#     main(10, "Images", "midSizedTest.jpg")
#     main(10, 'Images/CarbonFiber/', 'GM_LCF_EGP_23wt%_Middle_FLD1(circleLess).tif')
    d2 = datetime.datetime.now()
    print('Running time:', d2-d1)

#     os.system('say "beep"')


    

# def old():
#     fiberWidth = 9
#     im = Image.new('RGB', (500,500), (0,0,0))
#     for i1 in range(0,500):
#         for i2 in range(0,500):
#             im.putpixel((i1,i2), (int(255*i1/500),int(255*i2/500),int(255*(1-i1/500)*(1-i2/500))))
# 
# #     im = Image.open("tempSmaller.jpg")
#     
#     pixels = im.load()
#     imW, imH = im.size
# #     getStats(pixels, imW, imH)
# #     print("outside")
# #     return
#     imarray = np.array(im)
#     
# #     print(pixels[30,30])
# #     
# #     print(imarray[30,30])
# #     
# #     return
#     
#     #this is the number of lines to seperate the strip into - must be ~ 3*fiberWidth
#     stripW = fiberWidth*3*5
#     # this accounts for slight luminosity deviations along the fiber, will need a bunch of pixels to do so.
#     # must be short enough to assume all fibers are straight inside the strip though.
#     # ~60 maybe
#     stripH = 60*5
#     
#     stripCenX = 500/2
#     stripCenY = 500/2
#     
#     C = getStrip((stripCenX, stripCenY), stripW, stripH, 1)
#     
# #     print(C)
# #     B = np.empty((stripW, stripH),object)
# #     t = 67 / 180 * pi
# #     for y in range(0,stripW):
# #         for x in range(0,stripH):
# #             B[y, x] = rotate(C[y, x], t, (stripCenX, stripCenY))
# 
#     sum1 = 0
#     
#     im1 = im.copy()
#     import datetime
# #     d1 = datetime.datetime.now()
# #     # using an existing matrix of coordinates and rotating each one as you come to it
# #     for angle in range(0, 359,4):
# # #         maxP = (0,0)
# # #         minP = (1000,0)
# #         
# #         for y in range(0,stripH,2):
# #             for x in range(0,stripW,2):
# #                 t = angle / 180 * pi
# #                 p = rotate(C[x, y], t, (stripCenX, stripCenY))
# #                 p = int(p[0]), int(p[1])
# #                 sum1 += pixels[p][0]
# # #                 im1.putpixel(p,tuple( np.array([255,255,255])-np.array(pixels[p][:])))
# # #                 if p[0] > maxP[0]:
# # #                     maxP = p
# # #                 elif p[0] < minP[0]:
# # #                     minP = p
# # #         print(maxP, minP)
# #     d2 = datetime.datetime.now()
# #     im1.show()
# #     print(d2-d1, sum1)
#     sum1 = 0
#     
#     d2 = datetime.datetime.now()
#     for angle in range(0, 359,4):
#         
#         #corners are numbered in clockwise order, with the corner opposite c1 skipped 
#         t = angle / 180 * pi
#         c1 = np.array(rotate( np.array([stripCenX-stripW/2, stripCenY-stripH/2]), t, (stripCenX, stripCenY)))
#         c2 = np.array(rotate( np.array([stripCenX+stripW/2, stripCenY-stripH/2]), t, (stripCenX, stripCenY)))
#         c3 = np.array(rotate( np.array([stripCenX-stripW/2, stripCenY+stripH/2]), t, (stripCenX, stripCenY)))
#         v12 = np.array([(c2[0]-c1[0]), (c2[1]-c1[1])])
#         v13 = np.array([(c3[0]-c1[0]), (c3[1]-c1[1])])
#         v12hat = np.array([(c2[0]-c1[0])/stripW, (c2[1]-c1[1])/stripW])
#         v13hat = np.array([(c3[0]-c1[0])/stripH, (c3[1]-c1[1])/stripH])
# #         print(c1,c2,c3)
# #         print(c1,c1+v12,c1+v13)
# #         print(c1,c1+stripW*v12hat,c1+stripH*v13hat)
#         
#         for y in range(0, stripH, 2):
#             for x in range(0, stripW, 2):
#                 p = c1 + x * v12hat + y*v13hat
#                 p = (int(p[0]), int(p[1]))
# #                 try:
#                 sum1 += pixels[p][0]
# #                     im.putpixel(p,tuple( np.array([255,255,255])-np.array(pixels[p][:])))
# #                 except:
# #                     ()
#     
#     d3 = datetime.datetime.now()
# #     im.show()
#     print(d3-d2, sum1)
#     
#     sum2 = 0
#     
#     d3 = datetime.datetime.now()
#     for angle in range(0, 359,4):
#         
#         #corners are numbered in clockwise order, with the corner opposite c1 skipped 
#         t = angle / 180 * pi
#         c1 = np.array(rotate( np.array([stripCenX-stripW/2, stripCenY-stripH/2]), t, (stripCenX, stripCenY)))
#         c2 = np.array(rotate( np.array([stripCenX+stripW/2, stripCenY-stripH/2]), t, (stripCenX, stripCenY)))
#         c3 = np.array(rotate( np.array([stripCenX-stripW/2, stripCenY+stripH/2]), t, (stripCenX, stripCenY)))
#         v12 = np.array([(c2[0]-c1[0]), (c2[1]-c1[1])])
#         v13 = np.array([(c3[0]-c1[0]), (c3[1]-c1[1])])
#         v12hat = np.array([(c2[0]-c1[0])/stripW, (c2[1]-c1[1])/stripW])
#         v13hat = np.array([(c3[0]-c1[0])/stripH, (c3[1]-c1[1])/stripH])
# #         print(c1,c2,c3)
# #         print(c1,c1+v12,c1+v13)
# #         print(c1,c1+stripW*v12hat,c1+stripH*v13hat)
#         
#         for y in range(0, stripH, 2):
#             for x in range(0, stripW, 2):
#                 p = c1 + x * v12hat + y*v13hat
#                 p = (int(p[0]), int(p[1]))
# #                 try:
#                 sum2 += imarray[p][0]
# #                     im.putpixel(p,tuple( np.array([255,255,255])-np.array(pixels[p][:])))
# #                 except:
# #                     ()
#     
#     d4 = datetime.datetime.now()
# #     im.show()
#     print(d4-d3, sum2)
# 
# #     im.putpixel((250,250), (255,255,255))
# #     im.show()
#     return
#     class Strip:
#         def __init__(self, w, h, x, y, t, spacing ):
#             c = np.empty((w, h),object)
#             for y in range(0,h):
#                 for x in range(0,w):
#                     c[x, y] = np.array([float(spacing*(x-(w-1)/2) + x),float(spacing*(y-(h-1)/2) + y)])
#             return c
#         
#             self._w = w
#             self._h = h
#             self._x = x # x_center
#             self._y = y # y_center
#             self._t = t # angle from x-axis
#             
#         def set(self, x, y, val):
#             self._l[y, x] = val
#         
#         def get(self, x, y):
#             return self._l[y, x]
#     
#     
#     def getStrip(center, w, h, spacing):
#         c = np.empty((w, h),object)
#     
#         for y in range(0,h):
#             for x in range(0,w):
#                 c[x, y] = np.array([float(spacing*(x-(w-1)/2) + center[0]),float(spacing*(y-(h-1)/2) + center[1])])
#         return c
