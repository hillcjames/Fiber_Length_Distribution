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

    
'''
from PIL import Image, ImageDraw
from colorSpread import getStats
import numpy as np
from scipy import dot, array
from math import cos, sin, pi, sqrt, ceil, atan
import datetime
import os


class BigImage:
    def __init__(self, imDir, imName, numTiles):
        self.num_columns = int(ceil(sqrt(numTiles)))
        self.num_rows = int(ceil(numTiles / float(self.num_columns)))
        
        index = 0
        while imName[len(imName) - 1 - index] != '.':
            index += 1
        index += 1
        
        # create a new folder with that image's name, minus the extension
        folder = imName[:len(imName)-index]
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
                image_slicer.slice(imDir + imName, numTiles)
            except Exception as e:
                print(e)
            print("Finished slicing.")
            
            for j in range(0, self.num_rows):
                for i in range(0,self.num_columns):
                    # folder is the name of the image minus its extension
                    sliceName = folder + "_0"+str(j+1) + "_0"+str(i+1) + ".png"
                    newName = os.path.join(imDir, folder, sliceName)
                    os.rename(os.path.join(imDir, sliceName), newName)
                
        self.names = []
        self.imgs = []
        for j in range(0,self.num_rows):
            for i in range(0,self.num_columns):
                sliceName = folder + "_0"+str(j+1) + "_0"+str(i+1) + ".png"
                newName = os.path.join(imDir, folder, sliceName)
                self.names.append(newName)
                self.imgs.append(Image.open(newName))
                self.imgs[len(self.imgs)-1].convert("RGB")
        
        self.pxls = []
        for x in range(0, len(self.imgs)):
            self.pxls.append(self.imgs[x].load())
        self.sliceW, self.sliceH = self.imgs[0].size
        
    def pixels(self, p):
        # if the slices are arranged on a grid, the x-coord on that grid
        x = int(p[0] / self.sliceW)
        # if the slices are arranged on a grid, the y-coord on that grid
        y = int(p[1] / self.sliceH)
        
        p0 = (p[0] % self.sliceW, p[1] % self.sliceH)
        
        return self.pxls[ x + y*self.num_columns ][p0]
    
    def getStats(self):
        # this, for speed and simplicity's sake, just returns the stats for the center tile.
        return getStats(self.pxls[int((len(self.imgs)-1)/2)], self.sliceW, self.sliceH)
    
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
          

def sqrDist( p1, p2 ):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

class Fiber:
    def __init__(self, points, fiberW):
        self.pnts = points
        self.w = fiberW
        self.calcLength()
        
    def calcLength(self):
        self.length = 0
        for i in range(0, len(self.pnts)-1):
            self.length += sqrt(sqrDist(self.pnts[i], self.pnts[i+1]))
            
    
    def draw(self, im, offset, c):
        draw = ImageDraw.Draw(im)
        for i in range(0, len(self.pnts)-1):
            p1 = (self.pnts[i][0] - offset[0], self.pnts[i][1] - offset[1])
            p2 = (self.pnts[i+1][0] - offset[0], self.pnts[i+1][1] - offset[1])
            draw.ellipse([(p1[0]-self.w/2+1, p1[1]-self.w/2+1),(p1[0]+self.w/2-1, p1[1]+self.w/2-1)], fill = c)
#             draw.ellipse([(p1[0]-offset[0]-self.w/2+1, p1[1]-offset[1]-self.w/2+1),(p1[0]-offset[0]+self.w/2-1, p1[1]-offset[1]+self.w/2-1)], fill = c)
            draw.line([p1,p2], width = 10, fill = c)
        p0 = (self.pnts[len(self.pnts)-1][0] - offset[0], self.pnts[len(self.pnts)-1][1] - offset[1])
        draw.ellipse([(p0[0]-self.w/2+1, p0[1]-self.w/2+1),(p0[0]+self.w/2-1, p0[1]+self.w/2-1)], fill = c)
#tests drawFiber
# im = Image.new('L', (300,300), 0)
# l = [(30,100),(50,170),(70,180),(100,220)]
# f = Fiber(l, 10)
# f.drawFiber(im)
# im.show()
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
                    print(p, p0)
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
    stripW = 2 * fiberW
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
            val = (im.pixels(pInt) - avg)
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

def traceFiber(im, fiberW, initialP, avg, stdev):
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
        t, lightness = getBestStartAngle(im, initialP, int(fiberW/4), 4*fiberW, 1)
    except IndexError:
        return 0
#     t+=pi
    atEnd = False
    firstDirectionCompleted = False
    curP = initialP
    while not atEnd:
#         print(curP)
        nextP = (curP[0]+int(fiberW*jumpDist*cos(t)),curP[1]+int(fiberW*jumpDist*sin(t)))
        try:
            adjustedP = getStripCentroid(im, t, nextP, fiberW, 1, avg)
        except IndexError:
            adjustedP = nextP
            
        try:
            newT = atan((adjustedP[1]-curP[1])/(adjustedP[0]-curP[0]))
        except ZeroDivisionError:
#             print(curP, nextP, adjustedP)
            newT = pi/2+0.001
#             raise
#         print(t, newT)
        while newT - t > pi:
            newT -= pi
        while newT - t < -pi:
            newT += pi
        t = newT
#         print("**",t, newT)

        if getAvgSqrWhiteness(im, adjustedP, int(fiberW/2)) > avg + stdev/2:
            fiberPoints.append(adjustedP)
            curP = adjustedP
#             if len(fiberPoints) > 7:
#                 print("break", fiberPoints)
#                 break
        else:
            atEnd = True
#             print(adjustedP, getAvgSqrWhiteness(im, adjustedP, int(fiberW/2)))
#             try:
#                 im.putpixel(adjustedP, 180)
#             except Exception:
#                 ()
            
        if atEnd and not firstDirectionCompleted:
#             print("In second thing")
            atEnd = False
            firstDirectionCompleted = True
            fiberPoints.reverse()
            curP = initialP
            t -= pi
    
#     for p in fiberPoints:
#         im.putpixel(p, 255)
#     im.imgs[0].show()
#     print(len(fiberPoints))
    if len(fiberPoints) < 3:
        return 0
    fiber = Fiber(fiberPoints, fiberW)
    im.drawFiber(fiber, avg-stdev/2)
    return fiber 

def checkPoint(im, p, r, avg, stdev):
    x,y = p[:]
    sidesBiggerThan = 0
    # getAvgSqrWhiteness(im, (x,y), r) takes about 5/6 of the time that avgDot takes, but consistently (as expected) reports lower values
    # for light regions than avgDot does, and identical values for dark regions. So it kinda skews the graph a little, and would
    # require a lower tolerance of color variation among the fibers.
    # I'll wait and see if the speed is needed.
    currentAvg = getAvgDotWhiteness(im, p, r)
    if currentAvg < avg + stdev/2:
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


def printReport(fiberList):
    print(len(fiberList), "fibers were found.")
    
    total = 0
    for i in range(0, len(fiberList)):
        total += fiberList[i].length
    avg = total/len(fiberList)
    
    print("Total length of fibers:", total)
    print("Average length of a fiber:", avg)
    
    sum = 0
    for i in range(0, len(fiberList)):
        dif = (fiberList[i].length - avg)
        sum += dif*dif
    stdev = sqrt(sum/len(fiberList))
    
    print("Standard deviation of fiber length:", stdev)

def main(fiberW, imDir, imName):
    
#     im = Image.new('RGB', (500,500), (0,0,0))
#     for i1 in range(0,500):
#         for i2 in range(0,500):
#             im.putpixel((i1,i2), (int(255*i1/500),int(255*i2/500),int(255*(1-i1/500)*(1-i2/500))))
#     im = Image.open("testImgs/singleStraightGlass2.jpg") # has avg of 16 and stdev of 6
#     im = Image.open("tempSmaller.jpg") # has avg of 16.5 and stdev of 7
#     pixels = im.load()
    im = BigImage(imDir, imName, 2)
    imW, imH = im.size()
    print("Image size:", imW, "x", imH)
    
    stdev, avg = im.getStats()
    print("Average and Stdev:",avg,stdev)
    
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
    
    skipSize = 10
#     p = (467,170)
#     print(getAvgDotWhiteness(im, p, 10))
#     im.putpixel(p, 255)
#     im.imgs[0].show()
#     return

    start = int(fiberW*1.5)
    stop = imW - int(fiberW*1.5)
#     l1 = []
#     l2 = []
    for x in range(start, stop, skipSize):
        for y in range(int(fiberW*1.5), imH - int(fiberW*1.5), skipSize):
            p = (x,y)
#             if getAvgSqrWhiteness(im, p, int(fiberW/2)) > avg + stdev/2:
#                 l1.append(p)
#             if checkPoint(im, p, int(fiberW/2), avg, stdev)[0]:
#                 l2.append(p)

#             pAvg = getAvgSqrWhiteness(im, p, int(fiberW/2))
#             pointIsValid = pAvg > avg + stdev/2
            
            pointIsValid, pAvg = checkPoint(im, p, int(fiberW/2), avg, stdev)
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
                fiber = traceFiber(im, fiberW, p, avg, stdev)
#                 return
                if fiber != 0:
                    fiberList.append(fiber)
#                     if len(fiberList) > 1:
#                         im.imgs[0].show()
#                         return
        print("{}% completed".format(int(1000*x/(stop-start))/10))
    
    printReport(fiberList)
    for i in range(0, len(fiberList)):
        im.drawFiber(fiberList[i], 20+int(i/len(fiberList)*215))
        print(fiberList[i].length)
#     im.drawFiber(fiberList[5], 200)
    im.imgs[0].show()
    im.imgs[1].show()
    # http://stackoverflow.com/questions/403421
#     fiberList.sort()
    

if __name__ == "__main__":
    d1 = datetime.datetime.now()
#     im = BigImage("","rainbow.jpg",9)
#     im = BigImage('Images/smallTest.jpg')
    main(10, "Images/","colorAdjustedSmallTest.tif")
#     main(10, "Images/","smallTest.jpg")
#     main(10, 'Images/CarbonFiber/', 'GM_LCF_EGP_23wt%_Middle_FLD1(circleLess).tif')
    d2 = datetime.datetime.now()
    print('Running time:', d2-d1)


# 
# # im = BigImage('Images/CarbonFiber/', 'GM_LCF_EGP_23wt%_Middle_FLD1.tif', 9)
# im = Image.open('Images/CarbonFiber/GM_LCF_EGP_23wt%_Middle_FLD1/GM_LCF_EGP_23wt%_Middle_FLD1_02_02.png')
# im = im.convert('RGB')
# im0 = im.copy()
# w,h = im.size
# pixels = im.load()
# # print(w,h)
# # print(int(10*w/21), int(11*w/21))
# # print(1/0)
# r = 8
# print("Done loading")
# # for x in range(r+int(10*w/21), int(11*w/21)-r):
# #     for y in range(r+int(10*h/21), int(11*h/21)-r):
# for x in range(2*r,w-2*r,3):
#     for y in range(2*r,h-2*r,3):
#         sidesBiggerThan = 0
#         avg = getNearbyAvg(pixels, (x,y), r)
#         if avg < 30:
#             continue
#         for i in range(-1,2):
#             for j in range(-1,2):
#                 if getNearbyAvg(pixels, (x+r*i,y+r*j), r) < avg:
#                     sidesBiggerThan += 1
#         if sidesBiggerThan >= 5:
#             im0.putpixel((x,y), (0,255,0))
#             
#             
# #             im0.show()
# #             print(x,y, w,h)
# #             print(1/0)
# #     if x%3 == 0:
# #         print((x-r-int(10*w/21))/(int(11*w/21)-r-r-int(10*w/21)))
#     print(x/w)
# 
# # for f in range(0, 9):
# im0.show()



#     Image.getdata()
#     im = Image.open('Images/CarbonFiber/GM_LCF_EGP_23wt%_Middle_FLD1.tif')
#     import datetime
#     d1 = datetime.datetime.now()
#     main(9, "imName")
#     d2 = datetime.datetime.now()
#     print(d2-d1)
    
#     sum2 = 0
#     
#     w = 30
#     h = 90
#     Cx = 250
#     Cy = 250
#     im = Image.new('RGB', (500,500), (0,0,0))
#     for i1 in range(0,500):
#         for i2 in range(0,500):
#             im.putpixel((i1,i2), (int(255*i1/500),int(255*i2/500),int(255*(1-i1/500)*(1-i2/500))))
#     pixels = im.load()
#     
#     d3 = datetime.datetime.now()
#     
#     for angle in range(0, 359):
#         t = angle / 180 * pi
#         c1 = np.array(rotate( np.array([Cx-w/2, Cy-h/2]), t, (Cx, Cy)))
#         c2 = np.array(rotate( np.array([Cx+w/2, Cy-h/2]), t, (Cx, Cy)))
#         c3 = np.array(rotate( np.array([Cx-w/2, Cy+h/2]), t, (Cx, Cy)))
#         v12 = np.array([(c2[0]-c1[0]), (c2[1]-c1[1])])
#         v13 = np.array([(c3[0]-c1[0]), (c3[1]-c1[1])])
#         v12hat = np.array([(c2[0]-c1[0])/w, (c2[1]-c1[1])/w])
#         v13hat = np.array([(c3[0]-c1[0])/h, (c3[1]-c1[1])/h])
#         
#         for y in range(0, h, 2):
#             for x in range(0, w, 2):
#                 p = c1 + x * v12hat + y*v13hat
#                 p = (int(p[0]), int(p[1]))
#                 for c in range(0,3):
#                     sum2 += pixels[p][c]
#     
#     d4 = datetime.datetime.now()
# #     im.show()
#     print(d4-d3, sum2)

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
