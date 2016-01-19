from PIL import Image, ImageDraw
import math

# what follows will display a graph of the amount of each type of color in the picture
def drawData(graph, colorCountList, divs, numPixels, col):
    dataSet = []
    if not isinstance( colorCountList[0], int ):
        for i in range(0, 256):
            graph.putpixel( (i, 30), (0,0,0))
            x = i
            y = 30 + int(256 * colorCountList[int(i/divs)][col] / float(numPixels))
    #         print( i, int(colorCountList[i/divs][col]), y)
            dataSet.append((x,y))
            if ( i > 0 ):
                draw = ImageDraw.Draw(graph)
                draw.line([(dataSet[i-1][0],dataSet[i-1][1]), (dataSet[i][0],dataSet[i][1])],
                           (255*int(col == 0),255*int(col == 1),255*int(col == 2)), 1)
    else:
        for i in range(0, 256):
            graph.putpixel( (i, 30), 0)
            x = i
            y = 30 + int(256 * colorCountList[int(i/divs)] / float(numPixels))
    #         print( i, int(colorCountList[i/divs][col]), y)
            dataSet.append((x,y))
            if ( i > 0 ):
                draw = ImageDraw.Draw(graph)
                draw.line([(dataSet[i-1][0],dataSet[i-1][1]), (dataSet[i][0],dataSet[i][1])], 255, 1)

def graphData(colorCountArray, divs, numPixels):
    if not isinstance( colorCountArray[0], int ):
        graph = Image.new("RGB", (256, 300), "black")
        for i in range(0, 256):
            for j in range(0, 10):
                graph.putpixel( (i, j), (i,0,0))
                graph.putpixel( (i, j+10), (0,i,0))
                graph.putpixel( (i, j+20), (0,0,i))
                    
            if (i % divs == 0) & False:
                for c in range(30, 300):
                    graph.putpixel( (i, c), (255, 255, 255))
        for i in range(0,3):
            drawData( graph, colorCountArray, divs, numPixels, i )
    else:
        graph = Image.new("L", (256, 300), "black")
        for i in range(0, 256):
            for j in range(0, 10):
                graph.putpixel( (i, j), i)
                    
            if (i % divs == 0) & False:
                for c in range(30, 300):
                    graph.putpixel( (i, c), 255)
        drawData( graph, colorCountArray, divs, numPixels, i )
    graph.show()

def countColors( pixels, width, height, divs):
    # if there are three colors, as opposed to being BW
    if not isinstance( pixels[0,0], int ):
        # initialize an array with all zeros
        colorCountArray = []
        for i in range(0, int(256/divs) + 1):
            colorCountArray.append([0,0,0])
             
        for i in range(0,3):
            #increment each piece of the array when a color fits into that group
            for j in range(0, width):
                for k in range(0, height):
                    colorCountArray[ ((pixels[j,k])[i]) / divs ][i] += 1
                    
#         graphData(colorCountArray, divs, width * height)
    else:
        # initialize an array with all zeros
        colorCountArray = []
        for i in range(0, int(256/divs) + 1):
            colorCountArray.append(0)
             
        #increment each piece of the array when a color fits into that group
        for j in range(0, width):
            for k in range(0, height):
                colorCountArray[ int(pixels[j,k] / divs) ] += 1
                    
#         graphData(colorCountArray, divs, width * height)
    return colorCountArray
 
def getStats( pixels, width, height ):
    divs = 3
    colorArray = countColors( pixels, width, height, divs)
    if not isinstance( pixels[0,0], int ):
        avg = [0,0,0]
        stdev = [0,0,0]
        for i in range(0,3):
            for j in range(0, 256/divs):
                avg[i] += colorArray[j][i] * j * divs
            avg[i] /= (width * height)
    #         print("Average: ", avg[i])
    #     avgCol = Image.new("RGB", (100, 100), (avg[0], avg[1], avg[2]))
    #     avgCol.show()
    
        print("average found.")
        
        for i in range(0,3):
            for j in range(0, int(256/divs)):
                for k in range(0,colorArray[j][i]):
                    stdev[i] += pow(j * divs - avg[i], 2)
            stdev[i] /= (width*height)
            stdev[i] = math.sqrt( stdev[i] )
    else:
        
        avg = 0
        stdev = 0
        for j in range(0, int(256/divs)):
            avg += colorArray[j] * j * divs
        avg /= (width * height)
        
        print("average found.")
        
        for j in range(0, int(256/divs)):
            for k in range(0,colorArray[j]):
                stdev += pow(j * divs - avg, 2)
        stdev /= (width*height)
        stdev = math.sqrt( stdev )
        
#     avgCol = Image.new("RGB", (100, 100), (avg[0], avg[1], avg[2]))
#     avgCol.show()
#     stddev1Col = Image.new("RGB", (100, 100), (int(avg[0] + stdev[0]), int(avg[1] + stdev[1]), int(avg[2] + stdev[2])))
#     stddev1Col.show()
#     stddev2Col = Image.new("RGB", (100, 100), (int(avg[0] - stdev[0]), int(avg[1] - stdev[1]), int(avg[2] - stdev[2])))
#     stddev2Col.show()
#     stddev1Col = Image.new("RGB", (100, 100), (int(avg[0] + 2*stdev[0]), int(avg[1] + 2*stdev[1]), int(avg[2] + 2*stdev[2])))
#     stddev1Col.show()
#     stddev2Col = Image.new("RGB", (100, 100), (int(avg[0] - 2*stdev[0]), int(avg[1] - 2*stdev[1]), int(avg[2] - 2*stdev[2])))
#     stddev2Col.show()
    print("stdevs found.")
      
    return(stdev, avg)


