import numpy as np
import scipy.ndimage.morphology as smorph
from skimage.transform import hough_line
import cv2
import skimage.morphology as skm


################################################################################
#               RGB To Gray (also exists in skimage.color.rgb2gray
################################################################################

def rgb2gray(inIm): # also available in skimage

    outIm = 0.2126 * inIm[:,:,0]  + 0.7152 * inIm[:,:,1] + 0.0722 * inIm[:,:,2]
    return(outIm)

################################################################################
#               Gray To Binary With Threshold
################################################################################

def gray2bw(inData,thr=0.5):
    if (thr < 1):
        #normalize the user threshold
        thr = thr* np.max(np.max(inData))
    # normalize the data
    outData = np.array((inData > thr) * 255,dtype =np.uint8)

    return outData


################################################################################
#               BRESENHAM LINE ALGORITHM
################################################################################

def bresenham(start, end):
    # Setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1

    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)

    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1

    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1

    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx

    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return points

###################################################################################
#                   IMPLEMENTING MATLAB BWMORPH MISSING PIECES
###################################################################################

class bwmorph:
    @staticmethod
    def diag(imIn):
        strl = np.array([
        [[0,1,0],[1,0,0],[0,0,0]],
        [[0,1,0],[0,0,1],[0,0,0]],
        [[0,0,0],[1,0,0],[0,1,0]],
        [[0,0,0],[0,0,1],[0,1,0]],
        [[0,1,0],[1,0,0],[0,1,0]],
        [[0,1,0],[1,0,1],[0,0,0]],
        [[0,1,0],[0,0,1],[0,1,0]],
        [[0,0,0],[1,0,1],[0,1,0]]
        ],dtype=np.uint8)
        bwIm = np.zeros(imIn.shape,dtype=int)
        imIn = np.array(imIn)
        imIn = imIn/np.max(np.max(imIn)) #normalizing to be added later
        for i in range(7):
            bwIm = bwIm + smorph.binary_hit_or_miss(imIn,strl[i,:,:])

            bwIm = ((bwIm>0) + imIn)>0
        return bwIm # out put is boolean

    @staticmethod
    def clean(imIn):
        print("This is to be implemented in the future")
        return None
    @staticmethod
    def endpoints(imIn):
        print("This is to be implemented in the future")
        return None

    @staticmethod
    def hbreak(imIn):
        print("This is to be implemented in the future")
        return None
    @staticmethod
    def spur(imIn):
        print("This is to be implemented in the future")
        return None


###################################################################################
#                   IMPLEMENTING MATLAB BWMORPH MISSING PIECES
###################################################################################
# This requires a thinned image to be passed in
# The little openings between the connected lines can be fixed by changing the line width in
# "cv2.line(im, (x1, y1), (x2, y2), (0, 0, 0), thickness=2, lineType=8)" to "1"
# But that's only recommended if the lines are super straight.

def preciseHough(im,lineMinLen=10,lineMaxGap=5):
    # performing thinking on the image to create diagonal artifacts
    im = np.array(im * 255, dtype=np.uint8)
    labledLines = np.zeros(im.shape)
    numLines = 0
    strel = np.ones((4, 4))

    # Implementing Custome Hough Transform
    linesCenter = np.array([]).reshape(2, 0)
    lineEndCoords = np.array([]).reshape(0, 4)

    while True:  # Iterate untill all the walls are extracted
        hspace, _, _ = hough_line(im)  # Getting hspace to threshold and extract one line at a time
        lines = cv2.HoughLinesP(im, 1, np.pi / 180, np.max(np.max(hspace)), lineMinLen, lineMaxGap)
        if lines is None:
            break

        x = np.array([lines[:, 0, 0], lines[:, 0, 2]]) ;  y = np.array([lines[:, 0, 1], lines[:, 0, 3]])
        iterLineCent = np.array([np.mean(x, 0), np.mean(y, 0)])

        linesCenter = np.hstack([linesCenter, iterLineCent])
        lineEndCoords = np.vstack([lineEndCoords,lines[:,0,:]])

        counter = 0
        for x1, y1, x2, y2 in lines[:, 0, :]:
            cv2.line(im, (x1, y1), (x2, y2), (0, 0, 0), thickness=2, lineType=8)
            cv2.line(labledLines, (x1, y1), (x2, y2), (255 - (numLines+counter), 0, 0), thickness=1, lineType=8)
            counter = counter + 1
        # end of for

        im = skm.closing(im, strel)  # close image to fill holes
        numLines = numLines + lines.shape[0]

    else:  # When there are no other lines detected by the Hough transform
        pass
    # end of while
    return labledLines,linesCenter,lineEndCoords,numLines


################################################################################
#                         Heatmap Data to RGB IMAGE
################################################################################

def data2heatmap(data, dynamicRange = 'linear'):
    dataShape = data.shape

    # normalizing the data
    data = data.reshape((-1, 1))
    alpha = np.min(np.min(data))
    beta = np.max(np.max(data))
    gamma = beta - alpha
    data = data - alpha
    data = data / gamma


    # Intensity Transformation
    if dynamicRange.lower() == 'log':# be aware this manipulates the dynamic range
        # Constarst Streching
        for i in range(2):
            data = 1*np.log2(1+data)


    # Defining Colormap Transissions
    Rpx = np.array([0, .125, .38, .62, .88, 1])
    Gpx = Rpx
    Bpx = Rpx

    Rpy = np.array([0, 0, .00, 1, 1, .5])
    Gpy = np.array([0, 0, 1, 1, 0, 0])
    Bpy = np.array([.5, 1, 1, 0, 0, 0])

    RGBmap3D = np.zeros((1, data.size, 3))

    for i in range(data.size):
        if data[i] <= Rpx[1]:
            RGBmap3D[0, i, :] = [(np.diff(Rpy[0:2]) / np.diff(Rpx[0:2])) * (data[i] - Rpx[0]) + Rpy[0],
                                 (np.diff(Gpy[0:2]) / np.diff(Gpx[0:2])) * (data[i] - Gpx[0]) + Gpy[0],
                                 (np.diff(Bpy[0:2]) / np.diff(Bpx[0:2])) * (data[i] - Bpx[0]) + Bpy[0]]

        elif data[i] <= Rpx[2]:
            RGBmap3D[0, i, :] = [(np.diff(Rpy[1:3]) / np.diff(Rpx[1:3])) * (data[i] - Rpx[1]) + Rpy[1],
                                 (np.diff(Gpy[1:3]) / np.diff(Gpx[1:3])) * (data[i] - Gpx[1]) + Gpy[1],
                                 (np.diff(Bpy[1:3]) / np.diff(Bpx[1:3])) * (data[i] - Bpx[1]) + Bpy[1]]

        elif data[i] <= Rpx[3]:
            RGBmap3D[0, i, :] = [(np.diff(Rpy[2:4]) / np.diff(Rpx[2:4])) * (data[i] - Rpx[2]) + Rpy[2],
                                 (np.diff(Gpy[2:4]) / np.diff(Gpx[2:4])) * (data[i] - Gpx[2]) + Gpy[2],
                                 (np.diff(Bpy[2:4]) / np.diff(Bpx[2:4])) * (data[i] - Bpx[2]) + Bpy[2]]

        elif data[i] <= Rpx[4]:
            RGBmap3D[0, i, :] = [(np.diff(Rpy[3:5]) / np.diff(Rpx[3:5])) * (data[i] - Rpx[3]) + Rpy[3],
                                 (np.diff(Gpy[3:5]) / np.diff(Gpx[3:5])) * (data[i] - Gpx[3]) + Gpy[3],
                                 (np.diff(Bpy[3:5]) / np.diff(Bpx[3:5])) * (data[i] - Bpx[3]) + Bpy[3]]

        elif data[i] <= Rpx[5]:
            RGBmap3D[0, i, :] = [(np.diff(Rpy[4:6]) / np.diff(Rpx[4:6])) * (data[i] - Rpx[4]) + Rpy[4],
                                 (np.diff(Gpy[4:6]) / np.diff(Gpx[4:6])) * (data[i] - Gpx[4]) + Gpy[4],
                                 (np.diff(Bpy[4:6]) / np.diff(Bpx[4:6])) * (data[i] - Bpx[4]) + Bpy[4]]

    RGBmap = np.reshape(RGBmap3D, (dataShape[0], dataShape[1], 3))

    return RGBmap, RGBmap3D
