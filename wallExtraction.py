# Now need to use the Hough transform to find the walls and extract them
# To assure a high accuracy this is done ones for each line (wall) and then the extracted line (wall) is removed
import numpy as np
from skimage.morphology import thin
import skimage.color as skc
from skimage.transform import hough_line
import salimage as sl

##########################################################################
#                         INITIALIZING THE CODE
##########################################################################
def wallExtraction(img):

    if len(img.shape) > 2: # if image is not grayscale
        # Converting to grayscale if it is not already
        img = skc.rgb2gray(img)

    # Thresholding it to binary just in case it is not already
    bwImage = np.array((img > (img.max()/2)), dtype=bool)
    #bwImage = np.array(bwImage*255,dtype=np.uint8) # converting to a uint8 image
    # bwImage = cv2.cvtColor(bwImage,cv2.COLOR_BRG2GRAY) # figure is loded with mpl so it's RGB nor BRG

    # Make sure structures are in black (flase) and background is white (true)
    if (np.sum(bwImage) > np.sum(~bwImage)):
        bwImage = ~bwImage
        print('Image complemented')

    ############################################################################
    ###                 END OF INITIALIZATION
    ############################################################################

    # Pre processing the image to make it nice and clean for the jon :)
    bwImage = thin(bwImage) # converting to int after thining
    bwImage = sl.bwmorph().diag(bwImage)
    bwImage = np.array(bwImage*255,dtype=np.uint8)

    ############################################################################
    (labledImage,wallsCenter,_,numWalls) = sl.preciseHough(bwImage)
    wallLables = np.unique(labledImage) # each lable is an individual wall
    wallLables = np.delete(wallLables,0) # removing the first element as it's for the background

    ###########################################################################


    return labledImage,wallLables,wallsCenter