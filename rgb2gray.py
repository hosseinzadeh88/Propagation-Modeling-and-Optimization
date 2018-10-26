import matplotlib as plt


def rgb2gray(inIm):

    outIm = 0.2126 * inIm[:,:,0]  + 0.7152 * inIm[:,:,1] + 0.0722 * inIm[:,:,2]
    return(outIm)
