# Image acquisition and processing of the multi-wall model
# Created by: Salaheddin Hosseinzadeh
# Created on: 01.06.2018
# Last revision:
# Notes:
#####################################################################################
from tkinter import *
from tkinter import filedialog, messagebox, simpledialog
import cv2
import matplotlib.pyplot as plt
import numpy as np
import rgb2gray


def acquire(*args):
    filename = 'D:\\str.png'
    img = plt.imread(filename) # reads image in range of 0 to 1

    if img.shape[2] > 1: # if image is not grayscale
        # Converting to grayscale if it is not already
        img = rgb2gray.rgb2gray(img)

    # Thresholding it to binary just in case it is not already
    bwImage = np.array((img > (img.max()/2)), dtype=bool)
    # bwImage = cv2.cvtColor(bwImage,cv2.COLOR_BRG2GRAY) # figure is loded with mpl so it's RGB nor BRG

    # Make sure structures are in black (flase) and background is white (true)
    if (np.sum(bwImage) > np.sum(~bwImage)):
        bwImage = ~bwImage
        print('Image complemented')

    # Dilating the image to make it easier ot select the wall interactively
    # kernelSize = int(np.amin(bwImage.shape)/50)
    kernelSize = 8 # fixed size kernel
    kernel = np.ones((kernelSize, kernelSize), np.uint8)
    bwImDil = cv2.dilate(np.array(bwImage, dtype=np.uint8), kernel, iterations=1)

    return (bwImage, bwImDil)