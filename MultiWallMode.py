# Created by: Salaheddin Hosseinzadeh
# Created on: 20.05.2018
# Completed on:
# Last revision:
# Notes:  Need to check if it works for multiple a
#####################################################################################

from tkinter import *
from tkinter import filedialog, messagebox, simpledialog
import cv2
import matplotlib.pyplot as plt
import numpy as np
import rgb2gray
from skimage.morphology import thin
import skimage.color as skc
import salimage as sl
import scipy
from salmodule import NRMSE, RMSE
# from wallExtraction import wallExtraction

class multiWallModel:
    ####################################################################################
    #                              INITIALIZATION
    ####################################################################################
    lightVel = 3e8   # light velocity
    nodePerMeter = 0.5 # This identifies the resolution of the estimation
    pathLossExp = [1.55]  # path loss exponent of propagation ! Keep is as a list for FSM sake! :)
    shadowFadingStd = [0] # Standard deviation of log-distance shadow fading
    txPower = np.array([14]) # Power in dB or dBm
    propagationModel = "MW" #['FSPL' 'MW'] FSPL : free space path loss. MW: multi-wall model
    propFreq =  800e6 # Propagation frequency in Hertz
    d0 = 1           # reference distance of 1 meters

    ####### ASSIGNING WALL ATTENUATIONS MAUALLY (if required) #############
    wallsAtten = np.ones((1,256))*5 # 5 dB attenuation for each wall
    wallsAtten[0, 0] = 0  # do not change this (this for a case of clear LoS
    #######################################################################
    # Define any specific wall attenuations here
    wallsAtten[0,255-8] = 10 # 10 dB attenuation for wall #8

    ###### PROVIDING IMPIRICAL MEASUREMENTS  (if exists)  #################
    optimization = "OFF" # if set to 1 then walls attenuation will be derived through optimization
    TxSuperposition = "CW" # ['CW'& 'Ind']Continuous Waveform (results in standing wave), Independent
    # CW: adds multiple TX waves at the point of probing resulting in destructive/constructive impact of multiple TXs
    # Ind: Ignores the weaker RSS
    # measurements MUST follow this structure: RSS is the Received Signal Strength
    measuredRSS = [np.array([[12, 39, -75.7]])],  # Freq = 800e6 Exp = 1.55 / 60 M antenna at Corner


    ####################################################################################
    #                         END OF INITIALIZATION
    ####################################################################################


    ####################################################################################
    #                       Critical Input Sanity Check

    def __init__(self):
        if self.optimization.lower()=='on':
            if len(self.txPower) != len(self.measuredRSS):
                if len(self.txPower) < len(self.measuredRSS):
                    messagebox.showwarning("Error", "'txPower' and 'measuredRSS' dimensions are not consistent!"
                                                    "\nSome of measurements will be removed to make up for it.")
                    del (self.measuredRSS[len(self.txPower):])
                else:
                    messagebox.showwarning("Error", "'txPower' and 'measuredRSS' dimensions are not consistent!\n"
                                                    "This will be terminated")
                    assert False

        elif self.optimization.lower() == 'off':
            if np.sum(self.wallsAtten) < 3:
                messagebox.showwarning("Warning","Please make sure wall attenuations ('wallAtten') are defined correctly.")




    def acquireCSV(self):
        root = Tk()
        fileName = filedialog.askopenfilename(title="Select a CSV file")
        root.destroy()

        boundaryMargin = int(30)  # x margin and y margin in pixels

        # fileName = "structure.txt"
        # filePath = "D:\\"
        # The file delimiter is (,) "comma"
        # the file format is x1,y1,x2,y2,attenuation in dB

        fid = open(fileName, 'r')
        coords = fid.readlines()
        fid.close()
        x1 = np.zeros((len(coords), 1))
        y1 = x1.copy(); x2 = x1.copy(); y2 = y1.copy(); attenuation = x1.copy()
        attenuationMap = np.zeros((1, 256))
        try:
            for i in range(len(coords)):
                x1[i, 0], y1[i, 0], x2[i, 0], y2[i, 0], attenuation[i, 0] = coords[i].split(',')
                # Forming a picture with the walls being labled with different intensity levels
        except:
            messagebox.showerror("Error", "Incorrect CSV format, check the example please!")
            assert False

        # mapping the structure between [0, max(input)]

        xMax = np.max([np.max(np.max(x1)), np.max(np.max(x2))])
        xMin = np.min([np.min(np.min(x1)), np.min(np.min(x2))])

        yMax = np.max([np.max(np.max(y1)), np.max(np.max(y2))])
        yMin = np.min([np.min(np.min(y1)), np.min(np.min(y2))])

        i = 1
        intCount = 0
        while (np.max([xMax, yMax]) / i) >= 1:
            i = i * 10
            intCount = intCount + 1

        # putting the structure lengths into reasonable image dimensions
        x1 = np.array((x1 - xMin) * 10 ** (3 - intCount) + boundaryMargin, dtype=np.int64)
        x2 = np.array((x2 - xMin) * 10 ** (3 - intCount) + boundaryMargin, dtype=np.int64)

        y1 = np.array((y1 - yMin) * 10 ** (3 - intCount) + boundaryMargin, dtype=np.int64)
        y2 = np.array((y2 - yMin) * 10 ** (3 - intCount) + boundaryMargin, dtype=np.int64)

        # forming the labled image

        xMax = np.max([np.max(np.max(x1)), np.max(np.max(x2))])
        xMin = np.min([np.min(np.min(x1)), np.min(np.min(x2))])

        yMax = np.max([np.max(np.max(y1)), np.max(np.max(y2))])
        yMin = np.min([np.min(np.min(y1)), np.min(np.min(y2))])

        # Finding the centre of each wall
        wallCentres = np.zeros((2,x1.size))  # pre allocating the wallCentres
        for i in range(x1.size):
            wallCentres[0,i] = (x1[i,0] + x2[i,0])/2
            wallCentres[1,i] = (y1[i,0] + y2[i,0])/2

        labledImage = np.zeros((yMax + boundaryMargin, xMax + boundaryMargin), dtype=np.uint8)

        for i in range(x1.shape[0]):
            line = sl.bresenham((x1[i, 0], y1[i, 0]), (x2[i, 0], y2[i, 0]))
            line = np.asarray(line, dtype=np.int64)
            labledImage[line[:, 1], line[:, 0]] = (255 - i)
            attenuationMap[0, 255 - i] = attenuation[i]

        for i in range(x1.shape[0]):
            print("Wall #{0}: (x1={1}, y1={2}) , (x2={3}, y2={4}), Attenuation = {5}".\
                  format(i + 1, x1[i, 0], y1[i, 0],x2[i, 0], y2[i, 0], attenuation[i, 0]))


        bwImage = sl.gray2bw(labledImage,2)

        plt.imshow(bwImage, "gray")
        plt.title("Structur acquired from CSV")
        plt.show()

        # Dilating the image to make it easier ot select the wall interactively
        # kernelSize = int(np.amin(bwImage.shape)/50)
        kernelSize = 4 # fixed size kernel
        kernel = np.ones((kernelSize, kernelSize), np.uint8)
        bwImDil = cv2.dilate(np.array(bwImage, dtype=np.uint8), kernel, iterations=1)


        return bwImage, bwImDil, labledImage,attenuationMap,wallCentres

    ########################  IMAGE ACQUISITION  ########################################
    #####################################################################################
    def acquireImage(self):

        root = Tk()
        fileName = filedialog.askopenfilename(title="Select an image file")
        root.destroy()

        #filename = 'D:\\str.png'
        img = plt.imread(fileName) # reads image in range of 0 to 1

        try:
            if img.shape[2] > 1: # if image is not grayscale
                # Converting to grayscale if it is not already
                img = rgb2gray.rgb2gray(img)
        except:
            pass


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
    #####################################################################################
    ########################  IMAGE ACQUISITION  ########################################

    ########################    CALIBRATION      ########################################
    #####################################################################################
    def calibration(self,bwImDil):
        # Opening a dialog box asking the user to use a reference wall and enter its length
        root = Tk()
        messagebox.showinfo("Instructions!", """Please select a reference wall on the image by clicking on the two ends of it.
         \nLater you wil be asked to enter its corresponding length.""")

        root.destroy()
        fig = plt.figure(1)
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(bwImDil, 'gray')
        # Check if selected points are valid
        try:
            while True:
                calibWallPoints = np.array(np.floor(fig.ginput(2)), dtype=int)
                if bwImDil[calibWallPoints[0, 1], calibWallPoints[0, 0]] & bwImDil[
                    calibWallPoints[1][1], calibWallPoints[1][0]]:
                    # plotting the line on the image and exit
                    ax.plot((calibWallPoints[:, 0]), (calibWallPoints[:, 1]), color='red', linewidth=2)
                    plt.draw()
                    break
                else:
                    root = Tk()
                    messagebox.showinfo("Wrong selection!", """Error in selecting the wall.\nPlease select a reference wall on the image by clicking on the two ends of it.
                    \nLater you wil be asked to enter its corresponding length.""")
                    print('wrong points selected, try again')
                    root.destroy()
        except:  # This isn't working atm, I'll work on this later
            print('program terminated')
            raise

        # Getting wall length from user and finding the calibration factor

        wallLenMet = float(simpledialog.askstring("User Input", "Please enter the selected wall length in meters: "))
        wallLenPix = np.sqrt(np.diff(calibWallPoints[:, 0]) ** 2 + np.diff(calibWallPoints[:, 1]) ** 2)
        calUnit = wallLenMet / wallLenPix  # meter per pixel is the unit
    # Getting TXs Location
        TxNum = int(
            simpledialog.askstring("User Input", "Please enter the total number of transmitters in the environment: "))
        if TxNum < 1:
            TxNum = 1

        TxLocation = np.zeros((TxNum,2), dtype=np.int64)
        for i in range(0, TxNum):  # Prompt to get TX locations interactively
            TxLocation[i,0:2] = np.array(plt.ginput(1)).astype(int)
            ax.plot(TxLocation[0, 0], TxLocation[0, 1], color='red', marker='*', markersize='8')
        # Acquiring TX numbers from User
        ax.cla()
        plt.close(fig)  # closing the figure upon successful point registration

        numNodesX = bwImDil.shape[1] * calUnit * self.nodePerMeter
        numNodesY = bwImDil.shape[0] * calUnit * self.nodePerMeter

        nodesX = np.linspace(0, bwImDil.shape[1] - 1, numNodesX, dtype=np.int64)
        nodesY = np.linspace(0, bwImDil.shape[0] - 1, numNodesY, dtype=np.int64)
        gridX, gridY = np.meshgrid(nodesX, nodesY)  # Gridding the environment
        gridXCul = np.reshape(gridX, (-1, 1))
        gridYCul = np.reshape(gridY, (-1, 1))

        return calUnit, TxNum, TxLocation,gridX,gridY,gridXCul,gridYCul
    #####################################################################################
    ########################    CALIBRATION      ########################################

    ########################    Line of Sight    ########################################
    #####################################################################################
    def lineOfSight(self,TxNum,TxLocation,gridX,gridY):
        LoS = [None] * TxNum
        for i in range(TxNum):
            LoS[i] = np.reshape(np.sqrt((gridX - TxLocation[i, 0]) ** 2 + (gridY - TxLocation[i, 1]) ** 2), (-1, 1))
            # LoS[i] contains the distnace from Tx[i] to all the grid points (nodes)
        return LoS
    #####################################################################################
    ########################    Line of Sight    ########################################



    ############################    WALL EXTRACTION   ###################################
    #####################################################################################
    def wallExtraction(self,img):

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

        # Pre processing the image to make it nice and clean for the jon :)
        bwImage = thin(bwImage) # converting to int after thining
        bwImage = sl.bwmorph().diag(bwImage)
        bwImage = np.array(bwImage*255,dtype=np.uint8)

        ############################################################################
        (labledImage,wallsCenter,_,numWalls) = sl.preciseHough(bwImage)
        labledImage = np.asarray(labledImage,dtype=np.int64)
        wallsCenter = np.asarray(wallsCenter,dtype=np.int64)

        wallLables = np.unique(labledImage) # each lable is an individual wall
        wallLables = np.delete(wallLables,0) # removing the first element as it's for the background

        ###########################################################################


        return labledImage,wallLables,wallsCenter
    #####################################################################################
    ############################    WALL EXTRACTION   ###################################


    ##########################  FSPL PROPAGATION MODEL   ################################
    #####################################################################################


###############################     Finds Walls on Line of Sight #################################
    #####################################################################################
    def wallsOnLineOfSight(self,TxNum,TxLocation,gridXCul,gridYCul,labledImage):
        LoSImage = np.zeros(labledImage.shape, dtype=np.int8)
        wallsOnLoS = [0] * TxNum
        for i in range(TxNum):
            temp = []
            attEqTemp = []
            for j in range(gridXCul.size):
                LoSLineXY = sl.bresenham(TxLocation[i, :], (gridXCul[j, 0], gridYCul[j, 0]))
                LoSLineXY = np.asarray(LoSLineXY, dtype=np.int64)
                LoSImage = LoSImage * 0

                # print(j, " , ", elapsed)
                # for k in range(len(LoSLineXY)):
                LoSImage[LoSLineXY[:, 1], LoSLineXY[:, 0]] = 1  # Crearint an image of the LoS
                # End of for
                # Intersecting the LoS line image with the labled image of the walls
                temp.append(np.unique((LoSImage * labledImage)))  # temporary holding the walls in between
            # End of for
            wallsOnLoS[i] = temp
        # End of For
        return wallsOnLoS

    #####################################################################################
    #####################################################################################

    ######################    FSPL Propagation Model (log-distance) #####################
    #####################################################################################
    def fsplModel(self,lossExp,shadowFadingStd,txPower,LoS,measurements=None): # pass measurements as 0 if don't exist
        #  free space path loss is actually the log-distance model with shadowing (This is more practical)
        LoS[np.where(LoS<self.d0)] = self.d0  ## distances under 1 m are not acceptable (log10(<1) problem)
        shadowFading = np.random.normal(0, shadowFadingStd, LoS.shape)

        if (measurements is None) and self.optimization.lower() == 'on':
                raise SystemError # optimization defo requires measurements
        # end of if

        delaySpr = LoS / self.lightVel  # delay spread calculation

        if self.optimization.lower() == 'on': # finds both path loss exponent x[0], and shadow fading x[1] through optimization
            # Forming the optimization problem
            optimEq = ['a'] * LoS.shape[0]
            for i in range(LoS.shape[0]):
                optimEq[i] = "%f - (20*np.log10(%f) + 20*np.log10(%f) - 147.55) - 10 * x[0] * np.log10(%f) - (%f)" % (
                txPower, self.propFreq, self.d0, LoS[i, 0], measurements[i])
            def optimProblem(x):
                f = [None] * LoS.shape[0]
                for i in range(LoS.shape[0]):
                    f[i]=(eval(optimEq[i]))
                # print(f)
                return(f)

            # print(optimEq)
            # Executing the optimization
            sol = scipy.optimize.root(optimProblem, [1], method='lm', tol=1e-10, jac=False)

            # Finding Estimation Performance
            RSS = txPower - (20 * np.log10(self.propFreq) + 20 * np.log10(self.d0) - 147.55) \
                  - 10 * sol.x * np.log10(LoS)
            rmse = RMSE(measurements,RSS)
            nrmse = NRMSE(measurements,RSS)
            shadowFading = measurements-RSS

        else:
            RSS = txPower - (20*np.log10(self.propFreq) + 20*np.log10(self.d0) - 147.55) \
                  - 10 * lossExp * np.log10(LoS) + shadowFading

        if self.optimization.lower() == 'on':
            return RSS, delaySpr,sol.x,shadowFading,(rmse,nrmse)
        else:
            return RSS, delaySpr, lossExp,shadowFading, (None,None)

    #####################################################################################
    ##########################  FSPL PROPAGATION MODEL   ################################


    ##########################  Multiwall PROPAGATION MODEL   ###########################
    #####################################################################################
    def mwModel(self,lossExp,txPower,LoS,walls,wallsAtten,measurements=None):

        LoS[np.where(LoS<self.d0)] = self.d0  ## distances under 1 m are not acceptable (log10(<1) problem)

        assert ~((measurements is None) and self.optimization.lower() == 'on')
        # end of if

        delaySpr = LoS / self.lightVel  # delay spread calculation

        if self.optimization.lower() == 'on': # find
            # finding the wall involved in measurements
            walls2optim = np.array([],dtype=np.int64)
            for i in range(len(walls)):
                walls2optim = np.append(walls2optim,np.asarray(walls[i]).reshape((-1,1)))
            # end
            walls2optim = np.unique(walls2optim)
            walls2optim = np.sort(walls2optim)


            # if (walls2optim.size > 0) and (walls2optim[0] == 0):  # this may always not happen in case there is clear LoS
            #     walls2optim = np.delete(walls2optim,[0])
            #


            # Forming the optimization problem
            optimEq = ['a'] * LoS.shape[0]
            for i in range(LoS.shape[0]):
                optimEq[i] = "%f - (20*np.log10(%0.1f) + 20*np.log10(%0.1f) - 147.55) - 10 * x[0] * np.log10(%0.3f) - (%0.2f)" \
                             %(txPower, self.propFreq, self.d0, LoS[i, 0], measurements[i])
                if walls[i].size > 1:
                    wallLosses = ""
                    for j in range(1,walls[i].size):
                        wallLosses = wallLosses + "+ x[%d]" %np.where(walls2optim == walls[i][j])
                    # end
                # end

                    optimEq[i] = optimEq[i] + " - (%s)" %wallLosses
            # end

            # print(optimEq) # uncomment if you want to have the optimization equations

            def optimProblem(x):
                f = [None]*LoS.shape[0]
                for i in range(LoS.shape[0]):
                    f[i]=(eval(optimEq[i]))
                #end
                # print(f)
                return(f)


            print("Starting Multiwall optimization\nThis may take a while...")
            # Executing the optimization
            # sol = scipy.optimize.root(optimProblem, [1]*len(walls2optim+1), method='lm', tol=1e-10, jac=False)
            sol = scipy.optimize.least_squares(optimProblem,[2]*walls2optim.shape[0],ftol=1e-8,gtol=1e-8,xtol=1e-8,method='trf',bounds=(0,30))

            # remapping the optimized wall attenuation to their actual intensity index
            optimWallsAtten = np.zeros((1,256))
            for i in range(1,len(walls2optim)): # because sol.x[0] is the path loss exponent
                optimWallsAtten[0,walls2optim[i]] = sol.x[i]


            # calculating the total loss caused by walls (after optimization)
            totalOptimWallLosses = np.zeros((len(LoS), 1))
            for i in range(len(walls)):
                totalOptimWallLosses[i,0] = np.sum(optimWallsAtten[0,walls[i]])


            # Finding Estimation Performance
            RSS = txPower - (20 * np.log10(self.propFreq) + 20 * np.log10(self.d0) - 147.55) \
                  - 10 * sol.x[0] * np.log10(LoS) - totalOptimWallLosses
            rmse = RMSE(measurements,RSS)
            nrmse = NRMSE(measurements,RSS)

        else:
            # Calculating Total Loss Caused by Walls
            totalWallLoss = np.zeros((len(LoS), 1))
            for i in range(len(walls)):
                totalWallLoss[i, 0] = np.sum(wallsAtten[0, walls[i]])
            # end of for

            RSS = txPower - (20*np.log10(self.propFreq) + 20*np.log10(self.d0) - 147.55) \
                  - 10 * lossExp * np.log10(LoS) - totalWallLoss


        #
        if self.optimization.lower() == 'on':
            return RSS, delaySpr, sol.x[0], optimWallsAtten,(rmse,nrmse)
        else:
            return RSS, delaySpr, lossExp, wallsAtten,(None,None)

    #####################################################################################
    ##########################  Multiwall PROPAGATION MODEL   ###########################






################################    MAIN   ##############################################
#########################################################################################

# sim = multiWallModel()
# # Reading the paln image
# # bwImage, bwImDil = sim.acquireImage()
# bwImage, bwImDil,labledImage,wallsAtten = sim.acquireCSV()
#
# ############# Calibration & Meshing/Gridding  the probing environment
# calUnit, TxNum, TxLocation, gridX, gridY, gridXCul, gridYCul = sim.calibration(bwImDil)
#
# if TxNum != sim.txPower.size:
#     messagebox.showwarning("Wrong # of Tx Location or Tx Power", "Number of selected transmitters=%d and "
#                                                               "'txPower.size=%d' are size inconsisntant! \nTh" %(TxNum,sim.txPower.size))
#     if TxNum < sim.txPower.size: # only accepting the first TxNum-th locations that there is txPower for it
#         TxLocation = TxLocation[0:TxNum,:]
#     else:
#         raise SystemExit # Terminate the program
#
#
#
# # Calculating LoS from every Tx to Rx
# LoS = [None] * TxNum
# for i in range(TxNum):
#     LoS[i] = np.reshape(np.sqrt((gridX - TxLocation[i,0])**2 + (gridY - TxLocation[i,1])**2),(-1,1))
#     # LoS[i] contains the distnace from Tx[i] to all the grid points (nodes)
#
#
# ##### Find which node the measurements are belonging to
# if sim.optimization.lower() == 'on':
#     measuredNodeInd = [None] *TxNum
#     for j in range(TxNum):
#         temp = np.zeros((sim.measuredRSS[j].shape[0],1),dtype=np.int64)
#         for i in range(sim.measuredRSS[j].shape[0]):
#             # measuredNodeInd = np.min(np.sqrt((gridXCul-measuredRSS[i,0])**2 + (gridYCul-measuredRSS[i,1])**2 )
#               temp[i,0] = np.argmin(np.sqrt((gridXCul-sim.measuredRSS[j][i,0])**2 + (gridYCul-sim.measuredRSS[j][i,1])**2 ))
#         measuredNodeInd[j] = temp
#
# # Extracting the wall automagically for MW model & finding the intercepting walls
# if sim.propagationModel.lower() =='mw':
#     print("Starting line detection. This may thake a while...")
#     labledImage, wallLables, wallsCenter = sim.wallExtraction(bwImage)
#     print("Total of " + str(wallsCenter.shape[1]) + " walls are detected.")
#     # Put wall numbers on extracted walls
#     fig = plt.figure(1)
#     ax = fig.add_subplot(111)
#     ax.imshow(labledImage,'bone')
#     for i in range(wallsCenter.shape[1]): # put wall numbers next to the middle of them
#         ax.text(wallsCenter[0,i],wallsCenter[1,i],str(i),horizontalalignment='center',color='red',fontsize=13)
#     # end of for
#     plt.show()
#
#     # find the intercepting walls between each node and Tx
#     # wallsOnLoS = [None]* gridXCul.size
#     print("Starting multi-wall model line of sight analysis.\nThis will take a while...")
#     LoSImage = np.zeros(bwImage.shape, dtype=np.int8)
#     wallsOnLoS = [0]*TxNum
#     for i in range(TxNum):
#         temp = []
#         attEqTemp = []
#         for j in range(gridXCul.size):
#             LoSLineXY = sl.bresenham(TxLocation[i,:],(gridXCul[j,0],gridYCul[j,0]))
#             LoSLineXY = np.asarray(LoSLineXY,dtype=np.int64)
#             LoSImage = LoSImage * 0
#
#             # print(j, " , ", elapsed)
#             # for k in range(len(LoSLineXY)):
#             LoSImage[LoSLineXY[:,1],LoSLineXY[:,0]] = 1  # Crearint an image of the LoS
#             # End of for
#             # Intersecting the LoS line image with the labled image of the walls
#             temp.append(np.unique((LoSImage * labledImage))) # temporary holding the walls in between
#         # End of for
#         wallsOnLoS[i] = temp
#     # End of For
#     print("Line of sight analysis is completed.")
#
#
# RSS = np.zeros((gridXCul.size,TxNum)) ; optimLossExp = np.zeros((1,len(sim.measuredRSS)))
# estAccu = [None] * len(sim.measuredRSS); optimShadowFading = [None] * len(sim.measuredRSS);  optimWallsAtten = [None] *TxNum
#
# delaySpr = np.zeros((gridXCul.size,TxNum))
# for i in range(TxNum):
# ###############################     Multi-Wall Model       ################################
#     if sim.propagationModel.lower() =='mw':
#         if sim.optimization.lower() == 'on':
#             optimWallsOnLoS = []
#             for j in range(len(measuredNodeInd[i])):
#                 optimWallsOnLoS.append(wallsOnLoS[i][measuredNodeInd[i][j,0]])
#             # end
#             _ , _,optimLossExp[0,i:i+1], optimWallsAtten[i],estAccu[i]= \
#                 sim.mwModel(None, sim.txPower[i], LoS[i][measuredNodeInd[i][:,0]], optimWallsOnLoS,sim.wallsAtten,sim.measuredRSS[i][:,2:3])
#             sim.optimization='OFF'
#             RSS[:, i:i + 1], delaySpr[:, i:i + 1],_,_,_= \
#                 sim.mwModel(optimLossExp[0,i:i+1], sim.txPower[i], LoS[i], wallsOnLoS[i],optimWallsAtten[i])
#             sim.optimization='ON'
#         else:
#             RSS[:, i:i + 1], delaySpr[:, i:i + 1], _, _,_= sim.mwModel(sim.pathLossExp[i], sim.txPower[i], LoS[i],wallsOnLoS[i],sim.wallsAtten)
#
# ################################     Log Distance Model       ################################
#     elif sim.propagationModel.lower() =='fspl':
#         if sim.optimization.lower() == "on":
#             # returns the signal strength and delay spread for each node
#             _ , _ , optimLossExp[0, i:i + 1],optimShadowFading[i],estAccu[i] = sim.fsplModel(None,0,sim.txPower[i],LoS[i][measuredNodeInd[i][:,0]],sim.measuredRSS[i][:,2:3])
#             sim.optimization = "OFF"
#             RSS[:, i:i + 1], delaySpr[:, i:i + 1], _, _, _ = sim.fsplModel(optimLossExp[0, i:i + 1],np.std(optimShadowFading[i]), sim.txPower[i], LoS[i])
#             sim.optimization= "ON"
#         else:
#             # returns the signal strength and delay spread for each node
#              RSS[:,i:i+1], delaySpr[:,i:i+1],_,_,_ = sim.fsplModel(sim.pathLossExp[0, i:i + 1],sim.shadowFadingStd[i],sim.txPower[i],LoS[i])  #RSS, delaySpr, lossExp,shadowFading, (None,None)
#     else:
#         print("Incorrect Propagation Model Selected!")
#
#
#
# ########   PLOTTING AND DEMONSTRATION
#
# if sim.TxSuperposition.lower() == 'cw':
#     RSS = np.sum((10**(RSS/10)) * (np.cos(2*np.pi*sim.propFreq*delaySpr) + np.sin(2*np.pi*sim.propFreq*delaySpr)*1j),1)
#     RSS = 10* np.log10(np.abs(RSS))
# elif sim.TxSuperposition.lower() == 'ind':
#     RSS = np.amax(RSS,1)
#
# #  denormalization info
# alpha = np.min(np.min(RSS))
# beta  = np.max(np.max(RSS))
# gamma = beta - alpha
#
#
# # resizing to image size
# RSSImageResized = scipy.misc.imresize(np.reshape((RSS - alpha)/gamma,(gridX.shape)),bwImDil.shape,'bicubic')
# RSSImageResized = (RSSImageResized/np.max(np.max(RSSImageResized)) *gamma) + alpha
#
# # Converting to RGB image
# RSSRGBImage, _ = sl.data2heatmap(RSSImageResized, dynamicRange = 'log')
#
# # Overlay the structure image
# tempXY = np.where(bwImage==1)
# RSSRGBImage[tempXY[0],tempXY[1],:] = [0,0,0]
#
# # !
# fig1 = plt.figure(1)
# ax1 = fig1.add_subplot(111)
# plt.title("Signal strength heatmap")
# ax1.imshow(RSSRGBImage)
#
#
# fig2 = plt.figure(2)
# ax2 = fig2.add_subplot(111)
# im = ax2.imshow(RSSImageResized,'jet')
# plt.colorbar(im)
# plt.title("Hover mouse to see received signal strength (dB)")
#
#
#
# plt.show()
# print('Program complete')
#
# if sim.optimization.lower() == "on":
#     for i in range(TxNum):
#         print("for Tx #%d" %int(i+1))
#         if sim.propagationModel.lower()== "fspl":
#             print("""Path loss exponent = {0}\nShadow fading std = {1} dB
# Estimation RMSE & NRMSE = {2} & {3}\n""".format(optimLossExp[0,i],np.std(optimShadowFading[i]),estAccu[i][0],estAccu[i][1])) #sol.x,shadowFading,(rmse,nrmse)
#         elif sim.propagationModel.lower() == "mw":
#             print("""Path loss exponent = {0}\nEstimation RMSE & NRMSE = {1} & {2}\n""".format(optimLossExp[0, i],estAccu[i][0],estAccu[i][1]))  # sol.x,shadowFading,(rmse,nrmse)
#             print("Wall Numbers or Intensity    ,     Wall Attenuations  ")
#
#             for j in range(256-wallsCenter.shape[1],256):
#                 print(" Wall #{0} or Image Intensity of {1} ------>  {2:0.2f} dB   ".format(int(256-j),int(j+1), optimWallsAtten[i][0,j].tolist()))
#
#     #########################################################################################
#     #########################################################################################
#
