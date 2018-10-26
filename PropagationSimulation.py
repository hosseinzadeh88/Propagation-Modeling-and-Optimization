# Created by: Salaheddin Hosseinzadeh
# Created on: 20.05.2018
# Completed on:
# Last revision:
# Notes:
#####################################################################################

from tkinter import *
from tkinter import filedialog, messagebox, simpledialog
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import thin
import skimage.color as skc
import salimage as sl
import scipy
from salmodule import NRMSE, RMSE
from abc import ABC, abstractmethod
# from wallExtraction import wallExtraction

class propagationSimulation():
    ####################################################################################
    #                              INITIALIZATION
    ####################################################################################
    lightVel = 3e8   # light velocity
    # nodePerMeter = 3 # This identifies the resolution of the estimation
    # pathLossExp = 1.55  # path loss exponent of propagation ! Keep is as a list for FSM sake! :)
    # shadowStd   =  0      # Standard deviation of log-distance shadowing
    # propFreq =  800e6 # Propagation frequency in Hertz
    # d0 = 1           # reference distance of 1 meters
    #
    # #########
    # txPower = np.array([14]) # Power in dB or dBm
    # TxSuperposition = 'CW'  # ['CW'& 'Ind']Continuous Waveform (results in standing wave), Independent
    #
    # environmentFileType = 'IMAGE'   # Either image or CSV
    #
    #
    # propagationModel = 'MW' #['FSPL' 'MW'] FSPL : free space path loss. MW: multi-wall model
    #
    # ####### ASSIGNING WALL ATTENUATIONS MAUALLY (if required) #############
    # wallsAtten = np.ones((1,256))*5 # 5 dB attenuation for each wall
    # wallsAtten[0, 0] = 0  # do not change this (this for a case of clear LoS
    # #######################################################################
    # # Define any specific wall attenuations here
    # wallsAtten[0,255-8] = 10 # 10 dB attenuation for wall #8
    # wallsAtten[0, 255 - 20] = 15  # 15 dB attenuation for wall #20
    # wallsAtten[0, 255 - 4] = 6.5  # 10 dB attenuation for wall #4
    #
    # ###### PROVIDING IMPIRICAL MEASUREMENTS  (if exists)  #################
    # optimization = 'ON' # if set to 1 then walls attenuation will be derived through optimization
    #
    # # CW: adds multiple TX waves at the point of probing resulting in destructive/constructive impact of multiple TXs
    # # Ind: Ignores the weaker RSS
    # # measurements MUST follow this structure: RSS is the Received Signal Strength
    # measuredRSS = [np.array([[12, 39, -75.7],  # Freq = 800e6 Exp = 1.55 / 60 M antenna at Corner
    #                          [97, 96 , -60.9],
    #                          [116, 154, -53.7],
    #                          [20, 112, -74.4],
    #                          [5, 103, -80],
    #                          [54,  151, -68.5],
    #                          [72,  153,  -62.2],
    #                          [102, 208, -53.7],
    #                          [29, 253, -68.5],
    #                          [27, 281, -84.2],
    #                          [5, 307, -89.8],
    #                          [41, 324, -68.8],
    #                          [20, 353, -75],
    #                          [70, 320, -62.9],
    #                          [144, 353, -60.3],
    #                          [251, 324, -63.2],
    #                          [248, 280, -55],
    #                          [243, 240, -44.8],
    #                          [359,311, -60.9],
    #                          [377,279,-70.8],
    #                          [448, 305, -78.3],
    #                          [442, 229, -72.7],
    #                          [379,  202, -60.6],
    #                          [444, 92, -88.8],
    #                          [410, 112, -82.6],
    #                          [359, 97, -81.6],
    #                          [334, 125, -75],
    #                          [311, 140, -68.5],
    #                          [227, 128, -67.5],
    #                          [255, 147, -60.6],
    #                          [236, 155, -54.4],
    #                          [173, 93, -59.3],
    #                          [185, 45, -66.2]]),#
    #                np.array([[262,183,-144],     # X1,Y1,RSI1   # Measurements from/by second transceiver
    #                          [62, 331, -146],  # X2,Y2,RSS2
    #                          [181, 47, -133],
    #                          [324, 150, -150]])]


    # def __init__(self):
    #     if self.optimization.lower()=='on':
    #         if len(self.txPower) != len(self.measuredRSS):
    #             messagebox.showerror("Error","'txPower' and 'measuredRSS' dimensions are not consistent!")
    #             if len(self.txPower) < len(self.measuredRSS):
    #                 del (self.measuredRSS[len(self.txPower):])
    #             else:
    #                 raise SystemError
    #
    #     elif self.optimization.lower() == 'off':
    #         if np.sum(self.wallsAtten) < 3:
    #             messagebox.showwarning("Warning","Please make sure wall attenuations ('wallAtten') are defined correctly.")
    #



# defining abstract methods that required for any general proapgation model
#     @abstractmethod
    def boundaryCalibration(self):
        # needs to be returning the followings
        calUnit = TxNum = TxLocation = gridX = gridY = gridXCul =gridYCul = None
        return calUnit, TxNum, TxLocation, gridX, gridY, gridXCul, gridYCul


    def lineOfSightDistance(self):
        # Calculating LoS from every Tx to Rx
        self.los = [None] * self.TxNum
        print(self.TxLocation)
        for i in range(self.TxNum):
            self.los[i] = np.reshape(np.sqrt((self.gridX - self.TxLocation[i, 0]) ** 2 + (self.gridY - self.TxLocation[i, 1]) ** 2), (-1, 1))
            # LoS[i] contains the distnace from Tx[i] to all the grid points (nodes)

        return self.los

    # def measurementLocator(self):
    #     ##### Find which node the measurements are belonging to
    #     propagationSimulation.measuredNodeInd = [None] * propagationSimulation.TxNum
    #     for j in range(propagationSimulation.TxNum):
    #         temp = np.zeros((propagationSimulation.measuredRSS[j].shape[0], 1), dtype=np.int64)
    #         for i in range(propagationSimulation.measuredRSS[j].shape[0]):
    #             # measuredNodeInd = np.min(np.sqrt((gridXCul-measuredRSS[i,0])**2 + (gridYCul-measuredRSS[i,1])**2 )
    #             temp[i, 0] = np.argmin(np.sqrt(
    #                 (gridXCul - propagationSimulation.measuredRSS[j][i, 0]) ** 2 + (gridYCul - propagationSimulation.measuredRSS[j][i, 1]) ** 2))
    #         self.measuredNodeInd[j] = temp
    #
    #     return  self.measuredNodeInd,self.measuredRSS



    # @abstractmethod # Needs to overwritten based on the requried model
    def propagationModel(self):
        pass

    # @abstractmethod
    def analysisPlot(self):
        pass


    # @abstractmethod
    def defineEnvironment(self):
        pass
    ###############################  IMAGE ACQUISITION  #################################
    #####################################################################################





    #####################################################################################
    ########################  IMAGE ACQUISITION  ########################################



    ####################################################################################
    #                       Critical Input Sanity Check



class multiWallModel(propagationSimulation):

    pathLossExp = 1.5


    ########################    Evironment def    ########################################
    def defineEnvironment(self,environmentFileType):

        # should prompt to read a file here either image or txt
        fileName = 'D:\\str.png'
        labledImage = attenMap = None

        if environmentFileType.lower() == "image": # demands a pic which then requires calibration as well
            img = plt.imread(fileName)  # reads image in range of 0 to 1

        elif environmentFileType.lower() == "csv": # We will create not just an image but a labled image so Hough is not required
            boundaryMargin = int(30)
            fid = open(fileName, 'r')
            coords = fid.readlines()
            fid.close()
            x1 = np.zeros((len(coords), 1))
            y1, x2, y2, attenuation = x1.copy(), x1.copy(), x1.copy(), x1.copy()

            for i in range(len(coords)):
                x1[i, 0], y1[i, 0], x2[i, 0], y2[i, 0], attenuation[i, 0] = coords[i].split(',')
                # Forming a picture with the walls being labled with different intensity levels

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

            attenMap = np.array((1, 256))
            # forming the labled image

            xMax = np.max([np.max(np.max(x1)), np.max(np.max(x2))])
            xMin = np.min([np.min(np.min(x1)), np.min(np.min(x2))])

            yMax = np.max([np.max(np.max(y1)), np.max(np.max(y2))])
            yMin = np.min([np.min(np.min(y1)), np.min(np.min(y2))])

            labledImage = np.zeros((yMax + boundaryMargin, xMax + boundaryMargin), dtype=np.int8)

            for i in range(x1.shape[0]):
                line = sl.bresenham((x1[i, 0], y1[i, 0]), (x2[i, 0], y2[i, 0]))
                line = np.asarray(line, dtype=np.int64)
                labledImage[line[:, 1], line[:, 0]] = 255 - i
                attenMap[0, 255 - i] = attenuation[i, 0]  # wall attenuations are indexed in attenMap according to the wall's image intesity

            img =  labledImage
        # end of if

        if img.shape[2] > 1: # if image is RGB and not grayscale
            # Converting to grayscale if it is not already
            img = sl.rgb2gray(img)

        # Thresholding it to binary just in case it is not already
        bwImage = np.array((img > (np.max(np.max(img))/2)), dtype=bool)

        # Make sure structures are in black (flase) and background is white (true)
        if (np.sum(bwImage) > np.sum(~bwImage)):
            bwImage = ~bwImage
            print('Image complemented')

        # Dilating the image to make it easier ot select the wall interactively
        # kernelSize = int(np.amin(bwImage.shape)/50)
        kernelSize = 8 # fixed size kernel
        kernel = np.ones((kernelSize, kernelSize), np.uint8)
        bwImDil = cv2.dilate(np.array(bwImage, dtype=np.uint8), kernel, iterations=1)

        return bwImage, bwImDil,labledImage,attenMap


    ########################    CALIBRATION      ########################################
    def boundaryCalibration(self,bwImDil):
        # Opening a dialog box asking the user to use a reference wall and enter its length
        messagebox.showinfo("Instructions!", """Please select a reference wall on the image by clicking on the two ends of it.
         \nLater you wil be asked to enter its corresponding length.""")

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
                    messagebox.showinfo("Wrong selection!", """Error in selecting the wall.\nPlease select a reference wall on the image by clicking on the two ends of it.
                    \nLater you wil be asked to enter its corresponding length.""")
                    print('wrong points selected, try again')
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

    ############################    WALL EXTRACTION   ###################################
    #####################################################################################


    ##########################  Multiwall PROPAGATION MODEL   ###########################
    def propagationModel(self,lossExp,txPower,LoS,walls,wallsAtten,measurements=None):

        LoS[np.where(LoS<self.d0)] = self.d0  ## distances under 1 m are not acceptable (log10(<1) problem)

        if (measurements is None) and self.optimization.lower() == 'on':
                raise SystemError # optimization defo requires measurements
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
                optimEq[i] = "%f - (20*np.log10(%f) + 20*np.log10(%f) - 147.55) - 10 * x[0] * np.log10(%f) - (%f)" \
                             % (txPower, self.propFreq, self.d0, LoS[i, 0], measurements[i])
                if walls[i].size > 1:
                    wallLosses = ""
                    for j in range(1,walls[i].size):
                        wallLosses = wallLosses + "+ x[%d]" %np.where(walls2optim == walls[i][j])
                    # end
                # end

                    optimEq[i] = optimEq[i] + " - (%s)" %wallLosses
            # end

            print(optimEq)

            def optimProblem(x):
                f = [None]*LoS.shape[0]
                for i in range(LoS.shape[0]):
                    f[i]=(eval(optimEq[i]))
                #end
                print(f)
                return(f)



            # Executing the optimization
            # sol = scipy.optimize.root(optimProblem, [1]*len(walls2optim+1), method='lm', tol=1e-10, jac=False)
            sol = scipy.optimize.least_squares(optimProblem,[2]*23,ftol=1e-20,gtol=1e-20,xtol=1e-40,method='trf',bounds=(0,20))

            # Finding Estimation Performance
            RSS = txPower - (20 * np.log10(self.propFreq) + 20 * np.log10(self.d0) - 147.55) \
                  - 10 * sol.x * np.log10(LoS) + np.random.normal(0, self.shadowStd, LoS.shape)
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
            return RSS, delaySpr, sol.x[0], sol.x[1:]
        else:
            return RSS, delaySpr, lossExp, wallsAtten



    ############################    WALL EXTRACTION   ###################################
    def wallExtraction(self,img):

        if len(img.shape) > 2: # if image is not grayscale
            # Converting to grayscale if it is not already
            img = sl.rgb2gray(img)

        # Thresholding it to binary just in case it is not already
        bwImage = np.array((img > (np.max(np.max(img))/2)), dtype=bool)

        # Make sure structures are in black (flase) and background is white (true)
        if (np.sum(bwImage) > np.sum(~bwImage)):
            bwImage = ~bwImage
            print('Image complemented')

        # Pre processing the image to make it nice and clean for the job :)
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


    ##########################  Multiwall PROPAGATION MODEL   ###########################
    # def optimization(self):
    #     optimWallsOnLoS = []
    #     for j in range(len(self.measuredNodeInd[i])):
    #         optimWallsOnLoS.append(wallsOnLoS[i][self.measuredNodeInd[i][j,0]])
    #     # end
    #     _ , _,optimLossExp[0,i:i+1], wallsAtten[i] = \
    #         multiWallModel.propagationModel(pathLossExp, sim.txPower[i], self.los[i][self.measuredNodeInd[i][:,0]], optimWallsOnLoS,wallsAtten,self.measuredRSS[i][:,2:3])
    #     multiWallModel.optimization='OFF'
    #     RSS[:, i:i + 1], delaySpr[:, i:i + 1] = \
    #         multiWallModel.propagationModel(multiWallModel.pathLossExp, multiWallModel.txPower[i], self.los[i], wallsOnLoS[i],wallsAtten)
    #     multiWallModel.optimization='ON'
    #
    #     return

    ################################    Analysis   ########################################
    def analysisPlot(selfs):
        pass


class logDistanceModel (propagationSimulation):

    def __init__(self,boundaryRadious):
        # super(propagationSimulation,self).__init__()
        self.boundaryRadious = int(boundaryRadious)

    ##########################   Boundary Calibration    ################################
    def defineEnvironment(self):
        self.img = np.zeros((int(self.boundaryRadious * 1.4), int(self.boundaryRadious * 1.4)), dtype=np.int8)
        return self.img,None

    ##########################   Boundary Calibration    ################################
    def boundaryCalibration(self):
        calUnit = 1

        numNodesX = self.img.shape[1] * calUnit * self.nodePerMeter
        numNodesY = self.img.shape[0] * calUnit * self.nodePerMeter

        nodesX = np.linspace(0, self.img.shape[1] - 1, numNodesX, dtype=np.int64)
        nodesY = np.linspace(0, self.img.shape[0] - 1, numNodesY, dtype=np.int64)
        self.gridX, self.gridY = np.meshgrid(nodesX, nodesY)  # Gridding the environment
        self.gridXCul = np.reshape(self.gridX, (-1, 1))
        self.gridYCul = np.reshape(self.gridY, (-1, 1))

        # returning args just in case needed externally some of which are defined during the instantiation
        return calUnit, self.TxNum, self.TxLocation, self.gridX, self.gridY, self.gridXCul, self.gridYCul

    ##########################  FSPL Propagation OPtimization   ################################
    # def propagationModel(self): # pass measurements as 0 if don't exist
    #     #  free space path loss is actually the log-distance model with shadowing (This is more practical)
    #     self.los[np.where(self.los < self.d0)] = self.d0  ## distances under 1 m are not acceptable (log10(<1) problem)
    #     shadowFading = np.random.normal(0, shadowStd, self.los.shape)
    #
    #     if (measurements is None) and self.optimization.lower() == 'on':
    #             raise SystemError # optimization defo requires measurements
    #     # end of if
    #
    #     delaySpr = LoS / self.lightVel  # delay spread calculation
    #
    #     if self.optimization.lower() == 'on': # finds both path loss exponent x[0], and shadow fading x[1] through optimization
    #         # Forming the optimization problem
    #         optimEq = ['a'] * LoS.shape[0]
    #         for i in range(LoS.shape[0]):
    #             optimEq[i] = "%f - (20*np.log10(%f) + 20*np.log10(%f) - 147.55) - 10 * x[0] * np.log10(%f) - (%f)" % (
    #             txPower, self.propFreq, self.d0, LoS[i, 0], measurements[i])
    #         def optimProblem(x):
    #             f = [None] * LoS.shape[0]
    #             for i in range(LoS.shape[0]):
    #                 f[i]=(eval(optimEq[i]))
    #             print(f)
    #             return(f)
    #
    #         print(optimEq)
    #         # Executing the optimization
    #         sol = scipy.optimize.root(optimProblem, [1], method='lm', tol=1e-10, jac=False)
    #
    #         # Finding Estimation Performance
    #         RSS = txPower - (20 * np.log10(self.propFreq) + 20 * np.log10(self.d0) - 147.55) \
    #               - 10 * sol.x * np.log10(LoS)
    #         rmse = RMSE(measurements,RSS)
    #         nrmse = NRMSE(measurements,RSS)
    #         shadowFading = measurements-RSS
    #
    #     else:
    #         RSS = txPower - (20*np.log10(self.propFreq) + 20*np.log10(self.d0) - 147.55) \
    #               - 10 * pathLossExp * np.log10(LoS) + shadowFading
    #
    #     if self.optimization.lower() == 'on':
    #         return RSS, delaySpr,sol.x,shadowFading,(rmse,nrmse)
    #     else:
    #         return RSS, delaySpr, pathLossExp,shadowFading, (None,None)

    ################################    Analysis   ########################################
    def analysisPlot(selfs):
        pass

################################    MAIN   ##############################################
#########################################################################################
#
# sim = multiWallModel()
# # Reading the paln image
# bwImage, bwImDil  = sim.acquire()
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
# #
# # # Calculating LoS from every Tx to Rx
# # LoS = [None] * TxNum
# # for i in range(TxNum):
# #     LoS[i] = np.reshape(np.sqrt((gridX - TxLocation[i,0])**2 + (gridY - TxLocation[i,1])**2),(-1,1))
# #     # LoS[i] contains the distnace from Tx[i] to all the grid points (nodes)
# #
# #
# # ##### Find which node the measurements are belonging to
# # if sim.optimization.lower() == 'on':
# #     measuredNodeInd = [None] *TxNum
# #     for j in range(TxNum):
# #         temp = np.zeros((sim.measuredRSS[j].shape[0],1),dtype=np.int64)
# #         for i in range(sim.measuredRSS[j].shape[0]):
# #             # measuredNodeInd = np.min(np.sqrt((gridXCul-measuredRSS[i,0])**2 + (gridYCul-measuredRSS[i,1])**2 )
# #               temp[i,0] = np.argmin(np.sqrt((gridXCul-sim.measuredRSS[j][i,0])**2 + (gridYCul-sim.measuredRSS[j][i,1])**2 ))
# #         measuredNodeInd[j] = temp
# #
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
# estAccu = [None] * len(sim.measuredRSS); shadowFading = [None] * TxNum;  wallsAtten = [None] *TxNum
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
#             _ , _,optimLossExp[0,i:i+1], wallsAtten[i] = \
#                 multiWallModel.propagationModel(sim.pathLossExp, sim.txPower[i], LoS[i][measuredNodeInd[i][:,0]], optimWallsOnLoS,sim.wallsAtten,sim.measuredRSS[i][:,2:3])
#             multiWallModel.optimization='OFF'
#             RSS[:, i:i + 1], delaySpr[:, i:i + 1] = \
#                 multiWallModel.propagationModel(multiWallModel.pathLossExp, multiWallModel.txPower[i], LoS[i], wallsOnLoS[i],wallsAtten)
#             multiWallModel.optimization='ON'
#         else:
#             RSS[:, i:i + 1], delaySpr[:, i:i + 1], _, _= multiWallModel.propagationModel(sim.pathLossExp, sim.txPower[i], LoS[i],wallsOnLoS[i],sim.wallsAtten)
#
# ################################     Log Distance Model       ################################
#     elif sim.propagationModel.lower() =='fspl':
#         if sim.optimization.lower() == 'on':
#             # returns the signal strength and delay spread for each node
#             _ , _ , optimLossExp[0, i:i + 1],shadowFading[i],estAccu[i] = sim.fsplOptimization(None,sim.txPower[i],LoS[i][measuredNodeInd[i][:,0]],sim.measuredRSS[i][:,2:3])
#             sim.optimization = 'OFF'
#             RSS[:, i:i + 1], delaySpr[:, i:i + 1], _, _, _ = sim.fsplOptimization(optimLossExp[0, i:i + 1], sim.txPower[i], LoS[i])
#             sim.optimization= 'ON'
#         else:
#             # returns the signal strength and delay spread for each node
#              RSS[:,i:i+1], delaySpr[:,i:i+1],_ = sim.fsplOptimization(sim.pathLossExp[i],sim.txPower[i],LoS[i])
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
# # ax1.title("Signal strength heatmap")
# ax1.imshow(RSSRGBImage)
#
#
# fig2 = plt.figure(2)
# ax2 = fig2.add_subplot(111)
# ax2.imshow(RSSImageResized,'jet')
# # ax2.title("Hover mouse to see received signal strength (dB)")
#
# fig2.colorbar
#
# plt.show()
# print('so far so good.')
# Forming the LoS Attenuation Equations


# Calculate the attenuation/signal strength loss. This would be due to the distance and walls

# if optimization == 1:
#     pass # I'm going to add the optimization into it later
#
# fig = plt.figure()
    # ax = fig.add_subplot(1,1,1)
    # ax.imshow(LoS[0],'jet')
    # plt.imshow()

    #########################################################################################
    #########################################################################################

