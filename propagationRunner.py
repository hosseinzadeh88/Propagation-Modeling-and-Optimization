# Created by: Salaheddin Hosseinzadeh
# Created on: 20.05.2018
# Completed on:
# Last revision:
# Notes:  If you find any bugs please contact me at hosseinzadeh.88@gmail.com
#         I have a same file written in MATLAB, but this one is far more complete. I've redone this as the previous MATLAB code
#         happened to be quite popular. This Python submission has optimization both for the log-distance model and the multiwall model
#
# Need to bypass wall extraction if CSV is passed in
# When you're using it in optimization model, it first find the unknown parameters and then used the optimizated parameter to
# Generate a whole map of the propagation :) all in one go.
# Limitations: No more than 255 walls would be allowed in the structure.
#####################################################################################

from tkinter import *
from tkinter import filedialog, messagebox, simpledialog
import matplotlib.pyplot as plt
import numpy as np
import salimage as sl
import scipy
import MultiWallMode as mwm

################################    MAIN   ##############################################
#########################################################################################
sim = mwm.multiWallModel() # creating a propagation object instance

# only change hte nodePerMeter parameter of you require a higher resolution image
# be aware, the higher the number of nodes in a meter the higher the computation time.
# If you want to see the imapct of different Transmitters (Tx)s in the "CW" more, you need to adjust the following depending on the
# frequency of the propagation. Generally, the higher the frequency the higher the nodePerMeter should be.
sim.nodePerMeter = .25  # This identifies the resolution of the estimation
# Define the properties of the simulation here.
# Check multiWallModel class fields for further adjustable parameters

# Path loss exponent will be determined through optimizations anyways.
sim.pathLossExp = np.array([1.55, 1.55,1.55,1.55,1.4])  # path loss exponent of propagation! Keep is as a list for FSM sake! :)
sim.shadowFadingStd = [0,0,0,0,0]       # Standard deviation of log-distance shadow fading
# Check this link for some of the related parameters of Log Distance Model
# https://en.wikipedia.org/wiki/Log-distance_path_loss_model

sim.txPower = np.array([14, 14,14,14,14])  # Transmitter or transmitters power in dB or dBm
sim.propagationModel = "MW"       # Chose between 'FSPL' or 'MW' models. FSPL : free space path loss. MW: multi-wall model
sim.propFreq = 800e6              # Propagation frequency in Hertz
sim.d0 = 1                        # reference distance of 1 meters. This is usually 1 meters for indoor, but it may vary

structureFileType = "CSV"         # needs to be either "CSV" or "IMG". Either csv or image (preferably in PNG format).

# Assigning wall attenuations manually if required. If you have measurements you can ignore it and define all the
# atteniations to 0 or 1. Any value you set is going to be ignored if optimization is on, as it will be determined by the optimization. wallsAtten = np.zeros((1,256)).
# For Multi-Wall model the least-square optimization is used
# Also, if you're using a CSV file which shuold contain the attenuations, they will be assigned from the CSV file.
# When defining the CSV file, think of the environment as an image starting from coordinates (0,0) from the left top corner,
# if it's confusing, don't worry I coded some measures that would correct an ill-defined format, but try to be rational in case those fail.
sim.wallsAtten = np.ones((1, 256))*5# 5 dB attenuation for all wall
sim.wallsAtten[0, 0] = 0  # do not change this (this for a case of clear LoS)
#######################################################################
# This is how you would define the attenuation for each specific wall. For instance
sim.wallsAtten[0, 255 - 8] = 10  # 10 dB attenuation for wall #8
sim.wallsAtten[0, 255 - 20] = 15  # 15 dB attenuation for wall #20
sim.wallsAtten[0, 255 - 4] = 6.5  # 6.5 dB attenuation for wall #4

sim.optimization = "OFF"  # if set to "ON" then walls attenuation and path loss exponent will be derived through optimization
# if optimization is set ot "ON" then measurements must be supplied

# This impacts how the multiple transmitters in an environment would interact.
# CW   assumes that propagation is taking place with continuous waveforms of 0 phase-shift from each transmitter.
#      In case of more than 1 Tx, this will result in destructive or constructive
# Ind  Only considers the strongest and dominating RSS (received signal strength) received from a Tx. Ignores other RSS from weaker Txs
sim.TxSuperposition = "IND"  # ['CW'& 'Ind']Continuous Waveform (results in standing wave), Independent
# CW: adds multiple TX waves at the point of probing resulting in destructive/constructive impact of multiple TXs
# Ind: Ignores the weaker RSS

# measurements MUST have the following structure:
# measuredRSS = [np.array([[x1,y1,RSS1],    # Measurements from Tx1
#                          [x2,y2,RSS2],
#                          [x3,y3,RSS3]]),
#                np.array([[x1,y1,RSS1],   # Measurements from Tx2
#                          [x2,y2,RSS2]])]
# where x1 is the x/column of location of the measurement showed on the image of the structure and y1 is the y/row on the structure
# image where measurement 1 was take. and RSS1 is the recorded signal strength at that location and so on for location 2
# Below is an example of such measurements
# Bare in mind, the more walls you have the more measurements you require for the optimization ot have a unique answer!
# Also, all the walls need to be involved in the measurements otherwise optimization won't be unique. (This is maths nothing one can do about)
# Ultimately you can write your won method to read the measurements from file, but in a similar format
sim.measuredRSS = [np.array([[12, 39, -75.7],  #
                         [97, 96, -60.9],
                         [116, 154, -53.7],
                         [20, 112, -74.4],
                         [5, 103, -80],
                         [54, 151, -68.5],
                         [72, 153, -62.2],
                         [102, 208, -53.7],
                         [29, 253, -68.5],
                         [27, 281, -84.2],
                         [5, 307, -89.8],
                         [41, 324, -68.8],
                         [20, 353, -75],
                         [70, 320, -62.9],
                         [144, 353, -60.3],
                         [251, 324, -63.2],
                         [248, 280, -55],
                         [243, 240, -44.8],
                         [359, 311, -60.9],
                         [377, 279, -70.8],
                         [448, 305, -78.3],
                         [442, 229, -72.7],
                         [379, 202, -60.6],
                         [444, 92, -88.8],
                         [410, 112, -82.6],
                         [359, 97, -81.6],
                         [334, 125, -75],
                         [311, 140, -68.5],
                         [227, 128, -67.5],
                         [255, 147, -60.6],
                         [236, 155, -54.4],
                         [173, 93, -59.3],
                         [185, 45, -66.2]]),  #
               np.array([[262, 183, -144],  # X1,Y1,RSI1   # Measurements from/by second transceiver
                         [62, 331, -146],  # X2,Y2,RSS2
                         [181, 47, -133],
                         [324, 150, -150]])]


############################################################################################################
############################################################################################################
##################################   DONT MAKE ANY CHANGES BYOND THIS ######################################
############################################################################################################
############################################################################################################


# Acquiring the structure of the measurement environment (This applies to both indoor and outdoor)
# If an image is used make sure your image is tidy and it only consists of straight lines) this won't work otherwise.
# If you're using an image the walls will be extracted and labled automagically.
if structureFileType.lower() == "img":
    bwImage, bwImDil = sim.acquireImage()
# Second option is to use a CSV file contaning coordinates of each wall in the following format the x1_start,y1_start,x1_end,y1_end,attenuation1
elif structureFileType.lower() == "csv":
    bwImage, bwImDil,labledImage,sim.wallsAtten, wallsCenter = sim.acquireCSV()
    # Eventually you can write your own acquiring method as long as it returns the required params
else:
    assert 0, "\"structureFileType\" is incorrect. Needs to be either \"CSV\" or \"IMG\""


############# Calibration & Meshing/Gridding  the probing environment
calUnit, TxNum, TxLocation, gridX, gridY, gridXCul, gridYCul = sim.calibration(bwImDil)

############  Checking if the size of the input parameters and selected number of Txs are ok.
if TxNum != sim.txPower.size:
    root = Tk()
    messagebox.showwarning("Wrong # of Tx Location or Tx Power", "Number of selected transmitters=%d and "
                                                              "'txPower.size=%d' are size inconsisntant! \nTh" %(TxNum,sim.txPower.size))
    root.destroy()
    if TxNum < sim.txPower.size: # only accepting the first TxNum-th locations that there is txPower for it
        TxLocation = TxLocation[0:TxNum,:]
    else:
        assert 0,"Inconsistant number of \"txPower\" and selected number of transmitters (\"TxNum\")"


# Calculating LoS from every Tx to Rx
LoS = sim.lineOfSight (TxNum,TxLocation,gridX,gridY)

##### Find which node the measurements are belonging to
if sim.optimization.lower() == 'on':
    measuredNodeInd = [None] *TxNum
    for j in range(TxNum):
        temp = np.zeros((sim.measuredRSS[j].shape[0],1),dtype=np.int64)
        for i in range(sim.measuredRSS[j].shape[0]):
            # measuredNodeInd = np.min(np.sqrt((gridXCul-measuredRSS[i,0])**2 + (gridYCul-measuredRSS[i,1])**2 )
              temp[i,0] = np.argmin(np.sqrt((gridXCul-sim.measuredRSS[j][i,0])**2 + (gridYCul-sim.measuredRSS[j][i,1])**2 ))
        measuredNodeInd[j] = temp

######## Extracting the wall automagically for MW model & finding the intercepting walls
# This is a CPU taxing process!
# However, it is not needed when structure is defined in a CSV file.
if sim.propagationModel.lower() =="mw" and structureFileType.lower()== "img":
    print("Starting line detection. This may thake a while...")
    labledImage, wallLables, wallsCenter = sim.wallExtraction(bwImage)

if sim.propagationModel.lower() == "mw":
    print("Total of " + str(wallsCenter.shape[1]) + " walls are detected.")
    # Put wall numbers on extracted walls
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.imshow(labledImage,'bone')
    for i in range(wallsCenter.shape[1]): # put wall numbers next to the middle of them
        ax.text(wallsCenter[0,i],wallsCenter[1,i],str(i+1),horizontalalignment='center',color='red',fontsize=13)
    # end of for
    plt.show()


####################    This Piece can be embedded into the mltiwall class
    # find the intercepting walls between each node and Tx
    # wallsOnLoS = [None]* gridXCul.size
    print("Starting multi-wall model line of sight analysis.\nThis will take a while...")
    wallsOnLoS = sim.wallsOnLineOfSight(TxNum, TxLocation, gridXCul, gridYCul, labledImage)


RSS = np.zeros((gridXCul.size,TxNum)) ; optimLossExp = np.zeros((1,len(sim.measuredRSS)))
estAccu = [None] * len(sim.measuredRSS); optimShadowFading = [None] * len(sim.measuredRSS);  optimWallsAtten = [None] *TxNum
delaySpr = np.zeros((gridXCul.size,TxNum))

# Wrapping propagation models in a for loops
# In this instance the "optimization" value is changed from "on" to "off" when using potimization.
# This is so that the optimized parameters to be found first from measuremnets and then the propagation
# map will be generated.

print("Estimating (and optimizing) the propagation, this takes a while...")
for i in range(TxNum):
###############################     Multi-Wall Model       ################################
    if sim.propagationModel.lower() =='mw':
        if sim.optimization.lower() == 'on':
            optimWallsOnLoS = []
            for j in range(len(measuredNodeInd[i])):
                optimWallsOnLoS.append(wallsOnLoS[i][measuredNodeInd[i][j,0]])
            # end
            _ , _,optimLossExp[0,i:i+1], optimWallsAtten[i],estAccu[i]= \
                sim.mwModel(None, sim.txPower[i], LoS[i][measuredNodeInd[i][:,0]], optimWallsOnLoS,sim.wallsAtten,sim.measuredRSS[i][:,2:3])
            sim.optimization='OFF'
            RSS[:, i:i + 1], delaySpr[:, i:i + 1],_,_,_= \
                sim.mwModel(optimLossExp[0,i:i+1], sim.txPower[i], LoS[i], wallsOnLoS[i],optimWallsAtten[i])
            sim.optimization='ON'
        else:
            RSS[:, i:i + 1], delaySpr[:, i:i + 1], _, _,_= sim.mwModel(sim.pathLossExp[i], sim.txPower[i], LoS[i],wallsOnLoS[i],sim.wallsAtten)

################################     Log Distance Model       ################################
    elif sim.propagationModel.lower() =='fspl':
        if sim.optimization.lower() == "on":
            # returns the signal strength and delay spread for each node
            _ , _ , optimLossExp[0,i:i + 1],optimShadowFading[i],estAccu[i] = sim.fsplModel(None,0,sim.txPower[i],LoS[i][measuredNodeInd[i][:,0]],sim.measuredRSS[i][:,2:3])
            sim.optimization = "OFF"
            RSS[:, i:i + 1], delaySpr[:, i:i + 1], _, _, _ = sim.fsplModel(optimLossExp[0,i:i + 1],np.std(optimShadowFading[i]), sim.txPower[i], LoS[i])
            sim.optimization= "ON"
        else:
            # returns the signal strength and delay spread for each node
             RSS[:,i:i+1], delaySpr[:,i:i+1],_,_,_ = sim.fsplModel(sim.pathLossExp[i],sim.shadowFadingStd[i],sim.txPower[i],LoS[i])  #RSS, delaySpr, lossExp,shadowFading, (None,None)
    else:
        assert 0, "Incorrect Propagation Model Selected!"




########   PLOTTING AND DEMONSTRATION
print("Almost done! Just preparing the propagation map :)")
if sim.TxSuperposition.lower() == 'cw':
    RSS = np.sum((10**(RSS/10)) * (np.cos(2*np.pi*sim.propFreq*delaySpr) + np.sin(2*np.pi*sim.propFreq*delaySpr)*1j),1)
    RSS = 10* np.log10(np.abs(RSS))
elif sim.TxSuperposition.lower() == 'ind':
    RSS = np.amax(RSS,1)

delaySpr = np.amin(delaySpr,1)

#  denormalization the RSS map as for some reason the imresize scale the image to an actual image
alpha = np.min(np.min(RSS))
beta  = np.max(np.max(RSS))
gamma = beta - alpha


# resizing to image size
RSSImageResized = scipy.misc.imresize(np.reshape((RSS - alpha)/gamma,(gridX.shape)),bwImDil.shape,'bicubic')
RSSImageResized = (RSSImageResized/np.max(np.max(RSSImageResized)) *gamma) + alpha

# Converting to RGB image
# dynamicRange = "linear" is the default, however "log" compresses the dynamic range twice!
RSSRGBImage, _ = sl.data2heatmap(RSSImageResized, dynamicRange = 'log') # "dynamicRange = "log" compresses the dynamic range twice


# Overlay the structure image
tempXY = np.where(bwImage==1)
RSSRGBImage[tempXY[0],tempXY[1],:] = [0,0,0]

# !
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
plt.title("Signal strength heatmap")
ax1.imshow(RSSRGBImage)


fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
im2 = ax2.imshow(RSSImageResized,'jet')
plt.colorbar(im2)
plt.title("Hover mouse to see received signal strength (dB)")

fig3 = plt.figure(3)
ax3 = fig3.add_subplot(111)
im3 = ax3.imshow(np.reshape(delaySpr,(gridX.shape)),'jet')
plt.colorbar(im3)
plt.title("Delay spread (seconds)")



if sim.optimization.lower() == "on":
    fig4 = plt.figure(4)
    ax4 = fig4.add_subplot(111)
    for i in range(TxNum):
        print("for Tx #%d" %int(i+1))
        if sim.propagationModel.lower()== "fspl":
            print("""Path loss exponent = {0:0.3f}\nShadow fading std = {1:0.3f} dB
Estimation RMSE & NRMSE = {2:0.3f} & {3:0.3f}\n""".format(optimLossExp[0,i],np.std(optimShadowFading[i]),estAccu[i][0],estAccu[i][1])) #sol.x,shadowFading,(rmse,nrmse)
            ax4.plot(optimShadowFading[i],color =(np.random.rand(),np.random.rand(),1),label="Tx #{0}".format(i+1))
            plt.title("Optimized shadow-fading at each measured location")
            plt.ylabel("Shadow-fading (dB)")
            plt.xlabel("Measurement number")
        elif sim.propagationModel.lower() == "mw":
            print("""Path loss exponent = {0:0.3f}\nEstimation RMSE & NRMSE = {1:0.3f} & {2:0.3f}\n""".format(optimLossExp[0, i],estAccu[i][0],estAccu[i][1]))  # sol.x,shadowFading,(rmse,nrmse)
            print("Wall Numbers or Intensity    ,     Wall Attenuations  ")
            ax4.plot(optimWallsAtten[i][0,range(256-wallsCenter.shape[1],256)],color =(np.random.rand(),np.random.rand(),1),label="Tx #{0}".format(i+1))
            plt.title("Optimized effective attenuation of walls (dB)")
            plt.ylabel("Wall attenuation(dB)")
            plt.xlabel("Wall number")
            for j in range(256-wallsCenter.shape[1],256):
                print(" Wall #{0} or Image Intensity of {1} ------>  {2:0.2f} dB   ".format(int(256-j),int(j+1), optimWallsAtten[i][0,j].tolist()))

    ax4.legend()

    #########################################################################################
    #########################################################################################

plt.show()

print('Program complete')