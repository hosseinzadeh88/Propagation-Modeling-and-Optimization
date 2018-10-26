from PropagationSimulation import propagationSimulation, logDistanceModel
import numpy as np



sim = logDistanceModel(100)

sim.nodePerMeter = 3  # This identifies the resolution of the estimation
sim.pathLossExp = 1.55  # path loss exponent of propagation ! Keep is as a list for FSM sake! :)
sim.shadowStd = 0  # Standard deviation of log-distance shadowing
sim.propFreq = 800e6  # Propagation frequency in Hertz
sim.d0 = 1  # reference distance of 1 meters

#########
sim.txPower = np.array([14])  # Power in dB or dBm
sim.TxSuperposition = 'CW'  # ['CW'& 'Ind']Continuous Waveform (results in standing wave), Independent

sim.environmentFileType = 'IMAGE'  # Either image or CSV

sim.propagationModel = 'MW'  # ['FSPL' 'MW'] FSPL : free space path loss. MW: multi-wall model

####### ASSIGNING WALL ATTENUATIONS MAUALLY (if required) #############
sim.wallsAtten = np.ones((1, 256)) * 5  # 5 dB attenuation for each wall
sim.wallsAtten[0, 0] = 0  # do not change this (this for a case of clear LoS
#######################################################################
# Define any specific wall attenuations here
sim.wallsAtten[0, 255 - 8] = 10  # 10 dB attenuation for wall #8
sim.wallsAtten[0, 255 - 20] = 15  # 15 dB attenuation for wall #20
sim.wallsAtten[0, 255 - 4] = 6.5  # 10 dB attenuation for wall #4

###### PROVIDING IMPIRICAL MEASUREMENTS  (if exists)  #################
sim.optimization = 'ON'  # if set to 1 then walls attenuation will be derived through optimization

# CW: adds multiple TX waves at the point of probing resulting in destructive/constructive impact of multiple TXs
# Ind: Ignores the weaker RSS
# measurements MUST follow this structure: RSS is the Received Signal Strength
sim.measuredRSS = [np.array([[12, 39, -75.7],  # Freq = 800e6 Exp = 1.55 / 60 M antenna at Corner
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

sim.TxLocation = np.array([[10,10]],dtype=np.int64) # leave it as None if you want to chose it manually
sim.TxNum = 1 #


sim.defineEnvironment()
sim.boundaryCalibration()
sim.lineOfSightDistance()

