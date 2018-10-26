# This is the multi-wall model optimization porblem generator
# Created by: Salaheddin Hosseinzadeh
# Created on: 01.06.2018
# Last revision:
# Notes:
#####################################################################################
import numpy as np

def multiWallOptim(txPower, lossExp = 1,losDistance,LoSwalls,measurements,mode='cw'):
    statement = []
    losDistance [np.where(losDistance<1)] = 1 # distances under 1 m are not acceptable (log10(<1) problem)
    for i in range(losDistance.size):
        statement.append(txPower - lossExp*10*np.log10(losDistance) - )
