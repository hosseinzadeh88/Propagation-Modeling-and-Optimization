# A few of frequenctly used functions
# Created by: Salaheddin Hosseinzadeh
# Created on: 10.06.2018
# Last revision:
# Notes:
#########################################

import numpy as np


######################        Normalized Root Measn Sequare Root Errpr           ######################################
def NRMSE(observation,estimation):
    assert ~(observation.shape != estimation.shape)

    out = np.sqrt(np.sum(np.sum((observation - estimation)**2))) / (np.abs(np.max(np.max(observation)) - np.min(np.min(estimation))))

    return out




def RMSE (observation,estimation):
    assert ~(observation.shape != estimation.shape)

    out = np.sqrt(np.sum(np.sum((observation - estimation) ** 2)))

    return out





def MSE (observation,estimation):
    assert ~(observation.shape != estimation.shape)

    out = np.sum(np.sum((observation - estimation) ** 2))

    return out