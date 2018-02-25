import numpy as np
import os
import pandas as pd

# def preprocessData(data_in):
from ukf import processA,processW,caldQ

# 	return q_out

# def importData(path):
#     rawdata=pd.read_csv(path, sep='\t', header=None)
#
#     return q_out


def impData(path):
    data = pd.read_csv(path, sep='\t', header=None).values.T.astype(float)
    AW=data[1:,:]

    return AW.T

