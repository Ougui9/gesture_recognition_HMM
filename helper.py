import numpy as np
# import os
import pandas as pd

def impData(path):
    data = pd.read_csv(path, sep='\t', header=None).values.T.astype(float)
    AW=data[1:,:]

    return AW.T

