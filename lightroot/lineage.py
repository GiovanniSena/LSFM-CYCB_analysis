import pandas as pd
import numpy as np
import seaborn as sns
from . import SETTINGS

#data frame with t, id
def matrix_life_mask(times,max_time):
    ar = np.zeros(max_time)
    ar[times] = 1
    return ar

def make_life_mat(df):
    maxt = df.t.dropna().max()+1
    index = []
    mat = []
    for k,g in df.groupby("key"):
        l = list(g.t.dropna().values)
        vals = matrix_life_mask(l, maxt)
        index.append(k)
        mat.append(vals)
    return pd.DataFrame(mat, index=index)

def make_fluc_mat(df):
    return df[["epsilon", "key", "t"]].groupby(["t", "key"])[["epsilon"]].\
    mean().reset_index().pivot("key", "t", "epsilon").fillna(0)
    
    
def blobs_to_life_matrix(location="./cached_data/tracked_blobs.csv"):
    df=pd.read_csv(location, comment='#')

    with open("./cached_data/life_matrix.csv", 'a') as f:
        f.write('# Data from location {}\n'.format(SETTINGS["stack_files"]))
        if(len(df) > 0):
            df.t =df.t+1
            life_mat = make_life_mat(df)
            life_mat.to_csv(f)