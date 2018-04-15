import pandas as pd
import numpy as np
from scipy.spatial import KDTree as scKDTree

#things that are outside a giant cluster that is significant 
#things that are only just spotted and yet to be explained by the transformation
#check second and third order transitins for robustness i.e. robustness against dropped cells

def apply_cluster_id_over_time(df):
    import hdbscan
    for k,g in df.groupby("t"):
        #cluster
        g["cluster"] = None
    #mark giant cluster if exists and is larger than threshold - we can then optionally filter things that are not in it later
    return df

def propose_displacements_2d(b1,b2):
    disps = []
    mat1 = b1[["x", "y"]].as_matrix().tolist()
    mat2 = b2[["x", "y"]].as_matrix().tolist()
    for v1 in mat1:
        for v2 in mat2:
            d = np.array(v2) - np.array(v1)
            disps.append([*list(d), rank_disp(b1,b2,d)])
    return pd.DataFrame(disps, columns=["dx","dy","score"]).sort_values("score")
            
def rank_disp(b1,b2,d, return_pairing=False, epsilon=15):#epsilon should not be bigger than the transform anyway
    #print("testing alignment for shift", str(d))
    b1 = b1.copy().reset_index().drop("index",1)
    b2 = b2.copy().reset_index().drop("index",1)
    mat1 = b1[["x", "y"]].as_matrix() + d
    BTarget = scKDTree(b2[["x", "y"]].as_matrix())
    res=BTarget.query(mat1, k=1,distance_upper_bound=epsilon)
    ar = res[0]
    ids = res[1]
    ids[(np.isinf(ar)) ] = -1
    ar[(np.isinf(ar)) ] = epsilon+1 #default to the boundary, replacing inifinite values
    score = ar.sum().round(4) 
    if not return_pairing: return score
    b1["next"] = ids
    b1["epsilon"] = ar
    
    b2 = b2.join(b1.set_index("next"),rsuffix="prev")
    cols = [c for c in ["key", "x", "y", "z","t", "epsilon", "ffill"] if c in b2]
    return b2[cols], score
    #return b1.reset_index().set_index("next").join(b2,lsuffix="prev_", how='outer'), score

def continue_index(s,seed):
    seed = seed+1
    l  = len(s.dropna())
    return np.array(list(s.dropna().values) + [seed + i for i in range(len(s)-l)]).astype(int)

def get_pairing(blobs1,blobs2,known_key):
    def get_key(df):
        def _get_key(row):
            try:
                if "key" in blobs1.columns and row["key"] >=0:
                    return blobs1[blobs1.index==int(row["index"])].iloc[0]["key"]
            except:
                pass #temp - im tired
            return row["index"]
        return _get_key
            
    disps = propose_displacements_2d(blobs1,blobs2)
    conc = dict(take_concensus(disps, threshold=5))
    tr=np.array([conc["dx"], conc["dy"]])
    pairing = rank_disp(blobs1,blobs2,tr,return_pairing=True)[0].reset_index()
    pairing = pairing.sort_values("key")
    pairing["key"] = continue_index(pairing["key"],known_key)
    return pairing.sort_index(),pairing.key.max()

def take_concensus(scores,threshold=2):
    first_score = scores.iloc[0]["score"]
    return scores[np.abs(scores.score - first_score) < threshold].mean()
    