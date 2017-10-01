import json
import os
import pandas as pd
from matplotlib import pyplot as plt#currently use pyplot to overload the image and blobs - will change

class _SETTINGS(dict):
    """
    This singleton object is used to keep a global pipe context 
    """
    __shared_state = {}
    def __init__(self):
        self.__dict__ = self.__shared_state
        self.current = None
        
SETTINGS = _SETTINGS()

if not os.path.exists("./cached_data/"): os.makedirs("./cached_data/")
    
if not os.path.exists("./settings.json"):
    #create it with defaults
    #print("no settings file, using defaults")
    SETTINGS["stack_files"] = "C:/Users/sirsh/Documents/BioMedia/Todd/020817/Run0142_tp{:0>3}.tif"
    SETTINGS["maxint_files"] = "C:/Users/sirsh/Documents/BioMedia/Todd/020817/Run0142_tp{}_MIP.tif"
    data = json.dumps(SETTINGS)
    with open("./settings.json", "w") as f: f.write(data)

with open("./settings.json") as _f:
    try:
        #print("settings from settings.json")
        SETTINGS = json.load(_f)
        #print(SETTINGS)
    except Exception as ex:
        print("unable to parse the ./settings.json file", repr(ex))
        
from . import blobs
from . import io
from . import track

def get_spark_context():
    pass
    

def process_files(n,save_to="./cached_data/all_blobs.txt", pardo=False):
    if pardo:
        sc = get_spark_context()
        def process(i):
            #use the mounted path for the files
            stack_sample = io.get_stack(i)
            bl = blobs.detect(stack_sample)
            bl["t"] = i
            return bl
        dfs = sc.parallelize(list(range(n))).map(process(i)).collect()
        all_blobs = pd.concat(dfs)
        all_blobs.to_csv(save_to)
        return all_blobs                    
    else:
        all_blobs = []                     
        for i in range(n):
            print(i)
            stack_sample = io.get_stack(i)
            bl = blobs.detect(stack_sample)
            bl["t"] = i
            all_blobs.append(bl)
        all_blobs = pd.concat(all_blobs)
        all_blobs.to_csv(save_to,index=None)
        return all_blobs

def tracks_from_blobs(all_blobs,n,save_plot_loc=None,save_tracks_loc="./cached_data/augmented.txt"):#/"./sample_run6/p"+str(i)+".png"
    blobs1 = all_blobs[all_blobs.t==0]
    augmented_info = []
    blobs1 = blobs1.reset_index().drop("index",1)
    blobs1["key"] = blobs1.index #seed keys with the index (after we auto-increment keys for new cells)
    known_key = blobs1.key.max()
    for i in range(n):   
        print(i)
        augmented_info.append(blobs1.copy())
        blobs2 = all_blobs[all_blobs.t==i]
        blobs2 = blobs2.reset_index().drop("index",1)
        if save_plot_loc != None:
            ax = io.plotimg(io.get_max_int(i), colour_bar=False)
            plt.scatter(x=blobs1.x, y=blobs1.y, c='r', s=10)
        try:
            blobs1,known_key = track.get_pairing(blobs1,blobs2,known_key)
            if save_plot_loc != None:
                plt.scatter(x=blobs1.x, y=blobs1.y, c='g', s=30)
                for k,r in blobs1.iterrows(): plt.annotate(str(int(r["key"])), (r["x"],r["y"]+5),  ha='center', va='top', size=14)
            plt.savefig(save_plot_loc.format(i))
        except Exception as ex:  
            print(repr(ex))
            pass   
        plt.close()
    df=pd.concat(augmented_info)
    df.to_csv(save_tracks_loc,index=None)
    return df

