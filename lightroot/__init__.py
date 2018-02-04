import json
import os
import pandas as pd
from matplotlib import pyplot as plt#currently use pyplot to overload the image and blobs - will change
from datetime import datetime
import time
from glob import glob
import pathlib
using_tqdm = False
try:
    import tqdm
    import time
    using_tqdm = True
except: print("unable to load tqdm. you should 'pip install tqdm' for a better experience")
    
class _SETTINGS(dict):
    """
    This singleton object is used to keep a global pipe context 
    """
    __shared_state = {}
    def __init__(self):
        self.__dict__ = self.__shared_state
        self.current = None  
        self.current_frame_index = None
        
        
SETTINGS = _SETTINGS()
SETTINGS["current_frame_index"] = None

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


#################
###Primary
#################

def determine_file_count(format_path, nmax_files=100000):
    import os
    for i in range(nmax_files):
        if not os.path.isfile(format_path.format(i)):
            return i
        
def infer_formats(path_to_exp = "C:/Users/mrsir/Box Sync/uncut/images/310717/", token="_tp"):
    try:
        io.log("inferring file formats...")
        from glob import glob
        from os.path import basename
        import os
        from lightroot import SETTINGS #or .
        f0 = glob(path_to_exp+"/*.tif")[-1]
        f = basename(f0).split(".")[0]
        prefix = f[:f.index(token)+len(token)]
        stack_f = os.path.join(path_to_exp, prefix+ "{:0>3}.tif")
        maxint_f = os.path.join(path_to_exp, prefix+ "{}_MIP.tif")
        SETTINGS["stack_files"] = stack_f
        SETTINGS["maxint_files"] = maxint_f

        if not os.path.isfile(stack_f.format(0)):
            raise Exception("The format does not work - cannot find a file matching "+stack_f.format(0))
        return stack_f, maxint_f
             
    except Exception as ex:
        print (ex)
        print("Hence we are unable to infer file formats from the default cenvention.")
        print("You will need to enter the formats manually.")
        print("Either reconfigure convention or supply format for stack & max intensity files using the settings.json example.")
        
        
def process(folder,infer_file_formats=True,log_to_file=True, limit_count=None):
    save_plot_loc = "./cached_data/{}.png"
    SETTINGS["log_to_file"] = log_to_file
    if infer_file_formats: 
        infer_formats(folder)
    count = limit_count if limit_count != None else determine_file_count(SETTINGS["stack_files"])
    
    path = "./cached_data/"
    if os.listdir(path) != []: 
        if  input("The directory "+path+" should be empty. Do you want to clear it? (y/n)").lower() == "y":
            for i in glob(path+"*.*"): os.remove(i)
    
    if log_to_file: 
        print("Running lightroot. See log in cache_output for details...\n")
        using_tqdm = True
        
    io.log("Processing {} files in directory {}".format(count, SETTINGS["stack_files"]))

    tracks = []
    blobs_last = None
    known_key = None
    iterator = [i for i in range(count)]
    if using_tqdm: iterator =tqdm.tqdm(iterator) 
        
    for i in iterator:  
        stack = io.get_stack(i)
        #comes down to iterations and thresholds after we are given a clipped frame denoised - extract good parameters for every video
        current_blobs,stack = blobs.detect(stack, isol_threshold=0.125)
        current_blobs["t"] = i
        ax = io.overlay_blobs(stack,current_blobs)

        if i==0:
            current_blobs["key"] = current_blobs.index
            blobs_last = current_blobs
            known_key = current_blobs.key.max()
        else:
            try:
                plt.scatter(x=blobs_last.x, y=blobs_last.y, c='r', s=30)
                blobs_last,known_key = track.get_pairing(blobs_last,current_blobs,known_key)
                tracks.append(blobs_last.copy())
                pd.concat(tracks).drop("index",1).set_index("t").to_csv("./cached_data/tracked_blobs.cpt")
                for k,r in blobs_last.iterrows(): plt.annotate(str(int(r["key"])), (r["x"],r["y"]+5),  ha='center', va='top', size=14)
            except Exception as ex:  
                io.log(repr(ex),mtype="ERROR")

        plt.savefig(save_plot_loc.format(i))
        plt.close()
    io.log("writing final tracks")
    
    pd.concat(tracks).drop("index",1).set_index("t").to_csv("./cached_data/tracked_blobs.csv")
    os.remove("./cached_data/tracked_blobs.cpt")
    io.log("Done.")

################
####Sec
################

def get_spark_context():  pass
       
def _process_files_get_blobs(n,save_to="./cached_data/all_blobs.txt", pardo=False):
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
            bl,stack = blobs.detect(stack)
            #io.overlay_blobs(stack,blob_centroids)

            bl["t"] = i
            all_blobs.append(bl)
        all_blobs = pd.concat(all_blobs)
        all_blobs.to_csv(save_to,index=None)
        return all_blobs

def tracks_from_blobs(all_blobs,n,save_plot_loc=None,black_list = [],
                      save_tracks_loc="./cached_data/augmented.txt"):#/"./sample_run6/p"+str(i)+".png"
    blobs1 = all_blobs[all_blobs.t==0]
    augmented_info = []
    blobs1 = blobs1.reset_index().drop("index",1)
    blobs1["key"] = blobs1.index #seed keys with the index (after we auto-increment keys for new cells)
    known_key = blobs1.key.max()
    for i in range(n):   
        print(i)
        if i in black_list:continue# skip this frame because it contains unusable data - do we need to think about t
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

io.log("loaded lightroot")