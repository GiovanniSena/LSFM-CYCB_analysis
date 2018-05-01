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
from . import lineage


#################
###Primary
#################

def determine_file_count(format_path, nmax_files=100000, max_skip=50):
    import os
    missing = 0
    last_good = 0
    for i in range(nmax_files):
        if not os.path.isfile(format_path.format(i)):
            missing+=1
            if missing  >= max_skip: return last_good
        else:
            missing = 0#contiguous we care about
            last_good = i
    return last_good
        
def infer_formats(path_to_exp = "C:/Users/mrsir/Box Sync/uncut/images/310717/", token="_tp"):
    try:
           
        from glob import glob
        from os.path import basename
        import os
        from lightroot import SETTINGS #or .
        path_to_exp = path_to_exp.replace("\\", "/").replace("\"", "")
        search = path_to_exp+"/*.tif"
        print("searching",search )
        f0 = glob(search)[-1]
        f = basename(f0).split(".")[0]
        prefix = f[:f.index(token)+len(token)]
        stack_f = os.path.join(path_to_exp, prefix+ "{:0>3}.tif")
        maxint_f = os.path.join(path_to_exp, prefix+ "{}_MIP.tif")
        SETTINGS["stack_files"] = stack_f
        SETTINGS["maxint_files"] = maxint_f

        found_good = -1
        for i in range(50):#50 is the max gap value
            if os.path.isfile(stack_f.format(i)):
                found_good = i
                break
        
        if found_good == -1: raise Exception("The format does not work - cannot find a file matching "+stack_f.format(found_good))
        return stack_f, maxint_f
             
    except Exception as ex:
        print (ex, "- hence we are unable to infer file formats from the default cenvention.")
        print("You will need to enter the formats manually.")
        print("Either reconfigure convention or supply format for stack & max intensity files using the settings.json example.")
        raise ex
        
        
def process(folder,infer_file_formats=True,log_to_file=True, limit_count=None):
    
    save_plot_loc = "./cached_data/{:0>3}.png" #
    SETTINGS["log_to_file"] = log_to_file
    if infer_file_formats==True: 
        try:  infer_formats(folder)
        except: 
            print("FATAL ERROR: Unable to determine file formats - check that you have used a valid path")
            return
            
    path = "./cached_data/"
    chp_file = "./cached_data/tracked_blobs.cpt"
    
    if os.listdir(path) != []: 
        if  input("The directory "+path+" should be empty. Do you want to clear it? (y/n)").lower() == "y":
            for i in glob(path+"*.*"): os.remove(i)
    
    if log_to_file: 
        print("\nRunning lightroot. See log in cache_output for details...\n")
        using_tqdm = True
        
    count = limit_count if limit_count != None else determine_file_count(SETTINGS["stack_files"])
    io.log("processing {} files in directory {}".format(count, SETTINGS["stack_files"]))

    #loop state variables
    tracks = []
    old_out = []
    blobs_last = None
    known_key = None
    stack,current_blobs = None,None
    loaded_frames = -1
        
    iterator = [i for i in range(count)]
    if using_tqdm: iterator =tqdm.tqdm(iterator) 
    
    for i in iterator:
        out = []#
        try:
            stack = io.get_stack(i)
            current_blobs,stack = blobs.detect(stack,overlay_original_id=i,out=out)#, isol_threshold=0.125
            loaded_frames+=1
            old_out = list(out)#copy the state of overlay
        except:
            out = list(old_out)#i copy this because we need to use something for the overlay
            if stack is None: #if we have never seen a good stack, we can not do much
                io.log("waiting for first good frame...", mtype="WARN")
                continue
            io.log("failed to load next stack - filling forward",mtype="WARN")
            #if we have a good frame from before, we are going to just pretend that was our last but tell the user
            current_blobs["ffill"] = True
        current_blobs["t"] = i
        ax = io.overlay_blobs(stack,current_blobs,out)#<-use ffil to leave a message

        if loaded_frames==0: #was i==0 check but maybe the first one is not the one
            current_blobs["key"] = current_blobs.index
            blobs_last = current_blobs
            known_key = current_blobs.key.max()
        else:
            try:
                plt.scatter(x=blobs_last.x, y=blobs_last.y, c='r', s=30)
                blobs_last,known_key = track.get_pairing(blobs_last,current_blobs,known_key)
                tracks.append(blobs_last.copy())
                pd.concat(tracks).drop("index",1).set_index("t").to_csv(chp_file)
                for k,r in blobs_last.iterrows(): plt.annotate(str(int(r["key"])), (r["x"],r["y"]+5),  ha='center', va='top', size=14)
            except Exception as ex:  
                io.log(repr(ex),mtype="ERROR")

        plt.savefig(save_plot_loc.format(i))
        plt.close()
        
    io.log("writing final tracks")
    
    blob_file = "./cached_data/tracked_blobs.csv"
    with open(blob_file, 'a') as f:
        f.write('# Data from location {}\n'.format(SETTINGS["stack_files"]))
        if len(tracks) > 0: pd.concat(tracks).drop("index",1).set_index("t").to_csv(f)
        
    io.log("writing life matrix")
    lineage.blobs_to_life_matrix(blob_file)
    
    if os.path.isfile(chp_file): os.remove(chp_file)
    io.log("done.")

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