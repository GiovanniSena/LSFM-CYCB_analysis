import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from skimage import io,draw
from . import SETTINGS
#get fill loc default from sessions for format strings
import time

def log(message,index="", mtype="INFO"): 
    """Log to console or file"""
    index = "" if "current_frame_index" not in SETTINGS else "({})".format(SETTINGS["current_frame_index"])
    
    to_file = False if "log_to_file" not in SETTINGS else SETTINGS["log_to_file"]
    record = "{} {}{}:{}".format(time.strftime('%d/%m/%Y %H:%M:%S'), mtype, index, message)
    
    if to_file:
        with open("./cached_data/log.txt", "a") as f:
            f.write(record+"\n");
    else:  print(record)
    

def get_stack(i, formatof=None,norm=True,convert_and_clip=True):
    """Having set a folder template, load the 3d stack frame by index from that folder"""
    SETTINGS["current_frame_index"] = i
    
    if formatof == None: formatof= SETTINGS["stack_files"]
    file = formatof.format(i)
    im = io.imread(file)    
    if convert_and_clip==True: im = im.astype(int)#[:,300:900,:] #convert and clip  hard code for now
    if norm: im = im / im.max()
    if norm: im = im / im.max()
        
    log("Stack loaded from "+file)
        
    return im

def get_max_int(i, formatof=None, norm=True):
    """Having set a folder template, load the max intensity frame by index from that folder"""
    #todo - if there are no max int files make them using the stack or return none
    if formatof == None: formatof = SETTINGS["maxint_files"]
    file = formatof.format(i)
    try:
        im = io.imread(file)#[300:900,:]
        return im
    except:
        log("failed to find max intensity data forframe {} - using 3d-stack-sum instead".format(i), mtype="WARN")
        return get_stack(i).sum(0)
        
#todo add a generic image reader - simply wraps skimage

def stats(im,ylim=(0,500000), xlim=(0.05,0.8),normed_hist=True):
    """Dump out some statistics for the image"""
    
    import scipy
   
    hist, bin_edges = np.histogram(im, bins=100,normed=normed_hist)
    #print(hist,bin_edges)
    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
    plt.figure(figsize=(20,10))
    plt.subplot(131)
    #todo find sensible limits
    if ylim != None: plt.xlim(*xlim)
    if xlim != None: plt.ylim(*ylim)
    plt.plot(bin_centers, hist, lw=2)
    stack_sample = np.nonzero(im)[0]

    return  {"var" : scipy.ndimage.measurements.variance(stack_sample), 
             "min" : np.min(stack_sample),
             "max" : np.max(stack_sample),
            "mean" : scipy.ndimage.measurements.mean(stack_sample),
            "median" : scipy.ndimage.measurements.median(stack_sample),
            "snr" : scipy.stats.signaltonoise(stack_sample, axis=None),
            "sum" : scipy.ndimage.measurements.sum(stack_sample),
            "95_99Percentiles" : np.percentile(stack_sample, [95, 99])
            }

def show_xy_intensity(in_tile,i=None):
    #todo make this do xy and contour 
    plt.figure(figsize=(10,6))
    
    tile = in_tile if len(in_tile.shape) == 2 else in_tile.sum(axis=0)
    
    X = tile.sum(axis=0)
    X /= X.max()
    Y = tile.sum(axis=1)
    Y /= Y.max()
    plt.plot(X)
    plt.plot(Y)
    if i != None:
        save_plot_loc = "./sample_data/{:0>3}.png"
        plt.savefig(save_plot_loc.format(i))
        plt.close()
        
#add gradient display
#slab = stack.sum(0)
#grad = np.gradient(slab)[0]

def plotimg(im,default_slice=None,show_intensity=False,colour_bar=False):
    """Plot the image. If 3D plot the projection onto 3d by sunmming on axis 0"""
    if len(im.shape) == 3:#take a particular slice or max intensity
        plt.figure(figsize=(12,2))
        if show_intensity:plt.plot(im.sum(axis=1).sum(axis=1)/(np.array(im.shape[1:]).sum()))
        im= im.sum(axis=0) if default_slice == None else im[default_slice,:]
    #fig = plt.figure(figsize=(20,10))
    #ax= plt.imshow(im,)
    
    #return ax
    fig,ax = plt.subplots(1,figsize=(20,10))
    ax.imshow(im)
    #if colour_bar:fig.colorbar(ax)
    #if colour_bar:fig.colorbar(ax)
    return ax
    
def draw_bounding_box(image, bbox_coords_2d):
    r0 = bbox_coords_2d[0][0]
    c0 = bbox_coords_2d[0][1]
    r1 = bbox_coords_2d[0][2]
    c1 = bbox_coords_2d[0][3]
    border = 0
    rr,cc = draw.polygon_perimeter([r1+border, r0-border, r0-border, r1+border], 
                                   [c0-border, c0-border, c1+border, c1+border], 
                                   shape=image.shape, clip=False)
    image[rr, cc] = 200
    
def __has_ffill__(df):
    if "ffill" in df: return df["ffill"].astype(int).max() == 1
    return False
    
def overlay_blobs(image, blobs, rect=None):
    """scatter blobs from a dataframe t,x,y,z over the image"""
    ax = plotimg(image)
    
    if __has_ffill__(blobs): ax.text(30, 30, 'Raw Frame Missing Here! Using last good one',color='red', fontsize=15)

    if rect != None: 
        ax.add_patch( patches.Rectangle(  list(reversed(rect[0][0:2])),   rect[0][3]-rect[0][1],   rect[0][2]-rect[0][0],  fill=False, edgecolor='white' ))
    
    plt.scatter(x=blobs.x, y=blobs.y, c='white', s=30)
    
def plot_3d(image_in):
    from mpl_toolkits.mplot3d import Axes3D
    image = image_in if len(image_in.shape) <= 2 else image_in.sum(axis=0)       
    X = np.arange(0, image.shape[1], 1)
    Y = np.arange(0, image.shape[0], 1)
    X, Y = np.meshgrid(X, Y)
    Z = image
    fig = plt.figure(figsize=(20,10))
    ax = fig.gca(projection='3d')
    ax.set_axis_off()
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot', linewidth=0, antialiased=False)

def plot_blobs_at_t(df,t):
    maxin_sample = get_max_int(t)
    if maxin_sample == None: maxin_sample = get_stack(t)
    blobs=df[df.t==t]
    ax = plotimg(maxin_sample)
    plt.scatter(x=blobs.x, y=blobs.y, c='g', s=30)

def tile(s,x,y,divs=4,display=False):
    if x >= divs or y >= divs: raise Exception("out of range on tile request")
    if len(s.shape) > 2: s = s.sum(axis=0)
    L = (np.array(s.shape)/divs).astype(int)
    x1,x2 = x*L[1],x*L[1]+L[1]   
    y1,y2 = y*L[0],y*L[0]+L[0] 
    if x+1 == divs: x2=-1
    if y+1 == divs: y2=-1
    t =  s[y1:y2,x1:x2 ]
    if display: plotimg(t)
    return t

def tiling(data,divs=10, cmap=None):
    f, axarr = plt.subplots(divs,divs,figsize=(15,11.5))
    plt.axis('on')
    for x in range(divs):
        for y in range(divs):
            tile1 = tile(data,x,y,divs=divs)
            AX = axarr[y,x]
            AX.set_xticklabels([])
            AX.set_yticklabels([])
            text = "({},{})".format(x,y)
            if cmap != None: AX.imshow(tile1,cmap)  
            else:  AX.imshow(tile1)
            AX.text(20, 50, text, color='white', fontsize=14)
                
    plt.subplots_adjust(wspace=0, hspace=0)
    
    
def tile_function_plot(data, func, divs=10):
    f, axarr = plt.subplots(divs,divs,figsize=(15,11.5))
    plt.axis('on')#
    
    for x in range(divs):
        for y in range(divs):
            tile1 = tile(data,x,y,divs=divs)
    
            AX = axarr[y,x]
            AX.set_xticklabels([])
            AX.set_yticklabels([])
            
            try:
                tile1, text = func(tile1)
                
                #counts = dict(zip(*np.histogram(res[1].flatten())))
                #angle = counts[np.array(list(counts.keys())).max() ]
                #keys = sorted(list(counts.keys()))
                #top_keys = "{},{}".format(int(counts[keys[-2]]),int(counts[keys[-1]]))
                
                AX.imshow(tile1,'Spectral')    
                AX.text(20, 50, text, color='white', fontsize=14)
            except:
                continue
                
    plt.subplots_adjust(wspace=0, hspace=0)

