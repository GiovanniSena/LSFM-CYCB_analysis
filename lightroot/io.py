import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from . import SETTINGS
#get fill loc default from sessions for format strings

def get_max_int(i, formatof=None, norm=True):
    #todo - if there are no max int files make them using the stack or return none
    if formatof == None: formatof = SETTINGS["maxint_files"]
    file = formatof.format(i)
    im = io.imread(file)    
    if norm: im = im / im.max()
    return im

def get_stack(i, formatof=None,norm=True):
    if formatof == None: formatof= SETTINGS["stack_files"]
    file = formatof.format(i)
    im = io.imread(file)    
    if norm: im = im / im.max()
    return im

def plotimg(im,default_slice=None,show_intensity=False,colour_bar=True):
    if len(im.shape) == 3:#take a particular slice or max intensity
        plt.figure(figsize=(12,2))
        if show_intensity:plt.plot(im.sum(axis=1).sum(axis=1)/(np.array(im.shape[1:]).sum()))
        im= im.sum(axis=0) if default_slice == None else im[default_slice,:]
    fig = plt.figure(figsize=(20,10))
    ax= plt.imshow(im,)
    if colour_bar:fig.colorbar(ax)
    return ax
    
def plot_blobs_at_t(df,t):
    maxin_sample = get_max_int(t)
    if maxin_sample == None: maxin_sample = get_stack(t)
    blobs=df[df.t==t]
    ax = plotimg(maxin_sample)
    plt.scatter(x=blobs.x, y=blobs.y, c='g', s=30)
    
def overlay_blobs(image, blobs):
    ax = plotimg(image)
    plt.scatter(x=blobs.x, y=blobs.y, c='g', s=30)
    
    