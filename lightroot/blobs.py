import pandas as pd
#from PIL import ImageDraw,Image as PILI
import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import regionprops
from skimage.filters import gaussian, laplace
from scipy.ndimage import maximum_filter,gaussian_filter,label
from scipy.ndimage.morphology import binary_erosion
from skimage.filters import threshold_otsu, threshold_adaptive
from skimage import io
from skimage import data, feature
from skimage.draw import circle
from scipy import ndimage

#import warnings
#warnings.filterwarnings('ignore')
from skimage.morphology import erosion

DEFAULT_RANGE=[4,10]

def detect(stack, local_acceptance=0.8,sig_range=DEFAULT_RANGE):
    sharpened = sharpen(stack)
    markers = blob_labels(sharpened,sig_range=sig_range)
    blobs = blob_centroids(markers,local_threshold=local_acceptance,sig_range=sig_range)
    return blobs

def sharpen(sample,exageration=1000,sig=8):
    partial= ndimage.gaussian_laplace(sample,sigma=sig)
    acceptance = sample.mean() * 5#testing simple adaptive threshold based on "entropy"
    partial = sample-partial*exageration 
    partial = partial / partial.max()
    partial[partial<acceptance] = 0
    return partial

def blob_labels(partial,resharpen=True,sig_range=DEFAULT_RANGE):#[4,8]
    blobs=dog_blob_detect(partial, sigma_range=sig_range)
    if resharpen: blobs = sharpen(blobs)
    markers = label(blobs)[0]
    return markers
    
def erode(im,count=3):
    for i in range(count):
        im = erosion(im)
    return im / im.max()

def blob_centroids(markers, vol_threshold=100, local_threshold=0.8,sig_range=DEFAULT_RANGE,
                   should_plot=False,  use_countour_isolation=True, show_only_first_pass=False):
    centroids = []
    for P in regionprops(markers):
        img = P.image
        img = img / img.max()#SA adding scaling here
        bb = list(P.bbox)
        if len(bb) < 6:continue
        vol = abs(bb[0]-bb[3])*abs(bb[1]-bb[4])*abs(bb[2]-bb[5])
        #print(vol)
        parent_offset =bb[0:3]#these coords are (i think) the offset of this region that need to be "added" 
        r = np.sqrt( (bb[3]-bb[0])**2+(bb[4]-bb[1])**2+(bb[5]-bb[2])**2)
        #this is the centroid of the main blob which is only relevant if there are no subs
        crd = [(bb[3]+bb[0])/2,(bb[4]+bb[1])/2,(bb[5]+bb[2])/2, r]
        #print(vol,crd)
        #if vol > 500000: img = erode(img) #because sometimes the really large ones cause me trouble im eroding them only
        #contour_isolation(img)
        blobs = dog_blob_detect(img.copy(),sigma_range=sig_range, threshold=local_threshold) #local threshold can be aggresive
        #blobs = erode(P.image)
        sub_markers = label(blobs)[0]
        if len(sub_markers) == 1 or show_only_first_pass==True: #we do not need to isolate
            if vol > vol_threshold: centroids.append(crd)
            continue
        #plotimg(blobs)
        for p in regionprops(sub_markers):
            bb = list(p.bbox)
            if len(bb) < 6:continue
            #print(bb)
            offset_crd = [(bb[3]+bb[0])/2,(bb[4]+bb[1])/2,(bb[5]+bb[2])/2, r]
            sub_vol = abs(bb[0]-bb[3])*abs(bb[1]-bb[4])*abs(bb[2]-bb[5])                
            #print(offset_crd)
            offset_Crd = list(np.array(offset_crd)+np.array([*parent_offset,0]))
            #print(":>",offset_Crd)
            if should_plot: plotimg(p.image)
            if sub_vol > vol_threshold: centroids.append(offset_Crd)
    return pd.DataFrame(centroids, columns = ['z','y','x', 's']).sort_values(["x", "y"])

def dog_blob_detect(ci, sigma_range=[4,8], smoothing_sigma=5, threshold=0.01):
    g2 = gaussian_filter(ci,sigma=sigma_range[0]) - gaussian_filter(ci,sigma=sigma_range[1])
    g2=gaussian_filter(g2,sigma=smoothing_sigma)
    g2 = g2/g2.max()
    g2[g2<threshold]=0
    return g2
    
def contour_isolation(im):
    partial_lap = ndimage.laplace(im)
    partial_lap = im - partial_lap
    partial_lap = partial_lap / partial_lap.max()
    scaled = partial_lap * partial_lap #scale by self to emph
    g = gaussian_filter(scaled,sigma=3) #then smooth out
    g = g/g.max()
    return g



def sharpen_old(img,alpha = 1,thresh=0.2,gaus_sig=2):
    sharpened = img#.copy()
    sharpened[sharpened<thresh]=0 
    blurred_f = gaussian_filter(sharpened, gaus_sig)
    filter_blurred_f = gaussian_filter(blurred_f, 1)
    sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)
    sharpened = sharpened / sharpened.max()
    sharpened[sharpened<thresh]=0 #pre and post threshold - except we get more aggressive
    return sharpened
