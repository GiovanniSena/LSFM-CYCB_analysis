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
from . import io

#import warnings
#warnings.filterwarnings('ignore')
from skimage.morphology import erosion

DEFAULT_RANGE=[4,10]
DEFAULT_LOCAL_THRESHOLD = 0.7

def detect(stack,sig_range=DEFAULT_RANGE):
    stack = sharpen(stack)
    stack = blob_labels(stack,sig_range=sig_range)
    blobs = blob_centroids(stack)
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

def blob_centroids(markers, display=False,watch=None):
    centroids = []
    ax= None if not display else io.plotimg(markers,colour_bar=False)
    for p in _region.collection_from_markers(markers):
        if watch!= None and p.key != watch:continue
        if display: p.show(ax)
        data = weight_by_distance_transform(p.image)
        blobs = dog_blob_detect(data,sigma_range=DEFAULT_RANGE,threshold=DEFAULT_LOCAL_THRESHOLD)
        sub_markers = label(blobs)[0]  
        
        if len(sub_markers) == 1:
            centroids.append(p.coords)
            continue
        for p2 in _region.collection_from_markers(sub_markers, p):  
            if display: p2.show(ax)
            centroids.append(p2.coords)

    return pd.DataFrame(centroids,columns = ['z','y','x', 's']).sort_values(["x", "y"])
    
    
def blob_centroids_old(markers, vol_threshold=100, local_threshold=0.8,sig_range=DEFAULT_RANGE,
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

def weight_by_distance_transform(img,param=22.5):
    data =  ndimage.distance_transform_edt(img)
    data = data + (22.5 * img) # blob intensity is more important
    data = data / data.max()
    return data


colours = ["red", "green", "blue", "orange"]
class _region:
    
    def __init__(self,r,pr=None,i=None):
        self.r = r
        self.pr = pr
        self._key = i
        self.level = 0
        if self.pr != None:self.level = 1 + self.pr.level
    
    def show(self,ax):
        #if self.volume < 100000:return
        
        minz, minr, minc, maxz, maxr, maxc = self.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor=colours[self.level], linewidth=2)
        ax.add_patch(rect)
        ax.annotate(str(self.key)+"("+ str(self.volume)+")", (minc, minr-10),  ha='center', va='top', size=10,color='orange')
    
    @property
    def bbox(self):
        bbox = np.array(self.r.bbox)
        if self.pr != None:
            offset = np.zeros(len(self.pr.bbox))
            offset[:3] = self.pr.bbox[:3]#first coords
            offset[3:] = self.pr.bbox[:3]#first coords          
            bbox = bbox + offset            
        return list(bbox)
    
    @property
    def radius(self):
        bb= self.r.bbox
        return np.sqrt( (bb[3]-bb[0])**2+(bb[4]-bb[1])**2+(bb[5]-bb[2])**2)
    
    @property
    def volume(self):
        bb= self.r.bbox
        return abs(bb[0]-bb[3])*abs(bb[1]-bb[4])*abs(bb[2]-bb[5])  
    
    @property
    def coords(self):
        bb= self.bbox
        return [(bb[3]+bb[0])/2,(bb[4]+bb[1])/2,(bb[5]+bb[2])/2, self.radius]
    
    @property
    def image_stats(self):
        pass    

    @property
    def image(self):
        img= self.r.image.copy()
        img = img / img.max()
        return img
    
    @property
    def avg(self):
        return round(self.r.image.mean(),2)
    
    @property
    def sum(self):
        return round(self.r.image.sum(),2)
    
    @property
    def key(self):
        return self._key
    
    def __repr__(self):
        return str(self.coords)
    
    @staticmethod
    def collection_from_markers(markers,parent_region=None,having_min_vol=200):
        l = []
        for i, region in enumerate(regionprops(markers)):
            if len(region.bbox) >=6:#validate for volume data - could chnage this to a param
                r = _region(region, parent_region,i)
                if r.volume > having_min_vol:
                    l.append(r)
        return l
    
        