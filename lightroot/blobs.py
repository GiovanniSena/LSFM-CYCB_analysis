import pandas as pd
#from PIL import ImageDraw,Image as PILI
import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import regionprops
from skimage.filters import gaussian, laplace
from scipy.ndimage import maximum_filter,gaussian_filter,label
from scipy.ndimage.morphology import binary_erosion
from skimage.filters import threshold_otsu, threshold_adaptive,threshold_li
from skimage import io
from skimage import data, feature
from skimage.draw import circle
from scipy import ndimage
from . import io
import matplotlib.patches as mpatches

from . io import log
#import warnings
#warnings.filterwarnings('ignore')
from skimage.morphology import erosion

DEFAULT_RANGE=[1,2]
DEFAULT_LOCAL_THRESHOLD = 0.5
DEFAULT_BASE_THRESHOLD = 0.2
DEFAULT_THINNING_THRESHOLD = 0.3
#LOW_BAND_RANGE = [0.05,0.085]

def crop_by_region(data, region):
    if data is None: return None
    z1,y1,x1, z2,y2,x2 = region.bbox[0],region.bbox[1],region.bbox[2],region.bbox[3],region.bbox[4],region.bbox[5]
    return data[z1:z2, y1:y2,x1:x2]

def extract_largest_label(stack_sample,binary,  retain_size=False, out=[]):
    sub_markers = label(binary)[0]
    largest=0
    lab = 0
    rpr =  regionprops(sub_markers)
    PP = None
    for c, p in enumerate(rpr):
        if p.area > largest:
            largest = p.area
            lab = p.label
            PP = p
    
    out.append(PP.bbox)
    #create a new image
    output = np.zeros_like(stack_sample)
    output[sub_markers==lab] = stack_sample[sub_markers==lab]
    if retain_size: return output
    #clip the part of the output that is fitting the bounding box
    return crop_by_region(output, PP)

def low_pass_root_segmentation(stack, retain_size=False, low_band_range = None, out=[], final_filter=0.3):
    if low_band_range is None:
        low_band_range = [round(p,3) for p in np.percentile(stack, [95, 99,99])]
        #low_band_range= perc[0:2]
        log("using low band range from 95,98,99 data percentile {}".format(low_band_range))
    stack_sample = stack.copy()#copy is importany because we are creating a mask
    stack_sample[stack_sample>low_band_range[1]]=low_band_range[1]
    stack_sample[stack_sample<low_band_range[0]]=0
    perc_non_zero = len(np.nonzero(stack_sample)[0])/np.prod(stack_sample.shape)
    log("analysing image... amount of useable data for masking is {0:.2f}%".format(round(perc_non_zero*100, 2)))
    
    stack_sample=gaussian_filter(stack_sample,sigma=8)
    stack_sample /= stack_sample.max()
    stack_sample[stack_sample<low_band_range[1]] = 0 #remove any glitches after denoising
    stack_sample=gaussian_filter(stack_sample,sigma=8) # merge anything corroded due to filter or any small gaps in the mask
    
    perc_non_zero = len(np.nonzero(stack_sample)[0])/np.prod(stack_sample.shape)
    log("check if we need otsu... Percentage non zero is {0:.2f}% and we use if greater than 50%".format(round(perc_non_zero*100, 2)))
    thresh = threshold_otsu(stack_sample) if perc_non_zero > .50 else low_band_range[0]
    el = extract_largest_label(stack, stack_sample > thresh,retain_size,out)
    
    el[el<low_band_range[1]]=0 #not essential but suggests that it is the low band
    el /= el.max()
    
    shine = len(np.nonzero(el[el>final_filter])[0])
    #reduce shine
    log("checking shine @ {0:.2f}".format(shine))
    if shine > 25000:#considered to be too bright - purely heuristic for now
        log("bright frame detected. removing bottom", mtype="WARN")
        el[el<final_filter]=0 
        el/= el.max()
        
    perc_non_zero = len(np.nonzero(el)[0])/np.prod(el.shape)
    log("extracted root region with volume {0} with non-zero {1:.2f}%".format(np.prod(el.shape),round(perc_non_zero*100, 2)))
    
    return el

def detect(stack,cut_with_low_pass=True,sharpen_iter=2, isolate_iter=2,  isol_threshold=0.125, display_detections=False):
    out = []
    if cut_with_low_pass: 
        stack = low_pass_root_segmentation(stack, out=out)
        log("clipped root, offset at {} using pi times low_band upper".format(out))
    stack = sharpen(stack, iterations=sharpen_iter)
    overlay = stack.sum(axis=0)
    stack = isolate(stack, iterations=isolate_iter, threshold=isol_threshold)
    centroids = blob_centroids(stack, underlying_image=stack,display=display_detections)
  
    #offset correction here - move both the box and the coords
    
    return centroids,overlay # return the best overlay item and the centroids

def sharpen(sample,exageration=1000,sig=8, iterations=1):
    for i in range(iterations):    
        partial= ndimage.gaussian_laplace(sample,sigma=sig)
        partial /= partial.max()
        partial[partial>0] = 0
        sample = sample-partial*exageration 
        sample = sample / sample.max()    
                  
    return sample

def isolate(partial,resharpen=False,sig_range=DEFAULT_RANGE, threshold=0.125, iterations=1):#thing about threshold - should be adaptive
    perc_non_zero = len(np.nonzero(partial)[0])/np.prod(partial.shape)
    log("sharpening done. percentage non zero is {0:.2f}%".format(round(perc_non_zero*100, 2)))

    if perc_non_zero > 0.5:
        log("non-zero exceeds 50%, recommend dropping frame as root cannot be isolated. Probably no cells either")
        
    if perc_non_zero > 0.2:
        log("subtacting excessive bottom for noisy data")
        partial[partial<threshold] = 0 # this threshold is coupled to the one we used for hierarchical blobbing
        partial = partial/partial.max() 
            
    blobs=dog_blob_detect(partial, sigma_range=sig_range, threshold=threshold, iterations=iterations)

    return blobs
    
def erode(im,count=3):
    for i in range(count):   im = erosion(im)
    return im / im.max()

def blob_centroids(blobs, 
                   display=False,
                   watch=None, 
                   dt=True, 
                   max_final_ecc=0.95, 
                   min_final_volume=1000,
                   underlying_image=None, 
                   min_bright=2,
                   skip_large_regions=False,
                   root_offset= []):
    
    markers = label(blobs)[0]
    centroids = []
    ax= None if not display else io.plotimg(markers,colour_bar=False)

    for p in _region.collection_from_markers(markers,underlying_image=underlying_image):
        if watch!= None and p.key != watch:continue
        if display: p.show(ax)
           
        if p.volume > 200000 and skip_large_regions:
            log("Skipping excessive volume blob range - volume is {}".format(p.volume))
            continue
        data = weight_by_distance_transform(p.image) if dt else p.image
        blobs = dog_blob_detect(data,sigma_range=DEFAULT_RANGE,threshold=DEFAULT_LOCAL_THRESHOLD)
        sub_markers = label(blobs)[0]  
        
        #im commenting this out because i want to see everything go via the final filter
        #but presume it could fail the next test which means maybe we want to just add it now?
        #MAKESE SENSE THERE MIGHT BE DIFFERENT RULES ON FIRST PASS - THESE COULD BE THE DEFAULTS OF COLLECTION FUNC - SCALE AGNOSTIC CHECKS
        if len(sub_markers) == 1: # and some condition such as detection of roundness 
            log("adding large one with props", p.TwoDProps.eccentricity)
            centroids.append(p.coords)
            continue
            
        if len(sub_markers) == 0:
            log("failed to resolve blob in region")
            centroids.append(p.coords)
            continue
            
        found = False
        #now we are checking for eccentricity etc.
        for ct, p2 in enumerate(_region.collection_from_markers(sub_markers, p,
                                                  having_min_vol =min_final_volume, 
                                                  having_max_eccentricity=max_final_ecc, 
                                                  underlying_image=underlying_image,
                                                 min_bright = min_bright)):  
            if display: p2.show(ax)
            found = True
            centroids.append(p2.coords)
        
        if not found:##ading the big one because we found nothing - although, i need to condition - if the area is not rediculous
            #print("adding big region because no little region found")
            centroids.append(p.coords)
        
    log("Found {} centroids".format(len(centroids)))
            
    return pd.DataFrame(centroids,columns = ['z','y','x', 's']).sort_values(["x", "y"])


def dog_blob_detect(ci, sigma_range=[1,2], smoothing_sigma=7, threshold=0.2, iterations=1):
    g2 = gaussian_filter(ci,sigma=sigma_range[0]) - gaussian_filter(ci,sigma=sigma_range[1])
    g2=gaussian_filter(g2,sigma=smoothing_sigma)
    g2 = g2/g2.max() 
    g2[g2<threshold] = 0
    if iterations > 1: return dog_blob_detect(g2, sigma_range, smoothing_sigma, threshold, iterations-1)
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
    
    def __init__(self,r,pr=None,i=None,raw_data=None):
        self.r = r
        self.pr = pr
        self._key = i
        self.level = 0
        self.raw_data = raw_data
        if self.pr != None:self.level = 1 + self.pr.level
    
    def show(self,ax):
        #if self.volume < 100000:return
        
        minz, minr, minc, maxz, maxr, maxc = self.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor=colours[self.level], linewidth=2)
        ax.add_patch(rect)
        ax.annotate(str(self.key), (minc, minr-10),  ha='center', va='top', size=10,color='orange')
    
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
    def brightness(self):
        if self.raw_data is not None:
            self.raw_data.sum()/self.volume * 100
        return 1000# for now im not going to discriminate
    
    @property
    def max_intensity(self):
        if self.raw_data is not None:
            return self.raw_data.max()
        return 1000
    
    @property
    def TwoDProps(self):
        im = self.r.image.sum(axis=0)
        return regionprops(label(im)[0])[0] #return the first property as assumed one, also notice the offset for the API on the label result
        
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
    def collection_from_markers(markers,parent_region=None,having_min_vol=250, having_max_eccentricity=1.0,underlying_image=None, min_bright=0):
        l = []
        for i, region in enumerate(regionprops(markers)):
            if len(region.bbox) >=6:#validate for volume data - could chnage this to a param
                r = _region(region, parent_region,i, raw_data=crop_by_region(underlying_image, region))
                
                if r.brightness < min_bright:continue ##not round enough from a 2d perspective
                
                try:
                    if r.TwoDProps.eccentricity > having_max_eccentricity:continue ##not round enough from a 2d perspective
                except:
                    pass
                
                if r.volume > having_min_vol:
                    l.append(r)
        return l
    
        