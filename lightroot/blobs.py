import pandas as pd
#from PIL import ImageDraw,Image as PILI
import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import regionprops
from skimage.filters import gaussian, laplace
from scipy.ndimage import maximum_filter,gaussian_filter,label
from scipy.ndimage.morphology import binary_erosion
from skimage.morphology import watershed
from skimage.filters import threshold_otsu, threshold_adaptive,threshold_li
from skimage import io
from skimage import data, feature
from skimage.draw import circle
from scipy import ndimage
from . import io
import matplotlib.patches as mpatches

from . io import log
try:
    import phasepack
except:
    log("Could not load phasepack which is required for some features. Check that it is installed", mtype="WARN")
    pass

#import warnings
#warnings.filterwarnings('ignore')
from skimage.morphology import erosion

DEFAULT_RANGE=[1,2]
DEFAULT_LOCAL_THRESHOLD = 0.5
DEFAULT_BASE_THRESHOLD = 0.2
DEFAULT_THINNING_THRESHOLD = 0.3
#LOW_BAND_RANGE = [0.05,0.085]

def fft2d_lowpass_and_back(img, win=15):
    f = np.fft.fft2(img)
    # shift the center
    fshift = np.fft.fftshift(f)
    rows, cols = img.shape
    crow,ccol = int(rows/2) , int(cols/2)
    # remove the low frequencies by masking with a rectangular window of size 60x60
    # High Pass Filter (HPF)
    fshift[crow-win:crow+win, ccol-win:ccol+win] = 0
    # shift back (we shifted the center before)
    f_ishift = np.fft.ifftshift(fshift)
    # inverse fft to get the image back 
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    
def crop_by_region(data, region):
    """Given a region e.g something from a set of label, mask data using the index determined by the region"""
    if data is None: return None
    if len(region.bbox) == 6:
        z1,y1,x1, z2,y2,x2 = region.bbox[0],region.bbox[1],region.bbox[2],region.bbox[3],region.bbox[4],region.bbox[5]
        return data[z1:z2, y1:y2,x1:x2]
    else:# im assuming data is always 3d but i could put in more cases
        y1,x1, y2,x2 = region.bbox[0],region.bbox[1],region.bbox[2],region.bbox[3]
        return data[:, y1:y2,x1:x2]

def extract_largest_label(stack_sample,binary,  retain_size=False, out=[], clipIn2d=True):
    """Of all binary labels in the binary layer, find the largest one by volume and then crop the primary image using the largest label as a mask"""
   
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

    output = np.zeros_like(stack_sample)
    mask = sub_markers==lab
    
    if len(mask.shape) == 2 and len(output.shape)==3: 
        log("projecting 2d mask to 3d mask...")
        mask = np.tile(mask,(output.shape[0],1,1))
       
    #print(mask.shape,output.shape, stack_sample.shape)
    output[mask] = stack_sample[mask]
    
    if retain_size: return output
    #clip the part of the output that is fitting the bounding box
    return crop_by_region(output, PP)

def low_pass_2d_proj_root_segmentation(stack, retain_size=False, low_band_range = None, out=[], final_filter=-1):
    """Runs a composite recipe for isolating and clipping a root region"""
    stack_sample = stack.sum(axis=0)#copy is importany because we are creating a mask
    stack_sample /= stack_sample.max()
    low_band_range = [round(p,3) for p in np.percentile(stack_sample, [95, 99, 50])]
    log("using low band range for 2d data from 95,99, 50 data percentile {}".format(low_band_range))
    stack_sample[stack_sample>low_band_range[1]]=low_band_range[1]
    stack_sample[stack_sample<low_band_range[0]]=0
    perc_non_zero = len(np.nonzero(stack_sample)[0])/np.prod(stack_sample.shape)
    stack_sample=gaussian_filter(stack_sample,sigma=8)
    el = extract_largest_label(stack, stack_sample > 0.01,clipIn2d=True,out=out)
    
    el /= el.max()
    
    low_band_range = [round(p,3) for p in np.percentile(el, [95, 99, 80])]
    log("using low band range for 3d data from 95,99, 50 data percentile {}".format(low_band_range))
    
    if final_filter == -1:
        final_filter = low_band_range[1]
        log("using final filter from percentile low band range {}".format(final_filter))                              
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
    
    
def low_pass_root_segmentation(stack, retain_size=False, low_band_range = None, out=[], final_filter=0.3):
    """Runs a composite recipe for isolating and clipping a root region"""
    if low_band_range is None:
        low_band_range = [round(p,3) for p in np.percentile(stack, [95, 99,99])]
        #low_band_range= perc[0:2]
        log("using low band range from 95,98,99 data percentile {}".format(low_band_range))
    stack_sample = stack.copy() 
    stack_sample[stack_sample>low_band_range[1]]=low_band_range[1]
    stack_sample[stack_sample<low_band_range[0]]=0
    perc_non_zero = len(np.nonzero(stack_sample)[0])/np.prod(stack_sample.shape)
    log("analysing image... amount of useable data for masking is {0:.2f}%".format(round(perc_non_zero*100, 2)))
    
    stack_sample=gaussian_filter(stack_sample,sigma=8)
    stack_sample /= stack_sample.max()
    stack_sample[stack_sample<low_band_range[1]] = 0 #remove any glitches after denoising
    stack_sample=gaussian_filter(stack_sample,sigma=8) # merge anything corroded due to filter or any small gaps in the mask
    
    perc_non_zero = len(np.nonzero(stack_sample)[0])/np.prod(stack_sample.shape)
    log("check if we need otsu... Percentage non-zero is {0:.2f}% and we use if greater than 50%".format(round(perc_non_zero*100, 2)))
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

def transform_centroids(centroids, bbox):
    log("Transform back to original ref frame using clipped bounding blox", bbox)
    centroids["y"] += bbox[0][0]
    centroids["x"] += bbox[0][1]
    
    return centroids

def detect(stack,cut_with_low_pass=True,sharpen_iter=1, isolate_iter=1,  isol_threshold=0.125, display_detections=False,do_top_watershed=False,overlay_original_id=None,out=[]):
    """high level function to carry out a detection recipe"""
    #out = []
    if cut_with_low_pass: 
        stack = low_pass_2d_proj_root_segmentation(stack, out=out)
        log("clipped root, offset at {}".format(out))
    sig = 4 #if we are using 3d mode 8 works better
    stack = sharpen(stack, sig = sig, iterations=sharpen_iter)
    overlay = stack.sum(axis=0) if overlay_original_id ==None else io.get_max_int(overlay_original_id)
    
    stack = isolate(stack, iterations=isolate_iter, threshold=isol_threshold)
    #the attempt segment is more complicated and uncessary - the blob_centroids tries to segment and works reasonable well but there were twoo many variants in the data to complete
    #peaks are a safe and easier calculation
    centroids =peak_centroids(stack) # blob_centroids(stack, underlying_image=stack,display=display_detections,do_top_watershed=do_top_watershed)
    
    #if earlier in the process we have clipped and we want to pot on original, now we need to transform back 
    if overlay_original_id != None:  
        #in this mode i am going to mark the square on the overloy
        #io.draw_bounding_box(overlay,out)
        centroids = transform_centroids(centroids, out)
  
    return centroids,overlay # return the best overlay item and the centroids

def sharpen(sample,exageration=1000,sig=4, iterations=1):
    for i in range(iterations):    
        partial= ndimage.gaussian_laplace(sample,sigma=sig)
        partial /= partial.max()
        partial[partial>0] = 0
        sample = sample-partial*exageration 
        sample = sample / sample.max()    
                  
    return sample

def peak_centroids(im, size=10, min_distance=10):
    image_max = maximum_filter(im, size=size, mode='constant')
    coordinates = feature.peak_local_max(im, min_distance=min_distance)
    return pd.DataFrame(coordinates,columns=["z", "y", "x"])

def isolate(partial,resharpen=False,sig_range=DEFAULT_RANGE, threshold=0.125, iterations=1):#thing about threshold - should be adaptive
    perc_non_zero = len(np.nonzero(partial)[0])/np.prod(partial.shape)
    log("sharpening done. percentage non-zero is {0:.2f}%".format(round(perc_non_zero*100, 2)))

    if perc_non_zero > 0.5:
        log("non-zero exceeds 50%, recommend dropping frame as root cannot be isolated. Probably no cells either")
        
    if perc_non_zero > 0.2:
        log("subtacting excessive bottom for noisy data")
        partial[partial<threshold] = 0 # this threshold is coupled to the one we used for hierarchical blobbing
        partial = partial/partial.max() 
            
    
    blobs=dog_blob_detect(partial, sigma_range=sig_range, threshold=threshold, iterations=iterations)
    partial = ndimage.filters.sobel(partial)

    return blobs
    
def erode(im,count=3):
    for i in range(count):   im = erosion(im)
    return im / im.max()

def segment(image, threshold=0.7):
    sharp = image.copy()
    sharp[sharp<threshold] = 0
    sharp[sharp>0] = 1
    distance = ndimage.distance_transform_edt(image)
    local_maxi = feature.peak_local_max(  distance, indices=False, footprint=np.ones((10,10,10)), labels=image)
    markers = ndimage.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=image)
    return labels

def blob_centroids(blobs, 
                   display=False,
                   watch=None, 
                   dt=True, 
                   max_final_ecc=0.95, 
                   min_final_volume=1000,
                   underlying_image=None, 
                   min_bright=2,
                   skip_large_regions=False,
                   do_top_watershed=False,
                   root_offset= []):
    
    """Using a hierarchical decomposition, find blobs"""
    markers = label(blobs)[0]
    centroids = []
    ax= None if not display else io.plotimg(markers,colour_bar=False)

    #do i do it here or not - i dont think so because unless a volume violation we should split
    if do_top_watershed: markers = segment(markers)
    
    for p in _region.collection_from_markers(markers,underlying_image=underlying_image):
        if watch!= None and p.key != watch:continue
        if display: p.show(ax)
           
        if p.volume > 200000 and skip_large_regions:
            log("skipping excessive volume blob range - volume is {}".format(p.volume))
            continue
        #data = weight_by_distance_transform(p.image) if dt else p.image
        data = p.image
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
        
        #if doing watershed - we can split things out a bit
        #sub_markers = segment(sub_markers)
        
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
            if p.volume > min_final_volume and p.TwoDProps.eccentricity < max_final_ecc:
                log("adding big region because no little region found - note the vol of this item is {} and its ecc is {}".format(p.volume, p.TwoDProps.eccentricity))
                centroids.append(p.coords)
        
    log("Found {} centroids".format(len(centroids)))
            
    return pd.DataFrame(centroids,columns = ['z','y','x', 's']).sort_values(["x", "y"])


def dog_blob_detect(ci, sigma_range=[1,2], smoothing_sigma=7, threshold=0.2, iterations=1):
    """emphasise blobs via a differnece of gaussians and a lowerbound threshold. can be run iteratively"""
    g2 = gaussian_filter(ci,sigma=sigma_range[0]) - gaussian_filter(ci,sigma=sigma_range[1])
    g2=gaussian_filter(g2,sigma=smoothing_sigma)
    g2 = g2/g2.max() 
    g2[g2<threshold] = 0
    if iterations > 1: return dog_blob_detect(g2, sigma_range, smoothing_sigma, threshold, iterations-1)
    return g2
 
def contour_isolation(im):
    """deprecated"""
    partial_lap = ndimage.laplace(im)
    partial_lap = im - partial_lap
    partial_lap = partial_lap / partial_lap.max()
    scaled = partial_lap * partial_lap #scale by self to emph
    g = gaussian_filter(scaled,sigma=3) #then smooth out
    g = g/g.max()
    return g


def sharpen_old(img,alpha = 1,thresh=0.2,gaus_sig=2):
    """deprecated"""
    sharpened = img#.copy()
    sharpened[sharpened<thresh]=0 
    blurred_f = gaussian_filter(sharpened, gaus_sig)
    filter_blurred_f = gaussian_filter(blurred_f, 1)
    sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)
    sharpened = sharpened / sharpened.max()
    sharpened[sharpened<thresh]=0 #pre and post threshold - except we get more aggressive
    return sharpened


def root_finger_print(stack):
    low_band_range = [round(p,3) for p in np.percentile(stack, [55  , 99])]
    print(low_band_range)
    stack_sample = stack.copy()#copy is importany because we are creating a mask
    stack_sample[stack_sample>low_band_range[1]]=low_band_range[1]
    stack_sample[stack_sample<low_band_range[0]]=0
    stack_sample=gaussian_filter(stack_sample,sigma=20)
    lightroot.io.plotimg(stack_sample)
    stack_sample[stack_sample < low_band_range[0]] = 0
    lightroot.io.plotimg(stack_sample)
    stack_sample = ndimage.sobel(stack_sample)
    edges = feature.canny(stack_sample.sum(axis=0))
    lightroot.io.plotimg(edges*1000)
    lightroot.io.plotimg(stack_sample)
    

colours = ["red", "green", "blue", "orange"]
class _region:
    """wrapper class for labelled regions to get some hand properties"""
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
    
        