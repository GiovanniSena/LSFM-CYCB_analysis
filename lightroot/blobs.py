#trying to keep track of important params
# blob sigma range of 8 and 10 seems good for the cells we see
#the find_threshold threshold of 0.08 seems best ot pass through the detect wrapper -  i need it to be as high as possible without admitting junk e.g. 198 on the april cut 
# actually - it depends on uncertainity i.e. clipping region size. If we are clipping properly, it is probably good info. otherwise, SNR is bad and we sould sub out the bottom agressively
#peak local max should use a size of 10 and a min of 5

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
import warnings
warnings.filterwarnings("ignore")
from . io import log
try:
    import phasepack
except:
    log("Could not load phasepack which is required for some features. Check that it is installed", mtype="WARN")
    pass

from skimage.restoration import (denoise_wavelet, estimate_sigma)
from skimage import data, img_as_float
from skimage.util import random_noise
from skimage.measure import compare_psnr

NOISE_UPPER = 0.045

def denoise(stack,above=0.01):
    sigma_est = estimate_sigma(stack, multichannel=False, average_sigmas=True)
   
    log("estimated gaussian noise standard deviation = {}".format(sigma_est))
    if sigma_est> above and sigma_est < NOISE_UPPER: #no point denoising if its too noisy perhaps??
        log("estimated standard deviation of {} exceeds {} so we need to denoise. takes a moment...".format(sigma_est, above))
        return denoise_wavelet(stack, multichannel=False,  mode='soft'),sigma_est
    return stack, sigma_est

#import warnings
#warnings.filterwarnings('ignore')
from skimage.morphology import erosion

DEFAULT_RANGE=[1,2]
DEFAULT_LOCAL_THRESHOLD = 0.5
DEFAULT_BASE_THRESHOLD = 0.2
DEFAULT_THINNING_THRESHOLD = 0.3
DEFAULT_MAX_CELLS_ALLOWED = 100
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

def filter_isolated_cells(array, struct=np.ones((3,3,3)),min_size=500):
    """ Return array with completely isolated single cells removed
    :param array: Array with completely isolated single cells
    :param struct: Structure array for generating unique regions
    :return: Array with minimum region size > 1
    """
    filtered_array = array# np.copy(array) #im not copying becase im just going for it
    id_regions, num_ids = ndimage.label(filtered_array, structure=struct)
    id_sizes = np.array(ndimage.sum(array, id_regions, range(num_ids + 1)))
    area_mask = (id_sizes < min_size)
    filtered_array[area_mask[id_regions]] = 0
    return filtered_array,id_sizes

def find_threshold(im, gap=0.1,threshold=0.005):
    im = im.copy()
    #im /= im.sum()
    im /= im.max()
    cuts = np.arange(0.1,1.0,gap)
    vals = np.array([np.count_nonzero(im[im>t]) for t in cuts],dtype=np.float)
    df = pd.DataFrame(vals, columns=["v"])
    df["s"] = df["v"] / df["v"].sum()
    df["d"] = df["s"].diff(-2).fillna(method='ffill') 
    vals = list(df["d"].values)
    #algorithm wait till difference is greater than 1, then less then 1
    waiting = False
    for i,v in enumerate(vals):
        if v > threshold: waiting = True
        if waiting == True and v < threshold: return cuts[i]
    #this is very aggressive, it says if we did not find a cut point we are removing almost everything
    return cuts[-1]
       
def label_distribution(img):
    l = label(img.sum(0))[0]
    for r in regionprops(l):
        if r.perimeter != 0:yield r.perimeter
    
    
def find_level_and_dtrans(stack,min_perim=1000,max_perim=2000,mask_threshold=0.1, tmax=0.3):
    waiting=False
    
    for t in np.arange(0.01,tmax, 0.01):
        g2 = stack.copy()
        g2 /= g2.max()
        g2[g2<t]=0
        props = np.array(list(label_distribution(g2)))
        m = props.max()
        #print(m)
        if m > max_perim and not waiting: waiting=True
        if waiting and m < min_perim: 
            log("setting the clipped region adaptive threshold to {0:.2f} based on maximum label perimeters".format(t))
            g2 /=g2.max()
            mask = g2.copy()
            mask[mask<mask_threshold] = 0
            mask[mask>0] = 1
            #i could use just the distance transform i could weight with it
            return  ndimage.distance_transform_edt(g2) #* g2
        
    log("did not find a clipped region adaptive threshold based on maximum label perimeters - returning image as is")
    return stack

def low_pass_2d_proj_root_segmentation(stack, retain_size=False, low_band_range = None, out=[], final_filter=-1, find_threshold_val=0.2, 
                                       remove_specks_below=500,return_early=True,denoise_img=True, override_return_early_noise_thresh=NOISE_UPPER):
    """Runs a composite recipe for isolating and clipping a root region"""     
    stack,noise = denoise(stack) #denoise stack sampe
    stack_sample = stack.sum(axis=0)#copy is importany because we are creating a mask
    stack_sample /= stack_sample.max()
    low_band_range = [round(p,3) for p in np.percentile(stack_sample, [95, 99, 50])]
    
    if low_band_range[1] > 0.75: #this is crazy birhgt e.g. uncut 098..103
        stack_sample[stack_sample<0.5] = 0
        log("seems to be way too much light - these points cannot be trusted - change early return threshold")
        override_return_early_noise_thresh = 0
        
    
    log("using low band range for 2d data from 95,99, 50 data percentile {}".format(low_band_range))
    stack_sample[stack_sample>low_band_range[1]]=low_band_range[1]
    stack_sample[stack_sample<low_band_range[0]]=0
    perc_non_zero = len(np.nonzero(stack_sample)[0])/np.prod(stack_sample.shape)
    stack_sample=gaussian_filter(stack_sample,sigma=8)
    el = extract_largest_label(stack, stack_sample > 0.01,clipIn2d=True,out=out)
    
    el /= el.max()
    
    #doesnt make sense to denoise twice 
    #if denoise_img: el = denoise(el)
    
    el = find_level_and_dtrans(el)
    
    if return_early and not noise > override_return_early_noise_thresh: #sometimes we are so noisy we want to subtract the bottom out of it -so dont return
        return el
    
    low_band_range = [round(p,3) for p in np.percentile(el, [95, 99, 80])]
    log("using low band range for 3d data from 95,99, 50 data percentile {}".format(low_band_range))
    
    if final_filter == -1:
        final_filter = low_band_range[1]
        log("using final filter from percentile low band range {}".format(final_filter))                              
    shine = len(np.nonzero(el[el>final_filter])[0])
    #reduce shine
    log("checking shine @ {0:.2f}".format(shine))
    
    if shine < 100: #this is basically degenerate
        log("blocking degenerate frame - there are no non-zero cells after filtration".format(shine))
        el[el>0] = 0
        return el
    
    if shine > 0:#now it is purely noise dependent
        #log("bright frame detected. removing bottom agressively", mtype="WARN")
        if np.prod(el.shape) > 50000000:
            log("because clipping was not terribly successfull, I will reduce the threshold value to the low value of 0.08", mtype="WARN")
            find_threshold_val = 0.08
     
        th = find_threshold(el,threshold = find_threshold_val)
        
        if final_filter > th:
            log("switching adaptive threshold for final threshold as it is too small @ {0:.2f} < {1:.2f} ".format(th,final_filter ))
            th = final_filter
        
        log("applying adaptive cut threshold @ {0:.2f}".format(th))
        el[el<th] = 0
        perc_non_zero = len(np.nonzero(el)[0])/np.prod(el.shape)
        log("extracted root region with volume {0} with non-zero {1:.2f}%".format(np.prod(el.shape),round(perc_non_zero*100, 2)))
        
        el,sizes = filter_isolated_cells(el, min_size=remove_specks_below)

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

def peak_centroids(im, size=10, min_distance=5):
    image_max = maximum_filter(im, size=size, mode='constant')
    coordinates = feature.peak_local_max(im, min_distance=min_distance)
    df =  pd.DataFrame(coordinates,columns=["z", "y", "x"])
    return df


def simple_detector(g2, sigma_range = [8,10], bottom_threshold=0.1,min_size=1000):
    g2 = gaussian_filter(g2,sigma=sigma_range[0]) - gaussian_filter(g2,sigma=sigma_range[1])
    g2 /= g2.max()
    g2[g2<bottom_threshold] = 0
    g2,size = filter_isolated_cells(g2,min_size=min_size)
    blobs_centroids =peak_centroids(g2)   
    return g2, blobs_centroids

def detect(stack,cut_with_low_pass=True,find_threshold_val=0.1,  isol_threshold=0.125, display_detections=False,do_top_watershed=False,overlay_original_id=None,out=[]):
    """high level function to carry out a detection recipe"""
    #out = []
    if cut_with_low_pass: 
        stack = low_pass_2d_proj_root_segmentation(stack, find_threshold_val=find_threshold_val, out=out)
        log("clipped root, offset at {}".format(out))
    overlay = stack.sum(axis=0) if overlay_original_id ==None else io.get_max_int(overlay_original_id)
    
    #stack = sharpen(stack) #check the role of sharpen, might give us extra leeway to reduce the threshold and then sharpen back up - but make sure not to do both. Here i assume find thresho,d os say, 0.1
    stack, centroids = simple_detector(stack)
    
    log("detected {} centroids".format(len(centroids)))
    if len(centroids) > DEFAULT_MAX_CELLS_ALLOWED: 
        log("The number of cells {} exceeds the maximum {}. These will all be ignored".format(len(centroids),DEFAULT_MAX_CELLS_ALLOWED), mtype="WARN")
        centroids = centroids.iloc[0:0]

    #if earlier in the process we have clipped and we want to pot on original, now we need to transform back 
    if overlay_original_id != None:  
        #in this mode i am going to mark the square on the overloy
        #io.draw_bounding_box(overlay,out)
        centroids = transform_centroids(centroids, out)
    return centroids,overlay # return the bes

#def detect_last_one(stack,cut_with_low_pass=True,sharpen_iter=1, isolate_iter=1,  isol_threshold=0.125, display_detections=False,do_top_watershed=False,overlay_original_id=None,out=[]):
#    """high level function to carry out a detection recipe"""
#    #out = []
#    if cut_with_low_pass: 
#        stack = low_pass_2d_proj_root_segmentation(stack, out=out)
#        log("clipped root, offset at {} volume is {}".format(out))
#    
#    sig = 4 #if we are using 3d mode 8 works better
#    stack = sharpen(stack, sig = sig, iterations=sharpen_iter)
#    overlay = stack.sum(axis=0) if overlay_original_id ==None else io.get_max_int(overlay_original_id)
#    
##    stack = isolate(stack, iterations=isolate_iter, threshold=isol_threshold)
#   #the attempt segment is more complicated and uncessary - the blob_centroids tries to segment and works reasonable well but there were twoo many variants in the data to complete
#    #peaks are a safe and easier calculation
#    centroids =peak_centroids(stack) # blob_centroids(stack, underlying_image=stack,display=display_detections,do_top_watershed=do_top_watershed)
#    
#    #if earlier in the process we have clipped and we want to pot on original, now we need to transform back 
#    if overlay_original_id != None:  
#        #in this mode i am going to mark the square on the overloy
#        #io.draw_bounding_box(overlay,out)
#        centroids = transform_centroids(centroids, out)
#  
#    return centroids,overlay # return the best overlay item and the centroids


def sharpen(sample,exageration=1000,sig=4, iterations=1):
    for i in range(iterations):    
        partial= ndimage.gaussian_laplace(sample,sigma=sig)
        partial /= partial.max()
        partial[partial>0] = 0
        sample = sample-partial*exageration 
        sample = sample / sample.max()    
                  
    return sample


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
    
        