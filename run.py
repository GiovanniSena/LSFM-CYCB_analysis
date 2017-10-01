import sys
import lightroot

n = 5 #default

if __name__ == "__main__":
    l= len(sys.argv)
    if l >1:  
        n = int(sys.argv[1])
    print("processing", n, "files")
    all_blobs = lightroot.process_files(n)
    print("processing tracks")
    tracks = lightroot.tracks_from_blobs(all_blobs, n,"./cached_data/{}.png")
    
    