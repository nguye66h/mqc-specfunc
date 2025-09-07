import sys, os
import numpy as np
from glob import glob
print("-"*50)

try:
    fold = sys.argv[1]
except:
    fold = "output"


try:
    filename = sys.argv[2]
    filenames = [filename]
except:
    print ("Averaging all .txt files")
    fnames = glob(f"{fold}/*.txt")
    # get name from filenames: {name}-{number}.txt 
    # the name itself could include a dash
    fnames = [os.path.basename(f) for f in fnames]
    # print('fnames',fnames)
    names = ["-".join(f.split("/")[-1].split("-")[:-1]) for f in fnames]
    filenames = list(set(names)-set(['']))
    
    print(f"Averaging files that have the name:")
    for f in filenames:
        print(f"{f}-*.txt")
 
        
for filename in filenames:
    try:
        outName = fold + "/" + filename + ".txt"

        fnames = glob(f"{fold}/{filename}-*.txt")
        dat = 0j
        for i in fnames:
            dat += np.loadtxt(i, dtype=complex)
        N = len(fnames)
        np.savetxt(outName, dat/N)
        print ("Averaging done!")
    except Exception as e:
        print(f"Could not average {filename}-*.txt :")
        print(e)

