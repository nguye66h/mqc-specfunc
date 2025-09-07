import os,sys

def getInput(input,key):
    try:
        txt = [i for i in input if i.find(key)!=-1][0].split("=")[1].split("#", 1)[0].replace("\n","")
    except:
        txt = ""
    return txt.replace(" ","")


try:
    inputfile =  sys.argv[1]
    input = open(inputfile, 'r').readlines()
except:
    inputfile =  "input.txt"
    input = open(inputfile, 'r').readlines()

print(f"Reading {inputfile}")

fold = 'output'
try :
    fold = sys.argv[2]

except:
    pass

# System
system = getInput(input,"System")

os.system(f"rm -rf $SCRATCH/NAMD/{fold}")
os.system(f"mkdir -p $SCRATCH/NAMD/{fold}")

# SLURM
if system == "slurm" or system == "htcondor":
    #sys.path.append(os.popen("pwd").read().replace("\n","")+"/"+fold)
    print (f"Running jobs in a {system}")
    model = getInput(input,"Model")
    method = getInput(input,"Method")

    ncpus     = int(getInput(input,"Cpus"))
    # totalTraj = ntraj * ncpus
    print(f"Using {ncpus} CPUs")
    print("-"*20, "Default Parameters", "-"*20)

    print("-"*50)
    parameters = [i for i in input if i.split("#")[0].split("=")[0].find("$") !=- 1]
    for p in parameters:
        print(f"Overriding parameters: {p.split('=')[0].split('$')[1]} = {p.split('=')[1].split('#')[0]}")
    print("-"*50)

    os.system(f"cp Model/{model}.py $SCRATCH/NAMD/{fold}")
    os.system(f"cp Method/{method}.py $SCRATCH/NAMD/{fold}")
    os.system(f"cp input.txt $SCRATCH/NAMD/{fold}")
    if model == "lif":
        os.system(f"cp Model/LiF_g_nijkq_N5_bands_3_4_5.npz $SCRATCH/NAMD/{fold}")
        # os.system(f"cp Model/LiF_g_nijkq_N7_bands_3_4_5.npz $SCRATCH/NAMD/{fold}")
        # os.system(f"cp Model/LiF_g_nijkq_N9_bands_3_4_5.npz $SCRATCH/NAMD/{fold}")
        # os.system(f"cp Model/LiF_g_nijkq_N11_bands_3_4_5.npz $SCRATCH/NAMD/{fold}")
        # os.system(f"cp Model/LiF_g_nijkq_N13_bands_3_4_5.npz $SCRATCH/NAMD/{fold}")

    if system == "slurm":
        for i in range(ncpus):
            os.system(f"./submit.sh {inputfile} {fold} {i}")
# PC
else:
    print ("Running jobs in your local machine (like a PC)")
    os.system(f"python3 serial.py {inputfile} {fold}")
