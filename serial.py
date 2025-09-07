import sys, os
import numpy as np

try:
    fold = sys.argv[2]
except:
    fold = "./output"

#-------------------------
try:
    # Get the current working directory
    current_directory = os.getcwd()

    # Print the current working directory
    print("Current Directory:", current_directory)
    inputtxt = open(f"{fold}/{sys.argv[1]}", 'r').readlines()
    print(f"Reading {sys.argv[1]}")
except:
    print("Reading input.txt")
    inputtxt = open('input.txt', 'r').readlines()


def getInput(input,key):
    try:
        txt = [i for i in input if i.find(key)!=-1][0].split("=")[1].split("#", 1)[0].replace("\n","")
    except:
        txt = ""
    return txt.replace(" ","")

syst = getInput(inputtxt,"System")
if syst == "serial" or syst == "pc":
    sys.path.append(os.path.join(os.getcwd(), "Model"))
    sys.path.append(os.path.join(os.getcwd(), "Method"))
else:
    sys.path.append(os.path.join(os.getcwd(), fold))

model_ =  getInput(inputtxt,"Model")
method_ = getInput(inputtxt,"Method").split("-")
exec(f"import {model_} as model")
exec(f"import {method_[0]} as method")
try:
    stype = method_[1]
except:
    stype = "_"
#-------------------------
import time
import numpy as np

t0 = time.time()


os.system(f"mkdir -p {fold}")
ID = ''
try :
    ID = sys.argv[3]
    ID = "-" + ID
except:
    pass

t1 = time.time()

NTraj = model.parameters.NTraj
NStates = model.parameters.NStates

#------ Arguments------------------
par = model.parameters()
par.ID     = np.random.randint(0,100)
par.SEED   = np.random.randint(0,100000000)

#---- methods in model ------
if "kspace" in method_[0] or "k" in method_[0]:
    par.dHel_dq = model.dHel_dq
    par.dHel_dp = model.dHel_dp
else:
    par.dHel = model.dHel
par.dHel0 = model.dHel0
par.initR = model.initR
par.Hel   = model.Hel
par.stype = stype



#---- overriden parameters ------

parameters = [i for i in inputtxt if i.split("#")[0].split("=")[0].find("$") !=- 1]
for p in parameters:
    exec(f"par.{p.split('=')[0].split('$')[1]} = {p.split('=')[1].split('#')[0]}")
    print(f"Overriding parameters: {p.split('=')[0].split('$')[1]} = {p.split('=')[1].split('#')[0]}")
#--------------------------------

#------------------- run ---------------
rho_sum  = method.runTraj(par)
#---------------------------------------

try:
    PiiFile = open(f"{fold}/{method_[0]}-{method_[1]}-{model_}{ID}.txt","w+")
except:
    # PiiFile = open(f"{fold}/{method_[0]}-{model_}{ID}.txt","w+")
    PiiFile = open(f"{fold}/{method_[0]}-{model_}{ID}.txt","w+")

NTraj = par.NTraj

if method_[0] == 'mfe_kspace':
    for t in range(rho_sum.shape[0]):
        PiiFile.write(f"{t * par.nskip * par.dtE} \t")
        for ele in range(rho_sum.shape[-1]):
            PiiFile.write(str(rho_sum[t,ele] / ( NTraj ) ) + "\t")
        PiiFile.write("\n")
    PiiFile.close()
else:
    for t in range(rho_sum.shape[0]):
        PiiFile.write(f"{t * par.nskip * par.dtN} \t")
        for ele in range(rho_sum.shape[-1]):
            PiiFile.write(str(rho_sum[t,ele] / ( NTraj ) ) + "\t")
        PiiFile.write("\n")
    PiiFile.close()


t2 = time.time()-t1
print(f"Total Time: {t2}")
print(f"Time per trajectory: {t2/NTraj}")
