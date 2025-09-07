import numpy as np

class parameters():
    NSteps = 100 #int(2*10**6)
    NTraj = 100
    dtN = 0.05
    dtE = dtN/100
    nskip = 1

    NStates = 10
    ndof = NStates
    nK = NStates
    initState = 2

    nBP = 1
    nB = 1

    J = 1.0
    w = 1.0
    g = 1.0
    T = 0.0
    M = 1

    ## for MASH
    mo = 10**(-5)
    mk = np.sqrt(1-mo**2)

def Hel(R):
    V = np.zeros((parameters.NStates,parameters.NStates))
    for i in range(parameters.NStates-1):
        V[i,i+1] = -parameters.J
        V[i+1,i] = V[i,i+1]
    V[0, -1] = -parameters.J
    V[-1, 0] = -parameters.J

    g, w = parameters.g, parameters.w

    V[np.diag_indices(parameters.NStates)] = g * np.sqrt(2 * w**3) * R

    return V

def dHel0(R):
    return parameters.w**2 * R

def dHel(R):

    dVij = np.zeros((parameters.ndof,parameters.NStates,parameters.NStates))
    g, w = parameters.g, parameters.w

    for i in range(parameters.NStates):
        dVij[i, i, i] =   g * np.sqrt(2 * w**3)

    return dVij


def initR():

    w = parameters.w
    ndof = parameters.ndof

    if parameters.T == 0:
        sigP = np.sqrt( w / 2 )
    else:
        β  = 1/parameters.T
        sigP = np.sqrt( w / ( 2 * np.tanh( 0.5*β*w ) ) )

    sigR = sigP/w

    R = np.random.normal(size=ndof)*sigR
    P = np.random.normal(size=ndof)*sigP 

    return R, P
