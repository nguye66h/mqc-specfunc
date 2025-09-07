# From https://pubs.aip.org/aip/jcp/article/159/9/094115/2909882/A-multi-state-mapping-approach-to-surface-hopping

import numpy as np
import copy
import time


class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


def initElectronic(NStates, initState, mo, mk, Hij):
    k = np.arange(0,2*np.pi,2*np.pi/(NStates))
    NStates += 1

    sumN = np.sum(np.array([1/n for n in range(1,NStates+1)]))
    alpha = (NStates - 1)/(sumN - 1)
    beta = (alpha - 1)/NStates

    c = np.sqrt(beta/alpha) * np.ones((NStates), dtype = np.complex_)
    c[initState] = np.sqrt((1+beta)/alpha)
    uni = np.random.random(NStates)
    c = c * np.exp(1j*2*np.pi*uni)

    μ = np.identity(NStates) + 0j
    μ[initState, initState] = mk
    μ[initState, -1] = mo
    μ[-1, initState] = mo
    μ[-1, -1] = -mk

    ck = μ @ c
    c0 = ck[-1]
    cinitState = ck[initState]
    ck = ck[:-1]
    cn = np.einsum('kn,k->n',np.exp(np.outer(-1j*k,np.arange(NStates-1))),ck)/np.sqrt(NStates-1)

    E, U = np.linalg.eigh(Hij)
    c = np.conj(U).T @ cn
    return c, cinitState, c0

def Force(dHij, dH0, acst, U, ci):

    # dHij is in the diabatic basis !IMPORTANT
    F = -dH0
    # <a |dH | a> -->   ∑ij <a | i><i | dH |j><j| a>
    F -= np.einsum('j, ijk, k -> i', U[:, acst].conjugate(), dHij + 0j, U[:,acst]).real

    return F


def VelVer(ogdat, acst, dt) :

    dat = copy.deepcopy(ogdat)

    par =  dat.param
    v = dat.P/par.M
    F1 = dat.F1 * 1.0

    # half electronic evolution
    dat.ci = dat.ci * np.exp(-1j*dt*dat.E/2.0)
    cD =  dat.U @ dat.ci # to diabatic basis
    # ======= Nuclear Block =================================
    dat.R += v * dt + 0.5 * F1 * dt ** 2 / par.M

    #------ Do QM ----------------
    dat.Hij  = par.Hel(dat.R) + 0j
    dat.dHij = par.dHel(dat.R)
    dat.dH0  = par.dHel0(dat.R)
    #-----------------------------
    dat.E, dat.U = np.linalg.eigh(dat.Hij)
    F2 = Force(dat.dHij, dat.dH0, acst, dat.U, cD) # force at t2
    v += 0.5 * (F1 + F2) * dt / par.M
    dat.F1 = F2 * 1.0
    dat.P = v * par.M

    # ======================================================
    dat.ci = np.conj(dat.U).T @ cD # back to adiabatic basis

    # half electronic evolution
    dat.ci = dat.ci * np.exp(-1j*dt*dat.E/2.0)

    return dat


def pop(c,k): # returns the density matrix estimator (populations and coherences)
    NStates = len(c)+1

    sumN = np.sum(np.array([1/n for n in range(1,NStates+1)])) # constant based on NStates
    alpha = (NStates - 1)/(sumN - 1) # magnitude scaling
    beta = (1-alpha )/NStates # effective zero-point energy

    NStates -= 1
    ck = np.einsum('kn,n->k',np.exp(np.outer(1j*k,np.arange(NStates))),c)/np.sqrt(NStates)

    prod = np.outer(ck,np.conj(ck))
    return alpha * ck.conj() * ck + beta # works in any basis

def coeff_est (ci, cinitState, NStates, k):
    NStates += 1
    sumN = np.sum(np.array([1/n for n in range(1,NStates+1)])) # constant based on NStates
    alpha = (NStates - 1)/(sumN - 1) # magnitude scaling

    NStates -= 1
    ck = np.einsum('kn,n->k',np.exp(np.outer(1j*k,np.arange(NStates))),ci)/np.sqrt(NStates)
    return ck / cinitState

def checkHop(acst, c): # calculate current active state and store result
    # returns [hop needed?, previous active state, current active state]
    n_max = np.argmax(np.abs(c))
    if(acst != n_max):
        return True, acst, n_max
    return False, acst, acst


def hop(dat, a, b):

    if a != b:
        # a is previous active state, b is current active state
        P = dat.P/np.sqrt(dat.param.M) # momentum rescaled
        ΔE = np.real(dat.E[b] - dat.E[a])
        tol = 1E-10

        if (np.abs(ΔE) < tol) or dat.param.J==0:
            print("Trivial Crossing")
            return dat.P*1.0, True
        # dij is nonadiabatic coupling
        # <i | d/dR | j> = < i |dH | j> / (Ej - Ei)

        # # direction -> 1/√m ∑f Re (c[f] d [f,a] c[a] - c[f] d [f,b] c[b])  # c[f] = ∑m <m | Ψf>
        # #            =Re ( 1/√m ∑f ∑nm Ψ[m ,f]^ . (<m | dH/dRk | n> ) . Ψ[n ,a] /(E[a]-E[f])
        j = np.arange(len(dat.E))
        ΔEa, ΔEb = (dat.E[a] - dat.E), (dat.E[b] - dat.E)
        ΔEa[a], ΔEb[b] = 1.0, 1.0 # just to ignore error message
        rΔEa, rΔEb = (a != j)/ΔEa, (b != j)/ΔEb

        dHab =   np.einsum('ia, kij, jb -> kab', dat.U.conjugate(), dat.dHij, dat.U)
        term1 = np.einsum('n, jn, n -> j', dat.ci.conjugate(),  dHab[:, :, a] * dat.ci[a], rΔEa)
        term2 = np.einsum('n, jn, n -> j', dat.ci.conjugate(),  dHab[:, :, b] * dat.ci[b], rΔEb)

        δk = (term1 - term2).real

        #Project the momentum to the new direction
        P_proj = np.dot(P,δk) * δk / np.dot(δk, δk) if np.abs(np.dot(P,δk)) > tol else P * 0.0

        #Compute the orthogonal momentum
        P_orth = P - P_proj # orthogonal P

        #Compute projected norm, which will be useful later
        P_proj_norm = np.sqrt(np.dot(P_proj,P_proj))

        #print('P_proj',P_proj )

        #Compute the total kinetic energy in the projected direction
        if(P_proj_norm**2 < 2*ΔE): # rejected hop
            P_proj = -P_proj # reverse projected momentum
            P = P_orth + P_proj
            accepted = False
        else: # accepted hop
            P_proj = np.sqrt(P_proj_norm**2 - 2*ΔE)/P_proj_norm * P_proj if np.abs(P_proj_norm) > tol else P * 0.0 #re-scale the projected momentum
            P = P_orth + P_proj
            accepted = True
        P *= np.sqrt(dat.param.M)
        dat.P = P
        return dat.P, accepted
    return dat.P, False

def runTraj(parameters):
    #------- Seed --------------------
    try:
        np.random.seed(parameters.SEED)
    except:
        pass
    #------------------------------------
    ## Parameters -------------
    NSteps = parameters.NSteps
    NTraj = parameters.NTraj
    NStates = parameters.NStates
    initState = parameters.initState # intial state
    nskip = parameters.nskip
    dtN   = parameters.dtN
    mo = parameters.mo
    mk = parameters.mk
    k = np.arange(0,2*np.pi,2*np.pi/NStates)

    #---------------------------
    if NSteps%nskip == 0:
        pl = 0
    else :
        pl = 1
    coeff_ensemble = np.zeros((NSteps//nskip + pl, NStates), dtype=complex)
    # Ensemble
    for itraj in range(NTraj):
        # Trajectory data
        dat = Bunch(param =  parameters )
        dat.R, dat.P = parameters.initR()

        # set propagator
        vv  = VelVer

        #----- Initial QM --------
        dat.Hij  = parameters.Hel(dat.R)
        dat.dHij = parameters.dHel(dat.R)
        dat.dH0  = parameters.dHel0(dat.R)

        # Call function to initialize mapping variables
        dat.ci, dat.cinitState, dat.c0 = initElectronic(NStates, initState, mo, mk, dat.Hij) # np.array([0,1])
        acst = np.argmax(np.abs(dat.ci))
        dat.E, dat.U = np.linalg.eigh(dat.Hij)
        dat.F1 = Force(dat.dHij, dat.dH0, acst, dat.U, dat.U @ dat.ci) # Initial Force
        #----------------------------
        iskip = 0 # please modify
        t0 = time.time()
        for i in range(NSteps): # One trajectory
            #------- ESTIMATORS-------------------------------------
            if (i % nskip == 0):
                coeff_ensemble[iskip] += coeff_est (dat.U @ dat.ci, dat.cinitState, NStates, k)
                iskip += 1
            #-------------------------------------------------------
            dat0 = vv(dat, acst, dtN)

            maxhop = 10

            #if(checkHop(acst, dat0.ci)[0]==True):
            if (hop(dat0, acst, checkHop(acst, dat0.ci)[2])[1]):
                newacst = checkHop(acst, dat0.ci)[2]
                # lets find the bisecting point
                tL, tR = 0, dtN
                for _ in range(maxhop):
                    tm = (tL + tR)/2
                    dat_tm = vv(dat, acst, tm)
                    if checkHop(acst, dat_tm.ci)[0] == False:
                        tL = tm
                    else:
                        tR = tm

                P, accepted = hop(dat_tm, acst, newacst)
                if accepted:
                    dat_tm.P = P # momentum update
                    acst = newacst

                dat = vv(dat_tm, acst, dtN - tm)

            else:
                dat = dat0

        time_taken = time.time()-t0
        print(f"Time taken: {time_taken} seconds")
    return coeff_ensemble
