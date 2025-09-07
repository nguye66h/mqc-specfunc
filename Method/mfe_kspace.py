import numpy as np
import time
from numba import jit

class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

# Initialization of the electronic part
def initElectronic(Nstates, initState = 0, U = None):
    #global qF, qB, pF, pB, qF0, qB0, pF0, pB0
    c = np.zeros((Nstates), dtype='complex128')
    if np.any(U == None):
        c[initState] = 1.0
    else:
        c = U.T.conjugate()[:,initState]
        
    ## [1,0,0,...]
    return c

## propagate by schrodinger eqn for electronic state
## 4th order runge-kutta
@jit(nopython=False)
def propagateCi(ci,Vij, dt):
    c = ci * 1.0
    # https://thomasbronzwaer.wordpress.com/2016/10/15/numerical-quantum-mechanics-the-time-dependent-schrodinger-equation-ii/
    ck1 = (-1j) * (Vij @ c)
    ck2 = (-1j) * Vij @ (c + (dt/2.0) * ck1 )
    ck3 = (-1j) * Vij @ (c + (dt/2.0) * ck2 )
    ck4 = (-1j) * Vij @ (c + (dt) * ck3 )
    c = c + (dt/6.0) * (ck1 + 2.0 * ck2 + 2.0 * ck3 + ck4)

    # wave function propagated to current+dt

    return c

## force of quantum electronic states on classical nucleus
def force_on_p(ci, par, i):
    return (ci.conjugate()@par.dHel_dq(i, par.nB, par.nK, par.l_x, par.l_y, par.l_z, par.g_mn_nu_kq, par.w_nu_q, par.sites)@ci).real

def force_on_q(ci, par, i):
    return (ci.conjugate()@par.dHel_dp(i, par.nB, par.nK, par.l_x, par.l_y, par.l_z, par.g_mn_nu_kq, par.w_nu_q, par.sites)@ci).real

## force of quantum electronic states on classical nucleus
def Force_on_p(ci, par):
    fvec = np.vectorize(force_on_p, excluded=['ci', 'par'])
    return fvec(ci=ci, par=par, i=np.arange(par.nK * par.nBP))

def Force_on_q(ci, par):
    fvec = np.vectorize(force_on_q, excluded=['ci', 'par'])
    return fvec(ci=ci, par=par, i=np.arange(par.nK * par.nBP))


def VelVer(dat) :
    par =  dat.param

    # electronic wavefunction
    ci = dat.ci * 1.0

    ## number of electronic steps per 1 nuclear time step
    EStep = int(par.dtN/par.dtE)

    ## size of an electronic step
    dtE = par.dtN/EStep

    # half electronic evolution
    for t in range(int(np.floor(EStep/2))):
        dat.corr_ensemble[dat.iskip] += corr(dat,ci)
        dat.iskip += 1 
        #t0 = time.time()
        ci = propagateCi(ci, dat.Hij, dtE)
        #print('time for each electronic step',time.time()-t0)
        

    ## normalize electronic wavefunction
    #ci /= np.sum(ci.conjugate()*ci)

    ## update dat.ci
    dat.ci = ci * 1.0

    # ======= Nuclear Block ==================================
    t0 = time.time()
    fonp = Force_on_p(dat.ci, par)
    dat.P -= (par.w_nu_q.flatten()**2*dat.R + fonp) * (par.dtN/2)
    dat.R += (dat.P + Force_on_q(dat.ci, par)) * par.dtN
    dat.P -= (par.w_nu_q.flatten()**2*dat.R + fonp) * (par.dtN/2)

    #------ Do QM ----------------
    # ======================================================
    dat.Hij  = par.Hel(dat.R,dat.P,par.nBP,par.nB,par.nK,par.l_x,par.l_y,par.l_z,par.g_mn_nu_kq,par.w_nu_q,par.eh_n_k)
    # half electronic evolution
    for t in range(int(np.ceil(EStep/2))):
        dat.corr_ensemble[dat.iskip] += corr(dat,ci)
        dat.iskip += 1 
        ci = propagateCi(ci, dat.Hij, dtE)
        
    dat.ci = ci * 1.0

    return dat


def pop(dat):
    ci =  dat.ci
    return np.outer(ci.conjugate(),ci)


def corr(dat,ci):

    if np.any( dat.param.U == None ):
        return ci[dat.param.k_ind_of_c]
    else:
        c_dagger = dat.param.U[dat.param.k_ind_of_c,:]
        return c_dagger@ci


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
    dtN = parameters.dtN
    U = parameters.U
    ## number of electronic steps per 1 nuclear time step
    EStep = int(parameters.dtN/parameters.dtE)
    #---------------------------
    if NSteps%nskip == 0:
        pl = 0
    else :
        pl = 1
    #rho_ensemble = np.zeros((NStates,NStates,NSteps//nskip + pl), dtype=complex)
    corr_ensemble = np.zeros((NSteps*EStep//nskip + pl, 1), dtype=complex)
    # Ensemble
    for itraj in range(NTraj):
        # Trajectory data
        dat = Bunch(param =  parameters )
        dat.R, dat.P = parameters.initR()

        # set propagator
        vv  = VelVer

        # Call function to initialize mapping variables
        dat.ci = initElectronic(NStates, initState, U) # np.array([0,1])

        #----- Initial QM --------
        dat.Hij  = parameters.Hel(dat.R,dat.P,parameters.nBP,parameters.nB,parameters.nK,parameters.l_x,parameters.l_y,parameters.l_z,parameters.g_mn_nu_kq,parameters.w_nu_q,parameters.eh_n_k)
        # dat.dH0  = parameters.dHel0()
        #dat.F1 = Force(dat) # Initial Force
        #----------------------------
        dat.corr_ensemble = corr_ensemble
        dat.iskip = 0 # please modify

        for i in range(NSteps): # One trajectory
            dat = vv(dat)
    return corr_ensemble
