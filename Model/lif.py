import numpy as np
from numpy import array as A
import time
import os, sys
from numba import jit

class parameters():
    NSteps = 2 #int(2*10**6)
    NTraj = 1
    dtN = 250
    dtE = 1
    T = 0
    # mass
    M = 1

    #global l_x, l_y, l_z
    l_x = 5
    l_y = l_x
    l_z = l_x

    #global back_reaction
    back_reaction = True
    corr = True
    b = 2*l_x*l_y*l_z
    k_of_c = b
    k_of_c_dagger = b
    initState = b
    k_ind_of_c_dagger = b
    k_ind_of_c = b

    #global nB, nBP
    nB = 3
    nBP= 6

    #global nK
    nK = l_x * l_y * l_z
    nAcou = 3
    NStates = nK*nB
    ndof = nK*nBP
    H_to_meV = 27211.4

    i, j, kk = np.meshgrid(np.arange(l_x),np.arange(l_y),np.arange(l_z), indexing='ij')

    #global sites
    sites = np.column_stack((i.flatten(),j.flatten(),kk.flatten()))

    # Get the absolute path of the current file
    current_file_path = os.path.abspath(__file__)

    # Get the directory name from the absolute path
    current_directory = os.path.dirname(current_file_path)

    print("Current Directory:", current_directory)

    # Construct the full path to the 'lif.npz' file
    npz_file_path = os.path.join(current_directory, f'LiF_g_nijkq_N{l_x}_bands_3_4_5.npz')

    # Load the .npz file using its full path
    npz_file = np.load(npz_file_path)


    ## lif file
    #global eh_n_k, w_nu_q, g_mn_nu_kq, h_nq
    eh_n_k = (-1 * npz_file['e_ik']) / H_to_meV
    w_nu_q = npz_file['w_nq'] / H_to_meV
    w = w_nu_q.flatten()
    w = np.ones(w.shape)
    g_mn_nu_kq_e = npz_file['g_nijkq'].transpose(1,2,0,3,4) / H_to_meV

    g_mn_nu_kq = 0. * g_mn_nu_kq_e
    for k_ in range(nK):
        sitek = sites[k_]
        mk = -sitek % np.array([l_x,l_y,l_z])
        mki = l_z*(l_y*mk[0] + mk[1]) + mk[2]
        for q in range(nK):
            siteq = sites[q]
            mq = -siteq % np.array([l_x,l_y,l_z])
            mqi = l_z*(l_y*mq[0] + mq[1]) + mq[2]
            for m in range(nB):
                for n in range(nB):
                    g_mn_nu_kq[m, n, :, k_, q] = -np.conj(g_mn_nu_kq_e[n, m, :, mki, mqi])

    # setting acoustic phonon + zero mode couplings to zero
    g_mn_nu_kq[:, :, :nAcou, :, 0] = 0

    nskip = 1

    U = None
    #global sites
    k = np.column_stack((i.flatten(),j.flatten(),kk.flatten())) * 2*np.pi/l_x
    k = np.moveaxis(k,0,-1)

@jit(nopython=False)
def Hel_s(i,j,R,P,nBP,nK,l_x,l_y,l_z,g_mn_nu_kq,w_nu_q):

    R = R.reshape(nBP,nK)
    P = P.reshape(nBP,nK)

    # --- i ------------
    m  = (i//nK)
    kz1 = i % l_z
    ky1 = (i//l_y) % l_y
    kx1 = (i//(l_x**2)) % l_x

    #-----j ------------
    n = (j//nK)
    k2 = j % nK
    kz2 = j % l_z
    ky2 = (j//l_y) % l_y
    kx2 = (j//(l_x**2)) % l_x


    qx = (kx1 - kx2) % l_x
    qy = (ky1 - ky2) % l_y
    qz = (kz1 - kz2) % l_z

    mqx = (kx2 - kx1) % l_x
    mqy = (ky2 - ky1) % l_y
    mqz = (kz2 - kz1) % l_z

    k1_k2 = l_z*(l_y*qx + qy) + qz
    k2_k1 = l_z*(l_y*mqx + mqy) + mqz

    vij = np.sum(g_mn_nu_kq[m,n,:,k2,k1_k2]*(np.sqrt(w_nu_q[:,k2_k1]/2)*R[:,k2_k1]-1j*np.sqrt(1/(2*w_nu_q[:,k2_k1]))*P[:,k2_k1]+\
                             np.sqrt(w_nu_q[:,k1_k2]/2)*R[:,k1_k2]+1j*np.sqrt(1/(2*w_nu_q[:,k1_k2]))*P[:,k1_k2]))

    return vij


def Hel(R,P,nBP,nB,nK,l_x,l_y,l_z,g_mn_nu_kq,w_nu_q,eh_n_k):

    Helvec = np.vectorize(Hel_s, excluded=['R','P','nBP','nK','l_x','l_y','l_z','g_mn_nu_kq','w_nu_q'])

    i, j = np.arange(nK*nB), np.arange(nK*nB)

    # mesh with flat indices
    i, j = np.meshgrid(i, j, indexing='ij')

    ## this is H_system
    Vij = np.diag(eh_n_k.flatten()) + 0j

    ## this is V, interaction term
    Vij += Helvec(i, j, R = R, P = P, nBP = nBP, nK = nK, l_x = l_x, l_y = l_y, l_z = l_z, g_mn_nu_kq = g_mn_nu_kq, w_nu_q = w_nu_q)

    return Vij


def dHel0(R):
    w_nu_q = parameters.w_nu_q.flatten()
    return w_nu_q**2*R

@jit(nopython=False)
def dHel_dq(nuq, nB, nK, l_x, l_y, l_z, g_mn_nu_kq, w_nu_q, sites):

    nu = nuq // nK
    q = nuq % nK
    dhelq = np.zeros((nB,nK,nB,nK),dtype=np.complex64)
    for m in range(nB):
        for n in range(nB):
            for kp in range(nK):
                kp3d = sites[kp]
                q3d = sites[q]

                kp_p_q3d = (kp3d+q3d) % np.array([l_x, l_y, l_z])
                kp_p_q = l_z*(l_y*kp_p_q3d[0] + kp_p_q3d[1]) + kp_p_q3d[2]

                kp_q3d = (kp3d-q3d) % np.array([l_x, l_y, l_z])
                kp_q = l_z*(l_y*kp_q3d[0] + kp_q3d[1]) + kp_q3d[2]

                mq3d = -q3d % np.array([l_x, l_y, l_z])
                mq = l_z*(l_y*mq3d[0] + mq3d[1]) + mq3d[2]

                dhelq[m,kp_p_q,n,kp] += g_mn_nu_kq[m,n,nu,kp,q]*np.sqrt(w_nu_q[nu,q]/2)
                dhelq[m,kp_q,n,kp] += g_mn_nu_kq[m,n,nu,kp,mq]*np.sqrt(w_nu_q[nu,q]/2)

    return dhelq.reshape(dhelq.shape[0]*dhelq.shape[1],-1)

@jit(nopython=False)
def dHel_dp(nuq, nB, nK, l_x, l_y, l_z, g_mn_nu_kq, w_nu_q, sites):

    nu = nuq // nK
    q = nuq % nK
    dhelp = np.zeros((nB,nK,nB,nK),dtype=np.complex64)
    for m in range(nB):
        for n in range(nB):
            for kp in range(nK):
                kp3d = sites[kp]
                q3d = sites[q]

                kp_p_q3d = (kp3d+q3d) % np.array([l_x, l_y, l_z])
                kp_p_q = l_z*(l_y*kp_p_q3d[0] + kp_p_q3d[1]) + kp_p_q3d[2]

                kp_q3d = (kp3d-q3d) % np.array([l_x, l_y, l_z])
                kp_q = l_z*(l_y*kp_q3d[0] + kp_q3d[1]) + kp_q3d[2]

                mq3d = -q3d % np.array([l_x, l_y, l_z])
                mq = l_z*(l_y*mq3d[0] + mq3d[1]) + mq3d[2]

                dhelp[m,kp_p_q,n,kp] += 1j*g_mn_nu_kq[m,n,nu,kp,q]*np.sqrt(1/(w_nu_q[nu,q]*2))
                dhelp[m,kp_q,n,kp] += -1j*g_mn_nu_kq[m,n,nu,kp,mq]*np.sqrt(1/(w_nu_q[nu,q]*2))

    return dhelp.reshape(dhelp.shape[0]*dhelp.shape[1],-1)


def initR():

    w_nu_q = parameters.w_nu_q

    if parameters.T == 0:
        sigP = np.sqrt( w_nu_q / 2 )
    else:
        β  = 1/parameters.T
        sigP = np.sqrt( w_nu_q / ( 2 * np.tanh( 0.5*β*w_nu_q ) ) )

    sigR = sigP / w_nu_q

    R = ( np.random.normal(size=w_nu_q.shape)*sigR ).flatten()
    P = ( np.random.normal(size=w_nu_q.shape)*sigP ).flatten()

    return R, P
