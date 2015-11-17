#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy as np
cimport cython
cimport derivatives
from derivatives cimport computeGradient, computeDivergence, EDerivativeType, EBoundaryCondition, EDT_Forward, EDT_Backward, EBC_VonNeumanZero
cdef extern from "math.h" nogil:
    double sqrt(double)
    double fabs(double)

cdef inline double MAX(double a, double b)nogil:
    if a < b:
        return b
    return a

def filterTotalVariation_L2(double[:,:] g, double lmbd, double tau, 
                            double sigma, double theta, double[:,:] x):
    cdef:
        int nrows = g.shape[0]
        int ncols = g.shape[1]
        double[:,:] xbar = np.zeros((nrows, ncols), dtype=np.float64)
        double[:,:] yr = np.zeros((nrows, ncols), dtype=np.float64)
        double[:,:] yc = np.zeros((nrows, ncols), dtype=np.float64)
        double[:,:] dxdr = np.zeros((nrows, ncols), dtype=np.float64)
        double[:,:] dxdc = np.zeros((nrows, ncols), dtype=np.float64)
        double[:,:] divergence = np.zeros((nrows, ncols), dtype=np.float64)
        double tolerance = 1e-9
        double error = 1 + tolerance
        double nrm, diff
        int maxIter = 5000
        int npixels = nrows * ncols
        int iter=0
        int i, j
    with nogil:
        for i in range(nrows):
            for j in range(ncols):
                xbar[i, j] = x[i, j]
        while (tolerance<error) and (iter<=maxIter):
            iter += 1
            error = 0
            #update dual field
            computeGradient(xbar, dxdr, dxdc, EDT_Forward, EBC_VonNeumanZero)
            for i in range(nrows):
                for j in range(ncols):
                    yr[i, j] += sigma * dxdr[i, j]
                    yc[i, j] += sigma * dxdc[i, j]
                    nrm=sqrt(yr[i, j] * yr[i, j] + yc[i, j]*yc[i, j])
                    if nrm > 1:
                        yr[i, j]/=nrm
                        yc[i, j]/=nrm
            #update primal field
            computeDivergence(yr, yc, divergence, EDT_Backward, EBC_VonNeumanZero)
            for i in range(nrows):
                for j in range(ncols):
                    diff = -x[i, j]
                    x[i, j] += tau * divergence[i, j]
                    x[i, j] = (x[i, j] + tau * lmbd * g[i, j]) / (1.0 + tau * lmbd)
                    diff += x[i, j]
                    error += diff * diff
                    #update xbar
                    xbar[i, j] = x[i, j] + theta * diff
            error /= npixels


def filterTotalVariation_L1(double[:,:] g, double lmbd, double tau, double sigma, 
                            double theta, double[:,:] x):
    cdef:
        int nrows = g.shape[0]
        int ncols = g.shape[1]
        double[:,:] xbar = np.zeros((nrows, ncols), dtype=np.float64)
        double[:,:] yr = np.zeros((nrows, ncols), dtype=np.float64)
        double[:,:] yc = np.zeros((nrows, ncols), dtype=np.float64)
        double[:,:] dxdr = np.zeros((nrows, ncols), dtype=np.float64)
        double[:,:] dxdc = np.zeros((nrows, ncols), dtype=np.float64)
        double[:,:] divergence = np.zeros((nrows, ncols), dtype=np.float64)
        double tolerance=1e-5
        double nrm, diff, arg, obs
        double error = 1 + tolerance
        int maxIter = 5000
        int npixels = nrows * ncols
        int iter = 0
        int i, j
    with nogil:
        for i in range(nrows):
            for j in range(ncols):
                xbar[i, j] = x[i, j]
        while (tolerance<error) and (iter<=maxIter):
            iter += 1
            error = 0
            #update dual field
            computeGradient(xbar, dxdr, dxdc, EDT_Forward, EBC_VonNeumanZero)
            for i in range(nrows):
                for j in range(ncols):
                    yr[i, j] += sigma * dxdr[i, j]
                    yc[i, j] += sigma * dxdc[i, j]
                    #nrm=sqrt(yr[i*ncols+j]*yr[i*ncols+j] + yc[i*ncols+j]*yc[i*ncols+j])
                    nrm=MAX(fabs(yr[i, j]), fabs(yc[i, j]))
                    if nrm < 1:
                        nrm = 1
                    yr[i, j] /= nrm
                    yc[i, j] /= nrm
            #update primal field
            computeDivergence(yr, yc, divergence, EDT_Backward, EBC_VonNeumanZero)
            for i in range(nrows):
                for j in range(ncols):
                    diff = -x[i, j]
                    arg = x[i, j] + tau * divergence[i, j]
                    obs = g[i, j]
                    if (arg - obs) > (tau * lmbd):
                        x[i, j] = arg - tau * lmbd
                    elif (arg - obs) < -(tau * lmbd):
                        x[i, j] = arg + tau * lmbd
                    else:
                        x[i, j] = obs

                    diff += x[i, j]
                    error += diff * diff
                    #update xbar
                    xbar[i, j] = x[i, j] + theta * diff
            error /= npixels



def filterHuber_L2(double[:,:] g, double alpha, double lmbd, double tau, double sigma,
                   double theta, double[:,:] x):
    cdef:
        int nrows = g.shape[0]
        int ncols = g.shape[1]
        double[:,:] xbar = np.zeros((nrows, ncols), dtype=np.float64)
        double[:,:] yr = np.zeros((nrows, ncols), dtype=np.float64)
        double[:,:] yc = np.zeros((nrows, ncols), dtype=np.float64)
        double[:,:] dxdr = np.zeros((nrows, ncols), dtype=np.float64)
        double[:,:] dxdc = np.zeros((nrows, ncols), dtype=np.float64)
        double[:,:] divergence = np.zeros((nrows, ncols), dtype=np.float64)
        double tolerance = 1e-4
        double error = 1 + tolerance
        double nrm, diff
        int maxIter = 5000
        int npixels = nrows * ncols
        int iter = 0
        int i, j
    with nogil:
        for i in range(nrows):
            for j in range(ncols):
                xbar[i, j] = x[i, j]
        while (tolerance < error) and (iter <= maxIter):
            iter += 1
            error = 0
            #update dual field
            computeGradient(xbar, dxdr, dxdc, EDT_Forward, EBC_VonNeumanZero)
            for i in range(nrows):
                for j in range(ncols):
                    yr[i, j] += sigma * dxdr[i, j]
                    yc[i, j] += sigma * dxdc[i, j]
                    yr[i, j] /= (1 + sigma * alpha)
                    yc[i, j] /= (1 + sigma * alpha)
                    nrm = sqrt(yr[i, j] * yr[i, j] + yc[i, j]*yc[i, j])
                    if(nrm<1):
                        nrm = 1
                    yr[i, j] /= nrm
                    yc[i, j] /= nrm
            #update primal field
            computeDivergence(yr, yc, divergence, EDT_Backward, EBC_VonNeumanZero)
            for i in range(nrows):
                for j in range(ncols):
                    diff = -x[i, j]
                    x[i, j] += tau * divergence[i, j]
                    x[i, j] = (x[i, j] + tau * lmbd * g[i, j]) / (1.0 + tau * lmbd)
                    diff += x[i, j]
                    error += diff * diff
                    #update xbar
                    xbar[i, j] = x[i, j] + theta * diff
            error /= npixels


def filterTGV_L2(double[:,:] g, double lmbd, double alpha0, double alpha1, double tau, 
                 double sigma, double theta, double[:,:] u, double[:,:,:] v):
    cdef:
        int nrows = g.shape[0]
        int ncols = g.shape[1]
        #---primal variables---
        double[:,:] dubar_dr = np.zeros((nrows, ncols), dtype=np.float64)
        double[:,:] dubar_dc = np.zeros((nrows, ncols), dtype=np.float64)
        double[:,:] ubar = np.zeros((nrows, ncols), dtype=np.float64)
        double[:,:] vr = np.zeros((nrows, ncols), dtype=np.float64)
        double[:,:] vc = np.zeros((nrows, ncols), dtype=np.float64)
        double[:,:] vr_bar = np.zeros((nrows, ncols), dtype=np.float64)
        double[:,:] vc_bar = np.zeros((nrows, ncols), dtype=np.float64)
        double[:,:] dvr_bar_dr = np.zeros((nrows, ncols), dtype=np.float64)
        double[:,:] dvr_bar_dc = np.zeros((nrows, ncols), dtype=np.float64)
        double[:,:] dvc_bar_dr = np.zeros((nrows, ncols), dtype=np.float64)
        double[:,:] dvc_bar_dc = np.zeros((nrows, ncols), dtype=np.float64)
        #---dual variables-----
        double[:,:] pr = np.zeros((nrows, ncols), dtype=np.float64)
        double[:,:] pc = np.zeros((nrows, ncols), dtype=np.float64)
        double[:,:] qrr = np.zeros((nrows, ncols), dtype=np.float64)
        double[:,:] qcc = np.zeros((nrows, ncols), dtype=np.float64)
        double[:,:] qrc = np.zeros((nrows, ncols), dtype=np.float64)
        #-----------------
        double[:,:] divP = np.zeros((nrows, ncols), dtype=np.float64)
        double[:,:] divQr = np.zeros((nrows, ncols), dtype=np.float64)
        double[:,:] divQc = np.zeros((nrows, ncols), dtype=np.float64)
        double tolerance = 1e-9
        double error = 1 + tolerance
        double nrm, diff, diff_vr, diff_vc
        int maxIter = 500
        int iter = 0
        int i, j

    with nogil:
        while (tolerance < error) and (iter <= maxIter):
            iter += 1
            error = 0
            #update dual field
            computeGradient(ubar, dubar_dr, dubar_dc, EDT_Forward, EBC_VonNeumanZero)
            computeGradient(vr_bar, dvr_bar_dr, dvr_bar_dc, EDT_Forward, EBC_VonNeumanZero)
            computeGradient(vc_bar, dvc_bar_dr, dvc_bar_dc, EDT_Forward, EBC_VonNeumanZero)
            
            for i in range(nrows):
                for j in range(ncols):
                    #--update [pr,pc]--
                    pr[i, j] += sigma * (dubar_dr[i, j]-vr[i, j])
                    pc[i, j] += sigma * (dubar_dc[i, j]-vc[i, j])
                    nrm = pr[i, j] * pr[i, j] + pc[i, j] * pc[i, j]
                    nrm = sqrt(nrm) / alpha0
                    if(nrm > 1):
                        pr[i, j] /= nrm
                        pc[i, j] /= nrm
                    #--update [qrr,qrc,qcc]--
                    qrr[i, j] += sigma * (dvr_bar_dr[i, j])
                    qcc[i, j] += sigma * (dvc_bar_dc[i, j])
                    qrc[i, j] += sigma * 0.5 * (dvr_bar_dc[i, j] + dvc_bar_dr[i, j])
                    nrm = qrr[i, j] * qrr[i, j] + qcc[i, j] * qcc[i, j] + 2.0 * qrc[i, j] * qrc[i, j]
                    nrm = sqrt(nrm) / alpha1
                    if(nrm>1):
                        qrr[i, j] /= nrm
                        qcc[i, j] /= nrm
                        qrc[i, j] /= nrm
            #update primal field
            computeDivergence(pr, pc, divP, EDT_Backward, EBC_VonNeumanZero)
            computeDivergence(qrr, qrc, divQr, EDT_Backward, EBC_VonNeumanZero)
            computeDivergence(qrc, qcc, divQc, EDT_Backward, EBC_VonNeumanZero)
            
            for i in range(nrows):
                for j in range(ncols):
                    #--update u--
                    diff = -u[i, j]
                    u[i, j] += tau * divP[i, j]
                    u[i, j] = (u[i, j] + tau * lmbd * g[i, j]) / (1.0 + tau * lmbd)
                    #update ubar
                    diff += u[i, j]
                    error += diff * diff
                    ubar[i, j] = u[i, j] + theta * diff
                    #--update v--
                    diff_vr = -vr[i, j]
                    diff_vc = -vc[i, j]
                    vr[i, j] += tau * (pr[i, j] + divQr[i, j])
                    vc[i, j] += tau * (pc[i, j] + divQc[i, j])
                    #update vbar
                    diff_vr += vr[i, j]
                    diff_vc += vc[i, j]
                    vr_bar[i, j] = vr[i, j] + theta * diff_vr
                    vc_bar[i, j] = vc[i, j] + theta * diff_vc
        if v is not None:
            for i in range(nrows):
                for j in range(ncols):
                    v[i, j, 0] = vr[i, j]
                    v[i, j, 1] = vc[i, j]
