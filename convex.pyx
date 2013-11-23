# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:37:17 2013

@author: khayyam
"""
from cython.view cimport memoryview
from cython.view cimport array as cvarray
import numpy as np

cdef extern from "TotalVariation.h":
    void filterTotalVariation_L2(double *g, int nrows, int ncols, double lambdaParam, double tau, double sigma, double theta, double *x, void (*callback)(double*, int, int))
    void filterTotalVariation_L1(double *g, int nrows, int ncols, double lambdaParam, double tau, double sigma, double theta, double *x, void (*callback)(double*, int, int))
    void filterTGV_L2(           double *g, int nrows, int ncols, double lambdaParam, double alpha0, double alpha1, double tau, double sigma, double theta, double *u, double *v)
cdef extern from "linearprogramming.h":
    double linprog_pd(double *c, double *A, int n, int m, double *b, double *x, int maxIter, double tolerance)

def filter_tv_l2(double[:,:] g, double lambdaParam, double tau, double sigma, double theta):
    cdef int nrows=g.shape[0]
    cdef int ncols=g.shape[1]
    cdef double[:,:] filtered=np.zeros(shape=(nrows, ncols), dtype=np.double)
    filterTotalVariation_L2(&g[0,0], nrows, ncols, lambdaParam, tau, sigma, theta, &filtered[0,0], NULL)
    return filtered

def filter_tv_l1(double[:,:] g, double lambdaParam, double tau, double sigma, double theta):
    cdef int nrows=g.shape[0]
    cdef int ncols=g.shape[1]
    cdef double[:,:] filtered=np.zeros(shape=(nrows, ncols), dtype=np.double)
    filterTotalVariation_L1(&g[0,0], nrows, ncols, lambdaParam, tau, sigma, theta, &filtered[0,0], NULL)
    return filtered

def filter_tgv_l2(double[:,:] g, double lambdaParam, double alpha0, double alpha1, double tau, double sigma, double theta):
    cdef int nrows=g.shape[0]
    cdef int ncols=g.shape[1]
    cdef double[:,:] filtered=np.zeros(shape=(nrows, ncols), dtype=np.double)
    cdef double[:,:,:] filteredGrad=np.zeros(shape=(2,nrows, ncols), dtype=np.double)
    filterTGV_L2(&g[0,0], nrows, ncols, lambdaParam, alpha0, alpha1, tau, sigma, theta, &filtered[0,0], &filteredGrad[0,0,0])
    return filtered, filteredGrad

def linprog_pd_small(double[:] c, double[:,:] A, double[:] b, int maxIter, double tolerance):
    cdef int n=A.shape[0]
    cdef int m=A.shape[1]
    cdef double[:] x=np.zeros(shape=(m,), dtype=np.double)
    cdef double retVal
    retVal=linprog_pd(&c[0], &A[0,0], n, m, &b[0], &x[0], maxIter, tolerance)
    print 'RetVal:',retVal
    return x
