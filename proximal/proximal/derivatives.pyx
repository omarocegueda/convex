#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy as np
cimport cython
cimport derivatives
from derivatives cimport EDerivativeType, EBoundaryCondition

cdef int computeRowDerivative_Forward(double[:,:] f, double[:,:] dfdr, EBoundaryCondition ebc)nogil:
    cdef:
        int nrows = f.shape[0]
        int ncols = f.shape[1]
        int i, j
    for j in range(ncols):
        for i in range(nrows-1):
            dfdr[i,j]=f[i+1, j]-f[i, j]
    if ebc == EBC_Circular:
        for j in range(ncols):
            dfdr[nrows-1, j]=f[0, j]-f[nrows-1, j]
    elif ebc == EBC_DirichletZero:
        for j in range(ncols):
            dfdr[nrows-1, j]=-f[nrows-1, j]
    elif ebc == EBC_VonNeumanZero:
        for j in range(ncols):
            dfdr[nrows-1, j]=0
    else:
        return -1
    return 0


cdef int computeColumnDerivative_Forward(double[:,:] f, double[:,:] dfdc, EBoundaryCondition ebc)nogil:
    cdef:
        int nrows = f.shape[0]
        int ncols = f.shape[1]
        int i, j
    for i in range(nrows):
        for j in range(ncols-1):
            dfdc[i, j]=f[i, j+1]-f[i, j]
    if ebc == EBC_Circular:
        for i in range(nrows):
            dfdc[i, ncols-1] = f[i, 0]-f[i, ncols-1]
    elif ebc == EBC_DirichletZero:
        for i in range(nrows):
            dfdc[i, ncols-1] = -f[i, ncols-1]
    elif ebc == EBC_VonNeumanZero:
        for i in range(nrows):
            dfdc[i, ncols-1]=0
    else:
        return -1
    return 0


cdef int computeRowDerivative_Backward(double[:,:] f, double[:,:] dfdr, EBoundaryCondition ebc)nogil:
    cdef:
        int nrows = f.shape[0]
        int ncols = f.shape[1]
        int i, j
    for j in range(ncols):
        for i in range(1, nrows):
            dfdr[i, j]=f[i, j]-f[i-1, j]
    if ebc == EBC_Circular:
        for j in range(ncols):
            dfdr[0, j]=f[0, j] - f[nrows-1, j]
    elif ebc == EBC_DirichletZero:
        for j in range(ncols):
                dfdr[0, j]=f[0, j];
    elif ebc == EBC_VonNeumanZero:
        for j in range(ncols):
            dfdr[0, j] = 0
    return 0


cdef int computeColumnDerivative_Backward(double[:,:] f, double[:,:] dfdc, EBoundaryCondition ebc)nogil:
    cdef:
        int nrows = f.shape[0]
        int ncols = f.shape[1]
        int i, j
    for i in range(nrows):
        for j in range(1, ncols):
            dfdc[i, j]=f[i, j] - f[i, j-1];
    if ebc == EBC_Circular:
        for i in range(nrows):
            dfdc[i, 0]=f[i, 0] - f[i, ncols-1]
    elif ebc == EBC_DirichletZero:
        for i in range(nrows):
            dfdc[i, 0] = f[i, 0]
    elif ebc == EBC_VonNeumanZero:
        for i in range(nrows):
            dfdc[i, 0] = 0
    else:
        return -1
    return 0


cdef int computeGradient(double[:,:] f, double[:,:] dfdr, double[:,:] dfdc, 
                         EDerivativeType edt, EBoundaryCondition ebc)nogil:
    cdef:
        int nrows = f.shape[0]
        int ncols = f.shape[1]
        int i, j
        
    if nrows < 2 or ncols < 2:
        return -1
        
    if edt == EDT_Forward:
        computeRowDerivative_Forward(f, dfdr, ebc)
        computeColumnDerivative_Forward(f, dfdc, ebc)
    elif edt == EDT_Backward:
        computeRowDerivative_Backward(f, dfdr, ebc)
        computeColumnDerivative_Backward(f, dfdc, ebc)
    else:
        return -1
    return 0


cdef int computeDivergence_Forward(double[:,:] fr, double[:,:] fc,
                           double[:,:] div, EBoundaryCondition ebc)nogil:
    cdef:
        int nrows = fr.shape[0]
        int ncols = fr.shape[1]
        int i, j
    for i in range(nrows-1):
        for j in range(ncols-1):
            div[i, j] = (fr[i+1, j] - fr[i, j]) + (fc[i, j+1] - fc[i, j])
    if ebc == EBC_Circular:
        div[nrows-1, ncols-1]=(fr[0, ncols-1] - fr[nrows-1, ncols-1]) + (fc[nrows-1, 0] - fc[nrows-1, ncols-1])
        for i in range(nrows-1):
            div[i, ncols-1]=(fr[i+1, ncols-1]-fr[i, ncols-1]) + (fc[i, 0] - fc[i, ncols-1])
        for j in range(ncols-1):
            div[nrows-1, j]=(fr[0, j]-fr[nrows-1, j]) + (fc[nrows-1, j+1]-fc[nrows-1, j])
    elif ebc == EBC_DirichletZero:
        div[nrows-1, ncols-1]=(-fr[nrows-1, ncols-1]) + (-fc[nrows-1, ncols-1])
        for i in range(nrows-1):
            div[i, ncols-1]=(fr[i+1, ncols-1]-fr[i, ncols-1]) + (-fc[i, ncols-1])
        for j in range(ncols-1):
            div[nrows-1, j]=(-fr[nrows-1, j]) + (fc[nrows-1, j+1]-fc[nrows-1, j])
    elif ebc == EBC_VonNeumanZero:
        div[nrows-1, ncols-1] = 0
        for i in range(nrows-1):
            div[i, ncols-1]=(fr[i+1, ncols-1]-fr[i, ncols-1])
        for j in range(ncols-1):
            div[nrows-1, j] = (fc[nrows-1, j+1]-fc[nrows-1, j])
    else:
        return -1
    return 0


cdef int computeDivergence_Backward(double[:,:] fr, double[:,:] fc, double[:,:] div, EBoundaryCondition ebc)nogil:
    cdef:
        int nrows = fr.shape[0]
        int ncols = fr.shape[1]
        int i, j
    for i in range(1, nrows):
        for j in range(1, ncols):
            div[i, j] = (fr[i, j] - fr[i-1, j]) + (fc[i, j]-fc[i, j-1])
            
    if ebc == EBC_Circular:
        div[0, 0] = (fr[0, 0] - fr[nrows-1, 0]) + (fc[0,0] - fc[0, ncols-1])
        for i in range(1, nrows):
            div[i, 0] = (fr[i, 0] - fr[i-1, 0]) + (fc[i, 0] - fc[i, ncols-1])
        for j in range(1, ncols):
            div[0, j] = (fr[0, j] - fr[nrows-1, j]) + (fc[0, j] - fc[0, j-1])
    elif ebc == EBC_DirichletZero:
        div[0, 0]=fr[0, 0] + fc[0, 0]
        for i in range(1, nrows):
            div[i, 0] = (fr[i, 0] - fr[i-1, 0]) + fc[i, 0]
        for j in range(1, ncols):
            div[0, j]=fr[0, j] + (fc[0, j]-fc[0, j-1])
    elif ebc == EBC_VonNeumanZero:
        div[0, 0]=0
        for i in range(1, nrows):
            div[i, 0]=(fr[i, 0] - fr[i-1, 0])
        for j in range(1, ncols):
            div[0, j]=fc[0, j]-fc[0, j-1]
    else:
        return -1
    return 0


cdef int computeDivergence(double[:,:] fr, double[:,:] fc, double[:,:] div, EDerivativeType edt, EBoundaryCondition ebc)nogil:
    cdef:
        int nrows = fr.shape[0]
        int ncols = fr.shape[1]

    if nrows < 2 or ncols < 2:
        return -1

    if edt == EDT_Forward:
        computeDivergence_Forward(fr, fc, div, ebc)
    elif edt == EDT_Backward:
        computeDivergence_Backward(fr, fc, div, ebc)
    else:
        return -1
    return 0

