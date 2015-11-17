cdef enum EDerivativeType:
    EDT_Forward = 0 
    EDT_Backward = 1

cdef enum EBoundaryCondition:
    EBC_Circular = 0
    EBC_DirichletZero = 1
    EBC_VonNeumanZero = 2

cdef int computeGradient(double[:,:] f, double[:,:] dfdr, double[:,:] dfdc, EDerivativeType edt, EBoundaryCondition ebc)nogil
cdef int computeDivergence(double[:,:] fr, double[:,:] fc, double[:,:] div, EDerivativeType edt, EBoundaryCondition ebc)nogil
