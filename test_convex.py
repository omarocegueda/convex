import convex
import matplotlib.pyplot as plt
import numpy as np

def linprog(c,A,b,maxIter,tol):
    error=1+tol
    x=np.zeros(shape=(A.shape[1],))
    y=np.zeros(shape=(A.shape[0],))
    L=6
    tau=1.0/L
    sigma=1.0/L
    niter=0
    theta=1
    while((niter<maxIter)and(tol<error)):
        print niter, x, c.dot(x), b-A.dot(x)
        niter+=1
        newx=x-tau*(A.T.dot(y)+c)
        newx=np.maximum(newx,0)
        y=y+sigma*(A.dot(newx+theta*(newx-x))-b)
        error=np.max(np.abs(x-newx))
        x[...]=newx[...]
    return x
    
def test_tv_l2():
    fname="data/cameraman_256.jpg"
    lambdaParam=0.05
    L=8
    tau=1.0/L
    sigma=1.0/L
    theta=0.5
    inputImage=plt.imread(fname)[...,0]
    inputImage=inputImage.astype(np.double)
    filtered=convex.filter_tv_l2(inputImage, lambdaParam, tau, sigma, theta)
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(inputImage, cmap=plt.cm.gray)
    plt.title("Input")
    plt.subplot(1,2,2)
    plt.imshow(filtered, cmap=plt.cm.gray)
    plt.title("Output")

def test_linprog_sc50b():
    from numpy import genfromtxt
    Aname='data/sc50b/A.csv'
    Aeqname='data/sc50b/Aeq.csv'
    bname='data/sc50b/b.csv'
    beqname='data/sc50b/beq.csv'
    fname='data/sc50b/f.csv'
    A=genfromtxt(Aname, delimiter=',')
    Aeq=genfromtxt(Aeqname, delimiter=',')
    b=genfromtxt(bname, delimiter=',')
    beq=genfromtxt(beqname, delimiter=',')
    c=genfromtxt(fname, delimiter=',')
    I=np.eye(len(b))
    Z=np.zeros(shape=(len(beq), len(b)))
    AA=np.bmat([[A, I],[Aeq,Z]])
    cc=np.append(c,np.zeros(shape=(len(b),)))
    bb=np.append(b,beq).transpose()
    x_augmented=np.array(convex.linprog_pd_small(cc, AA, bb,1000000,1e-14))
    x=x_augmented[:A.shape[1]]
    print 'Solution:',x
    print 'Minimum objective:', c.dot(x)
    print 'Maximum equality violation:',np.abs(Aeq.dot(x)-beq).max()
    print 'Maximum inequality violation:',np.maximum(A.dot(x)-b,0).max()
    
def test_linprog_densecolumns():
    from numpy import genfromtxt
    Aeqname='data/densecolumns/Aeq.csv'
    beqname='data/densecolumns/beq.csv'
    fname='data/densecolumns/f.csv'
    lbname='data/densecolumns/lb.csv'
    ubname='data/densecolumns/ub.csv'
    Aeq=genfromtxt(Aeqname, delimiter=',')
    beq=genfromtxt(beqname, delimiter=',')
    c=genfromtxt(fname, delimiter=',')
    lb=genfromtxt(lbname, delimiter=',')
    ub=genfromtxt(ubname, delimiter=',')
    maxIter=200000
#    x=np.array(convex.linprog_pd_small(c, Aeq, beq,maxIter,1e-11))
#    print 'Dense:'
#    print 'Solution:',x
#    print 'Minimum objective:', c.dot(x)
#    print 'Maximum equality violation:',np.abs(Aeq.dot(x)-beq).max()
    n=Aeq.shape[0]
    m=Aeq.shape[1]
    x=np.array(convex.linprog_pd_large(fname, Aeqname, beqname,n, m, maxIter,1e-11))
    print 'Sparse:'
    print 'Solution:',x
    print 'Minimum objective:', c.dot(x)
    print 'Maximum equality violation:',np.abs(Aeq.dot(x)-beq).max()

if __name__=="__main__":
    #test_tv_l2()
    #test_linprog_sc50b()
    test_linprog_densecolumns()