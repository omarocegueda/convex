import convex
import matplotlib.pyplot as plt
import numpy as np
def test_convex(fname, lambdaParam, alpha0, alpha1, tau, sigma, theta):
    inputImage=plt.imread(fname)[...,0]
    inputImage=inputImage.astype(np.double)
    filtered, filteredGrad=convex.filter_tgv_l2(inputImage, lambdaParam, alpha0, alpha1, tau, sigma, theta)
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(inputImage, cmap=plt.cm.gray)
    plt.title("Input")
    plt.subplot(1,3,2)
    plt.imshow(filtered, cmap=plt.cm.gray)
    plt.title("Output")
    plt.subplot(1,3,3)
    filteredGrad=np.sum(np.array(filteredGrad)**2,0)
    plt.imshow(filteredGrad)
    plt.title("Smooth gradient norm")
if __name__=="__main__":
    fname="data/cameraman_256.jpg"
    lambdaParam=0.1
    alpha0=2.0
    alpha1=2.0
    L=8
    tau=1.0/L
    sigma=1.0/L
    theta=0.5
    test_convex(fname, lambdaParam, alpha0, alpha1, tau, sigma, theta)
