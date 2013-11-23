import convex
import matplotlib.pyplot as plt
import numpy as np
def test_convex(fname, lambdaParam, tau, sigma, theta):
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
if __name__=="__main__":
    fname="data/cameraman_256.jpg"
    lambdaParam=0.05
    L=8
    tau=1.0/L
    sigma=1.0/L
    theta=0.5
    test_convex(fname, lambdaParam, tau, sigma, theta)
