from total_variation import (filterTotalVariation_L2,
                             filterTotalVariation_L1,
                             filterHuber_L2,
                             filterTGV_L2)

def show_result(input, output):                             
    fig = figure()
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(input, cmap=cm.gray)
    ax2 = fig.add_subplot(1,2,2, sharex=ax1, sharey=ax1)
    ax2.imshow(output, cmap=cm.gray)

def test_tv_l2():                            
    input = imread('data/cameraman_256.jpg').astype(np.float64)
    input = input.mean(2)
    input = (input-input.min())/(input.max()-input.min())
    lmbd = 10
    tau = 0.5 / sqrt(12.0)
    sigma = 0.5 / sqrt(12.0)
    theta = 1
    output = np.zeros_like(input)
    filterTotalVariation_L2(input, lmbd, tau, sigma, theta, output)
    show_result(input, output)

def test_tv_l1():
    lmbd = 1.25
    tau = 0.5 / sqrt(12.0)
    sigma = 0.5 / sqrt(12.0)
    theta = 0.5
    input = imread('data/cameraman_256.jpg').astype(np.float64)
    input = input.mean(2)
    output = np.zeros_like(input)
    filterTotalVariation_L1(input, lmbd, tau, sigma, theta, output)
    show_result(input, output)

def test_huber_l2():
    alpha = 1.0
    lmbd = 0.03
    #lambda = 0.01 #for Salt & Pepper noise
    tau = 0.5 / sqrt(12.0)
    sigma = 0.5 / sqrt(12.0)
    theta = 0.5
    
    input = imread('data/cameraman_256.jpg').astype(np.float64)
    input = input.mean(2)
    output = np.zeros_like(input)
    filterHuber_L2(input, alpha, lmbd, tau, sigma, theta, output)
    show_result(input, output)

def test_tgv_l2():    
    alpha0 = .2
    alpha1 = .1
    lmbd = 0.01
    tau = 0.5 / sqrt(12.0)
    sigma = 0.5 / sqrt(12.0)
    theta = 1.0
    
    input = imread('data/cameraman_256.jpg').astype(np.float64)
    input = input.mean(2)
    output = np.zeros_like(input)
    filterTGV_L2(input, lmbd, alpha0, alpha1, tau, sigma, theta, output, None)
    show_result(input, output)
