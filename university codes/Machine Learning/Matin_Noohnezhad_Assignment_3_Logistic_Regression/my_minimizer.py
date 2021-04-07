import costFunctionReg as cf
import numpy as np

def minimizer(X,y,lambda_,initial_theta,xtol=1e-8,alpha = 0.01 , max_repeat=8000):
    # print(initial_theta.shape)
    # print(X.shape)
    # print(y.shape)
    theta = initial_theta
    oldJ=100
    diff = 1
    # max_repeat = 8000
    repeat = 0
    while(diff>xtol and repeat<max_repeat):
        if(repeat*20%max_repeat == 0):
            print(str(int((repeat/max_repeat) *100))+' %')
        repeat = repeat+1
        [J, grad] = cf.costFunctionReg(theta, X, y, lambda_)
        grad = np.reshape(grad, (len(grad), 1))
        if (oldJ - J > 0):
            theta = np.subtract(theta, (alpha * grad))
        diff = oldJ - J;
    return theta









