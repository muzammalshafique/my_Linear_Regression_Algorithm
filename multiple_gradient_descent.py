import numpy as np

def compute_cost(X, y, w, b):
    m = len(y)
    J = (np.sum((X.dot(w) + b - y)**2))/(2 * m)
    return J

# def compute_gradient(X,Y,w,b):
    m,n = X.shape
    dj_dw_s = []
    dj_db = 0
    for i in range(n):
        x_s = []
        for j in range(m):
            x = np.dot(X[j],w)+b-Y[j]
            x_s.append(x)
        y = np.dot(x_s, X[:,i])/m
        dj_db = np.sum(x_s)/m
        dj_dw_s.append(y)
    return dj_dw_s, dj_db

def compute_gradient(X,Y,w,b):
    m, n = X.shape
    dj_dw = np.zeros(n,)
    dj_db=0
    for i in range(m):
        err = np.dot(X[i],w)+b-y[i]
        for j in range(n):
                dj_dw[j]=dj_dw[j]+err * X[i,j]
        dj_db = dj_db + err
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, alpha, num_iters):
    b = b_in
    w = w_in
    for i in range(num_iters):
       dj_dw_s, dj_db = compute_gradient(x,y,w,b)
       
       dj_dw_s = np.array(dj_dw_s)
       w = w-alpha * dj_dw_s
       b = b-alpha * dj_db
    return w,b

X = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40],[1894, 2,1,43]])
y = np.array([460, 232, 330])
b = -4.8310239544923834e+123
w = np.array([-8.94017131e+126, -1.65577477e+124, -6.09468644e+123,-2.07697734e+125])
cost = compute_cost(X,y,w,b)
print("Cost is: ",cost)
gradient = compute_gradient(X, y, w, b)
print("Grdaient is: ", gradient)
gradient_descent = gradient_descent(X,y,w,0,0.0000006,60000)
print("Gradient Descent is: ", gradient_descent)