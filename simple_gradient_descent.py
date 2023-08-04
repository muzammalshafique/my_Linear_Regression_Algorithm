import numpy as np

# Cost Function
def compute_cost(x,y,w,b):
    m=x.shape[0]
    cost=0
    for i in range(m):
        f_wb=w*x[i]+b
        cost=cost+(f_wb-y[i])**2
    total_cost=1/(2*m)*(cost)
    return total_cost

# Computing Grdient
def compute_gradient(x,y,w,b):
    m=x.shape[0]
    dj_dw=0
    dj_db=0
    for i in range(m):
        f_wb=w*x[i]+b
        dj_dw_1=(f_wb-y[i])*x[i]
        dj_db_1=f_wb-y[i]
        dj_dw+=dj_dw_1
        dj_db+=dj_db_1
    dj_dw=dj_dw/m
    dj_db=dj_db/m
    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, alpha, num_iters):
   b=b_in
   w=w_in
   for i in range(num_iters):
       dj_dw,dj_db=compute_gradient(x,y,w,b)
       w=w-alpha*dj_dw
       b=b-alpha*dj_db
   return w,b


x_train=np.array([1,2])
y_train=np.array([300, 500])
w=200
b=100
a=compute_cost(x_train,y_train,w,b)
b=compute_gradient(x_train,y_train,w,b)
c=gradient_descent(x_train,y_train,w,b,1e-2, 10000)
print("The cost is: ", a)
print("The gradient is: ", b)
print("The gradient descent is: ", c)