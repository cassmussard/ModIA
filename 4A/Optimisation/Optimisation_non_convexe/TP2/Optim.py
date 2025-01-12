import numpy as np
from numpy import linalg as LA

def deriv_num(J,a,d,compute_grad=True,compute_Hess=True) :
    eps_range=[0.1**(i+1) for i in range(14)]
    for eps in  eps_range:
        s='eps {:1.3e}'.format(eps)
        if compute_grad:
            ratio = (J.eval(a+eps*d)-J.eval(a))/(eps*np.dot(J.grad(a),d))
            s+='grad {:1.1e}'.format(np.abs(ratio-1))
        if compute_Hess:
            v1 = (J.grad(a+eps*d)-J.grad(a))/eps
            v2 = J.Hess(a).dot(d)
            s+='ratio {:1.1e}'.format(np.abs(np.linalg.norm(v1)/np.linalg.norm(v2)-1))
            s+='angle {:1.1e}'.format(np.abs(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))-1))
            
        print(s)

def fixed_step(J,x_0,step=1.,itermax=500,tol=1.e-5):
    x_list = []
    cost_list=[]
    grad_list = []
    x = np.copy(x_0)
    k = 0
    err = tol+1
    
    while (err > tol) and (k < itermax) :
        cost = J.eval(x)
        grad = J.grad(x)
        err = np.linalg.norm(grad)
        
        grad_list.append(err)
        x_list.append(x)
        cost_list.append(cost)
        
        x = x-step*J.grad(x)
        k+=1
    cost = J.eval(x)
    grad = J.grad(x)
    err = np.linalg.norm(grad)
    
    grad_list.append(err)
    x_list.append(x)
    cost_list.append(cost)
    
    
    return x_list, cost_list, grad_list

def Wolfe(J,x_0,e1=1.e-4,e2=0.99,itermax=500,itermax_W=20,tol=1.e-5):
    
    
    x = np.copy(x_0)
    cost = J.eval(x_0)
    grad = J.grad(x_0)
    
    x_list = [np.copy(x)]
    cost_list=[cost]
    grad_list = [np.linalg.norm(grad)]
    step_list = []
    
    k = 0
    s = 1
    
    while (np.linalg.norm(grad) > tol) and (k < itermax) :
        
        k+=1
        d = -grad
        smin, smax = 0,np.inf
        niter_Wolfe, do_Wolfe = 0,True
        
        while do_Wolfe and (niter_Wolfe<itermax_W):
            niter_Wolfe+=1
            cost_new = J.eval(x+s*d)
            
            if cost_new > cost+e1*s*np.dot(grad,d):
                smax = s
                s = (smax+smin)/2

            else:
                grad_new = J.grad(x+s*d)
                if np.dot(grad_new,d) < e2*np.dot(grad,d):
                    
                    smin= s
                    s = min((smin+smax)/2, 2*s)
                else : 
                    do_Wolfe= False
                    
        if do_Wolfe:
            print("Wolfe n'a pas convergé")
            s= smin
            cost_new = J.eval(x+s*d)
            grad_new = J.grad(x+s*d)
        x = x+s*d
        cost = cost_new
        grad = np.copy(grad_new)
        x_list.append(np.copy(x))
        cost_list.append(cost)
        grad_list.append(np.linalg.norm(grad))
        step_list.append(s)
        
    return x_list,cost_list, grad_list, step_list
    