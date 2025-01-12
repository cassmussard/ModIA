import numpy as np
from numpy import linalg as LA
import scipy
import LibOptim

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

def Wolfe(J, x_0, e1=1.e-4, e2=0.99, itermax=500, itermax_W=20, tol=1.e-5):
    """
    Wolfe est un algo de choix de pas, la direction est deja donee (ici -grad).
    Ce qui coute cher c'est J.eval et J.grad, on cherche donc a minimiser ces appels.
    """
    x = np.copy(x_0)
    cost = J.eval(x_0)
    grad = J.grad(x_0)

    # init pour commencer la methode de gradient
    k = 0
    x_list = [np.copy(x)]
    cost_list = [cost]
    grad_list = [np.linalg.norm(grad)]
    step_list = []
    s = 1

    while k < itermax and np.linalg.norm(grad) > tol:
        k += 1
        d = -grad # choix de la direction de descente
        s, cost, grad = LibOptim.Wolfe(J, x, d, cost, grad, s, itermax=itermax_W, e1=e1, e2=e2)
        x = x + s*d

        x_list.append(np.copy(x))
        cost_list.append(cost)
        grad_list.append(np.linalg.norm(grad))
        step_list.append(s)

    iteration_data = {
        'x_list': x_list,
        'cost_list': cost_list,
        'norm_grad_list': grad_list,
        'step_list': step_list
    }

    return iteration_data

def Newton(J, x0, itermax=500, tol=1.e-12):
    
    x = np.copy(x0)
    cost=J.eval(x0)
    grad = J.grad(x0)
    x_list = [np.copy(x)]
    cost_list = [cost]
    grad_list = [np.linalg.norm(grad)]
    angle_list=[]
    s=1
    k=0
    while k < itermax and np.linalg.norm(grad)>tol:
        k+=1
        d= scipy.linalg.solve(J.Hess(x), -grad)
        x=x+s*d
        angle_list.append(np.dot(d, -grad)/(np.linalg.norm(d)*np.linalg.norm(grad)))
        cost= J.eval(x)
        grad=J.grad(x)
        x_list.append(np.copy(x))
        cost_list.append(cost)
        grad_list.append(np.linalg.norm(grad))
        r= {'x_list':x_list, 'cost_list':cost_list}
        r['norm_grad_list']=grad_list
        r['angle_list']=angle_list
    return r


def Newton_Wolfe(J, x0, itermax=500, tol=1.e-12):

    x = np.copy(x0)
    cost=J.eval(x0)
    grad = J.grad(x0)
    x_list = [np.copy(x)]
    cost_list = [cost]
    grad_list = [np.linalg.norm(grad)]
    angle_list=[]
    step_list=[]
    
    s=1
    k=0
    while k < itermax and np.linalg.norm(grad)>tol:
        
        k+=1
        d_newton= scipy.linalg.solve(J.Hess(x), -grad)
        angle = np.dot(d_newton, -grad)/(np.linalg.norm(d_newton)*np.linalg.norm(grad))
        angle_list.append(angle)
        if angle <0: 
            d = -grad
        else:
            d = d_newton
        
        s, cost, grad = LibOptim.Wolfe(J,x,d,cost,grad,1) ##Il faut bien mettre le pas à 1 sinon on prend le pas précédent
        
        x=x+s*d
        
        cost = J.eval(x)
        grad = J.grad(x)
        x_list.append(np.copy(x))
        cost_list.append(cost)
        grad_list.append(np.linalg.norm(grad))
        step_list.append(s)
        r= {'x_list':x_list, 'cost_list':cost_list}
        r['norm_grad_list']=grad_list
        r['angle_list']=angle_list
        r['step_list'] = step_list
    print(k)
    return r


def Newton_Wolfe_BFGS(J,x_0,itermax=500,tol=1.e-5) :
    BFGS = lib.BFGS()
    x = np.copy(x_0)
    x_list = []
    cost_list = []
    grad_list = []
    step_list = []
    angle_list = []
    k = 0
    err = tol+1
    s = 1.
    cost = J.eval(x)
    grad = J.grad(x)
    BFGS.push(x, grad)
    # SAVES
    x_list.append(np.copy(x))
    cost_list.append(cost)
    grad_list.append(npl.norm(grad))
    while k < itermax and npl.norm(grad) > tol :  
        k+=1
        d = BFGS.get(grad)
        angle = np.dot(d, -grad) / (npl.norm(d)*npl.norm(grad))
        angle_list.append(angle)
        d = d if angle > 1.e-4 else -grad

        s, cost, grad = lib.Wolfe(J, x, d, cost, grad)
        x = x + s*d # Méthode de gradient
        BFGS.push(x, grad)

        
        # SAVES
        x_list.append(np.copy(x))
        cost_list.append(cost)
        grad_list.append(npl.norm(grad))
        step_list.append(s)

    return {"x_list" : x_list, "cost_list" : cost_list, "grad_list" : grad_list, "step_list" : step_list, "angle_list" : angle_list}