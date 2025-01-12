import numpy as np

def Wolfe(J, x, d, cost, grad, s=1., itermax=20, e1=1.e-4, e2=0.99):
    smin, smax = 0., np.inf  # reinit
    niter_wolfe = 0
    do_wolfe = True
    cost_new = cost
    grad_new = grad

    while do_wolfe and niter_wolfe < itermax:
        niter_wolfe += 1
        cost_new = J.eval(x + s*d)
        # nouveau cout > ancien cout
        if cost_new > cost + e1 * s * np.dot(grad, d): # avoid big steps
            smax = s
            s = 0.5 * (smax + smin)
        else:
            grad_new = J.grad(x + s*d)
            if np.dot(grad_new, d) < e2 * np.dot(grad, d): #avoid small steps
                smin = s
                s = min(0.5 * (smin + smax), 2*s)
            else: # les 2 conditions sont vraies -> on sort
                do_wolfe = False
    if do_wolfe: # we exited wolfe because nb of max iteration reached
        # on a pas converge, alors on prefere choisir un petit pas plutot qu'un grand
        s = smin
        cost_new = J.eval(x + s*d)
        grad_new = J.grad(x + s*d)

    return s, cost_new, grad_new


class BFGS() :
    def __init__(self, nb_stock_max=8) :
        self.nb_stock_max = nb_stock_max
        self.stock = []
        self.last_iter = None
        
    def push(self, x, grad) :
        if not self.last_iter is None :
            x1, grad1 = self.last_iter
            
            s = x - x1
            g = grad - grad1
            rho = 1 / (np.dot(s, g))

            if rho > 0 :
                self.stock.append([s, g, rho])
                while len(self.stock) > self.nb_stock_max :
                    self.stock.pop(0)
                 
            else :
                print('Did you do a Wolfe ?')
                self.stock = []
        self.last_iter = [np.copy(x), np.copy(grad)]
            
            
    
    def get(self, grad) :
        if len(self.stock) == 0 :
            return -grad
        
        r = -np.copy(grad)
        
        l_alpha = []
        for i, v in enumerate(reversed(self.stock)) :
            s, g, rho = v
            alpha = rho*np.dot(s, r)
            r = r - alpha*g
            l_alpha.append(alpha)

        l_alpha = list(reversed(l_alpha))
        
        s, g, rho = self.stock[-1]
        gamma = np.dot(s, g) / np.dot(g, g)
        r =  gamma * r 
        
        for v, alpha in zip(self.stock, l_alpha) :
            s, g, rho = v
            beta = rho*np.dot(g, r)
            r = r + (alpha - beta) * s
        
        return r
            
    