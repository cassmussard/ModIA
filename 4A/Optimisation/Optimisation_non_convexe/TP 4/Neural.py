import numpy as np

class MLP() :
    def __init__(self,inp,outp,p=2,q=20,r=1) :
        self.inp=inp
        self.outp=outp
        self.p=p # taille de la première couche
        self.q=q # taille de la deuxième couche
        self.r=1 # taille de la troisième couche
        self.ind_s=(0,p*q,p*q+q,p*q+q+q*r) # indice de debut de stockage des variables
        self.ind_e=(p*q,p*q+q,p*q+q+q*r,p*q+q+q*r+r) # indice de fin de stockage des variables
        self.shapes=((q,p),(q,1),(r,q),(r,1)) #taille des variables
        self.nb_params=self.ind_e[-1] #taille totale du vecteur de paramètres
    def eval(self,theta) :
        (inp1,inp2,inp3,inp4)=self.forward(theta)
        return inp4
    def grad(self,theta) :
        state=self.forward(theta)
        gstate,gtheta=self.backward(theta,state)
        return gtheta

    def get_matrices(self,theta) :
        A = theta[self.ind_s[0]:self.ind_e[0]].reshape(self.shapes[0])
        b = theta[self.ind_s[1]:self.ind_e[1]].reshape(self.shapes[1])
        C = theta[self.ind_s[2]:self.ind_e[2]].reshape(self.shapes[2])
        d = theta[self.ind_s[3]:self.ind_e[3]].reshape(self.shapes[3])
        return (A,b,C,d)
    def get_theta(self,matrices) :
        theta=np.zeros(self.nb_params)
        for (i_s,i_e,m,s) in zip(self.ind_s,self.ind_e,matrices,self.shapes) :
            assert m.shape==s
            theta[i_s:i_e]=m.ravel()
        return theta
    def product(self,A,b,x) :
        return A.dot(x)+np.outer(b,np.ones(x.shape[1]))
    def forward(self,theta) :
        (A, b, C, d) = self.get_matrices(theta)
        x = self.inp
        inp1 = self.product(A,b,x)
        inp2 = 1/(1+np.exp(-inp1))
        inp3 = self.product(C,d, inp2)
        inp4 = 0.5*np.linalg.norm(inp3-self.outp)**2 
        
        return  (inp1,inp2,inp3,inp4)
    def tangent(self,theta,x,dtheta):
        (dA, db, dC, dd) = self.get_matrices(dtheta)
        (A, b, C, d) =self.get_matrices(theta)
        (x1, x2, x3, x4) = x
        dx1 = self.product(dA, db, self.inp)
        dx2 = np.exp(-x1)/(1+np.exp(-x1))**2*dx1
        dx3 = self.product(dC,dd, x2)+C@dx2
        dx4 = np.sum((x3-self.outp)*dx3)
        return  (dx1,dx2,dx3,dx4)
    
    
    def backward(self,theta,x) :
        (A,b,C,d) = self.get_matrices(theta)
        (x1,x2,x3,x4) = x
        gx4 = np.array([1])
        gx3 = gx4*(x3-self.outp)
        gx2 = C.T@gx3
        gx1 = np.exp(-x1)/(1+np.exp(-x1))**2*gx2
        
        gd = np.sum(gx3,axis=1).reshape(d.shape)
        gc = gx3@x2.T
        gb = np.sum(gx1, axis=1).reshape(b.shape)
        ga = gx1@self.inp.T
        gx = [gx1,gx2,gx3,gx4]
        gtheta = self.get_theta([ga, gb, gc, gd])
        return (gx, gtheta)
