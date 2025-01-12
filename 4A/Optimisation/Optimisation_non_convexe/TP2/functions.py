import numpy as np
import math

class square() :
    
    def __init__(self) :
        print("Fonction (x,y) --> x^2/2+7/2*y^2")
        self.size=2
        self.nb_eval = 0
        self.nb_grad=0
        self.nb_cost = 0
    def eval(self,x) :
        self.nb_eval+=1
        if not len(x)==self.size :
            print ("Erreur de taille de x, on a ",len(x)," au lieu de ",self.size)
        return 0.5*x[0]**2+7/2.*x[1]**2
    
    def grad(self,x) :
        self.nb_grad+=1
        return np.array([x[0],7*x[1]])
    
    def Hess(self,x) :
        self.nb_cost+=1
        to_return=np.zeros((2,2))
        to_return[0,0]=1
        to_return[1,1]=7
        return to_return

class Rosen():
    def __init__(self):
        self.size=2
        self.nb_eval = 0
        self.nb_grad=0
        self.nb_cost = 0
    def eval(self,x) :
        self.nb_eval +=1
        if not len(x)==self.size :
            print ("Erreur de taille de x, on a ",len(x)," au lieu de ",self.size)
        return 100*(x[1]-x[0]**2)**2+(1-x[0])**2
    def grad(self,x) :
        self.nb_grad+=1
        return np.array([400*(x[0]**3-x[1]*x[0])+2*(x[0]-1), 200*(x[1]-x[0]**2)])
    def Hess(self,x) :
        self.nb_cost+= 1
        to_return=np.zeros((2,2))
        to_return[0,0]=1200*x[0]**2-400*x[1]+2
        to_return[0,1]=-400*x[0]
        to_return[1,0]=-400*x[0]
        to_return[1,1]=200
        return to_return
    
class oscill():
    def __init__(self):
        self.size=2
        self.nb_eval = 0
        self.nb_grad=0
        self.nb_cost = 0
    def eval(self,x) :
        self.nb_eval+=1
        if not len(x)==self.size :
            print ("Erreur de taille de x, on a ",len(x)," au lieu de ",self.size)
        return 0.5*x[0]**2+x[0]*math.cos(x[1])
    def grad(self,x) :
        self.nb_grad+=1
        return np.array([x[0]+math.cos(x[1]),-x[0]*math.sin(x[1])])
    def Hess(self,x) :
        self.nb_cost+=1
        to_return=np.zeros((2,2))
        to_return[0,0]=1
        to_return[0,1]=-math.sin(x[1])
        to_return[1,0]=-math.sin(x[1])
        to_return[1,1]=-x[0]*math.cos(x[1])
        return to_return