import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from fenics import *
import dolfin as fe
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt 
import sys
from mshr import *

### Définir les constantes 
fire_temp = fe.Constant(1073.0) 
sigma = fe.Constant(5.64e-8)
mu = fe.Constant(1.9e-5)
wall_temp = fe.Constant(273)


def maillage_carre():
    '''
    Définition du maillage simplifié modélisé par un carré avec 30 points
    :param out : 
    ds, dx, u,v, u_gamma_dir,V, mesh
    '''
    ###############
    ###### Définition du domaine 
    NP = 30;
    mesh = UnitSquareMesh(NP, NP)

    ### Définition des murs de gauche de droite et du plafond
    class Wall(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and (
                (fe.near(x[0], 1)) ## mur de droite
                or (fe.near(x[0], 0)) ## mur de gauche 
                or (fe.near(x[1], 0)) ##plafond
            )
    ### Définition du domaine ou se trouve le feu (en bas à droite)
    class Gamma_Fire(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and (
                (fe.near(x[1], 0)  and (x[0] >= 0.5 and x[0]<= 1))
            )
    ### Définition du domaine avec dirichlet homogène (en bas à droite)
    class Gamma_Dir(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and (
                (fe.near(x[1], 0)  and (x[0] >= 0 and x[0]<= 0.5))
            )

    sub_boundaries = fe.MeshFunction("size_t", mesh, mesh.topology().dim()-1) # mesh.topology().dim()-1
    sub_boundaries.set_all(0)

    wall = Wall()
    wall.mark(sub_boundaries, 1)

    fire = Gamma_Fire()
    fire.mark(sub_boundaries, 2)

    gamma_dir = Gamma_Dir()
    gamma_dir.mark(sub_boundaries, 3)

    domains = fe.MeshFunction("size_t", mesh, mesh.topology().dim())  # CellFunction
    domains.set_all(0)

    # Redéfinissions des intégrales sur les frontières
    ds = fe.Measure('ds', domain=mesh, subdomain_data=sub_boundaries)

    # Domaine intérieur
    dx = fe.Measure("dx", domain=mesh, subdomain_data=domains)
    
    ############################################################
    ################################
    #### Construction des P-2 Lagrande et définition de u et v
    k = 2 ; print('Order of the Lagrange FE k = ', k)
    V = fe.FunctionSpace(mesh, "CG", int(k)) # Lagrange FE, order k
    # Fonctions trial et test 
    u = fe.TrialFunction(V)
    v = fe.TestFunction(V)


    ################################
    ###### Conditions de dirichlet
    u_diri_homog = Expression('0.', degree=u.ufl_element().degree())             
    u_diri_fire = fe.DirichletBC(V, u_diri_homog, fire)
    u_diri_wall = fe.DirichletBC(V, wall_temp, wall)
    u_gamma_dir  = fe.DirichletBC(V, u_diri_homog, gamma_dir)

    return ds, dx, u,v, u_gamma_dir,V, mesh




def stationnaire_carre():
    """
    Solution du problème stationnaire avec un maillage carré avec
    l'utilisation de l'algorithme de newton raphson pour linéariser 
    
    """
    #Récupération du maillage carré
    ds, dx, u,v, u_gamma_dir,V, mesh=maillage_carre()
    du = fe.TrialFunction(V)
    un, dun = fe.Function(V), fe.Function(V)
    #############################
    ########## Solution initiale
    u_init = fe.Expression("273.0", degree=1)
    u = fe.interpolate(u_init, V)
    un.assign(u)
    i = 0
    #### Définitions du nombre max d'itérations et de la tolérance pour sortir de la boucle
    i_max = 100 # max of iterations
    eps_du = 1e-9 # tolérance sur la norme 
    error = eps_du+1

    #### Boucle de l'algorithme Newton Raphson
    while (error>eps_du and i<i_max): 
        i+=1 # Update le nb itéraction actuel
        print("Newton-Raphson iteration #",i," commence...")
        # LHS
        a = mu*fe.inner(grad(du), grad(v))*dx-sigma*4*un**3*du*v*ds(2)
        # RHS
        L = - mu*fe.inner(grad(un), grad(v))*dx + sigma*(un**4-fire_temp**4)*v*ds(2)
        # Solve
        fe.solve(a == L, dun, u_gamma_dir)
        ## Calcul de l'erreur
        error = np.linalg.norm((dun.vector().get_local())) / np.linalg.norm(un.vector().get_local())
        un.assign(un+dun) # update la solution  
        print("Newton-Raphson iteration #",i,"; error = ", error)
        if (i == i_max):
            print("Warning: the algo exits because of the max number of ite ! error = ",error)
    if (i < i_max):
        print("* Newton-Raphson algorithm has converged: the expected stationarity has been reached. eps_du = ",eps_du)
    #
    # Plots
    #
    fe.plot(mesh)
    p=fe.plot(un, title='Solution par algorithme de Newton-Raphson')
    p.set_cmap("rainbow"); plt.colorbar(p); plt.show()
    plt.show()

def non_stationnaire_carre():
    """
    Solution du problème non stationnaire avec un maillage carré avec
    l'utilisation de l'algorithme de Newton Raphson pour linéariser 
    """
    #Récupération du maillage carre
    ds, dx, u,v, u_gamma_dir,V, mesh=maillage_carre()
    # ##### Initialisation du temps 
    t1 = 0.0
    t2 = 150000
    n = 100
    dt = (t2 - t1) / n 

    print('##################################################################')
    print('#')
    print('# Algorithme de Newton-Raphson:)')
    print('#')
    print('##################################################################')

    print('#')
    print('# Iterations')
    print('#') 

    #### Définition du nb max d'itérations et de la tolérance pour sortir de la boucle si convergence
    i_max = 100 
    eps_du = 1e-9 
    
    du = fe.TrialFunction(V)
    un, un_plus_1, dun = fe.Function(V), fe.Function(V), fe.Function(V)
    ### Initialisation de la solution
    u_init = fe.Expression("273.0", degree=1)
    u = fe.interpolate(u_init, V)
    un.assign(u)
    un_plus_1.assign(u)
    nb_pas = 100
    #Boucle sur le pas de temps
    for t_ in range(nb_pas):
        un_plus_1.assign(un)
        i = 0
        error = eps_du+1 
        ### Newton-Raphson
        while (error>eps_du and i<i_max): 
            i+=1 # update l'itération actuelle
            print("Newton-Raphson iteration #",i," begins...")
            # LHS
            a = du * v * dx + dt * mu * inner(grad(du), grad(v)) * dx - dt * sigma * 4 * un_plus_1**3 * du * v * ds(2)
            # RHS 
            L = - dt * sigma * fire_temp**4 * v * ds(2) + un * v * dx - un_plus_1 * v * dx -dt * mu * inner(grad(un_plus_1), grad(v)) * dx+ dt * sigma * un_plus_1**4 * v * ds(2)
            # Solve
            fe.solve(a == L, dun, u_gamma_dir)
            #Calcul de l'erreur
            error = np.linalg.norm((dun.vector().get_local())) / np.linalg.norm(un_plus_1.vector().get_local())
            un_plus_1.assign(un_plus_1+dun) #Update de la solution
            print("Newton-Raphson iteration #",i,"; error = ", error)
            if (i == i_max):
                print("Warning: the algo exits because of the max number of ite ! error = ",error)
        un.assign(un_plus_1) # update the solution
        fe.plot(mesh)
        p=fe.plot(un, title=f'Solution à t =  {t_}')
        p.set_cmap("rainbow")
        plt.pause(0.01)
    plt.show()
            
    if (i < i_max):
        print("* Newton-Raphson algorithm has converged: the expected stationarity has been reached. eps_du = ",eps_du)
    # #
    # # Plots de la solution sur la maillage
    # #
    fe.plot(mesh)
    p=fe.plot(un, title='Solution non stationnaire avec maillage carré')
    p.set_cmap("rainbow"); plt.colorbar(p); plt.show()
    plt.show()
    

def maillage_maison():

    '''
    Définition du maillage réaliste d'une maison
    u,v,dx,ds, u_dir, plancher, V, mesh
    '''
    mesh = Mesh()

    domain_vertices = [Point(0.0, 0.0),
                    Point(2.5, 0.0),
                    Point(2.5, 0.5),
                    Point(3.5, 0.5),
                    Point(3.5, 0.0),
                    Point(6.0, 0.0),
                    Point(6.0, 3.0),
                    Point(5.0, 3.0),
                    Point(5.0, 4.0),
                    Point(4.0, 4.0),
                    Point(4.0, 3.0),
                    Point(0.0, 3.0)]

    # Génération du mesh et du domaine
    domain = Polygon(domain_vertices)
    mesh   = generate_mesh(domain, 50)
    fe.plot(mesh)
    plt.show(block=True)
    class new_walls(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and (near(x[0], 0.) 
                                    or (near(x[0],6.) and (x[1]>=2.5))
                                    or (near(x[0],6.) and (0<=x[1]<=1.5 ))) 
    class rooftop(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and fe.near(x[1], 3.) and ((x[0]<=4) or (x[0] >= 5))
    class window_right(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and (fe.near(x[0], 6.) and ( 1.5 <= x[1] <=2.5))
    class plancher(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and fe.near(x[1], 0.) and (x[0] <= 2.5 or x[0] >= 3.5)
    class chimney(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and ((fe.near(x[0],4) and (x[1] >= 3))
                    or (fe.near(x[0],5) and (x[1]>=3))
                    or fe.near(x[1],4))

    class new_fire(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and (((x[1] <= 0.5 and fe.near(x[0],2.5))) or (x[1] <= 0.5 and fe.near(x[0],3.5)) or (fe.near(x[1],0.5) and (2.5<=x[0]<=3.5)))

    sub_boundaries =  MeshFunction("size_t", mesh, mesh.topology().dim()-1)

    sub_boundaries.set_all(0)

    new_wall = new_walls()
    new_wall.mark(sub_boundaries, 1)


    window = window_right()
    window.mark(sub_boundaries, 2)

    plancher = plancher()
    plancher.mark(sub_boundaries, 3)

    chimney = chimney()
    chimney.mark(sub_boundaries, 4)

    roof = rooftop()
    roof.mark(sub_boundaries, 5)

    fire2 = new_fire()
    fire2.mark(sub_boundaries, 6)

    domains = fe.MeshFunction("size_t", mesh, mesh.topology().dim())  # CellFunction
    domains.set_all(0)

    ds =  fe.Measure('ds', domain=mesh, subdomain_data=sub_boundaries)

    dx =  fe.Measure('dx', domain=mesh, subdomain_data=domains)
    ## Définition des P2-Lagrange
    k = 2 ; print('Order of the Lagrange FE k = ', k)
    V = fe.FunctionSpace(mesh, "CG", int(k))
    u = fe.TrialFunction(V)
    v = fe.TestFunction(V)
    ### conditions de dirichlet homogène
    u_diri_homog = Expression('0.', degree=u.ufl_element().degree())   
    u_dir  = fe.DirichletBC(V, u_diri_homog, plancher)
    return u,v,dx,ds, u_dir, plancher, V, mesh


def stationnaire_maison():
    """
    Solution du problème stationnaire avec le maillage réaliste d'une maison
    """
    ## Récupération du maillage réaliste
    u,v,dx,ds, u_dir, _, V, mesh= maillage_maison()
    i_max = 100 # max d'itérations
    eps_du = 1e-9 # tolerance de l'erreur
    du = fe.TrialFunction(V)
    un, dun = fe.Function(V), fe.Function(V)
    ### Initialisation de la solution
    u_init = fe.Expression("273.0", degree=2)
    u = fe.interpolate(u_init, V)
    un.assign(u)
    i = 0
    error = eps_du+1 
    ##Newton-Raphson
    while (error>eps_du and i<i_max): 
        i+=1 
        print("Newton-Raphson iteration #",i," commence...")
        # LHS
        a = mu*fe.inner(grad(du), grad(v))*dx-sigma*4*un**3*du*v*ds(6)
        # RHS
        L = - mu*fe.inner(grad(un), grad(v))*dx + sigma*(un**4-fire_temp**4)*v*ds(6)
        # Solve sur dirichlet homogène
        fe.solve(a == L, dun, u_dir)
        ## Calcul de l'erreur
        error = np.linalg.norm((dun.vector().get_local())) / np.linalg.norm(un.vector().get_local())
        un.assign(un+dun) # update the solution  
        print("Newton-Raphson iteration #",i,"; error = ", error)
        if (i == i_max):
            print("Warning: the algo exits because of the max number of ite ! error = ",error)
    if (i < i_max):
        print("* Newton-Raphson algorithm has converged: the expected stationarity has been reached. eps_du = ",eps_du)
    #
    # Plots
    #
    fe.plot(mesh)
    p=fe.plot(un, title='Solution du problème stationnaire avec maillage réaliste')
    p.set_cmap("rainbow"); plt.colorbar(p); plt.show()
    plt.show()


def non_staionnaire_maison():
    """
    Solution du problème non stationnaire avec le maillage d'une maison 
    """
    #Récupération du maillage de la maison
    u,v,dx,ds, _, plancher, V, mesh = maillage_maison()
    u = fe.TrialFunction(V)
    v = fe.TestFunction(V)
    ### Définition de la condition de dirichlet homogène
    u_diri_homog = Expression('0.', degree=u.ufl_element().degree()) 
    u_gamma_dir  = fe.DirichletBC(V, u_diri_homog, plancher)
    i_max = 100 # max d'itérations
    eps_du = 1e-9 # tolerance de l'erreur 

    du = fe.TrialFunction(V)
    un, un_plus_1, dun = fe.Function(V), fe.Function(V), fe.Function(V)
    ### Initialisation de la solution 
    u_init = fe.Expression("273.0", degree=1)
    u = fe.interpolate(u_init, V)
    un.assign(u)
    un_plus_1.assign(u)
    #Nb de pas de temps
    nb_pas = 100
    #Définition de delta t
    t1 = 0.0
    t2 = 150000
    n = 100
    dt = (t2 - t1) / n
    ##Boucle sur le pas de temps
    for t_ in range(nb_pas):
        un_plus_1.assign(un)
        i = 0
        error = eps_du+1
        ##Newton-Raphson
        while (error>eps_du and i<i_max): 
            i+=1 # update l'itération actuelle
            print("Newton-Raphson iteration #",i," commence...")
            # LHS 
            a = du * v * dx + dt * mu * inner(grad(du), grad(v)) * dx - dt * sigma * 4 * un_plus_1**3 * du * v * ds(6)
            # RHS 
            L = - dt * sigma * fire_temp**4 * v * ds(6) + un * v * dx - un_plus_1 * v * dx -dt * mu * inner(grad(un_plus_1), grad(v)) * dx+ dt * sigma * un_plus_1**4 * v * ds(6)
            # Solve sur dirichlet homogène
            fe.solve(a == L, dun, u_gamma_dir)
            #Calcul de l'erreur
            error = np.linalg.norm((dun.vector().get_local())) / np.linalg.norm(un_plus_1.vector().get_local())
            un_plus_1.assign(un_plus_1+dun)
            print("Newton-Raphson iteration #",i,"; error = ", error)
            if (i == i_max):
                print("Warning: the algo exits because of the max number of ite ! error = ",error)
        un.assign(un_plus_1) # update la solution
        fe.plot(mesh)
        p=fe.plot(un, title=f'Solution à t =  {t_}')
        p.set_cmap("rainbow")
        plt.pause(0.01)
    plt.show()
            
    if (i < i_max):
       print("* Newton-Raphson algorithm has converged: the expected stationarity has been reached. eps_du = ",eps_du)
    # #
    # # Plots
    # #
    fe.plot(mesh)
    p=fe.plot(un, title='Solution du problème non stationnaire sur le maillage de la maison')
    p.set_cmap("rainbow"); plt.colorbar(p); plt.show()
    plt.show()

def stationnaire_maison_avec_flux():
    """
    Solution stationnaire sur le maillage de la maison avec ajout des termes de conduction 
    """
    #Récupération du maillage de la maison
    u,v,dx,ds, _, _, V, mesh = maillage_maison()
    ##Définition des constantes qui sont maintenant non nulles
    f = fe.Constant(293) 
    c_wall = fe.Constant(1.1)  
    c_roof = fe.Constant(1.1)  
    c_chimney = fe.Constant(1.1)  
    c_plancher = fe.Constant(1.1)  
    c_window = fe.Constant(3) 

    v = fe.TestFunction(V)
    i_max = 100 # max d'iterations
    eps_du = 1e-6 # tolerance de l'erreur
    du = fe.TrialFunction(V)
    dun = fe.Function(V)
    un = fe.TrialFunction(V)
    ##Initialisation de la solution
    u_init = fe.Expression("273.0", degree=1)
    un = fe.interpolate(u_init, V)
    i = 0
    error = eps_du+1 
    ##Newton-Raphson
    while (error>eps_du and i<i_max): 
        i+=1 
        print("Newton-Raphson iteration #",i," commence...")
        a = fe.inner (mu * grad(du), grad(v)) * dx - sigma * 4 * un**3 * du * v * ds (6) + c_wall * du * v * ds (1)  
        + c_window * du * v * ds (2)  + c_roof * du * v * ds(5) 
        + c_chimney * du * v * ds (4) + c_plancher * du * v * ds(3) 
        L = -fe.inner (mu * grad(un), grad(v)) * dx + sigma * (un**4 - fire_temp**4) * v * ds (6) 
        + c_wall * (u_init-un) * v * ds (1)  + c_window * (u_init-un) * v * ds (2)  
        + c_roof * (u_init-un) * v * ds (5) 
        + c_chimney * (u_init-un) * v * ds (4) 
        + c_plancher * (u_init-un) * v * ds(3)+ fe.inner(v, f)*dx
        fe.solve(a == L, dun)
        #Calcul de l'erreur
        error = np.linalg.norm((dun.vector().get_local())) / np.linalg.norm(un.vector().get_local())
        un.assign(un+dun) # update the solution  
        print("Newton-Raphson iteration #",i,"; error = ", error)
        if (i == i_max):
            print("Warning: the algo exits because of the max number of ite ! error = ",error)
        
    if (i < i_max):
        print("* Newton-Raphson algorithm has converged: the expected stationarity has been reached. eps_du = ",eps_du)
    #
    # Plots
    #
    fe.plot(mesh)
    p=fe.plot(un, title='Solution stationnaire avec maillage de la maison et ajout des flux')
    p.set_cmap("rainbow"); plt.colorbar(p); plt.show()
    plt.show()

def non_stationnaire_maison_avec_flux():
    """
    Solution non stationnaire sur le maillage de la maison avec ajout des termes de conductions 
    """
    #Récupération du maillage de la maison
    u,v,dx,ds, _, plancher, V, mesh = maillage_maison()
    ##Définition des constantes non nulles
    f = fe.Constant(293) 
    c_wall = fe.Constant(1.1)  
    c_roof = fe.Constant(1.1)  
    c_chimney = fe.Constant(1.1)  
    c_plancher = fe.Constant(1.1)  
    c_window = fe.Constant(3) 
    
    u = fe.TrialFunction(V)
    v = fe.TestFunction(V)
    i_max = 100 # max of iterations
    eps_du = 1e-9 # tolerance on the relative norm
    du = fe.TrialFunction(V)
    dun = fe.Function(V)
    un, un_plus_1, dun = fe.Function(V), fe.Function(V), fe.Function(V)
    ## Initialisation de la solution 
    u_init = fe.Expression("273.0", degree=1)
    u = fe.interpolate(u_init, V)
    un = fe.interpolate(u_init, V)
    i = 0
    error = eps_du+1 
    #Pas de temps et delta t 
    nb_pas = 100
    t1 = 0.0
    t2 = 150000
    n = 100
    dt = (t2 - t1) / n
    ##Boucle sur le temps
    for t_ in range(nb_pas):
        un_plus_1.assign(un)
        i = 0
        error = eps_du+1
        ##Newton-Raphson
        while (error>eps_du and i<i_max): 
            i+=1 # update l'itération actuelle
            print("Newton-Raphson iteration #",i," begins...")
            # LHS 
            a = du * v * dx + dt * mu * inner(grad(du), grad(v)) * dx - dt * sigma * 4 * un_plus_1**3 * du * v * ds(6)
            + c_wall * du * v * ds (1)  
            + c_window * du * v * ds (2)  + c_roof * du * v * ds(5) 
            + c_chimney * du * v * ds (4) + c_plancher * du * v * ds(3) 
            # RHS 
            L = - dt * sigma * fire_temp**4 * v * ds(6) + un * v * dx - un_plus_1 * v * dx -dt * mu * inner(grad(un_plus_1), grad(v)) * dx+ dt * sigma * un_plus_1**4 * v * ds(6)
            + c_wall * (u_init-un) * v * ds (1)  + c_window * (u_init-un) * v * ds (2)  
            + c_roof * (u_init-un) * v * ds (5) 
            + c_chimney * (u_init-un) * v * ds (4) 
            + c_plancher * (u_init-un) * v * ds(3) + fe.inner(v, f)*dx
            # Solve
            fe.solve(a == L, dun)
            #Calcul de l'erreur 
            error = np.linalg.norm((dun.vector().get_local())) / np.linalg.norm(un_plus_1.vector().get_local())
            un_plus_1.assign(un_plus_1+dun)
            print("Newton-Raphson iteration #",i,"; error = ", error)
            if (i == i_max):
                print("Warning: the algo exits because of the max number of ite ! error = ",error)
        un.assign(un_plus_1) # update the solution
        fe.plot(mesh)
        p=fe.plot(un, title=f'Solution à t =  {t_}')
        p.set_cmap("rainbow")
        plt.pause(0.01)
    plt.show()
    if (i < i_max):
        print("* Newton-Raphson algorithm has converged: the expected stationarity has been reached. eps_du = ",eps_du)
    #
    # Plots
    #
    fe.plot(mesh)
    p=fe.plot(un, title='Solution non stationnaire maillage maison et ajout des flux')
    p.set_cmap("rainbow"); plt.colorbar(p); plt.show()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "stationnaire_carre":
        stationnaire_carre()
    elif len(sys.argv) == 2 and sys.argv[1] == "non_stationnaire_carre":
        non_stationnaire_carre()
    elif len(sys.argv) == 2 and sys.argv[1] == "stationnaire_maison":
        stationnaire_maison()
    elif len(sys.argv) == 2 and sys.argv[1] == "non_stationnaire_maison":
        non_staionnaire_maison()
    elif len(sys.argv) == 2 and sys.argv[1] == "stationnaire_maison_flux":
        stationnaire_maison_avec_flux()
    elif len(sys.argv) == 2 and sys.argv[1] == "non_stationnaire_maison_flux":
        non_stationnaire_maison_avec_flux()
    else:
        print("Argument invalide. Veuillez choisir parmi : stationnaire_carre, non_stationnaire_carre, stationnaire_maison, non_stationnaire_maison, stationnaire_maison_flux, non_stationnaire_maison_flux")
