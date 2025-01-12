close all
clear all


%%% Avec ces matrices on ne peut pas utiliser cholesky car pas symétrique
%%% et définie positive => LU
%load hydcar20.mat
%load pde225_5e-1.mat
load piston.mat
n= size(A,1);

b = [1:n]';

x_sol = A\b;

[L, U, Pn]= lu(A);

fprintf("Nb non zéros A : %4d \n", nnz(A));

y = L\Pn*b;

%%Erreur inverse
x1 = U\y;

norm(b-A*x1)/norm(b)

%Erreur directe 
norm(x_sol-x1)/norm(x_sol)


% Permutation
%P = symamd(A);
P = symrcm(A);
%P = amd(A);
%P = colamd(A);
%P = colperm(A);

B = A(P, P);

 
[count,h,parent,post,R] = symbfact(B);
BLU=R+R';

%%Factorisation de la matrice permutée
[L, U, Pn]= lu(B);
y = L\b(P);
z = L'\y;
x2(P) = z;
x2 = x2';

%Calcul du nombre d'opérations
fprintf("Nb opérations : %4d \n", 2*nnz(L) - 2*n+2*nnz(U)-n);

%Erreur directe 
fprintf("Erreur directe sur le résidu  : %4d \n", norm(b-A*x2)/norm(b));  
fprintf("Erreur directe sur la solution : %4d \n", norm(x_sol-x2)/norm(x_sol));  
