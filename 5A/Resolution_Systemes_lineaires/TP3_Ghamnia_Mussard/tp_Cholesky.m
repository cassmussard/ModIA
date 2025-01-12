close all
clear all

load mat0.mat;
%load bcsstk27.mat

%% Pour les matrices hydcar, piston et pde voir le deuxieme fichier tp_LU
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Première partie facto Cholesky %%%
n= size(A,1);

b = [1:n]';

% Calcul de la solution
x_sol = A\b;


%Facto de cholesky 
[L, flag]= chol(A, 'lower');

spy(L);
spy(triu(A)+L);

y = L\b;
%%Erreur inverse
x1 = L'\y;

norm(b-A*x1)/norm(b)

%Erreur directe 
norm(x_sol-x1)/norm(x_sol)

subplot(2,3,1);
spy(A);
title('Original matrix A');

 
[count,h,parent,post,R] = symbfact(A);
ALU=R+R';
subplot(2,3,2)
spy(ALU);
title('Factors of A')
fillin=nnz(ALU)-nnz(A)

% visualisation du fill
C=spones(A);
CLU=spones(ALU);
FILL=CLU-C;
subplot(2,3,3)
spy(FILL)
title('Fill on original A')

% Permutation
%P = symamd(A);
%P = symrcm(A);
%P = amd(A);
%P = colamd(A);
P = colperm(A);
B = A(P, P);
subplot(2,3,4);
spy(B);
title('Permuted matrix B');

 
[count,h,parent,post,R] = symbfact(B);
BLU=R+R';
subplot(2,3,5)
spy(BLU);
title('Factors of B')
fillin=nnz(BLU)-nnz(B)

% visualisation du fill
C=spones(B);
CLU=spones(BLU);
FILL=CLU-C;
subplot(2,3,6)
spy(FILL)
title('Fill premuted matrix B')

L = chol(B, 'lower');
y = L\b(P);
z = L'\y;
x2(P) = z;
x2 = x2';

%Calcul du nombre d'opérations
fprintf("Nb opérations : %4d \n", 4*nnz(L)-2*n);

%Erreur directe 
fprintf("Erreur directe sur le résidu  : %4d \n", norm(b-A*x2)/norm(b));  
fprintf("Erreur directe sur la solution : %4d \n", norm(x_sol-x2)/norm(x_sol));  
