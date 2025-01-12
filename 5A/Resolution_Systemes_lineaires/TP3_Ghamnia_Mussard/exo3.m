A = [1 0 0 1 0 0 0 1 0;
     0 1 0 0 1 1 0 0 0;
     0 0 1 0 0 0 0 1 0;
     1 0 0 1 0 1 1 0 0;
     0 1 0 0 1 0 0 0 1;
     0 1 0 1 0 1 0 1 0;
     0 0 0 1 0 0 1 0 0;
     1 0 1 0 0 1 0 1 0;
     0 0 0 0 1 0 0 0 1];
 
 %norm(A - A')
 
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
P = [3 7 9 5 2 1 4 6 8];
P = [9 5 2 6 4 8 3 7 1];
P = flip(P);
P = symrcm(A);
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
title('Fill on original B')


