#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cblas.h>
#include "utils.h"
#include "dsmat.h"
#include "gemms.h"

void p2p_transmit_A(int p, int q, Matrix *A, int i, int l) {
    int j;
    int me, my_row, my_col;
    MPI_Status status;

    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    node_coordinates_2i(p, q, me, &my_row, &my_col);

    Block *Ail;
    int node, tag, b;

    Ail = &A->blocks[i][l];
    b = A->b;
    tag = 0;
    if (me == Ail -> owner) {
        // Current process owns the block A[i,l]
        for (j = 0; j < q ; j++) {
            node = get_node(p, q, my_row, j);
            if(node != me){
                MPI_Ssend(Ail->c, b * b, MPI_FLOAT, node, tag, MPI_COMM_WORLD);
            } 
        }
    } else if (my_row == Ail->row) {
        // Current process is on the same column as the owner of the block
        Ail->c = malloc(b * b * sizeof(float));
        MPI_Recv(Ail->c, b * b, MPI_FLOAT, Ail -> owner, tag, MPI_COMM_WORLD, &status);
    }
}


void p2p_transmit_B(int p, int q, Matrix *B, int l, int j) {
    int i;
    int me, my_row, my_col;
    MPI_Status status;

    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    node_coordinates_2i(p, q, me, &my_row, &my_col);

    int node, tag, b;
    Block *Blj;

    Blj = &B->blocks[l][j];
    b = B->b;

    tag = 1;
    
    if (me == Blj -> owner) {
        // Current process owns the block B[l,j]
        for (i = 0; i < p; i++) {
            node =  get_node(p, q, i, my_col);
            if(node != me){
                MPI_Ssend(Blj->c, b * b, MPI_FLOAT, node, tag, MPI_COMM_WORLD);
            } 
        }
    } else if (my_col == Blj->col) {
        // Current process is on the same column as the owner of the block
        Blj->c = malloc(b * b * sizeof(float));
        MPI_Recv(Blj->c, b * b, MPI_FLOAT, Blj -> owner, tag, MPI_COMM_WORLD, &status);
    }
}


//Trace : on a 2 fois la meme chose car on fait 2 itérations 
