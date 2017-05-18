#ifndef METROPOLIS_H
#define METROPOLIS_H
int metropolis(int *lattice, int n, float *T);
int pick_site(int *lattice, int n);
int flip(int *lattice, int n, float *T, int idx);
int find_neighbors(int *lattice, int n, int idx, int *W, int *N, int *E, int *S);
int cost(int *lattice, int n, int idx, int *W, int *N, int *E, int *S);
#endif
