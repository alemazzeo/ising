#ifndef METROPOLIS_H
#define METROPOLIS_H
int metropolis(int *lattice, int n, float *T, int pasos, int *energy, int *magnet);
int pick_site(int *lattice, int n);
int flip(int *lattice, int n, float *T, int idx, int *energy, int *magnet);
int find_neighbors(int *lattice, int n, int idx, int *W, int *N, int *E, int *S);
int cost(int *lattice, int n, int idx, int *W, int *N, int *E, int *S);
int accept_flip(int *lattice, int n, int idx, int opposites, int *energy, int *magnet);
int calc_energy(int *lattice, int n, int idx);
int calc_energy(int *lattice, int n, int idx);
int calc_lattice(int *lattice, int n, int *energy, int *magnet);
#endif
