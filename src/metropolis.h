#ifndef METROPOLIS_H
#define METROPOLIS_H

struct _lat
{
    int *lattice;
    int n;
    int n2;
    int total_flips;
    float T, J, B;
    float exps[2];
    int W, N, E, S;
    int opposites;
    int *energy, *magnet;
};

typedef struct _lat Lattice;

int init(Lattice *self, int n);
int set(Lattice *self, float T, float J, float B);
int info(Lattice *self);
int metropolis(Lattice *self, int pasos);
int pick_site(Lattice *self);
int flip(Lattice *self, int idx);
int find_neighbors(Lattice *self, int idx);
int cost(Lattice *self, int idx);
int accept_flip(Lattice *self, int idx, int opposites);
int calc_energy(Lattice *self, int idx);
int calc_energy(Lattice *self, int idx);
int calc_lattice(Lattice *self);
#endif
