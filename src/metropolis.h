#ifndef METROPOLIS_H
#define METROPOLIS_H

struct _lat
{
    int *_p_lattice;
    int _n;
    int _n2;
    int _total_flips;
    float _T, _J, _B;
    float _exps[2];
    int _W, _N, _E, _S;
    int _opposites;
    int *_p_energy, *_p_magnet;
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
