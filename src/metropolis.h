#ifndef METROPOLIS_H
#define METROPOLIS_H

struct _ising
{
    int    *_p_lattice;
    int    _n;
    int    _n2;
    int    _flips;
    int    _total_flips;
    double _T, _J, _B;
    double _p_dEs[10];
    double _p_exps[10];
    int    _W, _N, _E, _S;
    int    _aligned;
    double _current_energy;
    int    _current_magnet;
    double *_p_energy;
    int    *_p_magnet;
};

typedef struct _ising Ising;

struct _sample
{
    int    _n;
    int    _sample_size;
    int    _step_size;
    double _tolerance;
    double _T, _J, _B;
    double *_p_energy;
    int    *_p_magnet;
    int    *_p_flips;
    int    *_p_total_flips;
    double *_p_q;
};

typedef struct _sample Sample;

int    init            (Ising *self, int n);
int    set_params      (Ising *self, double T, double J, double B);
int    info            (Ising *self);
int    metropolis      (Ising *self);
int    run             (Ising *self, int ntry);
double run_until       (Ising *self, int steps, double tolerance);
int    run_sample      (Ising *self, Sample *sample);
int    pick_site       (Ising *self);
int    flip            (Ising *self, int idx);
int    find_neighbors  (Ising *self, int idx);
int    cost            (Ising *self, int idx);
int    try_flip        (Ising *self, double pi);
int    accept_flip     (Ising *self, int idx, int aligned);
double calc_pi         (Ising *self, int idx, int aligned);
double calc_energy     (Ising *self, int idx);
int    calc_magnet     (Ising *self, int idx);
int    calc_lattice    (Ising *self);
int    autocorrelation (double *x, double *result, int n);
#endif
