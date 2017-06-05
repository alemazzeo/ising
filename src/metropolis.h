#ifndef METROPOLIS_H
#define METROPOLIS_H

struct _ising
{
    int   *_p_lattice;
    int   _n;
    int   _n2;
    int   _flips;
    int   _total_flips;
    float _T, _J, _B;
    float _p_dEs[10];
    float _p_exps[10];
    int   _W, _N, _E, _S;
    int   _aligned;
    float _current_energy;
    int   _current_magnet;
    float *_p_energy;
    int   *_p_magnet;
};

typedef struct _ising Ising;

struct _sample
{
	int   _sample_size;
	int   _step_size;
	float _tolerance;
	float _T, _J, _B;
	float *_p_energy;
	int   *_p_magnet;
	int   *_p_flips;
	int   *_p_total_flips;
	float *_p_q;
};

typedef struct _sample Sample;

int   init            (Ising *self, int n);
int   set_params      (Ising *self, float T, float J, float B);
int   info            (Ising *self);
int   metropolis      (Ising *self);
int   run             (Ising *self, int ntry);
float run_until       (Ising *self, int steps, float tolerance);
int   run_sample      (Ising *self, Sample *sample);
int   pick_site       (Ising *self);
int   flip            (Ising *self, int idx);
int   find_neighbors  (Ising *self, int idx);
int   cost            (Ising *self, int idx);
int   try_flip        (Ising *self, float pi);
int   accept_flip     (Ising *self, int idx, int aligned);
float calc_pi         (Ising *self, int idx, int aligned);
float calc_energy     (Ising *self, int idx);
int   calc_magnet     (Ising *self, int idx);
int   calc_lattice    (Ising *self);
int   autocorrelation (float *x, float *result, int n);
#endif
