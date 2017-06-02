#ifndef METROPOLIS_H
#define METROPOLIS_H

struct _lat
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

typedef struct _lat Lattice;

struct _result
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

typedef struct _result Result;

int   init            (Lattice *self, int n);
int   set_params      (Lattice *self, float T, float J, float B);
int   info            (Lattice *self);
int   metropolis      (Lattice *self);
int   run             (Lattice *self, int ntry);
float run_until       (Lattice *self, int steps, float tolerance);
int   run_sample      (Lattice *self, Result *results);
int   pick_site       (Lattice *self);
int   flip            (Lattice *self, int idx);
int   find_neighbors  (Lattice *self, int idx);
int   cost            (Lattice *self, int idx);
int   try_flip        (Lattice *self, float pi);
int   accept_flip     (Lattice *self, int idx, int aligned);
float calc_pi         (Lattice *self, int idx, int aligned);
float calc_energy     (Lattice *self, int idx);
int   calc_magnet     (Lattice *self, int idx);
int   calc_lattice    (Lattice *self);
int   autocorrelation (float *x, float *result, int n, float xt, float xt2);
#endif
