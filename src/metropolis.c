#include "metropolis.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

int metropolis(int *lattice, int n, float T) 
{
    int idx, dE;
    idx = pick_site(lattice, n);
    flip(lattice, n, T, idx);
    return 0;
}

int pick_site(int *lattice, int n) 
{
    return rand() % (n*n);
}

int flip(int *lattice, int n, float T, int idx) 
{
    int idx_l, idx_u, idx_r, idx_d, dE, r, pi, n2=n*n;
    
    idx_l = idx - 1;
    if (idx % n == 0) 
	idx_l += n;
    
    idx_u = idx - n;
    if (idx / n == 0) 
	idx_u += n2;
    
    idx_r = idx + 1;
    if (idx % n == n-1)
	idx_r -= n;

    idx_d = idx + n;
    if (idx / n == n-1)
	idx_d -= n2;    
    
    if (lattice[idx_l] == -lattice[idx]) 
	r++;
    if (lattice[idx_u] == -lattice[idx]) 
	r++;
    if (lattice[idx_r] == -lattice[idx]) 
	r++;
    if (lattice[idx_d] == -lattice[idx]) 
	r++;
	
    dE = (r-2) * 4;
    pi = exp(-dE / T);
    
    if (pi*RAND_MAX > rand()) 
    {
	lattice[idx] *= -1;
	return 1;
    }
    else
    {
	return 0;
    }
}

