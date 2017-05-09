#include "metropolis.h"

int metropolis(int *lattice, int n, float T) 
{
    pick_site(lattice, n);
    
    return 0;
}

int pick_site(int *lattice, int n) 
{
    return rand() % (n*n);
}

int flip(int *lattice, int n, float T, int idx) 
{
    int idx_l, idx_u, idx_r, idx_d, dE, r, n2=n*n;
    
    idx_l = idx - 1 + (idx % n == 0) * n;
    idx_u = idx - n + (idx / n == 0) * n2;
    idx_r = idx - 1 - (idx % n == n) * n;
    idx_d = idx - n - (idx / n == n) * n2;
    
    
    r = (*(lattice+idx_l)==-*(lattice+idx)) +
	(*(lattice+idx_u)==-*(lattice+idx)) +
	(*(lattice+idx_r)==-*(lattice+idx)) +
	(*(lattice+idx_d)==-*(lattice+idx)) - 2;
	
    dE = r * 4 * ((r>0)*(-2)+1);
	
    return 0;
}
