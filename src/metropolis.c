#include "metropolis.h"
#include <stdlib.h>
#include <stdio.h>

int metropolis(int *lattice, int n, float *T) 
{
    int idx, dE;
    idx = pick_site(lattice, n);
    flip(lattice, n, T, idx);
    return 0;
}

int pick_site(int *lattice, int n) 
{
    return (int) (((float) rand() / RAND_MAX) * n * n);
}

int flip(int *lattice, int n, float *T, int idx) 
{
    int W, N, E, S;
    int opposites, dE, pi, n2=n*n;
    
    // Find neighbors's indexs
    find_neighbors(lattice, n, idx, &W, &N, &E, &S);
    
    // Calculate the amount of opposing spins (flip's cost)
    opposites = cost(lattice, n, idx, &W, &N, &E, &S);
	
    // If the flip does not increase energy (2 or less)
    if (opposites<=2)
    {
	// Flip spin and confirm
	lattice[idx] *= -1;
	return 1;
    }
    // Otherwise (dE = 4 y 8)
    else
    {
	// Take the value of pi = exp(-dE/T) from table T:
	// T[0] = [temperature]
	// T[1] = exp(-4/T)
	// T[2] = exp(-8/T)
	pi = T[opposites - 2];
	// Flip spin with probability pi and report
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
}


int find_neighbors(int *lattice, int n, int idx, int *W, int *N, int *E, int *S)
{
    int n2 = n*n;

    // Periodic boundary conditions
    
    *W = (idx - 1) % n + (idx/n) * n;   // left
    *N = (idx - n + n2) % n2;           // up
    *E = (idx + 1) % n + (idx/n) * n;   // right
    *S = (idx + n + n2) % n2;           // down

    // (idx +/- 1) % n ---------> scroll over column number
    // (idx/n) * n -------------> first element of row
    // (idx +/- n + n2) % n2 ---> scroll over rows

    return 0;
}

int cost(int *lattice, int n, int idx, int *W, int *N, int *E, int *S)
{
    // Calculate the amount of opposing spins (flip's cost)
    // (4, 2, 0, -2, 4) / 2 + 2 --> (4, 3, 2, 1, 0)
    return (lattice[*W] * lattice[idx] +
	    lattice[*N] * lattice[idx] +
	    lattice[*E] * lattice[idx] +
	    lattice[*S] * lattice[idx]) / 2 + 2;
}
