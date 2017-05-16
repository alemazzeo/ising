#include "metropolis.h"
#include <stdlib.h>
#include <stdio.h>

int metropolis(int *lattice, int n, float T, float exp4, float exp8) 
{
    int idx, dE;
    idx = pick_site(lattice, n);
    flip(lattice, n, T, exp4, exp8, idx);
    return 0;
}

int pick_site(int *lattice, int n) 
{
    return (int) (((float) rand() / RAND_MAX) * n * n);
}

int flip(int *lattice, int n, float T, float exp4, float exp8, int idx) 
{
    int idx_l, idx_u, idx_r, idx_d, dE, pi;
    int r=0, n2=n*n;
    
    // Encuentra los indices de los vecinos
    idx_l = (idx - 1) % n + (idx/n) * n;   // izquierdo
    idx_u = (idx - n + n2) % n2;           // arriba
    idx_r = (idx + 1) % n + (idx/n) * n;   // derecho
    idx_d = (idx + n + n2) % n2;           // abajo
    
    // Calcula la cantidad de spins que se oponen
    r+= (lattice[idx_l] * lattice[idx] +
	 lattice[idx_u] * lattice[idx] +
	 lattice[idx_r] * lattice[idx] +
	 lattice[idx_d] * lattice[idx]) / 2 + 2;
	
    // Si se oponen 2 o menos
    if (r<=2)
    {
	// Da vuelta el spin y confirma el flip
	lattice[idx] *= -1;
	return 1;
    }
    // Para 3 y 4 spins en contra (dE = 4 y 8)
    else
    {
	// Toma el valor de pi = exp(-dE/T) ya calculado
	if (r==3)
	    pi = exp4;
	else
	    pi = exp8;
	
	// Da vuelta el spin con probabilidad pi
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
