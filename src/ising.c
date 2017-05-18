#include <stdio.h>
#include "stdlib.h"
#include <math.h>
#include "time.h"
#include "metropolis.h"
#include "lattice.h"

int main(int argc, char **argv) 
{
    int n = 32;
    int *lattice = malloc(n * n * sizeof(int));
    float prob=0.5;
    float T[3];
    int niter = 100;
    int pasos = 1000;
    int E = 0;
    int M = 0;
    int nflips = 0;

    T[0] = 2.0;
    
    if (argc > 1)
    {
	T[0] = atof(argv[1]);
	if (argc > 2)
	{
	    niter = atoi(argv[2]);
	}
    }

    T[1] = exp(-4/T[0]);
    T[2] = exp(-8/T[0]);
    
    srand(time(NULL));
    fill_lattice(lattice, n, prob);
    calc_lattice(lattice, n, &E, &M);
    
    printf("Tamaño: %d\n", n);
    printf("Probabilidad: %f\n", prob);
    printf("Temperatura: %f\n", T[0]);
    printf("Iteraciones: %d x %d\n", niter, pasos);
    
    for (int i = 0; i < niter; i++) {
	nflips += metropolis(lattice, n, T, pasos, &E, &M);
    }
    
    print_lattice(lattice, n);
    
    printf("Energía: %d\n", E);
    printf("Magnetización: %d\n", M);
    printf("Pasos válidos: %d\n", nflips);
    
    free(lattice);
    return 0;
}
