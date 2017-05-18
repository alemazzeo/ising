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
    int niter = 10000;

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
    printf("Tamaño: %d\n", n);
    printf("Probabilidad: %f\n", prob);
    printf("Temperatura: %f\n", T[0]);
    printf("Iteraciones: %d\n", niter);
    for (int i = 0; i < niter; i++) {
	metropolis(lattice, n, T);
    }
    print_lattice(lattice, n);
    free(lattice);
    return 0;
}
