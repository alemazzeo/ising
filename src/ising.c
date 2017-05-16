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
    float prob=0.5, T=2.0, exp4, exp8;
    int niter = 10000;
    
    if (argc > 0)
    {
	T = atof(argv[1]);
	if (argc > 1)
	{
	    niter = atoi(argv[2]);
	}
    }

    exp4 = exp(-4/T);
    exp8 = exp(-8/T);
    
    srand(time(NULL));
    fill_lattice(lattice, n, prob);
    printf("Tamaño: %d\n", n);
    printf("Probabilidad: %f\n", prob);
    printf("Temperatura: %f\n", T);
    printf("Iteraciones: %d\n", niter);
    for (int i = 0; i < niter; i++) {
	metropolis(lattice, n, T, exp4, exp8);
    }
    print_lattice(lattice, n);
    free(lattice);
    return 0;
}
