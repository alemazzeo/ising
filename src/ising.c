#include <stdio.h>
#include "stdlib.h"
#include <math.h>
#include "time.h"
#include "metropolis.h"
#include "lattice.h"


int main(int argc, char **argv)
{
    Lattice *ising;
    int n = 32;

    float prob=0.5;
    int niter = 1000;
    int pasos = 1000;

    float T = 2.0;
    float J = 1.0;
    float B = 0.0;

    if (argc > 1)
    {
	T = atof(argv[1]);
	if (argc > 2)
	{
	    niter = atoi(argv[2]);
	}
    }

    srand(time(NULL));

    ising = malloc(sizeof(Lattice));
    ising -> _p_lattice = malloc(n*n*sizeof(int));
	ising -> _p_energy = malloc(pasos * sizeof(int));
	ising -> _p_magnet = malloc(pasos * sizeof(int));

    init(ising, n);
    set_params(ising, T, J, B);

    fill_lattice(ising -> _p_lattice, n, prob);
    calc_lattice(ising);

    printf("Iteraciones: %d x %d\n", niter, pasos);

    for (int i = 0; i < niter; i++) {
		metropolis(ising);
    }

    print_lattice(ising -> _p_lattice, n);
    printf("\n");
    info(ising);

	free(ising -> _p_lattice);
	free(ising -> _p_energy);
	free(ising -> _p_magnet);
	free(ising);

    return 0;
}
