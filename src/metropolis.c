#include "metropolis.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int init(Lattice *self, int n)
{
    self -> _n = n;
    self -> _n2 = n * n;
    self -> _total_flips = 0;
    self -> _T = 2.0;
    self -> _J = 1;
    self -> _B = 0;
    self -> _exps[0] = exp(-4 / self -> _T);
    self -> _exps[1] = exp(-8 / self -> _T);
    self -> _opposites = 0;
    *(self -> _p_energy) = 0;
    *(self -> _p_magnet) = 0;
    return 0;
}

int set(Lattice *self, float T, float J, float B)
{
    self -> _T = T;
    self -> _J = J;
    self -> _B = B;
    self -> _exps[0] = exp(-4/T);
    self -> _exps[1] = exp(-8/T);
    return 0;
}

int info(Lattice *self)
{
    printf("Size: %d\n", self -> _n);
    printf("T: %f\n", self -> _T);
    printf("J: %f\n", self -> _J);
    printf("B: %f\n", self -> _B);
    printf("Energy: %d\n", *(self -> _p_energy));
    printf("Magnetization: %d\n", *(self -> _p_magnet));
    printf("Number of flips: %d\n", self -> _total_flips);
    return 0;
}

int metropolis(Lattice *self, int pasos)
{
    int i, idx, nflips = 0;

    // Realiza el número de pasos requerido
    for(i=0; i<pasos; i++)
    {
        // Pide la posición de un spin al azar
	idx = pick_site(self);
	// Trata de dar vuelta el spin
	nflips += flip(self, idx);
    }
    self -> _total_flips += nflips;
    // Devuelve el número de flips conseguidos
    return nflips;
}

int pick_site(Lattice *self)
{
    // Elige un spin al azar y devuelve su posición
    return (int) (((float) rand() / RAND_MAX) * (self -> _n2));
}

int flip(Lattice *self, int idx)
{
    int opposites;
    float pi;

    // Busca los indices de los vecinos
    find_neighbors(self, idx);

    // Cuenta los spins en contra (costo del flip)
    opposites = cost(self, idx);

    // Si el flip no aumenta la energía (2 o menos en contra)
    if (opposites<=2)
    {
	// Acepta el flip. Actualiza E y M
	accept_flip(self, idx, opposites);
	return 1;
    }
    // En otro caso (dE = 4 y 8)
    else
    {
	// Toma el valor de pi = exp(-dE/T) de la tabla T:
	// T[0] = [temperatura]
	// T[1] = exp(-4/T)
	// T[2] = exp(-8/T)
	pi = self -> _exps[opposites - 3];

	// Da vuelta el spin con probabilidad pi e informa
	if (pi*RAND_MAX > rand())
	{
	    // Acepta el flip. Actualiza E y M
	    accept_flip(self, idx, opposites);
	    return 1;
	}
	else
	{
	    return 0;
	}
    }
}


int find_neighbors(Lattice *self, int idx)
{

    int n, n2;
    n = self -> _n;
    n2 = self -> _n2;

    // Condiciones periódicas de contorno
    self -> _W = (idx - 1 + n) % n + (idx/n) * n;   // izquierda
    self -> _N = (idx - n + n2) % n2;               // arriba
    self -> _E = (idx + 1 + n) % n + (idx/n) * n;   // derecha
    self -> _S = (idx + n + n2) % n2;               // abajo

    // (idx +/- 1 + n) % n -----> se mueve de columna
    // (idx/n) * n -------------> primer elemento de la fila
    // (idx +/- n + n2) % n2 ---> se mueve de fila

    return 0;
}

int cost(Lattice *self, int idx)
{
    // Cuenta los spins en contra (costo del flip)

    // (4, 2, 0, -2, 4) / 2 + 2 --> (4, 3, 2, 1, 0)
    self -> _opposites = ((self -> _p_lattice[self -> _W]) *
			  (self -> _p_lattice[idx]) +
			  (self -> _p_lattice[self -> _N]) *
			  (self -> _p_lattice[idx]) +
			  (self -> _p_lattice[self -> _E]) *
			  (self -> _p_lattice[idx]) +
			  (self -> _p_lattice[self -> _S]) *
			  (self -> _p_lattice[idx])) / 2 + 2;

    return self -> _opposites;
}

int accept_flip(Lattice *self, int idx, int opposites)
{
    // Realiza el flip
    self -> _p_lattice[idx] *= -1;
    // Actualiza la energía
    *(self -> _p_energy) += ((opposites - 2) * 4);
    // Actualiza la magnetización
    *(self -> _p_magnet) += ((self -> _p_lattice[idx]) * 2);
    return 0;
}

int calc_energy(Lattice *self, int idx)
{
    // Identifica los vecinos
    find_neighbors(self, idx);
    // Aprovecha la función cost para calcular la energía
    return (cost(self, idx) - 2) * (-2);
}

int calc_magnet(Lattice *self, int idx)
{
    // Devuelve el valor del spin para calcular la magnetización
    return self -> _p_lattice[idx];
}

int calc_lattice(Lattice *self)
{
    int i;

    // Resetea E y M
    *(self -> _p_energy) = 0;
	*(self -> _p_magnet) = 0;

    // Recorre la red actualizando los valores de E y M
    for (i=0; i< self->_n2; i++)
    {
		*(self -> _p_energy) += calc_energy(self, i);
		*(self -> _p_magnet) += calc_magnet(self, i);
    }
    return 0;
}
