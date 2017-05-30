#include "metropolis.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int init(Lattice *self, int n)
{
    self -> _n = n;
    self -> _n2 = n * n;
    self -> _flips = 0;
    self -> _total_flips = 0;

    set_params(self, 2.0, 1.0, 0.0);

    self -> _aligned = 0;

    self -> _current_energy = 0.0;
    self -> _current_magnet = 0;

    *(self -> _p_energy) = 0.0;
    *(self -> _p_magnet) = 0;
    return 0;
}

int set_params(Lattice *self, float T, float J, float B)
{
    self -> _T = T;
    self -> _J = J;
    self -> _B = B;

    self -> _p_dEs[0] = -8*J - 2*B;
    self -> _p_dEs[1] = -4*J - 2*B;
    self -> _p_dEs[2] =  0*J - 2*B;
    self -> _p_dEs[3] = +4*J - 2*B;
    self -> _p_dEs[4] = +8*J - 2*B;

    self -> _p_dEs[5] = -8*J + 2*B;
    self -> _p_dEs[6] = -4*J + 2*B;
    self -> _p_dEs[7] =  0*J + 2*B;
    self -> _p_dEs[8] = +4*J + 2*B;
    self -> _p_dEs[9] = +8*J + 2*B;

    self -> _p_exps[0] = exp(-(self -> _p_dEs[0]) / T );
    self -> _p_exps[1] = exp(-(self -> _p_dEs[1]) / T );
    self -> _p_exps[2] = exp(-(self -> _p_dEs[2]) / T );
    self -> _p_exps[3] = exp(-(self -> _p_dEs[3]) / T );
    self -> _p_exps[4] = exp(-(self -> _p_dEs[4]) / T );

    self -> _p_exps[5] = exp(-(self -> _p_dEs[5]) / T );
    self -> _p_exps[6] = exp(-(self -> _p_dEs[6]) / T );
    self -> _p_exps[7] = exp(-(self -> _p_dEs[7]) / T );
    self -> _p_exps[8] = exp(-(self -> _p_dEs[8]) / T );
    self -> _p_exps[9] = exp(-(self -> _p_dEs[9]) / T );

    return 0;
}

int info(Lattice *self)
{
    printf("Size: %d\n", self -> _n);
    printf("T: %f\n", self -> _T);
    printf("J: %f\n", self -> _J);
    printf("B: %f\n", self -> _B);
    printf("Energy: %f\n", *(self -> _p_energy));
    printf("Magnetization: %d\n", *(self -> _p_magnet));
    printf("Number of flips: %d\n", self -> _total_flips);
    return 0;
}

int metropolis(Lattice *self, int ntry)
{
    int i, idx;
    self -> _flips = 0;

    // Realiza el número de pasos requerido
    for (i=0;i<ntry;i++)
    {
        // Pide la posición de un spin al azar
        idx = pick_site(self);
        // Trata de dar vuelta el spin
        if (flip(self, idx))
	{
	    self -> _p_energy[self -> _flips] = self -> _current_energy;
	    self -> _p_magnet[self -> _flips] = self -> _current_magnet;
	}
    }
    return self -> _flips;
}

int pick_site(Lattice *self)
{
    // Elige un spin al azar y devuelve su posición
    return (int) (((float) rand() / RAND_MAX) * (self -> _n2));
}

int flip(Lattice *self, int idx)
{
    int aligned;
    float pi;

    // Busca los indices de los vecinos
    find_neighbors(self, idx);

    // Cuenta los spins alineados
    aligned = cost(self, idx);

    // Calcula pi
    pi = calc_pi(self, idx, aligned);

    // Intenta realizar el flip
    if (try_flip(self, pi))
    {
        // Acepta el flip. Actualiza E y M
        accept_flip(self, idx, aligned);
        return 1;
    }
    else
    {
        // Da aviso del flip rechazado
        return 0;
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
    self -> _aligned = ((self -> _p_lattice[self -> _W]) *
			(self -> _p_lattice[idx]) +
			(self -> _p_lattice[self -> _N]) *
			(self -> _p_lattice[idx]) +
			(self -> _p_lattice[self -> _E]) *
			(self -> _p_lattice[idx]) +
			(self -> _p_lattice[self -> _S]) *
			(self -> _p_lattice[idx])) / 2 + 2;

    return self -> _aligned;
}

int try_flip(Lattice *self, float pi)
{
    if (pi > 1)
    {
        return 1;
    }
    else
    {
        if (pi*RAND_MAX > rand())
            return 1;
        else
            return 0;
    }
}

int accept_flip(Lattice *self, int idx, int aligned)
{
    float newE, newM;
    // Realiza el flip
    self -> _p_lattice[idx] *= -1;
    // Calcula los cambios
    if (self -> _p_lattice[idx] > 0)
    {
	newE = (self -> _current_energy +
		self -> _p_dEs[aligned]);
	newM = self -> _current_magnet + 2;
    }
    else
    {
	newE = (self -> _current_energy +
		self -> _p_dEs[aligned+5]);
	newM = self -> _current_magnet - 2;
    }
    // Aumenta el contador de flips
    self -> _flips += 1;
    self -> _total_flips += 1;
    // Actualiza la energía
    self -> _current_energy = newE;
    // Actualiza la magnetización
    self -> _current_magnet = newM;
    return 0;
}

float calc_pi(Lattice *self, int idx, int aligned)
{
    if (self -> _p_lattice[idx] < 0)
	// Si al cambiar el spin se alinea con B
	return self -> _p_exps[aligned];
    else
	// Si al cambiar el spin queda en contra de B
        return self -> _p_exps[aligned+5];
}

float calc_energy(Lattice *self, int idx)
{
    int opposites = 0;

    find_neighbors(self, idx);
    opposites = 4 - cost(self, idx);

    if (self -> _p_lattice[idx] > 0)
	// Si al cambiar el spin se alinea con B
	return self -> _p_dEs[opposites] / 2;
    else
	// Si al cambiar el spin queda en contra de B
	return self -> _p_dEs[opposites] / 2;
}

int calc_magnet(Lattice *self, int idx)
{
    // Devuelve el valor del spin para calcular la magnetización
    return self -> _p_lattice[idx];
}

int calc_lattice(Lattice *self)
{
    int i;

    self -> _current_energy = 0;
    self -> _current_magnet = 0;

    // Recorre la red actualizando los valores de E y M
    for (i=0; i< self->_n2; i++)
    {
        self -> _current_energy += calc_energy(self, i);
        self -> _current_magnet += calc_magnet(self, i);
    }
    return 0;
}

int autocorrelation(float *x, float *result, int n, float xt, float xt2)
{
    int j, k;
    float sum = 0.0;
    float mean = 0.0, sd = 0.0;

    mean = xt / n;
    sd = (xt2 / n) - (mean * mean);

    for (k=0; k<n; k++)
    {
	sum = 0.0;
	for (j=0; j<n-k; j++)
	{
	    sum += (x[j] - (xt / n)) * (x[j+k] - (xt / n));
	}
	result[k] = sum / ((n - k) * sd);
    }
    return 0;
}
