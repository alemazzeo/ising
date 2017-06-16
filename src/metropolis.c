#include "metropolis.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int init(Ising *self, int n)
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

int set_params(Ising *self, double T, double J, double B)
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

int info(Ising *self)
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


int metropolis(Ising *self)
{
	int idx;

	// Pide la posición de un spin al azar
	do
		idx = pick_site(self);
	while(idx==self -> _n2);
	// Trata de dar vuelta el spin y devuelve el resultado
	return flip(self, idx);
}

int run(Ising *self, int ntry)
{
    int i;
    self -> _flips = 0;

    // Realiza el número de pasos requerido
    for (i=0;i<ntry;i++)
    {
        if (metropolis(self))
		{
			self -> _p_energy[self -> _flips] = self -> _current_energy;
			self -> _p_magnet[self -> _flips] = self -> _current_magnet;
		}
    }
    return self -> _flips;
}

double run_until(Ising *self, int steps, double tolerance)
{
	int accept = 0, reject=0;
	self -> _flips = 0;
	while (accept < steps && (reject - accept) < tolerance * steps)
	{
		if (metropolis(self))
		{
			self -> _p_energy[self -> _flips] = self -> _current_energy;
			self -> _p_magnet[self -> _flips] = self -> _current_magnet;
			accept++;
		}
		else
		{
			reject++;
		}
	}
	return (double)(accept - reject) / (accept + reject);
}

int run_sample(Ising *self, Sample *result)
{
	int i, size, step_size;
	double q, tolerance;

	size = (result -> _sample_size);

	result -> _n = self -> _n;
	result -> _T = self -> _T;
	result -> _J = self -> _J;
	result -> _B = self -> _B;

	step_size = result -> _step_size;
	tolerance = result -> _tolerance;

	for (i=0; i<size; i++)
	{
		q = run_until(self, step_size, tolerance);
		result -> _p_magnet[i] = self -> _current_magnet;
		result -> _p_energy[i] = self -> _current_energy;
		result -> _p_flips[i] = self -> _flips;
		result -> _p_total_flips[i] = self -> _total_flips;
		result -> _p_q[i] = q;
	}
	return size;
}

int pick_site(Ising *self)
{
    // Elige un spin al azar y devuelve su posición
    return (int) (((double) rand() / RAND_MAX) * (self -> _n2));
}

int flip(Ising *self, int idx)
{
    int aligned;
    double pi;

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


int find_neighbors(Ising *self, int idx)
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

int cost(Ising *self, int idx)
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

int try_flip(Ising *self, double pi)
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

int accept_flip(Ising *self, int idx, int aligned)
{
    double newE, newM;
    // Realiza el flip
    self -> _p_lattice[idx] *= -1;
    // Calcula los cambios
    if (self -> _p_lattice[idx] > 0)
    {
		newE = ((self -> _current_energy) +
				(self -> _p_dEs[aligned]));
		newM = (self -> _current_magnet) + 2;
    }
    else
    {
		newE = ((self -> _current_energy) +
				(self -> _p_dEs[aligned+5]));
		newM = (self -> _current_magnet) - 2;
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

double calc_pi(Ising *self, int idx, int aligned)
{
    if (self -> _p_lattice[idx] < 0)
		// Si al cambiar el spin se alinea con B
		return self -> _p_exps[aligned];
    else
		// Si al cambiar el spin queda en contra de B
        return self -> _p_exps[aligned+5];
}

double calc_energy(Ising *self, int idx)
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

int calc_magnet(Ising *self, int idx)
{
    // Devuelve el valor del spin para calcular la magnetización
    return self -> _p_lattice[idx];
}

int calc_lattice(Ising *self)
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

int autocorrelation(double *x, double *result, int n)
{
    double sum = 0.0;
    double mean = 0.0, sd2 = 0.0;

	for (int i=0; i<n; i++)
	{
		mean += x[i];
		sd2 += x[i]*x[i];
	}

	mean = mean / n;
	sd2 = (sd2 / n) - mean*mean;

    for (int k=0; k<n; k++)
    {
		sum = 0.0;
		for (int j=0; j<n-k; j++)
		{
			sum += (x[j] - mean) * (x[j+k] - mean);
		}
		result[k] = sum / ((n - k) * sd2);
    }
    return 0;
}
