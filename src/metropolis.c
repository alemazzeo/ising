#include "metropolis.h"
#include <stdlib.h>
#include <stdio.h>

int metropolis(int *lattice, int n, float *T, int pasos,
	       int *energy, int *magnet) 
{
    int i, idx, nflips = 0;

    // Realiza el número de pasos requerido
    for(i=0; i<pasos; i++)
    {
        // Pide la posición de un spin al azar
	idx = pick_site(lattice, n);
	// Trata de dar vuelta el spin
	nflips += flip(lattice, n, T, idx, energy, magnet);
    }
    
    // Devuelve el número de flips conseguidos
    return nflips;
}

int pick_site(int *lattice, int n) 
{
    // Elige un spin al azar y devuelve su posición
    return (int) (((float) rand() / RAND_MAX) * n * n);
}

int flip(int *lattice, int n, float *T, int idx, int *energy, int *magnet) 
{
    int W, N, E, S;
    int opposites;
    float pi;
    
    // Busca los indices de los vecinos
    find_neighbors(lattice, n, idx, &W, &N, &E, &S);
    
    // Cuenta los spins en contra (costo del flip)
    opposites = cost(lattice, n, idx, &W, &N, &E, &S);
	
    // Si el flip no aumenta la energía (2 o menos en contra)
    if (opposites<=2)
    {
	// Acepta el flip. Actualiza E y M
	accept_flip(lattice, n, idx, opposites, energy, magnet);
	return 1;
    }
    // En otro caso (dE = 4 y 8)
    else
    {
	// Toma el valor de pi = exp(-dE/T) de la tabla T:
	// T[0] = [temperatura]
	// T[1] = exp(-4/T)
	// T[2] = exp(-8/T)
	pi = T[opposites - 2];
	
	// Da vuelta el spin con probabilidad pi e informa
	if (pi*RAND_MAX > rand()) 
	{
	    // Acepta el flip. Actualiza E y M
	    accept_flip(lattice, n, idx, opposites, energy, magnet);
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

    // Condiciones periódicas de contorno
    
    *W = (idx - 1 + n) % n + (idx/n) * n;   // izquierda
    *N = (idx - n + n2) % n2;               // arriba
    *E = (idx + 1 + n) % n + (idx/n) * n;   // derecha
    *S = (idx + n + n2) % n2;               // abajo

    // (idx +/- 1 + n) % n -----> se mueve de columna
    // (idx/n) * n -------------> primer elemento de la fila
    // (idx +/- n + n2) % n2 ---> se mueve de fila

    return 0;
}

int cost(int *lattice, int n, int idx, int *W, int *N, int *E, int *S)
{
    // Cuenta los spins en contra (costo del flip)

    // (4, 2, 0, -2, 4) / 2 + 2 --> (4, 3, 2, 1, 0)       
    return (lattice[*W] * lattice[idx] +
	    lattice[*N] * lattice[idx] +
	    lattice[*E] * lattice[idx] +
	    lattice[*S] * lattice[idx]) / 2 + 2;
}

int accept_flip(int *lattice, int n, int idx, int opposites, int *energy, int *magnet)
{
    // Realiza el flip
    lattice[idx] *= -1;
    // Actualiza la energía
    *energy = *energy + ((opposites - 2) * 4);
    // Actualiza la magnetización
    *magnet = *magnet + (lattice[idx] * 2);
    return 0;
}

int calc_energy(int *lattice, int n, int idx)
{
    int W, N, E, S;
    // Identifica los vecinos
    find_neighbors(lattice, n, idx, &W, &N, &E, &S);
    // Aprovecha la función cost para calcular la energía
    return cost(lattice, n, idx, &W, &N, &E, &S) + 2 * (-2);
}

int calc_magnet(int *lattice, int n, int idx)
{
    // Devuelve el valor del spin para calcular la magnetización
    return lattice[idx];
}

int calc_lattice(int *lattice, int n, int *energy, int *magnet)
{
    int i, n2=n*n;

    // Resetea E y M
    *energy = 0;
    *magnet = 0;

    // Recorre la red actualizando los valores de E y M
    for (i=0; i<n2; i++)
    {
	(*energy) += calc_energy(lattice, n, i);
	(*magnet) += calc_magnet(lattice, n, i);
    }
    return 0;
}
