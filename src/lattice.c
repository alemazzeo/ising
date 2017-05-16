#include <stdio.h>
#include <stdlib.h>
#include "lattice.h"
int fill_lattice(int *lattice, int n, float p) {
    int i, n2=n*n;
    for (i=0; i<n2; i++)
    {
	lattice[i] = (rand()> RAND_MAX/2)*(-2) + 1 ;
    }
    return 0;
}

int print_lattice(int *lattice, int n) {
    int i, j;
    for (i=0;i<n;i++)
    {
	for(j=0;j<n;j++)
	{
	    if (lattice[i*n+j]>0) 
		printf("+ ");
	    else
		printf("- ");
	}
	printf("\n");
    }
    return 0;
}
