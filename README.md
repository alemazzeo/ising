# Ising

## Contenido

### metropolis.c
Contiene las funciones necesarias para aplicar el método Metrópolis al modelo de Ising.

### lattice.c
Realiza el llenado de la red y la visualización de la misma por consola.

### ising.c
Contiene la rutina principal. Utiliza las funciones de metropolis y lattices. Permite obtener resultados por consola

### ising.py
Contiene la clase lattice que permite crear una red de spines y correr el algoritmo de metrópolis visualizando los cambios. Se encuentra en desarrollo.

## Modo de trabajo

El archivo Makefile contiene las reglas para generar los objetos, linkear y crear la biblioteca dinámica para integrar con python. 

Una vez ejecutado puede ejecutarse el archivo ./bin/ising.e para obtener resultados por consola.

ising.e admite como primer parametro la temperatura y como segundo parámetro el número de iteraciones.

### Ejemplo de uso para ising.py
Utilizando ipython3:
```
in [1]: from ising import lattice    # Importa la clase lattice
in [2]: lat1 = lattice(n=32)         # Crea el objeto
in [3]: lat1.view()                  # Activa la visualización
in [4]: lat1.fill_random()           # Llena la red aleatoriamente
in [5]: lat1.run(5000)               # Ejecuta 5000 pasos
in [6]: lat1.run(10000)              # Ejecuta otros 10000 pasos
```
