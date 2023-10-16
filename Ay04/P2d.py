import numpy as np
from gradient_scaled_quadratic import gradient_scaled_quadratic
from gradient_method_quadratic import gradient_method_quadratic
#Podemos usar la misma función de la ayudantia pasada, puesto que el problema es cuadratico
#Debemos construir la matriz de hilbert
#(f,g,x0,s,alpha,beta,epsilon)

#Tamaño de la matriz (5x5 en este caso)
n = 5

#Crear una matriz de Hilbert de 5x5
hilbert_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        hilbert_matrix[i][j] = 1.0 / (i + j + 1) #la formula cambia por los indices de python

print(hilbert_matrix)

D_m1 =  np.zeros((n, n))

for i in range(n):
    for j in range(n):
        if (i == j):
            D_m1[i][j] = (i + j + 1)
        
print(D_m1)


D_05 = np.sqrt(D_m1)

print(D_05)
#Se definen funciones lambda para f(x) y grad(f(x))
f = lambda x: x @ D_05 @ hilbert_matrix @ D_05 @ x
g = lambda x:  2*(D_05 @ hilbert_matrix @ D_05 @ x) 

x0 = np.array([1,2,3,4,5])
s = 1
alpha = 0.1
beta = 0.5
epsilon = 10**-4
b = np.array([0,0,0,0,0])
gradient_method_quadratic( D_05 @ hilbert_matrix @ D_05, b, x0, epsilon)