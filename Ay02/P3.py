import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0,1,num = 30)
w = 0.1 * np.sin(10 * np.arange(1,31)**3)
y = 2 * x**2 -3 * x + 1 + w

"""----------Propuesto--------"""
"""1. Empaquete el codigo en una función
2. Que funcione para n puntos
3. Que funcione para un polinomio de grado k"""


#Debemos calcular (c b a)^T = (X^T X)^-1 X^T Y

#Se generan matriz de ceros para ser rellenadas
X = np.zeros((len(x),3))
Y = np.zeros(len(x))

#Asigno a la primera columna de X, solo unos
X[:,0] = np.ones(len(x))
#print(X[:,0])

for i in range(len(x)):
    X[i,1] = x[i]
    X[i,2] = x[i]**2

print(f"La matriz X es: \n {X}")
#Se calcula el producto matricial por partes
Xt = np.transpose(X)
A = np.matmul(Xt,X)
C = np.matmul(np.matmul(np.linalg.inv(A),Xt),y)

print(f"Los coeficientes c, b, a, son: {C}")

def f(C,x):
    return [C[2]*xi**2 + C[1]*xi + C[0] for i,xi in enumerate (x)]



# plot
fig, ax = plt.subplots()
l = np.linspace(0,1,num = 100)
ax.plot(l, f(C,l), label='Aproximación')
ax.scatter(x, y, c = "r", label='Puntos')
ax.legend()
ax.set_title("Puntos (x,y) y aproximación de f(x) ")
plt.show()