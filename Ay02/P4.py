import matplotlib.pyplot as plt
import numpy as np


"""Creo que hay un error, la solución no me parece satisfactoria
Pareciera que un radio de r' = 1.5*r funciona mejor.
no logre identificar si habia error o no"""
a =np.array([[0,0],[0.5,0],[1,0],[1,1],[0,1]])
tam = len(a)
#a= np.array(a)
#y_sol = (A^T A)^-1  A^T b

#generemos A y b
A =  np.zeros((5,3))
b = np.zeros(5)
A[:,2] = -np.ones(5)

for i,xi in enumerate(a):
    A[i,[0,1]] = 2*xi
    #es equivalente a
    #A[i,0] = 2*xi[0]
    #A[i,1] = 2*xi[1]
    b[i] = np.matmul(xi,xi)
    

print(f"La matriz A es: \n{A}")
print(f"El vector b es: \n {b}")



#Realizamos el producto matricial
At = np.transpose(A)
C = np.linalg.inv(np.matmul(At,A))
y = np.matmul(np.matmul(C,At),b)

print(f"El vector y es: \n {y}")

r = np.sqrt(np.matmul(y[0:1],y[0:1]) -y[2])
print(f"El centro es ({y[0]},{y[1]}) con Radio {r}")


# Para graficar
X = np.zeros(tam)
Y = np.zeros(tam)
for i,xi in enumerate(a):
    X[i] = xi[0]
    Y[i] = xi[1]

fig, ax = plt.subplots()
ax.scatter(X,Y ,c = "r", label='Puntos')
ax.scatter(y[0],y[1], c = "b", label='Centro')
Drawing_uncolored_circle = plt.Circle( (y[0], y[1] ),
                                      r ,
                                      fill = False, label = "Circle fit" )
ax.set_aspect( 1 )
ax.add_artist( Drawing_uncolored_circle )


plt.xlim([-0.5, 1.5]) #Puede fallar si los puntos son diferentes
plt.ylim([-0.5, 1.5])
ax.legend(loc = "lower right")
ax.set_title("Circle fit")
plt.show()


#Calculo del error
sum = 0
for i,xi in enumerate(a):
    sum += np.matmul(xi-np.array(y[0],y[1]),xi-np.array(y[0],y[1]))

print(sum - r)
print(sum - 1.5*r)