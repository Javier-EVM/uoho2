import numpy as np
from scipy.optimize import minimize

#theta = t, es dado
#x = parametros

#Se define la función que representa al modelo real de donde se obtienen las observaciones
def modelo_real(t, x):
    val = x[0] * np.exp(x[1] * t) * np.cos(x[2] * t + x[3])
    return val


#Se calcula el gradiente del modelo real
def model_real_grad(t, x):
    val = np.zeros((4, 1))
    val[0] = np.exp(x[1] * t) * np.cos(x[2] * t + x[3])
    val[1] = x[0] * np.exp(x[1] * t) * t * np.cos(x[2] * t + x[3])
    val[2] = -x[0] * np.exp(x[1] * t) * np.sin(x[2] * t + x[3]) * t
    val[3] = -x[0] * np.exp(x[1] * t) * np.sin(x[2] * t + x[3])
    return val


#x variable
#tm lista de entradas teta, es el dominio de la entrada
#fn evaluacion de los tm en el modelo real
def g_error(x, tm, fn):
    val  = 0
    m = len(tm)

    #Se calcula el error
    for i,ti in enumerate(tm):
        val += np.linalg.norm(modelo_real(ti, x) - fn[i]) ** 2
    return val/m




#Funcion del gradiente del error estocastico
def g_error_grad(x, tm, fn):
    m = len(tm)
    ind = np.random.randint(0, m)  # Randomly sample from the dataset
    val = 2 * (modelo_real(tm[ind], x) - fn[ind]) * model_real_grad(tm[ind], x)
    return val



#Parametros verdaderos
xt = np.array([1, 2, np.pi, 0]) 

#Dominio de la variable independiente
tm = np.arange(-1, 1.001, 0.1)

#Semilla random, para que el experimento sea reproducible
np.random.seed(666)

#Se inicializa ft como vector de ceros
ft = np.zeros(len(tm)) 


#Se le dan los valores reales a ft
for i, t in enumerate(tm):
    ft[i] = modelo_real(t, xt)  # Evaluate model at each time point and store in ft


#Numero de muestras
m = len(tm)

#iteraciones maximas
maxiter = 10**4

#x0 inicial
x0 = np.array([1, 1, 1, 1])



xm = x0

#paso cte de descenso gradiente

learning_rate = 0.01

def SGD(learning_rate,xm,tm,ft):
    plot_y = [0 for i in range(maxiter)]
    plot_x = [0 for i in range(maxiter)]

    for j in range(maxiter):
        step = learning_rate * g_error_grad(xm, tm, ft)
        xp = np.zeros((4, 1))
        for i in range(4):
            xp[i] = xm[i] - step[i] 

        xm = xp

        plot_y[j] = g_error(xm, tm, ft)
        plot_x[j] = j

    print(plot_y)
    print(plot_x)
    print(xm)
    return plot_x, plot_y

import matplotlib.pyplot as plt

# Datos de ejemplo

plot_x1, plot_y1 = SGD(0.01,xm,tm,ft)
plot_x2, plot_y2 = SGD(0.001,xm,tm,ft)
plot_x3, plot_y3 = SGD(0.02,xm,tm,ft)
# Crear el gráfico de dispersión

plt.scatter(plot_x1, plot_y1, color='blue', marker='o',s = 1, label='Tasa de Aprendizaje: 0.01')
plt.scatter(plot_x2, plot_y2, color='red', marker='o',s = 1, label='Tasa de Aprendizaje: 0.001')
plt.scatter(plot_x3, plot_y3, color='purple', marker='o',s = 1, label='Tasa de Aprendizaje: 0.02')

# Agregar título y etiquetas de ejes
plt.title('Gráfico de Dispersión')
plt.xlabel('Iteraciones')
plt.ylabel('Error g(x)')

# Agregar leyenda
plt.legend()

plt.ylim(10**-10, 1)  # Restringir el eje y entre 2 y 4
# Mostrar el gráfico
plt.show()


