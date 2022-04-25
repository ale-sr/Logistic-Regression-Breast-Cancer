# -*- coding: utf-8 -*-
"""

### Práctica de Regresión Lógistica: Cancer de mama

1.   Elemento de lista

1.   Elemento de lista
2.   Elemento de lista


2.   Elemento de lista



***Por favor, trabaje en una copia de este  colab***

**Base de datos**: [Click](https://drive.google.com/file/d/1YccFuyedpJ7-rek4iR3MSviB0jv80uqZ/view?usp=sharing)

El conjunto de datos contiene casos de un estudio realizado entre 1958 y 1970 en el Hospital Billings de la Universidad de Chicago sobre la supervivencia de pacientes que se habían sometido a cirugía por cáncer de mama.

La base de datos está formada por 306 objetos, cada objeto tiene 3 características (Edad del paciente al momento de la operación, Años de operación y Número de ganglios axilares positivos detectados) y un predictor (variable a predecir estado de supervivencia, 1 si el paciente vivió, 2 si el paciente murío)

*Se pide predecir, en base a las características de un paciente,  si un paciente sobrevivirá o no*
"""

import numpy as np 
import math
import pandas as pd
import random
import matplotlib.pyplot as plt

"""

**Hipótesis**:

- Ecuación de la recta o Hiperplano
\begin{equation}
h(x_i) = w_0 + w_1x_i^1 +  w_2x_i^2 ... w_kx_i^k
\end{equation} \\

- Ecuación de la función sigmoidea (clasificador binario)
\begin{equation}
s(x_i) = \frac{1}{1 + e^{-h(x)}} 
\end{equation}




"""

def Hiperplano(x,w):
  # write your code here
  return np.dot(x, w.transpose())

def S(x,w):
  result = 1/(1 + math.exp(-Hiperplano(x, w)))
  return result

"""- **Loss Function** (Cross-Entropy)

\begin{equation}
L = -\frac{1}{n}\sum_{i=0}^n(y_ilog(s(x_i)) + (1-y_i)log(1-s(x_i)))  
\end{equation} \\

"""

def Loss_function(x, y, w):
  # write your code here 
  suma = 0
  n = y.size
  for i in range(n):
    temp = S(x[i], w)
    if temp == 0:
      temp = 0.000001
    elif temp == 1:
      temp = 0.999999
    suma += y[i]*math.log(temp) + (1-y[i])*math.log(1-temp)
  L = (-1/n)*suma
  return L

"""- **Derivatives**

\begin{equation}
\frac{\partial L}{\partial w_j} = \frac{1}{n}\sum_{i=0}^n(y_i - s(x_i))(-x_i^j)
\end{equation} \\

Nota:  $x_i^j$ se refiere a la característica $j-esima$ del objeto $i-esimo$ de entrenamiento

"""

def Der(x, y, w, b):
  return (y-S(x, w))*(-b)

def Derivate(x, y, w, j):
  n = y.size
  dw = (1/n)*sum([ Der(x[i], y[i], w, x[i][j]) for i in range(n)])
  return dw

def Derivatives_list(x,y,w):
  # write your code here
  n = y.size
  db = (1/n)*sum([Der(x[i], y[i], w, 1.0) for i in range(n)])
  return ([db]+[Derivate(x, y, w, j) for j in range(1, k)])

def Derivatives_example(x,y,w):
  # write your code here
  db = Der(x, y, w, -1.0)
  return [db]+[Der(x, y, w, x[j]) for j in range(1, k)]

"""- Change parameters 

\begin{equation}
 w_j = w_j - \alpha\frac{\partial L}{\partial w_j} 
\end{equation}
"""

def change_parameters(w, derivatives, alpha):
  # write your code here
  for j in range(k):
    w[j] = w[j] - alpha*derivatives[j]
  return w

def add_one(x):
  return np.insert(x, 0, 1, axis=1)

"""- Batches"""

def get_batches(x, y, batch_size=50):
  batches_x = []
  batches_y = []
  n = y.size
  pos = 0
  init = 0
  while n-pos > batch_size:
    pos += batch_size
    batches_x.append(x[init:pos])
    batches_y.append(y[init:pos])
    init += batch_size
  if n != pos:
    batches_x.append(x[init:])
    batches_y.append(y[init:])
  return np.array(batches_x), np.array(batches_y)

"""- **Training** 

Seleccione $70\%$ de los datos del dataset para entrenamiento y el resto para testing. Recuerde, los datos deben ser seleccionados de manera aleatoría.



"""

def training_batch(x,y, epochs, alpha, xt = None, yt = None):
  x = add_one(x)
  if xt.any():
    xt = add_one(xt)
  w = np.array([np.random.rand() for i in range(k+1)]) 
  errtr = []
  errts = []
  for i in range(epochs):
    L =  Loss_function(x,y, w)
    dw = Derivatives_list(x,y,w)
    w = change_parameters(w, dw, alpha)
    errtr.append(L)
    if xt.any():
      errts.append(Loss_function(xt,yt, w))
  return w, errtr, errts

def training_stochastic(x,y, epochs, alpha, xt = None, yt = None):
  x = add_one(x)
  if xt.any():
    xt = add_one(xt)
  w = np.array([np.random.rand() for i in range(k+1)]) 
  errtr = []
  errts = []
  for i in range(epochs):
    np.random.shuffle(x)
    np.random.shuffle(y)
    for j in range(n):
      L =  Loss_function(x,y, w)
      dw = Derivatives_example(x[j],y[j],w)
      w = change_parameters(w, dw, alpha)
      errtr.append(L)
      if xt.any():
        errts.append(Loss_function(xt,yt, w))
  return w, errtr, errts

def training_mini_batch(x,y, epochs, alpha, xt = None, yt = None):
  x = add_one(x)
  if xt.any():
    xt = add_one(xt)
  w = np.array([np.random.rand() for i in range(k+1)]) 
  errtr = []
  errts = []
  for i in range(epochs):
    np.random.shuffle(x)
    np.random.shuffle(y)
    batches_x, batches_y = get_batches(x, y, 60)
    ind = random.randint(0, batches_x.size-1)
    L =  Loss_function(x, y, w)
    dw = Derivatives_list(batches_x[ind],batches_y[ind],w)
    w = change_parameters(w, dw, alpha)
    errtr.append(L)
    if xt.any():
      errts.append(Loss_function(xt,yt, w))
  return w, errtr, errts

""""
def training_mini_batch(x,y, epochs, alpha, xt = None, yt = None):
  x = add_one(x)
  if xt.any():
    xt = add_one(xt)
  w = np.array([np.random.rand() for i in range(k+1)]) 
  errtr = []
  errts = []
  for i in range(epochs):
    np.random.shuffle(x)
    np.random.shuffle(y)
    batches_x, batches_y = get_batches(x, y, 100)
    for j in range(batches_y.size):
      L =  Loss_function(x, y, w)
      dw = Derivatives_list(batches_x[j],batches_y[j],w)
      w = change_parameters(w, dw, alpha)
      errtr.append(L)
      if xt.any():
        errts.append(Loss_function(xt,yt, w))
  return w, errtr, errts

- **Testing**

Utilize el $30\%$ de los datos restantes para el proceso de testing.
"""

def Testing(x_test, y_test, w):
  x_test = add_one(x_test)
  n = y_test.size
  k = x_test[0].size
  y_pred = []
  c = 0
  for i in range(n):
    yt = round(S(x_test[i], w))
    y_pred.append(yt)
    if y_test[i] == y_pred[i]:
      c += 1
  print("Número de datos correctos", round(c*100/n), "%") # sum(y_pred == y)
  return y_pred

"""**- Normalización**"""

def normalizacion(aux_x, aux_y):
  n = aux_y.size
  k = len(aux_x.columns)

  x = np.array([[i*1.0 for i in range(k)] for j in range(n)])
  y = np.array([])

  for i in range(n):
    lista = []
    for j in range(k):
      lista.append(aux_x.iloc[i][aux_x.columns[j]])
    lista = np.array(lista)
    x[i] = lista
    y = np.append(y, aux_y.iloc[i])

  mat = [[i*1.0 for i in range(n)] for j in range(k)]
  for i in range(k):
    col = x[:,i]
    max_x = max(col)
    min_x = min(col)
    lista = []
    for e in col:
      lista.append( (e-min_x) / (max_x - min_x) )
    lista = np.array(lista)
    mat[i] = lista

  x_norm = np.vstack((mat)).T

  max_y = max(y)
  min_y = min(y)
  y_norm = np.array([ ( e - min_y)/(max_y - min_y) for e in y])

  return x_norm, y_norm

df = pd.read_csv('dataset.csv', header=None, names = ['edad', 'anos', 'ganglios', 'estado'])

training_data = df.sample(frac=0.7, random_state=25)
testing_data = df.drop(training_data.index)

train_x = training_data[['edad', 'anos', 'ganglios']]
train_y = training_data[['estado']]

test_x = testing_data[['edad', 'anos', 'ganglios']] 
test_y = testing_data[['estado']]

n = train_y.size
k = len(train_x.columns)

# print(k)
train_x_norm, train_y_norm = normalizacion(train_x, train_y)
test_x_norm, test_y_norm = normalizacion(test_x, test_y)

#w, err_training, err_testing = training_batch(train_x_norm, train_y_norm, 1000, 0.01, test_x_norm, test_y_norm)
#w, err_training, err_testing = training_stochastic(train_x_norm, train_y_norm, 1000, 0.01, test_x_norm, test_y_norm)
w, err_training, err_testing = training_mini_batch(train_x_norm, train_y_norm, 400, 0.01, test_x_norm, test_y_norm)

n = test_y.size
y_pred = Testing(test_x_norm, test_y_norm, w)
plt.plot(err_training, "purple")
plt.plot(err_testing, "orange")
plt.legend(["Training", "Testing"])

fig = plt.figure()
ax = plt.axes(projection='3d')
xdata = train_x_norm[:, 0]
ydata = train_x_norm[:, 1]
zdata = train_x_norm[:, 2]
ax.scatter3D(xdata, ydata, zdata, c=train_y_norm, cmap='Greens')
ax.set_xlabel('edad')
ax.set_ylabel('años')
ax.set_zlabel('ganglios')

fig = plt.figure()
ax = plt.axes(projection='3d')
xdata = test_x_norm[:, 0]
ydata = test_x_norm[:, 1]
zdata = test_x_norm[:, 2]
ax.scatter3D(xdata, ydata, zdata, c=test_y_norm, cmap='Greens')
ax.set_xlabel('edad')
ax.set_ylabel('años')
ax.set_zlabel('ganglios')

"""##Desarrolle las siguientes actividades 

- Implemente funciones para graficar la función de pérdida. 
- Implemente la función para mostrar las funciones de error de training vs testing 
-¿Qué porcentaje de aciertos tiene el método?

  73%

- ¿Qué porcentaje de fallas tiene el método?
 
  27%

Un exelente libro: [click](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf)
"""