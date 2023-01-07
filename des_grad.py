import csv
import numpy as np
import matplotlib.pyplot as plt

def sigmoide(z):
  return 1 / (1 + np.exp(-z))

def coste(h, y):
  return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

def descenso_gradiente(X, y, theta, alpha, iteraciones):
  m = len(y)
  historial_costos = []
  for i in range(iteraciones):
    z = np.dot(X, theta)
    h = sigmoide(z)
    gradiente = np.dot(X.T, (h - y)) / m
    theta -= alpha * gradiente
    historial_costos.append(coste(h, y))
  return theta, historial_costos

def predecir(X, theta):
  z = np.dot(X, theta)
  h = sigmoide(z)
  return np.round(h)

def grafica_gradiente(historial_costos):
  plt.plot(historial_costos)
  plt.xlabel('Iteracion')
  plt.ylabel('Coste')
  plt.title('Descenso de la Gradiente')
  plt.show()

def main():
  # Leer los datos de entrenamiento
  with open('C:/Users/elias/Documents/ESPE/ABC/DescensoGrad_QuienEsQuien/training_data.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader) # salta los encabezados
    X, y, names = [], [], []
    for row in reader:
      caracteristicas = [float(x) for x in row[:8]]
      etiqueta = 0 if row[8] == 'Hombre' else 1
      nombre = row[9]
      X.append(caracteristicas)
      y.append(etiqueta)
      names.append(nombre)
  X = np.array(X)
  y = np.array(y)
  m, n = X.shape
  theta = np.zeros(n)
  alpha = 0.1 #tasa de aprendizaje
  iteraciones = 1000
  theta, historial_costos = descenso_gradiente(X, y, theta, alpha, iteraciones)
  grafica_gradiente(historial_costos)

  # Read in test data
  with open('C:/Users/elias/Documents/ESPE/ABC/DescensoGrad_QuienEsQuien/test_data.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader) # skip header row
    X_test, nombres_test = [], []
    for row in reader:
      caracteristicas = [float(x) for x in row[:8]]
      nombre = row[8]
      X_test.append(caracteristicas)
      nombres_test.append(nombre)
  X_test = np.array(X_test)
  predicciones = predecir(X_test, theta)

  # Impresión de los resultados
  print("{0:^25} \t\t\t {1:^10} \t\t\t {2:^10}".format("Entradas","Nombres","Prediccion"))
  for i in range(len(predicciones)):
    
    print("{0} \t\t\t {1:^10} \t\t\t {2:^10}".format(X_test[i], nombres_test[i], "Mujer" if predicciones[i] == 1 else "Hombre"))


if __name__ == '__main__':
  main()