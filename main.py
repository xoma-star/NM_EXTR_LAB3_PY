import numpy as np
import matplotlib.pyplot as plt
import math

L = 1
T = 1
a = 2
beta = 4
steps_l = 100
steps_t = 100
h = L / (steps_l - 1)
tau = T / (steps_t - 1)
p_min = 0
p_max = 4000

def phi(x):
  return 100 * x


def initial_management(t):
  return 1000 * t * t * (1 - t)


def start_management(t):
  return 1000 * math.sin(10 * t)


def tridiagonal_method(matrix1, matrix2):
  y = matrix1[0][0]
  N = len(matrix2)
  N1 = N - 1
  a = np.zeros(N)
  B = np.zeros(N)
  mat_res = np.zeros(N)
  a[0] = -matrix1[0][1] / y
  B[0] = matrix2[0] / y
  for i in range(1, N1):
    y = matrix1[i][i] + matrix1[i][i - 1] * a[i - 1]
    a[i] = -matrix1[i][i + 1] / y
    B[i] = (matrix2[i] - matrix1[i][i - 1] * B[i - 1]) / y
  mat_res[N1] = (matrix2[N1] - matrix1[N1][N1 - 1] * B[N1 - 1]) / (
    matrix1[N1][N1] + matrix1[N1][N1 - 1] * a[N1 - 1])
  for i in range(N1 - 1, -1, -1):
    mat_res[i] = a[i] * mat_res[i + 1] + B[i]
  return mat_res


def straight_task(current_management):
  management = np.zeros(steps_t)
  for i in range(steps_t):
    management[i] = current_management(i * tau)
  u = np.zeros((steps_t, steps_l))
  for i in range(steps_l):
    u[0][i] = phi(h * i)
  A = np.zeros((steps_l - 2, steps_l - 2))
  b = np.zeros(steps_l - 2)
  ku_1 = (a * a) / (h * h)
  ku_2 = -((2 * a * a) / (h * h) + 1 / tau)
  ku_3 = ku_1
  ku_4 = -1 / tau
  for i in range(1, steps_t):
    p1 = 0
    A[0][0] = ku_2 + ku_1
    A[0][1] = ku_3
    b[0] = ku_4 * u[i - 1][1] + ku_1 * p1 * h
    for j in range(1, len(A) - 1):
      A[j][j - 1] = ku_1
      A[j][j] = ku_2
      A[j][j + 1] = ku_3
      b[j] = ku_4 * u[i - 1][j + 1]
    A[len(A) - 1][len(A) - 1 - 1] = ku_1
    A[len(A) - 1][len(A) - 1] = ku_2 + ku_3 * (1 / (1 + beta * h))
    b[len(b) - 1] = ku_4 * u[i - 1][len(u[i - 1]) - 1 - 1] - ku_3 * (
      (1 / (1 + beta * h)) * beta * h * management[i])
    temp_u = tridiagonal_method(A, b)
    for j in range(0, len(temp_u)):
      u[i][j + 1] = temp_u[j]
    u[i][0] = u[i][1] - p1 * h
    u[i][len(u[i]) -
         1] = u[i][len(u[i]) - 1 -
                   1] * (1 / (1 + beta * h)) + beta * h * management[i]
  return u


def conjugate_task(u, y):
  psi = np.zeros((steps_t, steps_l))
  for j in range(len(psi[len(psi) - 1])):
    psi[len(psi) - 1][j] = 2 * (u[len(u) - 1][j] - y[j])
  A = np.zeros((steps_l - 2, steps_l - 2))
  b = np.zeros(steps_l - 2)
  ku_1 = -(a * a) / (h * h)
  ku_2 = -((-2 * a * a) / (h * h) - 1 / tau)
  ku_3 = ku_1
  ku_4 = 1 / tau
  for i in range(steps_t - 1 - 1, -1, -1):
    p1 = 0
    A[0][0] = ku_2 + ku_1
    A[0][1] = ku_3
    b[0] = ku_4 * u[i - 1][1] + ku_1 * p1 * h
    for j in range(1, len(A) - 1):
      A[j][j - 1] = ku_1
      A[j][j] = ku_2
      A[j][j + 1] = ku_3
      b[j] = ku_4 * psi[i + 1][j + 1]
    A[len(A) - 1][len(A) - 1 - 1] = ku_1
    A[len(A) - 1][len(A) - 1] = ku_2 + ku_3 * (1 / (1 + beta * h))
    b[len(b) - 1] = ku_4 * psi[i + 1][len(psi[i + 1]) - 1 - 1]
    temp_psi = tridiagonal_method(A, b)
    for j in range(len(temp_psi)):
      psi[i][j + 1] = temp_psi[j]
    psi[i][0] = psi[i][1] - p1 * h
    psi[i][len(psi[i]) - 1] = psi[i][len(psi[i]) - 1 - 1] * (1 /
                                                             (1 + beta * h))
  return psi

def calculation(y, epsilon=0.01):
  currentManagement = map(initial_management, [np.arange(0, steps_t)])
  # currentManagement = initial_management
  norm = 1e10
  iterations = 0

  while(norm > epsilon):
    iterations += 1
    u = straight_task(currentManagement)
    norm = l2_norm(u[len(u) - 1], y, h)

    psi = conjugate_task(u, y)

    tempManagement = []
    for j in range(len(psi)):
      if psi[j][psi[j].size - 1] >= 0:
        tempManagement.append(p_min)
      else:
        tempManagement.append(p_max)
    
    integrandOfNumerator = []
    for j in range(len(psi)):
      integrandOfNumerator.append(a ** 2 * beta * psi[j][psi[j].size - 1] * (tempManagement[j] - currentManagement[j]))
      
    numerator = rectangle_square(integrandOfNumerator, tau)
    tempU = straight_task(tempManagement)

    integrandOfDenominator = []
    for j in range(len(u[u.size - 1].size())):
      integrandOfDenominator.append((tempU[tempU.size - 1][j] - u[u.size - 1][j]) ** 2)

    denominator = rectangle_square(integrandOfDenominator, h)
    currentAlfa = min(-0.5 * (numerator / denominator), 1)

    for j in range(len(currentManagement)):
      currentManagement[j] = currentManagement[j] + currentAlfa * (tempManagement[j] - currentManagement[j])

    answerU = straight_task(currentManagement)
    
    return answerU[answerU.size() - 1]

def rectangle_square(vec, step):
  sum = 0
  for i in range(len(vec)):
    sum += vec[i] * step
  return sum

def l2_norm(vec, y, step):
  vec = []
  for i in range(len(vec)):
    vec.append((vec[i] - y[i]) ** 2)
  return rectangle_square(vec, step)

def print_file(u, name_of_file="ySolution.txt"):
  f = open(name_of_file, "a")
  if len(u.shape) == 1:
    for i in range(u.size):
      f.write(i * h + " " + u[i])
  else:
    for i in range(u.size):
      for j in range(u[i].size):
        f.write(i * tau + " " + j * h + " " + u[i][j])
  f.close()
  #проверка на u одномерный или нет и соотв. вывод в файл 

def plot_2d(filename='ySolution.txt'):
  f = open(filename, 'r') # требования к файлу: первая колонка - координата/время, вторая - значения функции
  tempX = list()
  tempY = list()

  for line in f:
      tempValue = [float(i) for i in line.strip().split()]
      tempX += [tempValue[0]]
      tempY += [tempValue[1]]
  f.close()

  tempX = np.array(tempX)
  tempY = np.array(tempY)
  plt.plot(tempX, tempY, ".-")
  plt.xlabel("ось X")
  plt.ylabel("ось Y")
  plt.show()

def plot_3d(filename='ySolution.txt'):
  f = open(filename, 'r') # требования к файлу: первая колонка - время, вторая - координата, третья - значения функции на них
  tempT = list()
  tempX = list()
  tempU = list()
  
  for line in f:
      tempValue = [float(i) for i in line.strip().split()]
      tempT += [tempValue[0]]
      tempX += [tempValue[1]]
      tempU += [tempValue[2]]
  f.close()
  
  tempT = np.array(tempT)
  tempX = np.array(tempX)
  tempU = np.array(tempU)
  
  xgrid, tgrid = np.meshgrid(np.unique(tempX), np.unique(tempT))
  
  ax = plt.axes(projection='3d')
  ax.plot_wireframe(xgrid, tgrid, tempU.reshape((len(np.unique(tempT)), len(np.unique(tempX)))))
  plt.xlabel("ось X")
  plt.ylabel("ось T")
  plt.show()

def main():
  true_u = straight_task(initial_management)
  print(true_u)
  print(true_u[len(true_u) - 1])
  approx_y = calculation(true_u[len(true_u) - 1])
  print_file(true_u[len(true_u) - 1], "trueY.txt")
  print_file(approx_y, "approxY.txt")

main()