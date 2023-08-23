import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp


def rk_solver(f, y0, t_span, p):
    '''
        Решает систему линейных ДУ явной схемой РК
        Входные параметры:
            f - функция, возвращающая правую часть СЛДУ 
            y0 - вектор с начальными условиями
            t_span - вектор [t0, tk]
            p - порядок точности
    '''
    N_MAX = 400  # количество интервалов
    dim = len(y0)  # размерность задачи

    t0 = t_span[0]
    T = t_span[1]

    t = np.linspace(t0, T, N_MAX + 1)  # массив для переменной t
    h = t[1] - t[0]  # размер шага (постоянный)

    if dim == 1:
        y = np.zeros(N_MAX + 1)  # массив для хранения численного решения
        y[0] = y0[0]
        w = np.zeros(p)
    else:
        # массив для хранения численного решения
        y = np.zeros((N_MAX + 1, dim))
        y[0] = y0  # начальные условия (y[0] - строка, присваиваем всей строке)
        w = np.zeros((p, dim))  # количество w[i] определяется точностью p

    if p == 1:
        # Метод Эйлера
        a = np.zeros((2, 2))
        b = np.zeros(2)
        c = np.zeros(2)
        a[1, :] = [0.5, 0]
        b[0] = 1

    elif p == 4:
        # РК4
        a = np.zeros((p, p))
        b = np.zeros(p)
        c = np.zeros(p)
        a[1, :] = [0.5, 0., 0., 0.]
        a[2, :] = [0., 0.5, 0., 0.]
        a[3, :] = [0., 0., 1., 0.]
        c = np.array([0., 0.5, 0.5, 1.])
        b = np.array([1/6, 2/6, 2/6, 1/6])
    else:
        print("Метод с заданной точностью не реализован")
        return [-1], [-1]

    for m in range(1, N_MAX + 1):
        sum_u = 0
        for k in range(0, p):
            sum_w = 0
            for i in range(0, k):
                sum_w = sum_w + a[k][i] * w[k]
            w[k] = f(t[m - 1] + h*c[k], y[m-1] + h*sum_w)

            sum_u = sum_u + b[k]*w[k]
        y[m] = y[m-1] + h*sum_u
    return t, y


def f(t, y):
    '''
        Одномерная задача dy/dt = cos(t)
    '''
    return np.cos(t)


def ff(t, y):
    '''
        ДВумерная задача.
        ДУ математического маятника
    '''

    g = 9.81
    l = 1

    res = np.zeros(2)
    res[0] = y[1]
    res[1] = -g/l*y[0]
    return res


# Решение одномерной задачи
# t, y = rk_solver(f, [0], [0, 30], p = 1)
# t4, y4 = rk_solver(f, [0], [0, 30], p = 4)
# y_scp = solve_ivp(f, [0, 30], [0], rtol=1e-9, atol=1e-9)

# plt.plot(t, y)
# plt.plot(t4, y4)
# plt.plot(y_scp.t, y_scp.y[0])
# plt.show()


# Решение двумерной задачи
t, y = rk_solver(ff, [np.pi/6, 0], [0, 10], p=1)
t4, y4 = rk_solver(ff, [np.pi/6, 0], [0, 10], p=4)
res = solve_ivp(ff, [0, 10], [np.pi/6, 0], rtol=1e-8, atol=1e-8)
plt.plot(t, y[:, 0])
plt.plot(t, y4[:, 0])
plt.plot(res.t, res.y[0])

plt.show()
