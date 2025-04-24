import math

import matplotlib.pyplot as plt
import numpy as np
import sympy
import sympy as sp


def create_graphic(func: sympy.core, dataset: list = None, minimum: tuple = None, min_x: int = None, max_x: int = None, hide_func_name: bool = False):
    # Определяем символы
    x = sp.symbols('x')

    if dataset is not None:
        min_x = math.floor(min(sublist[0] for sublist in dataset)) if min_x is None else min_x
        max_x = math.ceil(max(sublist[0] for sublist in dataset)) if max_x is None else max_x
        if min_x == max_x :
            raise Exception("The function cannot be drawn because the borders have coincided. Zoom in")

        # min_x = int(abs(dataset[0][0]) * -1)
        # max_x = -min_x
        minimum = dataset[len(dataset) - 1]
    else:
        min_x = -10 if min_x is None else min_x
        max_x = 10 if max_x is None else max_x

    # Создаем сетку для x и y
    x_vals = np.linspace(min_x, max_x, max(abs(min_x), abs(max_x), 100))

    # Вычисляем значения функции
    y_vals = sp.lambdify(x, func, 'numpy')(x_vals)

    plt.plot(x_vals, y_vals, label=f'{f"f = {func}" if not hide_func_name else ""}', color='b')

    # Добавляем подписи
    plt.xlabel('X')
    plt.ylabel('f')

    if minimum is not None:
        plt.scatter(*minimum, color='green', s=100, label=f'Минимум ({",".join(map(str, minimum))})')

    if dataset is not None:
        data = np.array(dataset)
        x_points = data[:, 0]
        y_points = data[:, 1]

        # Линия траектории
        plt.plot(x_points, y_points, color='red', marker='o', label='Градиентный спуск')

    plt.legend()

    # Показываем график
    plt.show()


