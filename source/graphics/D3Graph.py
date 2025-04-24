import math

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


def create_graphic(func: sp.core.expr.Expr, dataset: list = None, minimum: tuple = None,
                   min_x: int = None, max_x: int = None, min_y: int = None, max_y: int = None,
                   view_2d: bool = False, hide_func_name: bool = False):
    x, y = sp.symbols('x y')
    f = sp.lambdify((x, y), func, 'numpy')

    if dataset is not None:
        min_x = math.floor(min(sublist[0] for sublist in dataset)) if min_x is None else min_x
        max_x = math.ceil(max(sublist[0] for sublist in dataset)) if max_x is None else max_x
        min_y = math.floor(min(sublist[1] for sublist in dataset)) if min_y is None else min_y
        max_y = math.ceil(max(sublist[1] for sublist in dataset)) if max_y is None else max_y

        minimum = dataset[-1]
        if min_x == max_x or min_y == max_y:
            raise Exception("The function cannot be drawn because the borders have coincided. Zoom in")
    else:
        min_x = -10 if min_x is None else min_x
        max_x = 10 if max_x is None else max_x
        min_y = -10 if min_y is None else min_y
        max_y = 10 if max_y is None else max_y

    size = max(abs(min_x), abs(max_x), abs(min_y), abs(max_y), 100)
    x_vals = np.linspace(min_x, max_x, size)
    y_vals = np.linspace(min_y, max_y, size)
    x_mesh, y_mesh = np.meshgrid(x_vals, y_vals)
    z_vals = f(x_mesh, y_mesh)

    if view_2d:
        # ===== 2D режим: контурная карта =====
        plt.figure(figsize=(8, 6))
        contour = plt.contourf(x_mesh, y_mesh, z_vals, levels=50, cmap='viridis')
        plt.colorbar(contour, label='Z = f(x, y)')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('2D представление функции')

        if dataset is not None:
            data = np.array(dataset)
            x_points, y_points, _ = data.T
            colors = ['red', 'maroon']

            # Рисуем отрезки между точками с чередующимися цветами
            for i in range(1, len(data)):
                plt.plot(x_points[i - 1:i + 1], y_points[i - 1:i + 1],
                         color=colors[(i - 1) % 2], marker='o')

            plt.plot([], [], color='red', label='Траектория (чередуется)')

        if minimum is not None:
            plt.scatter(minimum[0], minimum[1], color='green', s=100, label='Минимум')

        plt.legend()
        plt.grid(True)
        plt.show()

    else:
        # ===== 3D режим =====
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x_mesh, y_mesh, z_vals, cmap='viridis', alpha=0.7 if dataset is not None else 1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        if minimum is not None:
            ax.scatter(*minimum, color='green', s=100, label=f'Минимум ({",".join(map(str, minimum))})')

        if dataset is not None:
            data = np.array(dataset)
            x_points = data[:, 0]
            y_points = data[:, 1]
            f_points = data[:, 2]
            ax.plot(x_points, y_points, f_points, color='red', marker='o', label='Градиентный спуск')

        plt.legend()
        plt.show()
