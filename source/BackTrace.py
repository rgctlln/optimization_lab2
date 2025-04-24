import math
import random
import time

import numpy as np
import sympy

from source.Function import Function
from source.graphics import d2_create_graphic
from source.graphics import d3_create_graphic
from source.one_dimensional_search.Dichotomy import dichotomy
from source.one_dimensional_search.GoldenSectionSearch import golden_section_search


def required_step_defined(method):
    """
    Обязательный декоратор для всех методов, кроме инициализирующих.

    Проверяет заданность шага.
    :param method:
    :return:
    """

    def wrapper(self, *args, **kwargs):
        if self.step is None:
            raise Exception("The step is not defined, call the step setup")
        return method(self, *args, **kwargs)

    return wrapper


def bfgs(
    grad_func,           # функция: принимает x, возвращает градиент ∇f(x)
    func,                # сама функция f(x), для логгирования/проверки
    x0,                  # начальная точка
    max_iter=100,
    tol=1e-6,
    alpha=1.0            # начальный шаг
):
    x = np.array(x0, dtype=float)
    n = len(x)
    H = np.eye(n)  # начальная аппроксимация обратного гессиана

    for i in range(max_iter):
        grad = grad_func(x)
        if np.linalg.norm(grad) < tol:
            break

        p = -H @ grad
        x_new = x + alpha * p
        grad_new = grad_func(x_new)

        s = x_new - x
        y = grad_new - grad

        rho = y @ s
        if rho == 0:
            break

        rho = 1.0 / rho
        I = np.eye(n)
        H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)

        x = x_new

    return x, func(x)


class BackTrace:

    def __init__(self, func: Function):
        self.create_graphic = None
        self.dichotomy_range = None
        self.start_step = None
        self.step = None  # требуется вызов установки шага
        self.func = func
        self.cnt_iterations = 0
        self.choose_step_mode = None
        self.full_output = False
        self.exponential_decay = None

        self.polynomial_decay_alpha = None
        self.polynomial_decay_beta = None
        return

    def set_full_output(self):
        self.full_output = True
        return self

    def set_create_graphic(self):
        self.create_graphic = True
        return self

    def set_constant_step(self, constant_step: int | float):
        """
        Устанавливает шаг как константу

        :param constant_step:
        :return:
        """

        self.choose_step_mode = "constant"
        self.step = constant_step
        return self

    def set_piecewise_constant(self, start_step: int | float = 1):
        """
        Устанавливаем шаг как кусочно-постоянную функцию

        :return:
        """
        self.choose_step_mode = "piecewise_constant"
        self.step = start_step
        return self

    def set_exponential_decay(self, exponential_decay: float, start_step: int | float):
        """
        Устанавливаем шаг как кусочно-постоянную функцию

        :return:
        """
        self.choose_step_mode = "exponential_decay"
        self.exponential_decay = exponential_decay
        self.step = start_step
        self.start_step = start_step
        return self

    def set_polynomial_decay(self, alpha: float = 0.5, beta: float = 1.0, start_step: float = 1.0):
        """
        Устанавливает полиномиальное затухание: h(k) = h0 / (βk + 1)^α

        :param alpha: показатель степени α
        :param beta: множитель при k
        :param start_step: начальный шаг h0
        """
        self.choose_step_mode = "polynomial_decay"
        self.polynomial_decay_alpha = alpha
        self.polynomial_decay_beta = beta
        self.step = start_step
        self.start_step = start_step
        return self

    def set_dichotomy_step(self, a=1e-5, b=1.0):
        """
        Устанавливаем метод выбора шага через дихотомию

        :param a: левая граница отрезка
        :param b: правая граница отрезка
        """
        self.choose_step_mode = "dichotomy"
        self.dichotomy_range = (a, b)
        self.step = 1
        return self

    def set_golden_section_step(self, a=1e-5, b=1.0):
        """
        Устанавливаем метод выбора шага через метод золотого сечения

        :param a: левая граница отрезка
        :param b: правая граница отрезка
        """
        self.choose_step_mode = "golden_section"
        self.dichotomy_range = (a, b)
        self.step = 1
        return self

    def set_bfgs(self, alpha=1.0):
        self.choose_step_mode = "bfgs"
        self.step = alpha
        return self

    @required_step_defined
    def start_back_trace(self, start_point: list = None, graphic_borders: list = None,
                         show_func_graph_name: bool = True) -> list:
        """
        Запуск градиентного спуска
        :return:
        """
        start_time = time.time()
        # epsilon = np.finfo(float).eps
        epsilon = 1e-6
        history_last_norma = -1
        cnt_history_last_norma = 0

        dataset = []  # сюда собираются точки для графика

        last_point = {}
        tuple_of_variables = self.func.get_tuple_of_variables()

        if start_point is not None:
            if len(start_point) != len(tuple_of_variables):
                raise Exception("start_point and tuple_of_variables must have same length")

        for index in range(len(tuple_of_variables)):
            var = tuple_of_variables[index]
            if start_point is None:
                last_point[var] = random.Random().randint(1, 1000)
            else:
                last_point[var] = start_point[index]

        derivatives = self.func.count_all_partial_derivative()

        hessian: sympy.Matrix = self.func.count_all_second_derivatives_matrix()
        hessian_inv = hessian.inv()  # инвертируем гессиан

        temp_save_kv_derivative_val = 0

        if self.full_output:
            print("Iterations|Step|", end="")
            for el in self.func.list_var:
                print(el.name, end="|")
            # print("|".join(derivatives.keys()), end = "|")
            print("norma|F")

        if self.choose_step_mode == "bfgs":
            return self.run_bfgs(start_point)

        for i in range(0, 10000):
            self.cnt_iterations += 1
            new_point = {}
            if self.full_output:
                print(self.cnt_iterations, self.step, sep="|", end="|")

            self.append_point_to_graphic(dataset, last_point)

            grad = derivatives.subs(last_point).evalf()

            hessian_at_point = hessian_inv.subs(last_point).evalf()

            # Определяем знак определённости гессиана
            eigenvals = list(hessian.subs(last_point).eigenvals().keys())
            is_positive_definite = all(e > 0 for e in eigenvals)
            is_negative_definite = all(e < 0 for e in eigenvals)

            if is_positive_definite:
                direction = -hessian_at_point * grad
            elif is_negative_definite:
                direction = hessian_at_point * grad
            else:
                raise Exception(
                    "Гессиан не определённый — функция имеет седловую точку. Метод Ньютона может не работать.")

            RENAME_ME_PLS_GOD = direction

            for j in range(len(RENAME_ME_PLS_GOD)):
                symbol = self.func.list_var[j]
                eval_derivative = RENAME_ME_PLS_GOD[j]
                # grad[symbol] = eval_derivative
                temp_save_kv_derivative_val += grad[j] ** 2

                if self.choose_step_mode not in ["dichotomy", "golden_section"]:
                    new_point[symbol] = last_point[symbol] + self.step * eval_derivative
                    if self.full_output:
                        print(new_point[symbol], end="|")

            if self.choose_step_mode in ["dichotomy", "golden_section"]:
                def phi(alpha):
                    # last_point_values = last_point.values()
                    temp_symbols = {}
                    temp2_index = 0
                    for key2, value in last_point.items():
                        temp_symbols[key2] = last_point[key2] - alpha * grad[temp2_index]
                        temp2_index += 1

                    # temp_symbols = {k: last_point[k] - alpha * grad[k] for k in grad}
                    return self.func.evaluate(temp_symbols)

                if self.choose_step_mode == "dichotomy":
                    self.step = dichotomy(phi, self.dichotomy_range[0], self.dichotomy_range[1])
                else:
                    self.step = golden_section_search(phi, self.dichotomy_range[0], self.dichotomy_range[1])

                temp_index = 0
                for key in self.func.list_var:
                    new_point[key] = last_point[key] - self.step * grad[temp_index]
                    temp_index += 1

                if self.full_output:
                    for key in new_point.keys():
                        print(new_point[key], end="|")

            new_norma = math.sqrt(temp_save_kv_derivative_val)
            if self.full_output:
                print(new_norma, end="|")
                print(self.func.evaluate(new_point))

            # stop
            if new_norma ** 2 < epsilon:
                self.append_point_to_graphic(dataset, new_point)
                break

            if self.choose_step_mode == "exponential_decay":
                self.step = self.start_step * np.exp((-self.exponential_decay) * self.cnt_iterations)
            elif self.choose_step_mode == "polynomial_decay":
                self.step = self.start_step / (
                        (self.polynomial_decay_beta * self.cnt_iterations + 1) ** self.polynomial_decay_alpha)

            temp_save_kv_derivative_val = 0
            if new_norma == history_last_norma:
                cnt_history_last_norma += 1
            else:
                cnt_history_last_norma = 0
                history_last_norma = new_norma

            if cnt_history_last_norma >= 10:
                cnt_history_last_norma = 0
                history_last_norma = -1
                print("Мы обнаружили зацикливание")
                print(f"Повторяющаяся норма: {new_norma}")
                print(f"Последнее значение функции: {self.func.evaluate(new_point)}")
                if self.choose_step_mode == "constant":
                    raise Exception(
                        f"A short circuit has been detected, and the constant step mode does not allow changing the step. Run the method again, but reduce the step. Current step: {self.step}")
                elif self.choose_step_mode == "piecewise_constant":
                    self.step /= 2
                elif self.choose_step_mode == "exponential_decay":
                    pass
                else:
                    raise Exception(
                        f"A short circuit has been detected")

            last_point = new_point
        else:
            raise Exception(
                f"Protection has been activated. More than {self.cnt_iterations} iterations have been done.")

        end_time = time.time()
        self.print_results(last_point, start_time, end_time)

        if self.create_graphic:
            print(f"Стартовая точка {dataset[0]}")

            cnt_vars = len(self.func.list_var)
            if cnt_vars == 2:
                if graphic_borders not in [None, []] and len(graphic_borders) == 4:
                    d3_create_graphic(self.func.function, dataset=dataset, min_x=graphic_borders[0],
                                      max_x=graphic_borders[1], min_y=graphic_borders[2], max_y=graphic_borders[3],
                                      hide_func_name=show_func_graph_name)
                    d3_create_graphic(self.func.function, dataset=dataset, min_x=graphic_borders[0],
                                      max_x=graphic_borders[1], min_y=graphic_borders[2], max_y=graphic_borders[3],
                                      view_2d=True, hide_func_name=show_func_graph_name)
                else:
                    d3_create_graphic(self.func.function, dataset=dataset, hide_func_name=show_func_graph_name)
                    d3_create_graphic(self.func.function, dataset=dataset, view_2d=True,
                                      hide_func_name=show_func_graph_name)

            elif cnt_vars == 1:

                if graphic_borders not in [None, []] and len(graphic_borders) == 2:
                    d2_create_graphic(self.func.function, dataset=dataset, min_x=graphic_borders[0],
                                      max_x=graphic_borders[1], hide_func_name=not show_func_graph_name)
                else:
                    d2_create_graphic(self.func.function, dataset=dataset, hide_func_name=not show_func_graph_name)

            else:
                raise Exception(f"It is not possible to create a graph for an {cnt_vars} dimensional function")
        return list(last_point.values())

    def append_point_to_graphic(self, dataset, symbols_dict):
        if self.create_graphic:
            new_list = list(symbols_dict.values())
            new_list.append(self.func.function.subs(symbols_dict).evalf())
            dataset.append(list(map(float, new_list)))  # todo мб dict не гарантирует порядок, проверить

    def run_bfgs(self, start_point):
        def grad_numpy(x_np):
            point = {sym: val for sym, val in zip(self.func.list_var, x_np)}
            grad_sym = self.func.count_all_partial_derivative()
            grad_vals = grad_sym.subs(point).evalf()
            return np.array(grad_vals).astype(float).flatten()

        def func_numpy(x_np):
            point = {sym: val for sym, val in zip(self.func.list_var, x_np)}
            return float(self.func.evaluate(point))

        def bfgs_with_trace(grad_func, func, x0, max_iter=100, tol=1e-6, alpha=1.0):
            x = np.array(x0, dtype=float)
            n = len(x)
            H = np.eye(n)
            trace = []

            for _ in range(max_iter):
                grad = grad_func(x)
                f_val = func(x)
                trace.append(list(x) + [f_val])

                if np.linalg.norm(grad) < tol:
                    break

                p = -H @ grad
                x_new = x + alpha * p
                grad_new = grad_func(x_new)

                s = x_new - x
                y = grad_new - grad
                rho = y @ s
                if rho == 0:
                    break
                rho = 1.0 / rho
                I = np.eye(n)
                H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)

                x = x_new

            return x, func(x), trace

        x0 = [0.5] * len(self.func.list_var) if start_point is None else start_point
        x_min, f_min, trace = bfgs_with_trace(grad_numpy, func_numpy, x0, alpha=self.step)

        print(f"Минимум найден: {x_min}")
        print(f"Значение функции: {f_min}")

        if self.create_graphic:
            cnt_vars = len(self.func.list_var)
            if cnt_vars == 2:
                d3_create_graphic(self.func.function, dataset=trace, hide_func_name=False)
                d3_create_graphic(self.func.function, dataset=trace, view_2d=True, hide_func_name=False)
            elif cnt_vars == 1:
                d2_create_graphic(self.func.function, dataset=trace, hide_func_name=False)
            else:
                print("График невозможен — функция имеет более 2 переменных.")

        return x_min

    def print_results(self, symbols_dict, start_time, end_time):
        """
        Вывод результатов

        :param end_time: время конца запуска алгоритма
        :param start_time : время начала запуска алгоритма
        :param symbols_dict: словарь из значений переменных
        :return:
        """
        print(f"Расчёт минимума закончен. Был использован {self.choose_step_mode} шаг")
        print(f"Функция: {self.func}")
        print(f"Время выполнения: {end_time - start_time} секунд")
        print(f"Использованное количество итераций: {self.cnt_iterations}")
        print(symbols_dict)  # todo remove
        for sym, value in symbols_dict.items():
            try:
                print(f"{sym}: {round(value, 10)}")
            except ZeroDivisionError as ex:
                print(f"{sym}: {0}")
        print(f"Минимальное значение функции: {self.func.evaluate(symbols_dict)}")
        print("=================")
