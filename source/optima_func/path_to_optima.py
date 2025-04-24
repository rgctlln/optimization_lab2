import time

from source.BackTrace import BackTrace
from source.Function import Function


def try_backtrace(backtrace: BackTrace, start_point: list, func: Function):
    # НЕ создаём графики при автоподборе (можно отключить)
    # backtrace.set_create_graphic()
    start_timer = time.time()

    try:
        result = backtrace.start_back_trace(start_point=start_point)
    except Exception:
        return float("inf"), float("inf")

    variables = func.list_var
    point_dict = dict(zip(variables, result))

    value = func.evaluate(point_dict)
    duration = time.time() - start_timer

    return value, duration # минимальное значение и время


def objective_constant_step(trial, create_func:Function, start_point: list):
    # Подбираемый шаг градиентного спуска
    alpha = trial.suggest_float("constant_step", 1e-4, 1.0, log=True)

    # Создаём функцию и запускаем спуск
    backtrace = BackTrace(create_func).set_constant_step(constant_step=alpha)

    return try_backtrace(backtrace, start_point, create_func)


def objective_dichotomy_step(trial, create_func:Function, start_point: list):
    # Подбираемый шаг градиентного спуска
    alpha = trial.suggest_float("left_border", 1e-4, 1.0, log=True)

    beta = trial.suggest_float("right_border", 1e-4, 1.0, log=True)

    # Создаём функцию и запускаем спуск
    backtrace = BackTrace(create_func).set_dichotomy_step(a = alpha, b = beta)

    return try_backtrace(backtrace, start_point, create_func)

