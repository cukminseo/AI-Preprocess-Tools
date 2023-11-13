import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.optimize import minimize, approx_fprime


def obj_fun_quartic(x):
    return x ** 4 - 4 * x ** 3 + 4 * x ** 2 + x - 4


def plt_opt_res(res):
    plt.figure(figsize=[16, 10])
    x_val = np.linspace(-1.0, 3.0, 100)
    plt.plot(x_val, obj_fun_quartic(x_val), '--')
    plt.scatter(res.x, obj_fun_quartic(res.x), s=100, c='r', alpha=0.75, marker='o')
    plt.annotate("minimum", (res.x, obj_fun_quartic(res.x) + 0.2), size=23)
    plt.annotate("x^4 - 4*x^3 + 4*x^2 + x - 4", (1.5, 0.5), size=20)
    plt.title('Find Minimum of Quartic Function', fontsize=22)
    plt.xlabel('x', fontsize=20)
    plt.ylabel('func(x)', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()


x0 = -0.5  # 초기값
res = minimize(obj_fun_quartic, x0, method="SLSQP", callback=print, options={'disp': True})

plt_opt_res(res)
print(res.x)
