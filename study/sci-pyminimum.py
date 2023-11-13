import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.optimize import minimize, approx_fprime


def objective(X1):
    return X1 ** 4 + X1**3 +(-8) * (X1 ** 2) + 6

plt.title('Linear Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-5, 5)
plt.ylim(-25, 25)
plt.grid(True)
x = np.linspace(-4, 4, 100)
plt.plot(x, objective(x), '--', label='function')


# x0 = 5
# res = minimize(objective, x0, method="SLSQP", callback=print, options={'disp': True})



# local optimize
param_bounds = [(-4, 4)]
global_result = differential_evolution(objective, param_bounds,
                                           strategy='best1bin', maxiter=1000, popsize=15)
print("Global optimization number of iterations:", global_result.nit)
print("Global optimization number of iterations:", global_result.x)
# 전역 최적화의 결과를 초기값으로 사용하여 지역 최적화
local_result = minimize(objective, global_result.x,
                            method="L-BFGS-B")
print("Local optimization number of iterations:", local_result.nit)
optimized_params = local_result.x
print(optimized_params)



# 전역 최적화의 결과를 초기값으로 사용하여 지역 최적화
local_result = minimize(objective, global_result.x,
                            method="L-BFGS-B", callback=print)
print("Local optimization number of iterations:", local_result.nit)
plt.scatter(local_result.x, objective(local_result.x),
            s=100, c='r', alpha=0.75, marker='o', label='optimized')
plt.legend()
plt.show()


