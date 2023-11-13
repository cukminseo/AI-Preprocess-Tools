import numpy as np
import matplotlib.pyplot as plt

# Gradient Descent 알고리즘 구현
def gradient_descent(function, derivative, start_point, lr, iterations=100):
    x = start_point
    history = [x]
    for _ in range(iterations):
        x = x - lr * derivative(x)
        history.append(x)
    return np.array(history)

# 주어진 함수와 도함수 정의
def f(x):
    return x * (x - 1) * (x - 2) * (x - 4)

def df(x):
    return 4 * x**3 - 21 * x**2 + 22 * x - 4

# 시작점, 학습률, 반복 횟수 설정
start_point = 0
lr = 0.1
iterations = 3

# Gradient Descent 수행
x_history = gradient_descent(f, df, start_point, lr, iterations)

# 함수와 GD 과정 시각화
x_values = np.linspace(-0.4, 4.4, 400)
y_values = f(x_values)

plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, label="Function: $x(x-1)(x-2)(x-4)$")
plt.scatter(x_history, f(x_history), color='red', label="GD Iterations")
plt.title(f"Gradient Descent Optimization(lr={lr})")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()
