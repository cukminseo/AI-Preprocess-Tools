import matplotlib.pyplot as plt
import numpy as np

# Gradient Descent 알고리즘 구현 (2D)
def gradient_descent_2d(function, derivative, start_point, lr, iterations=50):
    x, y = start_point
    history = [(x, y)]
    for _ in range(iterations):
        dx, dy = derivative(x, y)
        x, y = x - lr * dx, y - lr * dy
        history.append((x, y))
    return np.array(history)

# 주어진 2D 함수와 도함수 정의
def f(x, y):
    return x**2 / 20 + y**2

def df(x, y):
    return (x / 10, 2 * y)

# 시작점, 학습률 설정
start_point = (-5, 5)
lr = 0.1

# Gradient Descent 수행
history = gradient_descent_2d(f, df, start_point, lr)

# 함수의 등고선 및 GD 경로 시각화
x = np.linspace(-10, 10, 400)
y = np.linspace(-10, 10, 400)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

plt.figure(figsize=(8, 6))
contours = plt.contour(X, Y, Z, 20, cmap='viridis')
plt.clabel(contours, inline=True, fontsize=8)
plt.scatter(*history.T, color='red', s=10, label="GD Path")
plt.title("Gradient Descent on $f(x, y) = x^2/20 + y^2$")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
