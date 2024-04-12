import numpy as np
from sklearn.decomposition import NMF

# 사용자-아이템 행렬 생성 (누락된 값은 np.nan으로 표시)
R = np.array([
    [5, 3, np.nan, 1],
    [4, np.nan, np.nan, 1],
    [1, 1, np.nan, 5],
    [1, np.nan, np.nan, 4],
    [np.nan, 1, 5, 4],
])

# np.nan을 0으로 대체
R[np.isnan(R)] = 0

# NMF 모델 생성
model = NMF(n_components=2, init='random', random_state=0)

# 모델 훈련
W = model.fit_transform(R)
H = model.components_

# 예측 행렬 생성
R_hat = np.dot(W, H)

print("예측 행렬:")
print(R_hat)
