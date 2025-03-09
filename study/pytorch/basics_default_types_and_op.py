import torch
import numpy as np

print(f"torch version : {torch.__version__}, is enable mps ? : {torch.backends.mps.is_available()}")  # 설치된 PyTorch 버전 확인

# 1차원 (Vector)
x = torch.tensor([1, 2, 3, 4])
print(x)

# 2차원 (Matrix)
y = torch.tensor([[1, 2], [3, 4]])
print(y)

# 랜덤 텐서
z = torch.rand(3, 3)  # 3x3 행렬 (0~1 사이 랜덤 값)
print(z)

print(x.shape)  # (4,)
print(y.shape)  # (2,2)
print(z.dtype)  # float32 (기본값)

# 덧셈
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
c = a + b
print(c)  # tensor([5, 7, 9])

# 행렬 곱셈
A = torch.rand(2, 3)
B = torch.rand(3, 2)
C = torch.mm(A, B)  # 행렬 곱 (2x3) @ (3x2) -> (2x2)
print(C)

x = torch.tensor([1.0, 2.0, 3.0])
x_np = x.numpy()
print(x_np, type(x_np))  # [1. 2. 3.] <class 'numpy.ndarray'>

# NumPy 배열을 PyTorch Tensor로 변환
y_np = np.array([4.0, 5.0, 6.0])
y_torch = torch.from_numpy(y_np)
print(y_torch, type(y_torch))  # tensor([4., 5., 6.]) <class 'torch.Tensor'>
