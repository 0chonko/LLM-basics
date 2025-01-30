# Import PyTorch
import torch

# 1. Create Tensors
# A tensor is a multi-dimensional array, similar to NumPy arrays but with GPU support.
scalar = torch.tensor(3.14)  # Scalar (0D tensor)
vector = torch.tensor([1, 2, 3])  # Vector (1D tensor)
matrix = torch.tensor([[1, 2], [3, 4]])  # Matrix (2D tensor)

print("Scalar:", scalar)
print("Vector:", vector)
print("Matrix:\n", matrix)

# 2. Tensor Operations
# Perform basic arithmetic operations on tensors <button class="citation-flag" data-index="5">.
addition = vector + vector
multiplication = matrix * 2
dot_product = torch.dot(vector, vector)  # Dot product of two vectors

print("\nAddition:", addition)
print("Multiplication:\n", multiplication)
print("Dot Product:", dot_product)

# 3. Matrix Multiplication
# Use `torch.matmul` for matrix multiplication <button class="citation-flag" data-index="7">.
matrix_a = torch.tensor([[1, 2], [3, 4]])
matrix_b = torch.tensor([[2, 0], [1, 3]])
matmul_result = torch.matmul(matrix_a, matrix_b)

print("\nMatrix A:\n", matrix_a)
print("Matrix B:\n", matrix_b)
print("Matrix Multiplication Result:\n", matmul_result)

# 4. Tensor Attributes
# Explore attributes like shape, data type, and device <button class="citation-flag" data-index="9">.
print("\nShape of Matrix:", matrix.shape)
print("Data Type of Matrix:", matrix.dtype)
print("Device of Matrix:", matrix.device)

# 5. Move Tensor to GPU (if available)
# PyTorch allows tensors to be moved to GPU for faster computation <button class="citation-flag" data-index="10">.
if torch.cuda.is_available():
    gpu_tensor = matrix.to('cuda')  # Move tensor to GPU
    print("\nTensor on GPU:", gpu_tensor)
else:
    print("\nGPU not available.")

# 6. Gradient Tracking
# PyTorch supports automatic differentiation using `requires_grad`.
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1
y.backward()  # Compute gradients
print("\nGradient of y w.r.t. x:", x.grad)

# 7. Load Data into Tensors
# Convert a Python list or NumPy array into a PyTorch tensor.
import numpy as np
numpy_array = np.array([1, 2, 3])
tensor_from_numpy = torch.from_numpy(numpy_array)

print("\nNumpy Array:", numpy_array)
print("Tensor from Numpy:", tensor_from_numpy)