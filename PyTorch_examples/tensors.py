# This code corresponds to the PyTorch Tensors tutorial.

import torch
import numpy as np


# Initializing a Tensor

# Initializing a tensor directly from data:

data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

print("x_data: ", x_data)

# Initializing a tensor from a NumPy array:

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

print("x_np: ", x_np)

# Initializing a tensor from another tensor:

x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")


# Initializing a tensor with random or constant values:

shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor} \n")


# Attributes of a Tensor

# Tensor attributes describe their shape, datatype, and the device on which they are stored.

tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device} \n")


# Operations on Tensors

# By default, tensors are created on the CPU. We need to explicitly move tensors to the GPU
# using .to method (after checking for GPU availability). Keep in mind that copying large tensors
# across devices can be expensive in terms of time and memory!

# We move the previously created tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

print(f"Device tensor is now stored on: {tensor.device} \n")

# Once tensors are on ghe GPU, operations on them can be run, typically at higher speeds than on a CPU.

# Standard numpy-like indexing and slicing:

tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor, "\n")

# Joining tensors:
# You can use torch.cat to concatenate a sequence of tensors along a given dimension.
# See also torch.stack, another tensor joining op that is subtly different from torch.cat.

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print("t1:", t1, "\n")


# Arithmetic operations

# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value.
# Note that @ is the matrix multiplication operator in PyTorch;  it is equivalent to matmul.
# See also https://docs.python.org/3/library/operator.html#mapping-operators-to-functions
# .T results in the transpose of the tensor.

y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)

print("tensor.T: ", tensor.T, "\n")

print("y1: ", y1, "\n")
print("y2: ", y2, "\n")
print("y3: ", y3, "\n")

# This computes the element-wise product. z1, z2, z3 will have the same value.

z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

print("z1: ", z1, "\n")
print("z2: ", z2, "\n")
print("z3: ", z3, "\n")

# Single-element tensors:

# If you have a one-element tensor, for example by aggregating all values of a tensor into one value,
# you can convert it to a Python numerical value using item():

agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item), "\n")

# In-place operations:
# Operations that store the result into the operand are called in-place. They are denoted by a _ suffix.
# For example: x.copy_(y), x.t_(), will change x.

print(f"{tensor} \n")
tensor.add_(5)
print(tensor, "\n")

# Note:  in-place operations save some memory, but can be problematic when computing derivatives
# because of an immediate loss of history. Hence, their use is discouraged.


# Bridge with NumPy

# Tensors on the CPU and NumPy arrays can share their underlying memory locations,
# and changing one will change the other.

# Tensor to NumPy array

t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

# A change in the tensor reflects in the NumPy array.

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# NumPy array to tensor

n1 = np.ones(5)
t1 = torch.from_numpy(n1)
print(f"n1: {n1}")
print(f"t1: {t1}")

# Changes in the NumPy array reflects in the tensor.

np.add(n1, 1, out=n1)
print(f"t1: {t1}")
print(f"n1: {n1}")




