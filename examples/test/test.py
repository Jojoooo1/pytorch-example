import torch

x = torch.zeros(2, dtype=torch.long)
print(x)
# tensor([0, 0])


y = torch.ones(2, 3, dtype=torch.long)
print(y)
# tensor([[0, 0, 0],
#         [0, 0, 0]])


z = torch.ones(2, 3, 5, dtype=torch.long)
print(z)

# tensor([[[1, 1, 1, 1, 1],
#          [1, 1, 1, 1, 1],
#          [1, 1, 1, 1, 1]],

#         [[1, 1, 1, 1, 1],
#          [1, 1, 1, 1, 1],
#          [1, 1, 1, 1, 1]]])

z = torch.ones(2, 3, 4, 5, dtype=torch.long)
print(z)

# tensor([[[1, 1, 1, 1, 1],
#          [1, 1, 1, 1, 1],
#          [1, 1, 1, 1, 1]],

#         [[1, 1, 1, 1, 1],
#          [1, 1, 1, 1, 1],
#          [1, 1, 1, 1, 1]]])