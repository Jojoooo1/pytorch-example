# Using torch
import torch
tensor_cpu = torch.ones(2, 2)

# CPU TO GPU
if torch.cuda.is_available():
    tensor_cpu.cuda()

# GPU to CPU
tensor_cpu.cpu()

a = torch.ones(2, 3)  # RxC

# + / torch.add(a, b)
# - / torch.sub
#   / torch.div

# _ => in place modification

a = torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(a.size())
print(a.mean(dim=0))
print(a.std(dim=0))

a = torch.Tensor([[0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6]])
print(a.mean(dim=1))
print(a.size())
