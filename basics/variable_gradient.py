import torch
from torch.autograd import Variable

# 1. Create Variables
x = Variable(torch.ones(2), requires_grad=True)
print(x)

# 2. Define equation with a x value
y = 5 * (x + 1)**2
print(y)

# 3. Reduce to scalar output, "o" through mean (divide by number of a value which is 2)
o = (1 / 2) * torch.sum(y)

# 4. Calculate gradient 
o.backward() 

# 5. Access gradient of x
gradient = x.grad
print (gradient)

1. 
2. 
3. 