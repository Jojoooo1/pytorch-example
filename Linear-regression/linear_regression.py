# Linear model => y = ax + b # a = coeficient, b = intercept

# Linear Regression definition: minimize the distance between the points and the line

# calculate distance trough MSE, calculate gradients (vector), update parameters, slowly update parameters with a & b

import numpy as np
import matplotlib.pyplot as plt

# Creates a List of value
x_values = [i for i in range(11)]  # O to 10 = 11 value

# Tranforms it to np Array object  ( of 1 dimension )
x_train = np.array(x_values, dtype=np.float32)
x_train.shape  # 11

# print(x_train)  # [ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10.]

# 3. Transform it to 2 Dimensions "1" => numpy define the number to input when the second dimension is "1"
x_train = x_train.reshape(
    -1, 1
)  # "-1", **1** = Let numpy define the number to input when the second dimension is **1**
x_train.shape
# if **1**  (11, 1) if **2** (6, 2) but need to modify the range to 12 (because 11 / 2 is not possible)

# print(x_train)
# case reshape is 1 [[ 0.]  is 2 [[0. 1.]
#                   ...           ...
#                   [9.]]        [8. 9.]]

# True Output based on a function that the model should find
y_values = [2 * i + 1
            for i in x_values]  # [ 1.  3.  5.  7.  9. 11. 13. 15. 17. 19. 21.]

# Transform it to np Array object
y_train = np.array(y_values, dtype=np.float32)
y_train.shape
# print(y_train)

# Transform it to 2 Dimensions
y_train = y_train.reshape(-1, 1)
y_train.shape
# print(y_train)

import torch
import torch.nn as nn
from torch.autograd import Variable

# STEP 1: CREATE MODEL CLASS


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel,
              self).__init__()  # inherit LinearRegressionModel in nn module
        self.linear = nn.Linear(input_dim,
                                output_dim)  # instantiate self.linear

    def forward(self, x):
        out = self.linear(x)
        return out


# STEP 2: INSTANTIATE MODEL CLASS
input_dim = 1
output_dim = 1

model = LinearRegressionModel(input_dim, output_dim)

#######################
#  USE GPU FOR MODEL  #
#######################
if torch.cuda.is_available():
    model.cuda()

# STEP 3: INSTANTIATE LOSS CLASS
# Instantiate a Loss Class ( how we want to minimize the distance between the point and the line )
# We use a Mean Squared Error Loss ( Y' (predicted value) - Y labels (true value) / n (number of y value) )
criterion = nn.MSELoss()

# STEP 4: INSTANTIATE OPTIMIZER CLASS
# Instantiate Optimizer Class
learning_rate = 0.01  # how fast we want our model to learn
# model.parameters() equivalent of ax + b
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# STEP 5: TRAIN THE MODEL
epochs = 100  # 100 times [ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10.]

for epoch in range(epochs):
    epoch += 1

    # #######################
    # #  USE GPU FOR MODEL  #
    # #######################
    if torch.cuda.is_available():
        inputs = Variable(torch.from_numpy(x_train).cuda())
        labels = Variable(torch.from_numpy(y_train).cuda())
    else:
        inputs = torch.from_numpy(x_train).requires_grad_(
        )  # Variable as requires_grad=True by default
        labels = torch.from_numpy(y_train)

    # Clear gradients  # dont want gradient from previous epoch
    optimizer.zero_grad()

    # Forward to get output
    outputs = model(inputs)

    # Calculate Loss between true & calculated
    loss = criterion(outputs, labels)  # calculate loss of the model

    # Calculate the gradients ( after getting scalar loss )
    loss.backward() # ** compute gradient of loss ** w.r.t all the parameters in loss that have requires_grad = True and store them in parameter.grad attribute for every parameter.

    # Updating parameters
    optimizer.step() # ** update parameter based on parameter.grad ** (optimizer iterate over all parameters to update their values)

    print('epoch {}, loss {}'.format(epoch, loss.item()))

# Get predictions
predicted = model(Variable(
    torch.from_numpy(x_train))).data.numpy()  # feeding our x data to our model
# print(predicted)
print(np.squeeze(predicted))  # squeeze it into 1 dimension

####################
## DRAW PLOT LINE ##
####################

# y_predicted (squeezed) [ 0.9127647  2.9253275  4.93789    6.950453   8.963016  10.975578  12.988141  15.000704  17.013268  19.02583   21.038393 ]
# y_train (squeezed) [1.  3.  5.  7.  9. 11. 13. 15. 17. 19. 21.]

# Clear figure
plt.clf()

# Plot true data
plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)

# Plot predictions
plt.plot(x_train, predicted, '--', label='Predictions', alpha=0.5)

# Legend and plot
plt.legend(loc='best')
plt.show()

# Save Model
save_model = False
if save_model is True:
    # Saves only parameters
    # alpha & beta
    torch.save(model.state_dict(), 'awesome_model.pkl')

load_model = False
# Load Model
if load_model is True:
    model.load_state_dict(torch.load('awesome_model.pkl'))