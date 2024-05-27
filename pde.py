# gonna write some PINN to solve some random pde; maybe the heat equation.
# Let's introduce some boundary conditions
'''
1. u(0, t) = 0
2. u(1, t) = 0
3. u(x, 0) = sin(pi*x)
'''

# The idea is to define a new loss function that will evaluate the loss in the model when the 
# boudnary conditions are met and then update the parameters accordingly.

import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

alpha = 0.03
epochs = 10000

class PINN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PINN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, output_size)

    def forward(self, x, t):
        x = torch.concat([x, t], dim=1)
        out = self.linear1(x)
        out = self.tanh(out)
        out = self.linear2(out)
        out = self.tanh(out)
        out = self.linear3(out)
        out = self.tanh(out)
        out = self.linear4(out)
        return out
    

def pinn_loss(model, alpha, x, t):
    x.requires_grad = True
    t.requires_grad = True
    u = model(x, t)
    u_t = torch.autograd.grad(u, t, create_graph=True, grad_outputs=torch.ones_like(t_colloc))[0]
    u_x = torch.autograd.grad(u, x, create_graph=True, grad_outputs=torch.ones_like(x_colloc))[0]
    u_xx = torch.autograd.grad(u_x, x, create_graph=True, grad_outputs=torch.ones_like(x_colloc))[0]

    physics_loss = u_t - (alpha)*u_xx # The wave equation modified such that the ideal outuput would be 0
    physics_loss = torch.mean(physics_loss**2)
    return physics_loss

def bc_training(model, optimizer, num_epochs):
    # Trains for the first BC
    zero_space = torch.zeros((1000, 1), requires_grad=True)
    time = torch.rand((1000, 1), requires_grad=True)
    print("Training on BCs..........")
    BC_training_compact(model, optimizer, zero_space, time, num_epochs, 1)

    final_coords = torch.ones((1000, 1), requires_grad=True)
    time = torch.rand((1000, 1), requires_grad=True)
    BC_training_compact(model, optimizer, final_coords, time, num_epochs, 2)

    space_coords = torch.rand((1000, 1), requires_grad=True)
    time = torch.zeros((1000, 1), requires_grad=True)
    for epoch in range(num_epochs):
        loss = BC3_loss(model, space_coords, time)
        loss.backward()
        optimizer.step()
        if epoch%100 == 0:
            print(f"BC: 3, epoch: {epoch}, BC_loss: {loss.item()}")


def BC_training_compact(model, optimizer, x, t, num_epochs, BC): # for whenever the boundary conditions imply that the value of the function is 0. conditions 1 and 2
    for epoch in range(num_epochs):
        u = model(x, t)
        loss = torch.mean(torch.square(u))
        loss.backward()
        optimizer.step()
        if epoch%100 == 0:
            print(f"BC: {BC}, epoch: {epoch}, BC_loss: {loss.item()}")

def BC3_loss(model, x, t):
    u = model(x, t)
    correct_vals = torch.sin(math.pi * x)
    loss = torch.mean((correct_vals - u)**2)
    return loss

def check_parameters(model):
    # to verify that training did produce changes in the parameters.
    params = {}
    for name, param in model.named_parameters():
        params[name] = param.clone().detach()
    return params

        
model = PINN(2, 25, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

initial_params = check_parameters(model)

def analytical_solution(x, t, alpha):
    return torch.exp(-alpha * (torch.pi ** 2) * t) * torch.sin(torch.pi * x)


# Training on BCs first
bc_training(model, optimizer, num_epochs=epochs)

final_params = check_parameters(model)


# Generating random data -- Have no idea if any of these points will actually satisfy this equation or nah but I'll iterate.
x = torch.rand(100) * (1 - 0) + 0
t = torch.rand(100) * (1 - 0) + 0
X, T = torch.meshgrid(x, t)
x_colloc = torch.tensor(X.flatten(), dtype=torch.float32).view(-1, 1)
t_colloc = torch.tensor(T.flatten(), dtype=torch.float32).view(-1, 1)

print("x_colloc shape: ", x_colloc.shape)
print("t_colloc shape: ", t_colloc.shape)


# Training on the pde loss
for epoch in range(epochs):
    optimizer.zero_grad()
    loss = pinn_loss(model, alpha, x_colloc, t_colloc) # calcualtes physics loss per epoch
    loss.backward()
    optimizer.step()
    if epoch%100 == 0:
        print(f"epoch: {epoch}, loss: {loss.item()}")

# Plot results
x_plot = torch.linspace(0, 1, 500).view(-1, 1)
t_plot = torch.linspace(0, 1, 500).view(-1, 1)
X_plot, T_plot = torch.meshgrid(x_plot.view(-1), t_plot.view(-1))

# Flatten the grid for model input
x_flat = X_plot.flatten().view(-1, 1)
t_flat = T_plot.flatten().view(-1, 1)

# Get model predictions
u_pred = model(x_flat, t_flat).detach().numpy()
U_plot = u_pred.reshape(500, 500)

# Get analytical solution
u_analytic = analytical_solution(X_plot, T_plot, alpha=alpha).numpy()

# Plotting
fig = plt.figure(figsize=(12, 6))

# PINN Solution
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X_plot.numpy(), T_plot.numpy(), U_plot, cmap='viridis')
ax1.set_xlabel('x')
ax1.set_ylabel('t')
ax1.set_zlabel('u')
ax1.set_title('PINN Solution')

# Analytical Solution
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X_plot.numpy(), T_plot.numpy(), u_analytic, cmap='viridis')
ax2.set_xlabel('x')
ax2.set_ylabel('t')
ax2.set_zlabel('u')
ax2.set_title('Analytical Solution')

plt.show()