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

# Generating data points for training
x_colloc = torch.rand(10000, 1, requires_grad=True)
t_colloc = torch.rand(10000, 1, requires_grad=True)

x_boundary = torch.cat([torch.zeros(5000, 1), torch.ones(5000, 1)], dim=0)
t_boundary = torch.rand(10000, 1)

x_initial = torch.rand(10000, 1)
t_initial = torch.zeros(10000, 1)

def pde_loss(model, x, t):
    x.requires_grad = True
    t.requires_grad = True
    u = model(x, t)
    u_t = torch.autograd.grad(u, t, create_graph=True, grad_outputs=torch.ones_like(t_colloc))[0]
    u_x = torch.autograd.grad(u, x, create_graph=True, grad_outputs=torch.ones_like(x_colloc))[0]
    u_xx = torch.autograd.grad(u_x, x, create_graph=True, grad_outputs=torch.ones_like(x_colloc))[0]

    physics_loss = u_t - (alpha)*u_xx # The wave equation modified such that the ideal outuput would be 0
    physics_loss = torch.mean(physics_loss**2)
    return physics_loss

def boundary_loss(model, x, t):
    x.requires_grad = True
    t.requires_grad = True
    u = model(x, t)
    loss = torch.mean(torch.square(u))
    return loss

def initial_loss(model, x, t):
    x.requires_grad = True
    t.requires_grad = True
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

def analytical_solution(x, t, alpha):
    return torch.exp(-alpha * (torch.pi ** 2) * t) * torch.sin(torch.pi * x)


# Training on the pde loss
for epoch in range(epochs):
    optimizer.zero_grad()
    eq_loss = pde_loss(model, x_colloc, t_colloc)
    bc_loss = boundary_loss(model, x_boundary, t_boundary)
    init_loss = initial_loss(model, x_initial, t_initial)

    total_loss = eq_loss + bc_loss + init_loss
    total_loss.backward()
    optimizer.step()
    if epoch%100 == 0:
        print(f"epoch: {epoch}, loss: {total_loss.item()}")

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