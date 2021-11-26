import torch
from scipy.optimize import linprog
import numpy as np
import cvxpy as cp
import scipy.optimize as opt
import matplotlib.pyplot as plt
import time
from torch import distributions as dist

class perturbations():

    def __init__(self,func,num_samples,sigma,noise,batched):
        self.func = func
        self.num_samples = num_samples
        self.sigma = sigma
        self.noise = noise
        self.batched = batched
        
    def sample_noise_with_gradients(noise,shape):
        _GUMBEL = 'gumbel'
        _NORMAL = 'normal'
        SUPPORTED_NOISES = (_GUMBEL, _NORMAL)
        if noise not in SUPPORTED_NOISES:
            raise ValueError('{} noise is not supported. Use one of [{}]'.format(
            noise, SUPPORTED_NOISES))
        if noise == _GUMBEL:
            sampler = dist.gumbel.Gumbel(0.0, 1.0)
            samples = sampler.sample(shape)
            gradients = 1 - torch.exp(-samples)
        elif noise == _NORMAL:
            sampler = dist.normal.Normal(0.0, 1.0)
            samples = sampler.sample(shape)
            gradients = samples
        return samples, gradients

    def forward(self,input_tensor):
        original_input_shape = input_tensor.size()
        if self.batched:
            if original_input_shape[0] < 2:
                raise ValueError('Batched inputs must have at least rank two')
        else:  # Adds dummy batch dimension internally.
            input_tensor = torch.unsqueeze(input_tensor,0)
        input_shape = torch.tensor(input_tensor.size(),dtype=torch.int)  # [B, D1, ... Dk], k >= 1, Dimension of the tensor, in a sigle number.
        #input_shape = input_tensor.size()
        perturbed_input_shape = torch.cat((torch.tensor([self.num_samples]), input_shape)) #The storage space size for all perturbed sequence.

        noises = perturbations.sample_noise_with_gradients(self.noise, perturbed_input_shape) #Create noises
        additive_noise, noise_gradient = tuple([noise for noise in noises]) #Cast the noise to make the noise and input tensor operable
        perturbed_input = torch.unsqueeze(input_tensor, 0) + self.sigma * additive_noise #Add noises to the input tensor, if the expansion in dimension necessary tho?

        # [N, B, D1, ..., Dk] -> [NB, D1, ..., Dk].
        flat_batch_dim_shape = torch.cat((torch.tensor([-1]), input_shape[1:]))
        perturbed_input = torch.reshape(perturbed_input, tuple(flat_batch_dim_shape))
        # Calls user-defined function in a perturbation agnostic manner.
        perturbed_output = self.func(perturbed_input)
        # [NB, D1, ..., Dk] ->  [N, B, D1, ..., Dk].
        perturbed_input = torch.reshape(perturbed_input, tuple(perturbed_input_shape))
        # Either
        #   (Default case): [NB, D1, ..., Dk] ->  [N, B, D1, ..., Dk]
        # or
        #   (Full-reduce case) [NB] -> [N, B]
        perturbed_output_shape = torch.cat(
        (torch.tensor([self.num_samples]), torch.tensor([-1]), torch.tensor(perturbed_output.size()[1:])))
        perturbed_output = torch.reshape(perturbed_output, tuple(perturbed_output_shape))
        forward_output = torch.mean(perturbed_output,0)

        if not self.batched:
            forward_output = forward_output[0]
        self.original_input_shape = original_input_shape
        self.perturbed_input = perturbed_input
        self.noise_gradient = noise_gradient
        self.perturbed_output = perturbed_output
        return forward_output

    def backward(self,grad_output):
        if len(self.perturbed_input.size()) > len(self.perturbed_output.size()):
            self.grad_output = torch.unsqueeze(self.grad_output,-1)
            self.perturbed_output = torch.unsqueeze(self.perturbed_output,-1)
        if not self.batched:
            grad_output = torch.unsqueeze(grad_output,0)
        flatten = lambda t: torch.reshape(t, (t.size()[0], t.size()[1], -1))
        grad_output = torch.reshape(grad_output,(grad_output.size()[0],-1))
        self.perturbed_output = flatten(self.perturbed_output)
        noise_grad = flatten(self.noise_gradient)
        g = torch.einsum('nbi,nbj->nij', self.perturbed_output, self.noise_gradient)
        g = torch.mean(g,0)
        g /= self.sigma
        return g

#Objective function and its partial derivatives
def objective(t,x,u):
    return -t*x@u-np.log(u[0]-1)-np.log(30.-u[1]-u[0])-np.log(u[1])
def objective_dy(t,x,u):
    return np.array([-t*x[0]-1/(u[0]-1)+1/(30-u[0]-u[1]),-t*x[1]-1/(u[1])+1/(30-u[0]-u[1])])
def objective_dxy(t,x,u):
     return np.matrix([[t,0],\
                      [0,t]])
def objective_dyy(x,u):
    return np.matrix([[np.power(u[0]-1,-2)+np.power(30-u[0]-u[1],-2),np.power(30-u[0]-u[1],-2)],\
                      [np.power(30-u[0]-u[1],-2),np.power(30-u[0]-u[1],-2)+1/np.power(u[1],2)]])
def objective_gradient(t,x,u):
    return -np.linalg.inv(objective_dyy(x,u))*objective_dxy(t,x,u)

def plot_LP_1_BF(t):
    u = cp.Variable(2)
    x = np.array([3.,-4.])
    x_data = -1*np.linspace(1,8,200)
    u_data0 = []
    u_data1 = []
    gradient0 = []
    gradient1 = []

    time_cost_solution = []
    time_cost_gradient = []
    for x[0] in x_data:
        #Objective function of barriered problem
        objec = lambda u: cp.Minimize(t*x@u-cp.log(u[0]-1)-cp.log(30.-u[1]-u[0])-cp.log(u[1]))
        #Solving the problem with CVXPY
        prob = cp.Problem(objec(u))
        start = time.time()
        prob.solve()
        end = time.time()
        time_cost_solution.append(end-start)
        u_data0.append(u.value[0])
        u_data1.append(u.value[1])
        #Determine the gradient of the solution with the derived equation
        start = time.time()
        gradient = objective_gradient(t,x,u.value)
        end = time.time()
        time_cost_gradient.append(end-start)
        gradient0.append(gradient[0,0])
        gradient1.append(gradient[1,0])
    plot1 = plt.figure(1)
    plt.plot(x_data,u_data0,label = 't = %s'%t)
    plt.title('$x_1$ against $\Theta$')
    plt.legend()
    plt.xlabel('$\Theta$')
    plt.ylabel('$x_1$')
    plt.savefig('../Figures/LP_sample1_BF_x1_sol.png')
    plot2 = plt.figure(2)
    plt.plot(x_data,u_data1,label = 't = %s' %t)
    plt.title('$x_2$ against $\Theta$')
    plt.legend()
    plt.xlabel('$\Theta$')
    plt.ylabel('$x_2$')
    plt.savefig('../Figures/LP_sample1_BF_x2_sol.png')
    plot3 = plt.figure(3)
    plt.plot(x_data,gradient0,label = 't = %s'%t)
    plt.title('Derivative of $x_1$ against $\Theta$')
    plt.xlabel('$\Theta$')
    plt.ylabel('$\partial x_1 / \partial \Theta$')
    plt.legend()
    plt.savefig('../Figures/LP_sample1_BF_x1_grad.png')
    plot4 = plt.figure(4)
    plt.plot(x_data,gradient1,label = 't = %s'%t)
    plt.legend()
    plt.title('Derivative of $x_2$ against $\Theta$')
    plt.xlabel('$\Theta$')
    plt.ylabel('$\partial x_2 / \partial \Theta$')
    plt.savefig('../Figures/LP_sample1_BF_x2_grad.png')
    print('Avg. time cost for solution with t = {} is {} seconds'.format(t,np.mean(time_cost_solution)))
    print('Avg. time cost for gradient with t = {} is {} seconds'.format(t,np.mean(time_cost_gradient)))


def Sample_LinProg1(x):
#     x = x.clone().detach().numpy()
    solution = torch.zeros_like(x)
    for i in range(x.shape[0]):
        u = cp.Variable(2)
        coe = x[i].clone().detach().numpy()
        objec = lambda coe,u: cp.Minimize(coe@u)
        constraints = [u[0] >= 1,u[0]+u[1] <= 30, u>=0]
        prob = cp.Problem(objec(coe,u),constraints)
        prob.solve()
        solution[i] = torch.tensor(u.value)
    return solution

def plot_LP_1_PO(sample_size,epsilon,noise_type,p_int):
    parameter_name = ['sample_size','epsilon','noise_type']
    parameter_val = [sample_size,epsilon,noise_type]
    p_int_name = parameter_name[p_int]
    p_int_val = parameter_val[p_int]

    perturbed_Sample_Linprog1 = perturbations(Sample_LinProg1,sample_size,epsilon,noise_type,False)
    x = torch.tensor([-3.,-4.])
    x_data = -1*np.linspace(1,8,20)
    time_cost_solution = []
    time_cost_gradient = []
    solution = np.zeros((len(x_data),2))
    gradient = np.zeros_like(solution)
    dy = torch.tensor([1.,1.])
    for i in range(len(x_data)):
        x[0] = x_data[i]
        x_new = x.clone().detach()
        start = time.time()
        pert_output  = perturbed_Sample_Linprog1.forward(x_new)
        end = time.time()
        time_cost_solution.append(end-start)
        start = time.time()
        x_gradient = perturbed_Sample_Linprog1.backward(dy)
        end = time.time()
        time_cost_gradient.append(end-start)
        solution[i] = np.array(pert_output)
        gradient[i] = [x_gradient[0,0],x_gradient[1,0]]
    plot1 = plt.figure(5)
    plt.plot(x_data,solution[:,0],label = '{} = {}'.format(p_int_name,p_int_val))
    plt.title('$x_1$ against $\Theta$')
    plt.legend()
    plt.xlabel('$\Theta$')
    plt.ylabel('$x_1$')

    plt.savefig('../Figures/LP_sample1_PO_x1_sol_{}_{}.png'.format(p_int_name,noise_type))
    plot2 = plt.figure(6)
    plt.plot(x_data,solution[:,1],label = '{} = {}'.format(p_int_name,p_int_val))
    plt.title('$x_2$ against $\Theta$')
    plt.legend()
    plt.xlabel('$\Theta$')
    plt.ylabel('$x_2$')
    plt.savefig('../Figures/LP_sample1_PO_x2_sol_{}_{}.png'.format(p_int_name,noise_type))
    plot3 = plt.figure(7)
    plt.plot(x_data,gradient[:,0],label = '{} = {}'.format(p_int_name,p_int_val))
    plt.title('Derivative of $x_1$ against $\Theta$')
    plt.xlabel('$\Theta$')
    plt.ylabel('$\partial x_1 / \partial \Theta$')
    plt.legend()
    plt.savefig('../Figures/LP_sample1_PO_x1_grad_{}_{}.png'.format(p_int_name,noise_type))
    plot4 = plt.figure(8)
    plt.plot(x_data,gradient[:,1],label = '{} = {}'.format(p_int_name,p_int_val))
    plt.legend()
    plt.title('Derivative of $x_2$ against $\Theta$')
    plt.xlabel('$\Theta$')
    plt.ylabel('$\partial x_2 / \partial \Theta$')
    plt.savefig('../Figures/LP_sample1_PO_x2_grad_{}_{}.png'.format(p_int_name,noise_type))
    print('Avg. time cost for solution with {} = {} is {} seconds'.format(p_int_name,p_int_val,np.mean(time_cost_solution)))
    print('Avg. time cost for gradient with {} = {} is {} seconds'.format(p_int_name,p_int_val,np.mean(time_cost_gradient)))

noise_g = 'gumbel'
noise_n = 'normal'
sz_val = np.array([20,200,2000,20000])
epsilon_val = np.array([0.1,0.5,0.7,1,2,5])
t_value = np.logspace(-3,0,4)
plt.close('all')
for t in t_value:
    plot_LP_1_BF(t)

plt.close('all')

sample_size = 2000
epsilon = 0.5
for epsilon in epsilon_val:
    plot_LP_1_PO(sample_size,epsilon,noise_g,1)

plt.close('all')

sample_size = 2000
epsilon = 0.5
for epsilon in epsilon_val:
    plot_LP_1_PO(sample_size,epsilon,noise_n,1)

plt.close('all')

sample_size = 2000
epsilon = 0.5
for sample_size in sz_val:
    plot_LP_1_PO(sample_size,epsilon,noise_g,0)

plt.close('all')
sample_size = 2000
epsilon = 0.5
for sample_size in sz_val:
    plot_LP_1_PO(sample_size,epsilon,noise_n,0)

