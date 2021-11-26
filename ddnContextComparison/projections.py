import torch
import numpy as np
import cvxpy as cp
from torch import distributions as dist

class LP_Stage2_SchemeA_barrier():
    @staticmethod
    def project(v):
        input_size = v.shape
        v = v.flatten(end_dim=-2)
        w = torch.zeros_like(v)
        t = 0.5
        for i in range(input_size[0]):
            sub_vec = v[i]
            u = cp.Variable(len(sub_vec))
            coe = sub_vec.cpu().clone().detach().numpy()
            objec = lambda coe,u: cp.Minimize(-t*coe@u - cp.sum(cp.log(u)) - cp.sum(cp.log(1-u)))
            prob = cp.Problem(objec(coe,u))
            prob.solve(warm_start= True, solver = cp.SCS)
            w[i] = torch.tensor(u.value)
        w = w.reshape(input_size)
        return w, None

    @staticmethod
    def gradient(grad_output, output, input):
        output_size = output.size()
        output = output.flatten(end_dim=-2)
        input = input.flatten(end_dim=-2)
        grad_output = grad_output.flatten(end_dim=-2)
        input_shape = input.shape
        grad_input = torch.zeros_like(grad_output)
        t = 0.5
        for i in range(input_shape[0]):
            input0 = input[i].detach().clone()
            input0.requires_grad = True
            input1 = output[i].detach().clone()
            input1.requires_grad = True
            dxy = -t*torch.eye(len(input0))
            dyy = torch.diag(((input1)**(-2)+(1-input1)**(-2))**(-1))
            gradient = -torch.matmul(dyy,dxy)
            grad_input[i] = torch.matmul(grad_output[i],gradient)

        grad_input = grad_input.reshape(output_size)
        return grad_input


class LP_Stage2_SchemeB_barrier():
    @staticmethod
    def project(v):
        input_size = v.shape
        v = v.flatten(end_dim=-2)
        w = torch.zeros_like(v)
        t = 0.5
        for i in range(input_size[0]):
            sub_vec = v[i]
            alpha = np.floor(0.5*len(sub_vec))
            u = cp.Variable(len(sub_vec))
            coe = sub_vec.clone().detach().numpy()
            objec = lambda coe,u: cp.Minimize(t*coe@u - cp.sum(cp.log(u)) - cp.sum(cp.log(1-u)) - cp.log(cp.sum(u)-alpha))
            prob = cp.Problem(objec(coe,u))
            prob.solve(warm_start= True,solver=cp.SCS)
            w[i] = torch.tensor(u.value)
        w = w.reshape(input_size)
        return w, None

    @staticmethod
    def gradient(grad_output, output, input):
        output_size = output.size()
        output = output.flatten(end_dim=-2)
        input = input.flatten(end_dim=-2)
        grad_output = grad_output.flatten(end_dim=-2)
        input_shape = input.shape
        grad_input = torch.zeros_like(grad_output)
        t = 0.5
        alpha = np.floor(0.5*len(input[0]))
        for i in range(input_shape[0]):
            input0 = input[i].detach().clone()
            input0.requires_grad = True
            input1 = output[i].detach().clone()
            input1.requires_grad = True
            dxy = t*torch.eye(len(input0))
            dyy = torch.zeros_like(dxy)
            m,n = dyy.shape
            dyy = torch.ones([m,n])*(torch.sum(input1)-alpha)**(-2)+torch.diag((input1)**(-2)+(1-input1)**(-2))
            U_Matrix = torch.cholesky(dyy,upper=True)
            gradient = -torch.cholesky_solve(dxy,U_Matrix,upper = True)
            grad_input[i]=grad_output[i]@gradient
        grad_input = grad_input.reshape(output_size)
        return grad_input

class LP_Stage3_SchemeC_barrier():
    @staticmethod
    def project(v):
        input_size = v.shape
        v = v.flatten(end_dim=-2)
        w = torch.zeros_like(v)
        t = 0.5
        for i in range(input_size[0]):
            sub_vec = v[i]
            alpha = np.floor(0.5*len(sub_vec))
            u = cp.Variable(len(sub_vec))
            coe = sub_vec.clone().detach().numpy()
            objec = lambda coe,u: cp.Minimize(t*coe@u - cp.sum(cp.log(u+1)) - cp.sum(cp.log(1-u)))
            prob = cp.Problem(objec(coe,u))
            prob.solve(warm_start= True,solver=cp.SCS)
            w[i] = torch.tensor(u.value)
        w = w.reshape(input_size)
        return w, None

    @staticmethod
    def gradient(grad_output, output, input):
        output_size = output.size()
        output = output.flatten(end_dim=-2)
        input = input.flatten(end_dim=-2)
        grad_output = grad_output.flatten(end_dim=-2)
        input_shape = input.shape
        grad_input = torch.zeros_like(grad_output)
        t = 0.5
        alpha = np.floor(0.5*len(input[0]))
        for i in range(input_shape[0]):
            input0 = input[i].detach().clone()
            input0.requires_grad = True
            input1 = output[i].detach().clone()
            input1.requires_grad = True
            dxy = t*torch.eye(len(input0))
            dyy = torch.zeros_like(dxy)
            dyy_inverse = torch.diag(((input1+1)**(-2)+(1-input1)**(-2))**(-1))
            gradient = -dyy_inverse@dxy
            grad_input[i]=grad_output[i]@gradient

        grad_input = grad_input.reshape(output_size)
        return grad_input


class LP_Stage2_SchemeA_perturbed():
    @staticmethod
    def func(x):
        solution = torch.zeros_like(x)
        for i in range(x.shape[0]):
            u = cp.Variable(len(x[i]))
            coe = x[i].clone().detach().numpy()
            objec = lambda coe,u: cp.Minimize(-coe@u)
            constraints = [u >= 0,u <= 1]
            prob = cp.Problem(objec(coe,u),constraints)
            prob.solve(solver=cp.SCS)
            if u.value is None:
                solution[i] = solution[i]
            else:
                solution[i] = torch.tensor(u.value)
        return solution

    def __init__(self):
        self.num_samples = 200
        self.sigma = 0.5
        self.noise = 'gumbel'
        self.batched = True
        
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
        else: 
            input_tensor = torch.unsqueeze(input_tensor,0)
        input_shape = torch.tensor(input_tensor.size(),dtype=torch.int) 
        perturbed_input_shape = torch.cat((torch.tensor([self.num_samples]), input_shape)) 

        noises = LP_Stage2_SchemeA_perturbed.sample_noise_with_gradients(self.noise, perturbed_input_shape) 
        additive_noise, noise_gradient = tuple([noise for noise in noises]) 
        perturbed_input = torch.unsqueeze(input_tensor, 0) + self.sigma * additive_noise 
        flat_batch_dim_shape = torch.cat((torch.tensor([-1]), input_shape[1:]))
        perturbed_input = torch.reshape(perturbed_input, tuple(flat_batch_dim_shape))
        perturbed_output = self.func(perturbed_input)
        perturbed_input = torch.reshape(perturbed_input, tuple(perturbed_input_shape))
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
        g = torch.einsum('nbd,nb->bd', self.noise_gradient,
                      torch.einsum('nbd,bd->nb', self.perturbed_output, grad_output))
        g /= self.sigma * self.num_samples
        return torch.reshape(g, tuple(self.original_input_shape))

class LP_Stage2_SchemeB_perturbed():
    @staticmethod
    def func(x):
        solution = torch.zeros_like(x)
        for i in range(x.shape[0]):
            u = cp.Variable(len(x[i]))
            coe = x[i].clone().detach().numpy()
            objec = lambda coe,u: cp.Minimize(coe@u)
            alpha = cp.floor(0.5*len(x[i]))
            constraints = [u >= 0,u <= 1, cp.sum(u)>= alpha ]
            prob = cp.Problem(objec(coe,u),constraints)
            prob.solve(solver=cp.SCS)
            if u.value is None:
                solution[i] = solution[i]
            else:
                solution[i] = torch.tensor(u.value)
        return solution

    def __init__(self):
        self.num_samples = 200
        self.sigma = 0.5
        self.noise = 'gumbel'
        self.batched = True
        
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
        else: 
            input_tensor = torch.unsqueeze(input_tensor,0)
        input_shape = torch.tensor(input_tensor.size(),dtype=torch.int) 
        perturbed_input_shape = torch.cat((torch.tensor([self.num_samples]), input_shape)) 

        noises = LP_Stage2_SchemeA_perturbed.sample_noise_with_gradients(self.noise, perturbed_input_shape) 
        additive_noise, noise_gradient = tuple([noise for noise in noises]) 
        perturbed_input = torch.unsqueeze(input_tensor, 0) + self.sigma * additive_noise 
        flat_batch_dim_shape = torch.cat((torch.tensor([-1]), input_shape[1:]))
        perturbed_input = torch.reshape(perturbed_input, tuple(flat_batch_dim_shape))
        perturbed_output = self.func(perturbed_input)
        perturbed_input = torch.reshape(perturbed_input, tuple(perturbed_input_shape))
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
        g = torch.einsum('nbd,nb->bd', self.noise_gradient,
                      torch.einsum('nbd,bd->nb', self.perturbed_output, grad_output))
        g /= self.sigma * self.num_samples
        return torch.reshape(g, tuple(self.original_input_shape))

class LP_Stage3_SchemeC_Perturbed():
    @staticmethod
    def func(x):
        solution = torch.zeros_like(x)
        for i in range(x.shape[0]):
            u = cp.Variable(len(x[i]))
            coe = x[i].clone().detach().numpy()
            objec = lambda coe,u: cp.Minimize(coe@u)
            constraints = [u >= -1,u <= 1]
            prob = cp.Problem(objec(coe,u),constraints)
            prob.solve(solver=cp.SCS)
            if u.value is None:
                solution[i] = solution[i]
            else:
                solution[i] = torch.tensor(u.value)
        return solution

    def __init__(self):
        self.num_samples = 200
        self.sigma = 0.5
        self.noise = 'gumbel'
        self.batched = True
        
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
        input_shape = torch.tensor(input_tensor.size(),dtype=torch.int)
        perturbed_input_shape = torch.cat((torch.tensor([self.num_samples]), input_shape)) 
        noises = LP_Stage3_SchemeC_Perturbed.sample_noise_with_gradients(self.noise, perturbed_input_shape) #Create noises
        additive_noise, noise_gradient = tuple([noise for noise in noises]) 
        perturbed_input = torch.unsqueeze(input_tensor, 0) + self.sigma * additive_noise 
        flat_batch_dim_shape = torch.cat((torch.tensor([-1]), input_shape[1:]))
        perturbed_input = torch.reshape(perturbed_input, tuple(flat_batch_dim_shape))
        perturbed_output = self.func(perturbed_input)
        perturbed_input = torch.reshape(perturbed_input, tuple(perturbed_input_shape))
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
        g = torch.einsum('nbd,nb->bd', self.noise_gradient,
                      torch.einsum('nbd,bd->nb', self.perturbed_output, grad_output))
        g /= self.sigma * self.num_samples
        return torch.reshape(g, tuple(self.original_input_shape))


#### Projection modules
class EuclideanProjectionFn(torch.autograd.Function):
    """
    A function to project a set of features to an Lp-sphere or Lp-ball
    """
    @staticmethod
    def forward(ctx, input, method):
        output = method.project(input)[0]
        ctx.method = method
        ctx.save_for_backward(output.clone(), input.clone())
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, input= ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = ctx.method.gradient(grad_output, output, input)
        return grad_input, None, None

class EuclideanProjection(torch.nn.Module):
    def __init__(self, method):
        super(EuclideanProjection, self).__init__()
        self.method = method

    def forward(self, input):
        return EuclideanProjectionFn.apply(input,
                                           self.method,
                                           )
                            
class PerturbationFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, method):
        instance = method()
        output = instance.forward(input)
        ctx.method = instance
        ctx.save_for_backward(output.clone(), input.clone())
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, input = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = ctx.method.backward(grad_output)
        return grad_input, None, None

class PerturbationsNet(torch.nn.Module):
    def __init__(self, method):
        super(PerturbationsNet, self).__init__()
        self.method = method

    def forward(self, input):
        return PerturbationFn.apply(input,
 self.method
                                           )
