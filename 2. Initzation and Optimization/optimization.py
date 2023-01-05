import torch

class OptimizerTemplate:
    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        # Set gradients of all parameters to zero
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()  # For second-order optimizers important
                p.grad.zero_()
    
    @torch.no_grad()
    def step(self):
        # Apply update step to all parameters
        for p in self.params:
            if p.grad is None:  # We skip parameters without any gradients
                continue
            self.update_param(p)
    def update_param(self, p):
        # To be implemented in optimizer-specific classes
        raise NotImplementedError



class SGD(OptimizerTemplate):
    """
        w[t]  =  w[t-1] - lr.g[t]
    """
    def __init__(self, params, lr):
        super().__init__(params, lr)
    
    def update_param(self, p):
        p_update = -self.lr * p.grad
        p.add_(p_update)
    
class SGDMomentum(OptimizerTemplate):
    """
        m[t] = b1.m[t-1] + (1-b1).g[t]
        w[t] = w[t-1] - lr.m[t]
    """
    def __init__(self, params, lr, momentum=0.0):
        super().__init__(params, lr)
        self.momentum = momentum  # Corresponds to beta_1 in the equation above
        self.param_momentum = {p: torch.zeros_like(p.data) for p in self.params}  # Dict to store m_t
    
    def update_param(self, p):
        self.param_momentum[p] = (1-self.momentum)*p.grad + self.momentum*self.param_momentum[p]
        p_update = -self.lr*self.param_momentum[p]
        p.add_(p_update)

class Adam(OptimizerTemplate):
    def __init__(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.param_step = {p: 0 for p in self.params}  # Remembers "t" for each parameter for bias correction
        self.param_momentum = {p: torch.zeros_like(p.data) for p in self.params}
        self.param_2nd_momentum = {p: torch.zeros_like(p.data) for p in self.params}
    
    def update_param(self, p):
        self.param_step[p] += 1
        self.param_momentum[p] = (1 - self.beta1) * p.grad + self.beta1 * self.param_momentum[p]
        self.param_2nd_momentum[p] = (1 - self.beta2) * (p.grad) ** 2 + self.beta2 * self.param_2nd_momentum[p]
        
        bias_correction_1 = 1 - self.beta1 ** self.param_step[p]
        bias_correction_2 = 1 - self.beta2 ** self.param_step[p]

        p_2nd_mom = self.param_2nd_momentum[p] / bias_correction_2
        p_mom = self.param_momentum[p] / bias_correction_1
        p_lr = self.lr / (torch.sqrt(p_2nd_mom) + self.eps)

        p_update = -p_lr * p_mom
        p.add_(p_update)

#How to use
# model = Model()
# initializer(model)
# optimizer = Adam(model.parameters(), lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8)