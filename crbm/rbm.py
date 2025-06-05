import numpy as np
from scipy.special import expit   # sigmoid

class CRBM():
    def __init__(self, v_dim=28*28, h_dim=100):
        self.v_dim = v_dim
        self.h_dim = h_dim
        # Parameters
        self.W = 0.1 * np.random.randn(v_dim, h_dim)
        self.vbias = np.zeros(v_dim)
        self.hbias = -4.0 * np.ones(h_dim)
        # Gradients
        self.dW = np.zeros(self.W.shape)
        self.dvbias = np.zeros(v_dim)
        self.dhbias = np.zeros(h_dim)
        # Velocities - for momentum
        self.W_inertia = np.zeros(self.W.shape)
        self.vbias_inertia = np.zeros(v_dim)
        self.hbias_inertia = np.zeros(h_dim)
    
    def get_hidden(self, v):
        '''
        input:
            - v: (batch_size, v_dim)
        output:
            - p(H|v) = sigmoid(W^Tv + hbias): (batch_size, h_dim)
            - samples from p(H|v): (batch_size, h_dim)
        '''
        p = expit(np.matmul(v, self.W) + self.hbias)
        return (p, np.random.binomial(1, p=p))
    
    def get_visible(self, h):
        '''
        input:
            - h: (batch_size, h_dim)
        output:
            - p(V|h) = sigmoid(Wh + vbias): (batch_size, v_dim)
            - samples from p(V|h): (batch_size, v_dim)
        '''
        p = expit(np.matmul(h, self.W.T) + self.vbias)
        return (p, np.random.binomial(1, p=p))
    
    def compute_grad(self, batch):
        '''
        Function to compute the gradient of parameters and store in param_grad variables
        and get_reconstructionion error.
        input:
            - batch: (batch_size, v_dim)
            - burn_in: Number of burn in steps for Gibbs sampling
            - num_steps: Number of steps for Gibbs sampling chain to run
            - method: Method for computing gradients. Available options:
                    - "cd": Contrastive Divergence
        output:
            - recon_error: get_reconstructionion error
        '''
        b_size = batch.shape[0]
        v0 = batch.reshape(b_size, -1)
        
        # Compute gradients - Positive Phase
        ph0, h0 = self.get_hidden(v0)
        dW = np.matmul(v0.T, ph0)
        dvbias = np.sum(v0, axis=0)
        dhbias = np.sum(ph0, axis=0)
        
        # Compute gradients - Negative Phase
        
        # only contrastive with k = 1, i.e., method="cd"

        pv1, v1 = self.get_visible(h0)
        ph1, h1 = self.get_hidden(pv1)
        
        dW -= np.matmul(pv1.T, ph1)
        dvbias -= np.sum(pv1, axis=0)
        dhbias -= np.sum(ph1, axis=0)
        
        self.dW = dW/b_size
        self.dhbias = dhbias/b_size
        self.dvbias = dvbias/b_size
        
        recon_err = np.mean(np.sum((v0 - pv1)**2, axis=1), axis=0) # sum of squared error averaged over the batch
        return recon_err
    
    def update(self, lr, momentum=0):
        '''
        Function to update the parameters based on the stored gradients.
        input:
            - lr: Learning rate
            - momentum
        '''
        self.W_inertia *= momentum
        self.W_inertia += (1.-momentum) * lr * self.dW
        self.W += self.W_inertia
        
        self.vbias_inertia *= momentum
        self.vbias_inertia += (1.-momentum) * lr * self.dvbias
        self.vbias += self.vbias_inertia
        
        self.hbias_inertia *= momentum
        self.hbias_inertia += (1.-momentum) * lr * self.dhbias
        self.hbias += self.hbias_inertia
        
    def get_reconstruction(self, v):
        '''
        get_reconstructioning visible units from given v.
        v -> h0 -> v1
        input:
            - v: (batch_size, v_dim)
        output:
            - prob of get_reconstructioned v: (batch_size, v_dim)
        '''
        ph0, h0 = self.get_hidden(v)
        pv1, v1 = self.get_visible(ph0)
        return pv1
    
    def free_energy(self, v):
        '''
        Compute the free energy of v averaged over the batch.
        input:
            - v: (batch_size, v_dim)
        output:
            - average of free energy: where free energy = - v.vbias - Sum_j (log(1 + exp(hbias + v_j*W_:,j)) )
        '''
        x = self.hbias + np.matmul(v, self.W)
        free_energy_batch = -np.matmul(v, self.vbias) - np.sum(np.log(1 + np.exp(x)), axis=1)
        return np.mean(free_energy_batch)
    
    def sample(self, start=None, num_iters=1000):
        '''
        Generate random samples of visible unit from the model using Gibbs sampling.
        input:
            - start: Any starting value of v.
            - num_iters: Number of iterations of Gibbs sampling.
        '''
        if(start is None):
            v = np.random.randn(self.v_dim)
        else:
            v = start
        for _ in range(num_iters):
            ph, h = self.get_hidden(v)
            pv, v = self.get_visible(h)
        return v