import numpy as np
from scipy.special import expit   # sigmoid

class RestrictedBM():
    def __init__(self, num_visible=28*28, num_hidden=100):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        # Initialize parameters
        self.weights = 0.1 * np.random.randn(num_visible, num_hidden)
        self.vbias = np.zeros(num_visible)
        self.hbias = -4.0 * np.ones(num_hidden)
        # Initialize gradients
        self.weightgrad = np.zeros(self.weights.shape)
        self.vbiasgrad = np.zeros(num_visible)
        self.hbiasgrad = np.zeros(num_hidden)
        # Initialize velocities for momentum
        self.weightinertia = np.zeros(self.weights.shape)
        self.vbiasinertia = np.zeros(num_visible)
        self.hbiasinertia = np.zeros(num_hidden)
    

    def generate_reconstruction(self, visible_input):
        '''
        Make the reconstructed visible units from the given visible input.
        
        input:
            - visible_input: (batch_size, num_visible)
        
        output:
            - reconstructed_visible_prob: Probability of reconstructed visible units, shape (batch_size, num_visible)
        '''
        hidden_prob, _ = self.hidden_given_visible(visible_input)
        reconstructed_visible_prob, _ = self.visible_given_hidden(hidden_prob)
        return reconstructed_visible_prob 


    def hidden_given_visible(self, visible_input):
        '''
        Compute the probability of hidden units given visible units and sample from it.
        
        input:
            - visible_input: (batch_size, num_visible)
        
        output:
            - hidden_prob: Probability of hidden units, sigmoid(weights^T * visible_input + hidden_bias), shape (batch_size, num_hidden)
            - hidden_sample: Binary samples from hidden_prob, shape (batch_size, num_hidden)
        '''
        hidden_prob = expit(np.matmul(visible_input, self.weights) + self.hbias)
        hidden_sample = np.random.binomial(1, p=hidden_prob)
        return (hidden_prob, hidden_sample)
    
    def visible_given_hidden(self, hidden_input):
        '''
        Compute the probability of visible units given hidden units and sample from it.
        
        input:
            - hidden_input: (batch_size, num_hidden)
        
        output:
            - visible_prob: Probability of visible units, sigmoid(weights * hidden_input + visible_bias), shape (batch_size, num_visible)
            - visible_sample: Binary samples from visible_prob, shape (batch_size, num_visible)
        '''
        visible_prob = expit(np.matmul(hidden_input, self.weights.T) + self.vbias)
        visible_sample = np.random.binomial(1, p=visible_prob)
        return (visible_prob, visible_sample)
    
    def apply_parameter_updates(self, lr, momentum_factor=0):
        '''
        Update the model parameters using the computed gradients and momentum.
        
        input:
            - lr: Step size for parameter updates
            - momentum_factor: Factor for momentum term
        '''
        
        self.vbiasinertia *= momentum_factor*self.vbiasinertia + lr * (1. - momentum_factor) * self.vbiasgrad
        self.vbias += self.vbiasinertia

        self.weightinertia = momentum_factor * self.weightinertia + lr * (1. - momentum_factor) * self.weightgrad
        self.weights += self.weightinertia
        
        self.hbiasinertia *= momentum_factor * self.hbiasinertia + lr * (1. - momentum_factor) * self.hbiasgrad
        self.hbias += self.hbiasinertia
    

    def sample_from_model(self, initial_visible=None, gibbs_iterations=1000):
        '''
        Generate a sample from the model using Gibbs sampling.
        
        input:
            - initial_visible: Starting visible units, if None, initialize randomly
            - gibbs_iterations: Number of Gibbs sampling steps to perform
        
        output:
            - visible_sample: Sampled visible units after gibbs_iterations steps
        '''
        if initial_visible is None:
            visible_sample = np.random.randn(self.num_visible)
        else:
            visible_sample = initial_visible
        for _ in range(gibbs_iterations):
            _, hidden_sample = self.hidden_given_visible(visible_sample[None, :])  # Add batch dimension
            _, visible_sample = self.visible_given_hidden(hidden_sample)
            visible_sample = visible_sample[0]  # Remove batch dimension
        return visible_sample
    

    def compute_average_free_energy(self, visible_units):
        '''
        Compute the average free energy of the given visible units over the batch.
        
        input:
            - visible_units: (batch_size, num_visible)
        
        output:
            - average_free_energy: Scalar value of the mean free energy
        '''
        hidden_activation = self.hbias + np.matmul(visible_units, self.weights)
        free_energy_per_sample = -np.matmul(visible_units, self.vbias) - np.sum(np.log(1 + np.exp(hidden_activation)), axis=1)
        average_free_energy = np.mean(free_energy_per_sample)
        return average_free_energy



    def compute_gradients_and_error(self, data_batch):
        '''
        Calculate the gradients of the parameters and the reconstruction error using the specified learning method.
        
        input:
            - data_batch: (batch_size, num_visible)
        '''
        batch_size = data_batch.shape[0]
        v0 = data_batch.reshape(batch_size, -1)
        
        hp0, hidden_sample_initial = self.hidden_given_visible(v0)
        wgrad_pos = np.matmul(v0.T, hp0)
        vbgrad_pos = np.sum(v0, axis=0)
        hbgrad_pos = np.sum(hp0, axis=0)
        
        vprecon, _ = self.visible_given_hidden(hidden_sample_initial)
        hprecon, _ = self.hidden_given_visible(vprecon)
        
        wgrad_neg = np.matmul(vprecon.T, hprecon)
        vbgrad_neg = np.sum(vprecon, axis=0)
        hbgrad_neg = np.sum(hprecon, axis=0)
        
        self.weightgrad = (wgrad_pos - wgrad_neg) / batch_size
        self.vbiasgrad = (vbgrad_pos - vbgrad_neg) / batch_size
        self.hbiasgrad = (hbgrad_pos - hbgrad_neg) / batch_size
        
        reconstruction_error = np.mean(np.sum((v0 - vprecon)**2, axis=1))
        return reconstruction_error
    
