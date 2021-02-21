import os
import torch as th
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
import pickle

#TODO:
# - Better to predict logvar or logstd?
# - Learn logvar or keep it constant?
# - Holdout loss: best ratio? save best checkpoint in epoch? individual improvement?

class EnsembleLayer(nn.Module):

    def __init__(self, ensemble_size, input_dim, output_dim):
        super().__init__()
        self.W = nn.Parameter(th.empty((ensemble_size, input_dim, output_dim)), requires_grad=True).float()
        nn.init.xavier_uniform_(self.W, gain=nn.init.calculate_gain('relu'))
        self.b = nn.Parameter(th.zeros((ensemble_size, 1, output_dim)), requires_grad=True).float()

    def forward(self, x):
        # assumes x is 3D: (ensemble_size, batch_size, dimension)
        return x @ self.W + self.b

class ProbabilisticEnsemble(nn.Module):

    def __init__(self, input_dim, output_dim, ensemble_size=5, arch=(200,200,200,200), activation=F.relu, learning_rate=0.001, num_elites=2, device='auto'):
        super().__init__()

        self.ensemble_size = ensemble_size
        self.input_dim = input_dim
        self.output_dim = output_dim * 2 # mean and std
        self.activation = activation
        self.arch = arch
        self.num_elites = num_elites
        self.elites = [i for i in range(self.ensemble_size)]

        self.layers = nn.ModuleList()
        in_size = input_dim
        for hidden_size in self.arch:
            self.layers.append(EnsembleLayer(ensemble_size, in_size, hidden_size))
            in_size = hidden_size
        self.layers.append(EnsembleLayer(ensemble_size, self.arch[-1], self.output_dim))

        self.inputs_mu = nn.Parameter(th.zeros(input_dim), requires_grad=False).float()
        self.inputs_sigma = nn.Parameter(th.zeros(input_dim), requires_grad=False).float()

        self.max_logvar = nn.Parameter(th.ones(1, output_dim, dtype=th.float32) / 2.0).float()
        self.min_logvar = nn.Parameter(-th.ones(1, output_dim, dtype=th.float32) * 10.0).float()

        self.decays = [0.000025, 0.00005, 0.000075, 0.000075, 0.0001]
        self.optim = th.optim.Adam([{'params': self.layers[i].parameters(), 'weight_decay': self.decays[i]} for i in range(len(self.layers))] +
                                   [{'params': self.max_logvar}, {'params': self.min_logvar}], lr=learning_rate)
        if device == 'auto':
            self.device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')
        else:
            self.device = device
        self.to(self.device)

    def forward(self, input, deterministic=False, return_dist=False):
        dim = len(input.shape)
        # input normalization
        h = (input - self.inputs_mu) / self.inputs_sigma
        # repeat h to make amenable to parallelization
        # if dim = 3, then we probably already did this somewhere else (e.g. bootstrapping in training optimization)
        if dim < 3:
            h = h.unsqueeze(0)
            if dim == 1:
                h = h.unsqueeze(0)
            h = h.repeat(self.ensemble_size, 1, 1)

        for layer in self.layers[:-1]:
            h = layer(h)
            h = self.activation(h)
        output = self.layers[-1](h)

        # if original dim was 1D, squeeze the extra created layer
        if dim == 1:
            output = output.squeeze(1) # output is (ensemble_size, output_size)

        mean, logvar = th.chunk(output, 2, dim=-1)

        # Variance clamping to prevent poor numerical predictions
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if deterministic:
            if return_dist:
                return mean, logvar
            else:
                return mean
        else:
            std = th.sqrt(th.exp(logvar))
            samples = mean + std * th.randn(std.shape, device=std.device)
            if return_dist:
                return samples, mean, logvar
            else:
                return samples

    def compute_loss(self, x, y):
        mean, logvar = self.forward(x, deterministic=True, return_dist=True)
        inv_var = th.exp(-logvar)

        if len(y.shape) < 3:
            y = y.unsqueeze(0).repeat(self.ensemble_size, 1, 1)
        
        mse_losses = (th.square(mean - y) * inv_var).mean(-1).mean(-1)
        var_losses = logvar.mean(-1).mean(-1)
        total_losses = (mse_losses + var_losses).sum()
        total_losses += 0.01*self.max_logvar.sum() - 0.01*self.min_logvar.sum()
        return total_losses

    def compute_mse_losses(self, x, y):
        mean = self.forward(x, deterministic=True, return_dist=False)
        if len(y.shape) < 3:
            y = y.unsqueeze(0).repeat(self.ensemble_size, 1, 1)
        mse_losses = (mean - y)**2
        return mse_losses.mean(-1).mean(-1)
    
    def save(self, path):
        save_dir = 'weights/'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        th.save({'ensemble_state_dict': self.state_dict(),
                 'ensemble_optimizer_state_dict': self.optim.state_dict()}, path + '.tar')

    def load(self, path):
        params = th.load(path)
        self.load_state_dict(params['ensemble_state_dict'])
        self.optim.load_state_dict(params['ensemble_optimizer_state_dict'])

    def fit_input_stats(self, data):
        mu = np.mean(data, axis=0, keepdims=True)
        sigma = np.std(data, axis=0, keepdims=True)
        sigma[sigma < 1e-12] = 1.0
        self.inputs_mu.data = th.from_numpy(mu).to(self.device).float() # Can I ommit .data?
        self.inputs_sigma.data = th.from_numpy(sigma).to(self.device).float()

    def train_ensemble(self, X, Y, batch_size=256, holdout_ratio=0.1, max_holdout_size=5000, max_epochs_no_improvement=5, max_epochs=200):
        self.fit_input_stats(X)
        
        num_holdout = min(int(X.shape[0] * holdout_ratio), max_holdout_size)
        permutation = np.random.permutation(X.shape[0])
        inputs, holdout_inputs = X[permutation[num_holdout:]], X[permutation[:num_holdout]]
        targets, holdout_targets = Y[permutation[num_holdout:]], Y[permutation[:num_holdout]]
        holdout_inputs = th.from_numpy(holdout_inputs).to(self.device).float()
        holdout_targets = th.from_numpy(holdout_targets).to(self.device).float()

        idxs = np.random.randint(inputs.shape[0], size=[self.ensemble_size, inputs.shape[0]])
        num_batches = int(np.ceil(idxs.shape[-1] / batch_size))

        def shuffle_rows(arr):
            idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
            return arr[np.arange(arr.shape[0])[:, None], idxs]

        num_epochs_no_improvement = 0
        epoch = 0
        best_holdout_losses = [float('inf') for _ in range(self.ensemble_size)]
        while num_epochs_no_improvement < max_epochs_no_improvement and epoch < max_epochs:
            self.train()
            for batch_num in range(num_batches):
                batch_idxs = idxs[:, batch_num * batch_size : (batch_num + 1) * batch_size]
                batch_x, batch_y = inputs[batch_idxs], targets[batch_idxs]
                batch_x, batch_y = th.from_numpy(batch_x).to(self.device).float(), th.from_numpy(batch_y).to(self.device).float()

                loss = self.compute_loss(batch_x, batch_y)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            idxs = shuffle_rows(idxs)

            self.eval()
            with th.no_grad():
                holdout_losses = self.compute_mse_losses(holdout_inputs, holdout_targets)
            holdout_losses = [l.item() for l in holdout_losses]
            #print('Epoch:', epoch, 'Holdout losses:', [l.item() for l in holdout_losses])

            self.elites = np.argsort(holdout_losses)[:self.num_elites]

            improved = False
            for i in range(self.ensemble_size):
                if epoch == 0 or (best_holdout_losses[i] - holdout_losses[i]) / (best_holdout_losses[i]) > 0.01:
                    best_holdout_losses[i] = holdout_losses[i]
                    num_epochs_no_improvement = 0
                    improved = True
            if not improved:
                num_epochs_no_improvement += 1

            epoch += 1

        print('Epoch:', epoch, 'Holdout losses:', ', '.join(["%.4f"%hl for hl in holdout_losses]))
        return np.mean(holdout_losses)

if __name__ == '__main__':

    with open('/home/lucas/Desktop/drl-cd/weights/drlcd-cheetah-ns-paper1data0', 'rb') as f:
        memory = pickle.load(f)
    X, Y = memory.to_train_batch()

    model = ProbabilisticEnsemble(X.shape[1], Y.shape[1])
    model.train_ensemble(X, Y, max_epochs=200)