import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


def mlp(sizes, activation, output_activation=nn.Identity, Glorot_init=True, Glorot_gain=1.0):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        if Glorot_init:
            L = nn.Linear(sizes[j], sizes[j+1])
            nn.init.xavier_uniform_(L.weight,gain=Glorot_gain)
            # nn.init.xavier_uniform_(L.weight,gain=nn.init.calculate_gain('relu'))
            # nn.init.xavier_normal_(L.weight,gain=1.0)
            layers += [L, act()]
        else:
            layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def compute_loss_pi(learned_policy, data):
    o, a = data['sta'], data['act']
    a_pred = learned_policy(o)
    return torch.mean(torch.sum((a - a_pred)**2, dim=1))


class MLPActor(nn.Module):
    def __init__(self, sta_dim=1, act_dim=1, hidden_sizes=(256, 256), 
            activation=nn.Hardswish, act_scale=1.0, Glorot_init=True, Glorot_gain=1.0):
        super().__init__()
        nn_policy_sizes = [sta_dim] + list(hidden_sizes) + [act_dim]
        # self.nn_policy = mlp(sizes=nn_policy_sizes, activation=activation,
        #                      output_activation=nn.Tanh, Glorot_init=Glorot_init)
        # Due to action limit is +-1, output_activation is tanh
        self.nn_policy = mlp(sizes=nn_policy_sizes, activation=activation,
                             output_activation=nn.Identity, 
                             Glorot_init=Glorot_init,
                             Glorot_gain=Glorot_gain)

    def forward(self, sta):
        # Return output from network scaled to action space limits * 1.
        return self.nn_policy(sta) * 1.


class MLPClassifier(nn.Module):

    def __init__(self, sta_dim, act_dim, hidden_sizes, activation, device):
        super().__init__()
        nn_policy_sizes = [sta_dim] + list(hidden_sizes) + [act_dim]
        self.nn_policy = mlp(nn_policy_sizes, activation, output_activation=nn.Sigmoid)
        self.device = device

    def forward(self, sta):
        # Return output from network scaled to action space limits
        return self.nn_policy(sta).to(self.device)


class MLPQFunction(nn.Module):

    def __init__(self, sta_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([sta_dim + act_dim] + list(hidden_sizes) +
                     [1], activation, nn.Sigmoid)

    def forward(self, sta, act):
        q = self.q(torch.cat([sta, act], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.

# NN with ReplayBuffer ------------------------------------------------------------


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer.
    """

    def __init__(self, sta_dim, act_dim, size, device):
        self.sta_buf = np.zeros(combined_shape(
            size, sta_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(
            size, act_dim), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.device = device

    def store(self, sta, act):
        self.sta_buf[self.ptr] = sta
        self.act_buf[self.ptr] = act
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(sta=self.sta_buf[idxs],
                     act=self.act_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in batch.items()}

    def fill_buffer(self, sta, act):
        for i in range(len(sta)):
            self.store(sta[i], act[i])

    def save_buffer(self, name='replay'):
        pickle.dump({'sta_buf': self.sta_buf, 'act_buf': self.act_buf,
                     'ptr': self.ptr, 'size': self.size}, open('{}_buffer.pkl'.format(name), 'wb'))
        print('buf size', self.size)

    def load_buffer(self, name='replay'):
        p = pickle.load(open('{}_buffer.pkl'.format(name), 'rb'))
        self.sta_buf = p['sta_buf']
        self.act_buf = p['act_buf']
        self.ptr = p['ptr']
        self.size = p['size']

    def clear(self):
        self.ptr, self.size = 0, 0

# --------------------------------------------------------------------------

# NN with prior ------------------------------------------------------------


class SinPrior(nn.Module):
    def forward(self, input):
        return torch.sin(3 * input)


class ModelWithPrior(nn.Module):
    def __init__(self,
                 base_model: nn.Module,
                 prior_model: nn.Module,
                 prior_scale: float = 0.1):
        super().__init__()
        self.base_model = base_model
        self.prior_model = prior_model
        self.prior_scale = prior_scale

    def forward(self, inputs):
        with torch.no_grad():
            prior_out = self.prior_model(inputs)
            prior_out = prior_out.detach()
        model_out = self.base_model(inputs)
        return model_out + (self.prior_scale * prior_out)
# NO more need
# -------------------------------------------------------------------------


class MLP(nn.Module):

    def __init__(self,
                 state_space, action_space, device, hidden_sizes=(256, 256),
                 activation=nn.ReLU):
        super().__init__()

        sta_dim = state_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = MLPActor(sta_dim, act_dim, hidden_sizes,
                           activation, act_limit).to(device)
        self.pi_safe = MLPClassifier(
            sta_dim, 1, (128, 128), activation, device).to(device)
        self.device = device

    def act(self, sta):
        sta = torch.as_tensor(sta, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.pi(sta).cpu().numpy()

    def classify(self, sta):
        sta = torch.as_tensor(sta, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.pi_safe(sta).cpu().numpy().squeeze()


class Ensemble(nn.Module):
    # Multiple policies
    def __init__(self, state_space, action_space, device, hidden_sizes=(256, 256),
                 activation=nn.ReLU, num_nets=5, Glorot_init=True, Glorot_gain=1.0, prior_scale=0.):
        super().__init__()

        # sta_dim = state_space.shape[1]
        # act_dim = action_space.shape[1]
        # act_limit = action_space.max()
        sta_dim = state_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]
        self.num_nets = num_nets
        self.device = device
        # build policy and value functions
        self.pis = [MLPActor(sta_dim, act_dim, hidden_sizes, activation,
                             act_limit, Glorot_init, Glorot_gain).to(device) for _ in range(num_nets)]
        self.q1 = MLPQFunction(
            sta_dim, act_dim, hidden_sizes, activation).to(device)
        self.q2 = MLPQFunction(
            sta_dim, act_dim, hidden_sizes, activation).to(device)
        self.pis_prior = [ModelWithPrior(base_model=self.pis[i], prior_model=SinPrior(
        ).to(device), prior_scale=prior_scale).to(device) for i in range(num_nets)]

    def act(self, sta, i=-1):
        sta = torch.as_tensor(sta, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if i >= 0:  # optionally, only use one of the nets.
                return self.pis[i](sta).cpu().numpy()
            vals = list()
            for pi in self.pis:
                vals.append(pi(sta).cpu().numpy())
            return np.mean(np.array(vals), axis=0)

    def variance(self, sta):
        sta = torch.as_tensor(sta, dtype=torch.float32, device=self.device)
        if len(sta.shape) == 1:
            sta = sta[None, :]
        with torch.no_grad():
            vals = list()
            for pi in self.pis:
                vals.append(pi(sta).cpu().numpy())
            return np.square(np.std(np.array(vals), axis=0)).mean(axis=1)

    def safety(self, sta, act):
        # closer to 1 indicates more safe.
        sta = torch.as_tensor(sta, dtype=torch.float32, device=self.device)
        act = torch.as_tensor(act, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return float(torch.min(self.q1(sta, act), self.q2(sta, act)).cpu().numpy())
